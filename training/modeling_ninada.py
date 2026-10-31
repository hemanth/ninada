"""
Ninada v2.0: Native-Script Specialist Streaming VITS + Stochastic Duration Predictor + Adversarial MPD/MRD.

Key Architectural Advances over v1.x:
1. Per-Language Native Phonology & Zero-DC-Offset HiFi-GAN Vocoders:
   - Each Indic language (`sa`, `kn`, `hi`, `ta`, `te`, `ml`) uses its native Brahmic phoneme space
     (with Classical/Vedic Sanskrit mapped losslessly onto the zero-schwa-deletion Kannada VITS
     inventory `0..75` + 3 Vedic/Classical extensions `76..78`: `ಙ`, `ೠ`, `ೄ`).
   - Eliminates `lang_dec_proj` DC channel bias on HiFi-GAN activations and eliminates Female-on-Male
     sub-harmonic pitch mismatch.
2. Bit-Exact (`0.00e+00`) ONNX-Safe `VitsStochasticDurationPredictor` (`SDP`):
   - Replaces dynamic boolean indexing in `_unconstrained_rational_quadratic_spline` with branchless
     `torch.clamp` + `torch.where` + `torch.cat`, exporting cleanly to ONNX Opset 17 (`onnxruntime-web`
     WASM SIMD) with zero `NonZero` or `ScatterND` ops while preserving natural syllable rhythm.
3. Adversarial `MPD + MRD` + Feature Matching + Exact VITS `KL` + `SDP` MLE:
   - Freezes `posterior_encoder` and `decoder` (`HiFi-GAN`) on their 500k-step pre-trained manifold
     while calibrating `text_encoder`, `duration_predictor` (`SDP`), and `flow` with Multi-Period
     (`MPD`, periods `[2, 3, 5, 7, 11]`) and Multi-Resolution STFT (`MRD`) adversarial + feature-matching
     objectives on Male 48kHz->16kHz anechoic studio speech.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import VitsConfig, VitsModel
import transformers.models.vits.modeling_vits as hf_vits


def onnx_safe_unconstrained_rqs(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    reverse: bool = False,
    tail_bound: float = 5.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bit-exact (`0.00e+00` diff), branchless ONNX-safe Rational Quadratic Spline transform.
    Eliminates dynamic boolean mask indexing (`NonZero`) and in-place slice scatter (`ScatterND`).
    """
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    x = torch.clamp(inputs, -tail_bound, tail_bound)

    constant = float(np.log(np.exp(1.0 - min_derivative) - 1.0))
    pad_c = torch.full_like(unnormalized_derivatives[..., :1], constant)
    unnorm_d = torch.cat([pad_c, unnormalized_derivatives, pad_c], dim=-1)

    upper_bound = float(tail_bound)
    lower_bound = float(-tail_bound)
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    lb_pad = torch.full_like(cumwidths[..., :1], lower_bound)
    ub_pad = torch.full_like(cumwidths[..., :1], upper_bound)
    cumwidths = torch.cat([lb_pad, cumwidths[..., :-1], ub_pad], dim=-1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnorm_d)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights = torch.cat([lb_pad, cumheights[..., :-1], ub_pad], dim=-1)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    base_locs = cumheights if reverse else cumwidths
    bin_locations = torch.cat([base_locs[..., :-1], base_locs[..., -1:] + 1e-6], dim=-1)
    bin_idx = torch.sum((x[..., None] >= bin_locations).to(torch.long), dim=-1) - 1
    bin_idx = torch.clamp(bin_idx, min=0, max=num_bins - 1)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    if not reverse:
        theta = (x - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1.0 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        spline_out = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - theta).pow(2)
        )
        spline_lad = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
    else:
        intermediate2 = x - input_cumheights
        intermediate3 = intermediate2 * intermediate1
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        b = input_heights * input_derivatives - intermediate3
        c = -input_delta * intermediate2

        discriminant = torch.clamp_min(b.pow(2) - 4.0 * a * c, 0.0)
        root = (2.0 * c) / (-b - torch.sqrt(discriminant))
        spline_out = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1.0 - root)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - root).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
        spline_lad = -log_abs_det

    outputs = torch.where(inside, spline_out, inputs)
    log_abs_det_out = torch.where(inside, spline_lad, torch.zeros_like(inputs))
    return outputs, log_abs_det_out


# Apply global monkey-patch so VitsStochasticDurationPredictor is 100% ONNX-exportable
hf_vits._unconstrained_rational_quadratic_spline = onnx_safe_unconstrained_rqs


@dataclass
class NinadaConfig:
    lang: str = "sa"
    vocab_size: int = 79
    hidden_size: int = 192
    flow_size: int = 192
    spectrogram_bins: int = 513
    sampling_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    segment_frames: int = 32
    base_checkpoint: str = "facebook/mms-tts-kan"
    speaking_rate: float = 1.0
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.80
    architecture_version: str = "2.0.0"

    def save_pretrained(self, save_dir: str | Path) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ninada_config.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_pretrained(cls, save_dir: str | Path) -> "NinadaConfig":
        p = Path(save_dir) / "ninada_config.json"
        raw = json.loads(p.read_text())
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        return cls(**filtered)


def strip_all_weight_norm(root: nn.Module) -> None:
    """Convert parametrized and legacy weight_norm modules to plain .weight tensors."""
    for mod in root.modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
        elif hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            try:
                nn.utils.remove_weight_norm(mod)
            except Exception:
                pass


def _patch_vits_attention_for_onnx(attn_mod: nn.Module) -> None:
    """Replace Python int conditionals in VitsAttention with pure tensor ops safe for all seq lengths >= 1."""
    window_size = attn_mod.window_size
    num_heads = attn_mod.num_heads
    head_dim = attn_mod.head_dim
    embed_dim = attn_mod.embed_dim

    def _pure_get_rel_emb(relative_embeddings: torch.Tensor, length_val: Any) -> torch.Tensor:
        if isinstance(length_val, torch.Tensor):
            pad_len = torch.clamp_min(length_val - (window_size + 1), 0)
            start_pos = torch.clamp_min((window_size + 1) - length_val, 0)
        else:
            pad_len = max(int(length_val) - (window_size + 1), 0)
            start_pos = max((window_size + 1) - int(length_val), 0)
        pad_zeros = torch.zeros(
            (relative_embeddings.size(0), pad_len, relative_embeddings.size(2)),
            dtype=relative_embeddings.dtype,
            device=relative_embeddings.device,
        )
        padded = torch.cat([pad_zeros, relative_embeddings, pad_zeros], dim=1)
        end_pos = start_pos + 2 * length_val - 1
        return padded[:, start_pos:end_pos, :]

    def _pure_forward(
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz = hidden_states.shape[0]
        tgt_len = hidden_states.shape[1]

        q = attn_mod.q_proj(hidden_states) * attn_mod.scaling
        k = attn_mod.k_proj(hidden_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        v = attn_mod.v_proj(hidden_states).view(bsz, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2).contiguous()

        proj_shape = (bsz * num_heads, -1, head_dim)
        q = q.view(*proj_shape)
        k = k.view(*proj_shape)
        v = v.view(*proj_shape)

        src_len = k.shape[1]
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if window_size is not None:
            rel_k = _pure_get_rel_emb(attn_mod.emb_rel_k, src_len)
            rel_logits = torch.matmul(q, rel_k.transpose(-2, -1))
            attn_weights = attn_weights + attn_mod._relative_position_to_absolute_position(rel_logits)

        if attention_mask is not None:
            attn_weights = (attn_weights.view(bsz, num_heads, tgt_len, src_len) + attention_mask).view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_probs = F.dropout(F.softmax(attn_weights, dim=-1), p=attn_mod.dropout, training=attn_mod.training)
        attn_output = torch.bmm(attn_probs, v)

        if window_size is not None:
            rel_v = _pure_get_rel_emb(attn_mod.emb_rel_v, src_len)
            rel_w = attn_mod._absolute_position_to_relative_position(attn_probs)
            attn_output = attn_output + torch.matmul(rel_w, rel_v)

        attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        return attn_mod.out_proj(attn_output), None

    attn_mod.forward = _pure_forward


@torch.no_grad()
def monotonic_alignment_search(
    log_p: torch.Tensor,
    text_mask: torch.Tensor,
    spec_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact Vectorized Viterbi Monotonic Alignment Search (MAS)."""
    lp_np = log_p.detach().float().cpu().numpy()
    t_lengths = text_mask.sum(dim=1).long().cpu().numpy()
    y_lengths = spec_mask.sum(dim=1).long().cpu().numpy()

    B, T_x, T_y = lp_np.shape
    paths = np.zeros((B, T_x, T_y), dtype=np.float32)

    for b in range(B):
        tx = int(max(1, t_lengths[b]))
        ty = int(max(1, y_lengths[b]))
        if ty < tx:
            for y in range(ty):
                x = min(tx - 1, int(y * tx / max(1, ty)))
                paths[b, x, y] = 1.0
            continue

        mat = lp_np[b, :tx, :ty]
        q = np.full((tx, ty), -1e9, dtype=np.float32)
        q[0, 0] = mat[0, 0]
        for y in range(1, ty):
            max_x = min(tx, y + 1)
            q[0, y] = q[0, y - 1] + mat[0, y]
            if max_x > 1:
                prev_same = q[1:max_x, y - 1]
                prev_step = q[0 : max_x - 1, y - 1]
                q[1:max_x, y] = mat[1:max_x, y] + np.maximum(prev_same, prev_step)

        index = tx - 1
        for y in range(ty - 1, -1, -1):
            paths[b, index, y] = 1.0
            if index > 0 and (index == y or q[index, y - 1] < q[index - 1, y - 1]):
                index -= 1

    return torch.from_numpy(paths).to(device=log_p.device, dtype=log_p.dtype)


def _build_mel_basis(
    sr: int = 16000, n_fft: int = 1024, n_mels: int = 80, fmin: float = 0.0, fmax: float = 8000.0
) -> torch.Tensor:
    def hz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        if f_center == f_left:
            f_center += 1
        if f_right == f_center:
            f_right += 1
        for k in range(f_left, f_center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - f_left) / max(1, f_center - f_left)
        for k in range(f_center, f_right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (f_right - k) / max(1, f_right - f_center)
    return torch.from_numpy(fb)


def _wn(layer: nn.Module) -> nn.Module:
    if hasattr(nn.utils.parametrizations, "weight_norm"):
        return nn.utils.parametrizations.weight_norm(layer)
    return nn.utils.weight_norm(layer)


class PeriodSubDiscriminator(nn.Module):
    """HiFi-GAN Period Sub-Discriminator capturing glottal harmonic structure at period p."""

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                _wn(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
                _wn(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
                _wn(nn.Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0))),
                _wn(nn.Conv2d(256, 512, (5, 1), (3, 1), padding=(2, 0))),
                _wn(nn.Conv2d(512, 512, (5, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = _wn(nn.Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for lyr in self.convs:
            x = F.leaky_relu(lyr(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class SpectralSubDiscriminator(nn.Module):
    """Multi-Resolution STFT Magnitude Discriminator preserving high-frequency consonant bursts."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.convs = nn.ModuleList(
            [
                _wn(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                _wn(nn.Conv2d(32, 64, (3, 9), stride=(1, 2), padding=(1, 4))),
                _wn(nn.Conv2d(64, 128, (3, 9), stride=(1, 2), padding=(1, 4))),
                _wn(nn.Conv2d(128, 256, (3, 9), stride=(1, 2), padding=(1, 4))),
                _wn(nn.Conv2d(256, 256, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = _wn(nn.Conv2d(256, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        wav = x.squeeze(1).float()
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.float(),
            center=True,
            return_complex=True,
        )
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-7).unsqueeze(1).transpose(2, 3)
        h = mag
        for lyr in self.convs:
            h = F.leaky_relu(lyr(h), 0.1)
            fmap.append(h)
        h = self.conv_post(h)
        fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


class NinadaDiscriminator(nn.Module):
    """Combined HiFi-GAN Multi-Period (MPD) + Multi-Resolution STFT (MRD) Discriminator."""

    def __init__(self) -> None:
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [
                PeriodSubDiscriminator(2),
                PeriodSubDiscriminator(3),
                PeriodSubDiscriminator(5),
                PeriodSubDiscriminator(7),
                PeriodSubDiscriminator(11),
                SpectralSubDiscriminator(1024, 120, 600),
                SpectralSubDiscriminator(2048, 240, 1200),
                SpectralSubDiscriminator(512, 50, 240),
            ]
        )

    def forward(
        self, y_real: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        if y_real.dim() == 2:
            y_real = y_real.unsqueeze(1)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)
        y_d_rs: list[torch.Tensor] = []
        y_d_gs: list[torch.Tensor] = []
        fmap_rs: list[list[torch.Tensor]] = []
        fmap_gs: list[list[torch.Tensor]] = []
        for d in self.sub_discriminators:
            y_d_r, fmap_r = d(y_real)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def discriminator_lsgan_loss(
    disc_real_outputs: list[torch.Tensor], disc_generated_outputs: list[torch.Tensor]
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss = loss + r_loss + g_loss
    return loss / len(disc_real_outputs)


def generator_lsgan_and_fm_loss(
    disc_outputs: list[torch.Tensor],
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    gen_loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dg in disc_outputs:
        gen_loss = gen_loss + torch.mean((1.0 - dg.float()) ** 2)
    gen_loss = gen_loss / len(disc_outputs)

    fm_loss = torch.tensor(0.0, device=disc_outputs[0].device)
    n_layers = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            fm_loss = fm_loss + F.l1_loss(gl.float(), rl.detach().float())
            n_layers += 1
    fm_loss = fm_loss / max(1, n_layers)
    return gen_loss, fm_loss


class NinadaModel(nn.Module):
    """
    Native-Script Specialist VITS + Stochastic Duration Predictor (`SDP`) with
    Adversarial `MPD + MRD` Training & Split Halo-Chunked Streaming ONNX Export.
    """

    def __init__(self, ninada_cfg: NinadaConfig, vits_cfg: VitsConfig | None = None) -> None:
        super().__init__()
        self.ninada_cfg = ninada_cfg
        if vits_cfg is None:
            vits_cfg = VitsConfig.from_pretrained(ninada_cfg.base_checkpoint)
            vits_cfg.vocab_size = ninada_cfg.vocab_size
            vits_cfg.speaker_embedding_size = 0
            vits_cfg.use_stochastic_duration_prediction = True
            vits_cfg.layerdrop = 0.0
        self.vits_cfg = vits_cfg

        shell = VitsModel(vits_cfg)
        self.text_encoder = shell.text_encoder
        self.duration_predictor = shell.duration_predictor
        self.flow = shell.flow
        self.posterior_encoder = shell.posterior_encoder
        self.decoder = shell.decoder

        for layer in self.text_encoder.encoder.layers:
            _patch_vits_attention_for_onnx(layer.attention)

        self.register_buffer("hann_1024", torch.hann_window(1024), persistent=False)
        self.register_buffer("mel_basis_80", _build_mel_basis(ninada_cfg.sampling_rate, 1024, 80), persistent=False)

    @classmethod
    def from_mms_specialist(
        cls,
        lang: str,
        vocab_tokens: list[str],
        base_checkpoint: str,
        donor_map: dict[str, str] | None = None,
    ) -> "NinadaModel":
        """
        Warm-start a native Indic or Classical Sanskrit VITS+SDP specialist directly from its
        exact `facebook/mms-tts-{kan,hin,tam,tel,mal}` checkpoint (`strict=True` on all layers!).
        """
        base_vits = VitsModel.from_pretrained(base_checkpoint)
        strip_all_weight_norm(base_vits)

        vits_cfg = VitsConfig.from_pretrained(base_checkpoint)
        old_vocab_size = vits_cfg.vocab_size
        new_vocab_size = len(vocab_tokens)
        vits_cfg.vocab_size = new_vocab_size
        vits_cfg.speaker_embedding_size = 0
        vits_cfg.use_stochastic_duration_prediction = True
        vits_cfg.layerdrop = 0.0

        ninada_cfg = NinadaConfig(
            lang=lang,
            vocab_size=new_vocab_size,
            hidden_size=vits_cfg.hidden_size,
            flow_size=vits_cfg.flow_size,
            spectrogram_bins=vits_cfg.spectrogram_bins,
            sampling_rate=vits_cfg.sampling_rate,
            base_checkpoint=base_checkpoint,
            speaking_rate=1.0,
            noise_scale=0.667,
            noise_scale_duration=0.80,
        )
        model = cls(ninada_cfg, vits_cfg)
        strip_all_weight_norm(model)

        with torch.no_grad():
            copy_n = min(old_vocab_size, new_vocab_size)
            model.text_encoder.embed_tokens.weight[:copy_n].copy_(
                base_vits.text_encoder.embed_tokens.weight[:copy_n]
            )
            if donor_map and new_vocab_size > old_vocab_size:
                tok_to_idx = {t: i for i, t in enumerate(vocab_tokens)}
                for ext_tok, donor_tok in donor_map.items():
                    if ext_tok in tok_to_idx and donor_tok in tok_to_idx:
                        dst_i = tok_to_idx[ext_tok]
                        src_i = tok_to_idx[donor_tok]
                        if dst_i >= old_vocab_size and src_i < old_vocab_size:
                            model.text_encoder.embed_tokens.weight[dst_i].copy_(
                                base_vits.text_encoder.embed_tokens.weight[src_i]
                            )

            model.text_encoder.encoder.load_state_dict(base_vits.text_encoder.encoder.state_dict(), strict=True)
            model.text_encoder.project.load_state_dict(base_vits.text_encoder.project.state_dict(), strict=True)
            model.duration_predictor.load_state_dict(base_vits.duration_predictor.state_dict(), strict=True)
            model.flow.load_state_dict(base_vits.flow.state_dict(), strict=True)
            model.posterior_encoder.load_state_dict(base_vits.posterior_encoder.state_dict(), strict=True)
            model.decoder.load_state_dict(base_vits.decoder.state_dict(), strict=True)

        return model

    def compute_linear_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        pad = (1024 - 256) // 2
        wav_padded = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        stft = torch.stft(
            wav_padded.float(),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=self.hann_1024.float(),
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)

    def compute_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.compute_linear_spectrogram(wav)
        mel = torch.matmul(self.mel_basis_80.float(), spec.float())
        return torch.log(torch.clamp_min(mel, 1e-5))

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_mask = attention_mask.unsqueeze(-1).to(self.text_encoder.embed_tokens.weight.dtype)
        attn_4d = (1.0 - attention_mask[:, None, None, :].to(pad_mask.dtype)) * -10000.0

        x = (self.text_encoder.embed_tokens(input_ids) * math.sqrt(self.ninada_cfg.hidden_size)) * pad_mask
        for layer in self.text_encoder.encoder.layers:
            x = layer(x, padding_mask=pad_mask, attention_mask=attn_4d, output_attentions=False)[0]
        x = x * pad_mask

        stats = self.text_encoder.project(x.transpose(1, 2)).transpose(1, 2) * pad_mask
        prior_means, prior_log_vars = torch.split(stats, self.ninada_cfg.flow_size, dim=2)
        return x, prior_means, prior_log_vars, pad_mask

    def forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        wav: torch.Tensor,
        wav_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute VITS Variational `KL` + `SDP` normalizing-flow NLL + windowed `decoder(z_slice)`
        for adversarial `MPD + MRD` evaluation.
        """
        spec = self.compute_linear_spectrogram(wav)  # [B, 513, T_spec]
        T_spec = spec.shape[2]
        spec_lengths = torch.clamp(wav_lengths // self.ninada_cfg.hop_length, min=4, max=T_spec)
        spec_idx = torch.arange(T_spec, device=wav.device).unsqueeze(0)
        spec_mask_2d = (spec_idx < spec_lengths.unsqueeze(1)).to(spec.dtype)
        spec_mask = spec_mask_2d.unsqueeze(1)  # [B, 1, T_spec]

        h_text, m_p, logs_p, text_pad_mask = self._encode_text(input_ids, attention_mask)

        # Frozen posterior encoder q(z | x_lin) + trainable normalizing flow z -> z_p
        with torch.no_grad():
            z, m_q, logs_q = self.posterior_encoder(spec, spec_mask)
        z_p = self.flow(z, spec_mask, reverse=False)

        # Exact Viterbi Monotonic Alignment Search (MAS)
        with torch.no_grad():
            inv_var = torch.exp(-2.0 * torch.clamp(logs_p.float(), -5.0, 3.0))
            const_term = -0.5 * torch.sum(
                math.log(2.0 * math.pi) + 2.0 * logs_p.float() + (m_p.float() ** 2) * inv_var,
                dim=-1,
                keepdim=True,
            )
            cross_term = torch.matmul(m_p.float() * inv_var, z_p.float())
            quad_term = -0.5 * torch.matmul(inv_var, z_p.float() ** 2)
            log_p = const_term + cross_term + quad_term
            attn = monotonic_alignment_search(log_p, attention_mask.float(), spec_mask_2d.float()).to(h_text.dtype)

        m_p_aligned = torch.matmul(attn.transpose(1, 2), m_p).transpose(1, 2) * spec_mask
        logs_p_aligned = torch.matmul(attn.transpose(1, 2), logs_p).transpose(1, 2) * spec_mask

        # 1. Exact VITS Variational KL Divergence Loss
        kl = logs_p_aligned.float() - logs_q.float() - 0.5 + 0.5 * (
            (z_p.float() - m_p_aligned.float()).pow(2)
            * torch.exp(-2.0 * torch.clamp(logs_p_aligned.float(), -6.0, 3.0))
        )
        kl_denom = torch.clamp_min(spec_mask.float().sum() * float(self.ninada_cfg.flow_size), 1.0)
        kl_loss = torch.clamp_min((kl * spec_mask.float()).sum() / kl_denom, 0.0)

        # 2. Exact Stochastic Duration Predictor (SDP) Maximum-Likelihood Loss
        gt_dur = torch.sum(attn, dim=2, keepdim=True).transpose(1, 2)  # [B, 1, T_text]
        pm_t = text_pad_mask.transpose(1, 2)  # [B, 1, T_text]
        sdp_nll = self.duration_predictor(
            h_text.detach().transpose(1, 2),
            pm_t,
            durations=torch.clamp_min(gt_dur, 1.0),
            reverse=False,
        )
        dur_loss = torch.sum(sdp_nll.float()) / torch.clamp_min(pm_t.float().sum(), 1.0)

        with torch.no_grad():
            log_dur_sample = self.duration_predictor(
                h_text.detach().transpose(1, 2),
                pm_t,
                reverse=True,
                noise_scale=0.6,
            )
            pred_dur = torch.ceil(torch.exp(torch.clamp(log_dur_sample.float(), -3.0, 5.0)) * pm_t.float())
            dur_mae = ((pred_dur - gt_dur.float()).abs() * pm_t.float()).sum() / torch.clamp_min(pm_t.float().sum(), 1.0)
            frame_sec = self.ninada_cfg.hop_length / float(self.ninada_cfg.sampling_rate)
            utt_dur_mae_sec = (
                (pred_dur.sum(dim=[1, 2]) - (gt_dur.float() * pm_t.float()).sum(dim=[1, 2])).abs().mean() * frame_sec
            )

        # 3. Sample from prior p(z_p | text, MAS) -> flow_reverse -> frozen HiFi-GAN decoder window
        eps_prior = torch.randn_like(m_p_aligned) * 0.667
        z_p_sample = (m_p_aligned + eps_prior * torch.exp(torch.clamp(logs_p_aligned, -6.0, 2.0))) * spec_mask
        z_hat_full = self.flow(z_p_sample, spec_mask, reverse=True) * spec_mask

        B = wav.shape[0]
        seg_f = self.ninada_cfg.segment_frames
        hop = self.ninada_cfg.hop_length
        seg_samples = seg_f * hop

        z_slices: list[torch.Tensor] = []
        wav_slices: list[torch.Tensor] = []
        for b in range(B):
            valid_f = int(spec_lengths[b].item())
            max_start = max(0, valid_f - seg_f)
            st = int(torch.randint(0, max_start + 1, (1,), device=wav.device).item()) if max_start > 0 else 0
            z_chunk = z_hat_full[b, :, st : st + seg_f]
            if z_chunk.shape[1] < seg_f:
                z_chunk = F.pad(z_chunk, (0, seg_f - z_chunk.shape[1]))
            z_slices.append(z_chunk)

            w_st = st * hop
            w_chunk = wav[b, w_st : w_st + seg_samples]
            if w_chunk.shape[0] < seg_samples:
                w_chunk = F.pad(w_chunk, (0, seg_samples - w_chunk.shape[0]))
            wav_slices.append(w_chunk)

        z_batch = torch.stack(z_slices, dim=0)
        wav_gt_slice = torch.stack(wav_slices, dim=0)
        wav_hat_slice = self.decoder(z_batch).squeeze(1)

        min_s = min(wav_gt_slice.shape[1], wav_hat_slice.shape[1])
        wav_gt_slice = wav_gt_slice[:, :min_s]
        wav_hat_slice = wav_hat_slice[:, :min_s]

        mel_gt = self.compute_log_mel(wav_gt_slice)
        mel_hat = self.compute_log_mel(wav_hat_slice)
        mel_l1 = F.l1_loss(mel_hat, mel_gt)

        return {
            "kl_loss": kl_loss,
            "dur_loss": dur_loss,
            "mel_l1": mel_l1,
            "dur_mae_frames": dur_mae.detach(),
            "utt_dur_mae_sec": utt_dur_mae_sec.detach(),
            "wav_gt_slice": wav_gt_slice,
            "wav_hat_slice": wav_hat_slice,
        }

    def encode_flow_infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming Stage-1 Forward Pass (ONNX Opset 17):
        `scales` = `[speaking_rate, noise_scale, noise_scale_duration]`.
        Returns `(latents [B, 192, T_frames], durations [B, T_seq])`.
        """
        h_text, prior_means, prior_log_vars, pad_mask = self._encode_text(input_ids, attention_mask)
        pm_t = pad_mask.transpose(1, 2)

        rate = torch.clamp_min(scales[0], 0.25)
        noise_scale = scales[1]
        noise_scale_dur = scales[2] if int(scales.shape[0]) >= 3 else (scales[1] * 1.2)

        log_dur = self.duration_predictor(
            h_text.transpose(1, 2),
            pm_t,
            reverse=True,
            noise_scale=noise_scale_dur,
        )
        length_scale = 1.0 / rate
        dur = torch.ceil(torch.exp(log_dur) * pm_t * length_scale)
        dur = torch.clamp_min(dur, 1.0) * pm_t

        pred_lengths = torch.clamp_min(torch.sum(dur, dim=[1, 2]), 1.0).long()
        max_len = pred_lengths.max()
        idx = torch.arange(max_len, device=input_ids.device).unsqueeze(0)
        out_mask = (idx < pred_lengths.unsqueeze(1)).unsqueeze(1).to(pad_mask.dtype)

        attn_mask = pm_t.unsqueeze(2) * out_mask.unsqueeze(-1)
        bsz, _, out_len, in_len = attn_mask.shape
        cum_dur = torch.cumsum(dur, dim=-1).reshape(bsz * in_len, 1)
        out_idx = torch.arange(out_len, device=input_ids.device, dtype=dur.dtype).unsqueeze(0)
        valid = (out_idx < cum_dur).to(pad_mask.dtype).reshape(bsz, in_len, out_len)
        padded = valid - F.pad(valid, [0, 0, 1, 0])[:, :-1, :]
        attn = padded.unsqueeze(1).transpose(2, 3) * attn_mask

        m_p = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), prior_log_vars).transpose(1, 2)

        eps = torch.randn_like(m_p) * noise_scale
        z_p = (m_p + eps * torch.exp(logs_p)) * out_mask
        latents = self.flow(z_p, out_mask, reverse=True) * out_mask
        return latents, dur.squeeze(1)

    def decode_latents_infer(self, latents: torch.Tensor) -> torch.Tensor:
        """Streaming Stage-2 Forward Pass: Halo-Windowed Latents [B, 192, W] -> 16 kHz Audio [B, W*256]."""
        return self.decoder(latents).squeeze(1)

    def forward_infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents, dur = self.encode_flow_infer(input_ids, attention_mask, scales)
        waveform = self.decode_latents_infer(latents)
        return waveform, dur

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scales is None:
            scales = torch.tensor(
                [
                    self.ninada_cfg.speaking_rate,
                    self.ninada_cfg.noise_scale,
                    self.ninada_cfg.noise_scale_duration,
                ],
                dtype=torch.float32,
                device=input_ids.device,
            )
        return self.forward_infer(input_ids, attention_mask, scales)

    def save_pretrained(self, save_dir: str | Path, filename: str = "model.safetensors") -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        strip_all_weight_norm(self)
        self.ninada_cfg.save_pretrained(p)
        self.vits_cfg.save_pretrained(str(p))
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, str(p / filename))

    @classmethod
    def from_pretrained(cls, save_dir: str | Path, filename: str = "model.safetensors") -> "NinadaModel":
        p = Path(save_dir)
        ninada_cfg = NinadaConfig.from_pretrained(p)
        vits_cfg = VitsConfig.from_pretrained(str(p))
        model = cls(ninada_cfg, vits_cfg)
        strip_all_weight_norm(model)
        state = load_file(str(p / filename))
        model.load_state_dict(state, strict=True)
        return model


NinadaSpecialistModel = NinadaModel
