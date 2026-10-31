"""
Ninada: Disentangled Pan-Indic & Classical Sanskrit Variational End-to-End Neural TTS (VITS + HiFi-GAN).

Architectural Advances over Vanilla Multilingual VITS Fine-Tuning:
1. 87-Token Pan-Indic & Classical/Vedic Sanskrit Phonology:
   Warm-starts tokens 0..72 from `facebook/mms-tts-hin` and initializes tokens 73..86
   (Dravidian retroflexes/alveolars + Sanskrit independent vocalic `r` / `rr`, velar `nga`,
   and danda prosodic boundaries) from articulatory donors.
2. Speaker-Disentangled Training + Zero-Overhead Inference Conditioning:
   - `UtteranceStyleEncoder` extracts utterance-level Mel statistics `[mean_80, std_80] -> R^192`
     during `forward_train` so multi-speaker pitch/cadence variance across 9 corpora does not
     blur consonant articulatory trajectories in `text_encoder` or `flow`.
   - `lang_voice_embed` (`[6, 192]` buffer) tracks the EMA studio voice centroid per language
     during training and supplies clean studio timbre in `forward_infer` (`0` extra ONNX ops).
   - Native VITS `global_conditioning` (`speaker_embedding_size = 192`) injected into every
     WaveNet coupling layer of `flow` and `dur_predictor` (zero-initialized at warm-start).
3. Crisp Latent-Manifold VITS Objective:
   Combines Exact VITS `KL(q(z|spec) || p(z_p|text, MAS, g))`, full-utterance Latent Feature
   Matching `SmoothL1(flow_rev(m_p, g), sg[m_q])` in the frozen adversarial HiFi-GAN latent
   manifold, token + utterance-level Duration Huber loss, and high-frequency (`1.5-8 kHz`)
   consonant-weighted STFT loss.
4. Trace-Pure ONNX Opset 17 Inference (`forward_infer`) with zero Python-boolean TracerWarnings.
"""
from __future__ import annotations

import copy
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
from transformers.models.vits.modeling_vits import (
    VitsDurationPredictor,
    VitsResidualCouplingBlock,
)


@dataclass
class NinadaConfig:
    vocab_size: int = 87
    num_languages: int = 6
    hidden_size: int = 192
    flow_size: int = 192
    speaker_embedding_size: int = 192
    spectrogram_bins: int = 513
    sampling_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    segment_frames: int = 32
    base_checkpoint: str = "facebook/mms-tts-hin"
    speaking_rate: float = 1.0
    noise_scale: float = 0.32

    def save_pretrained(self, save_dir: str | Path) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "ninada_config.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_pretrained(cls, save_dir: str | Path) -> "NinadaConfig":
        p = Path(save_dir) / "ninada_config.json"
        return cls(**json.loads(p.read_text()))


def strip_all_weight_norm(root: nn.Module) -> None:
    """Convert both modern parametrized and legacy weight_norm modules to plain .weight tensors."""
    for mod in root.modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
        elif hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            try:
                nn.utils.remove_weight_norm(mod)
            except Exception:
                pass


def _patch_vits_attention_for_onnx(attn_mod: nn.Module) -> None:
    """Replace Python int conditionals in VitsAttention with pure tensor ops (zero TracerWarnings)."""
    window_size = attn_mod.window_size
    num_heads = attn_mod.num_heads
    head_dim = attn_mod.head_dim
    embed_dim = attn_mod.embed_dim

    def _pure_get_rel_emb(relative_embeddings: torch.Tensor, length_tensor: torch.Tensor) -> torch.Tensor:
        pad_len = length_tensor - (window_size + 1)
        pad_zeros = torch.zeros(
            (relative_embeddings.size(0), pad_len, relative_embeddings.size(2)),
            dtype=relative_embeddings.dtype,
            device=relative_embeddings.device,
        )
        padded = torch.cat([pad_zeros, relative_embeddings, pad_zeros], dim=1)
        end_pos = 2 * length_tensor - 1
        return padded[:, :end_pos, :]

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


class UtteranceStyleEncoder(nn.Module):
    """Training-only reference utterance encoder ([mel_mean_80, mel_std_80] -> R^192) for speaker disentanglement."""

    def __init__(self, n_mels: int = 80, out_dim: int = 192) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_mels * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, mel: torch.Tensor, spec_mask: torch.Tensor) -> torch.Tensor:
        # mel: [B, 80, T], spec_mask: [B, 1, T]
        denom = torch.clamp_min(spec_mask.float().sum(dim=-1), 1.0)  # [B, 1]
        m = (mel.float() * spec_mask.float()).sum(dim=-1) / denom  # [B, 80]
        var = ((mel.float() - m.unsqueeze(-1)).pow(2) * spec_mask.float()).sum(dim=-1) / denom
        std = torch.sqrt(var + 1e-5)
        feat = torch.cat([m, std], dim=-1)
        h = self.norm(F.silu(self.fc1(feat)))
        return torch.tanh(self.fc2(h)) * 0.5


class NinadaModel(nn.Module):
    """Disentangled Pan-Indic & Classical Sanskrit VITS Model with Differentiable Training & ONNX Export."""

    def __init__(self, ninada_cfg: NinadaConfig, vits_cfg: VitsConfig | None = None) -> None:
        super().__init__()
        self.ninada_cfg = ninada_cfg
        if vits_cfg is None:
            vits_cfg = VitsConfig.from_pretrained(ninada_cfg.base_checkpoint)
            vits_cfg.vocab_size = ninada_cfg.vocab_size
            vits_cfg.speaker_embedding_size = 0
            vits_cfg.use_stochastic_duration_prediction = False
            vits_cfg.layerdrop = 0.0
        self.vits_cfg = vits_cfg

        # Base shell keeps posterior_encoder & decoder unconditioned to match base_vits 1-to-1
        shell_cfg = copy.deepcopy(vits_cfg)
        shell_cfg.speaker_embedding_size = 0
        shell = VitsModel(shell_cfg)
        self.text_encoder = shell.text_encoder
        self.posterior_encoder = shell.posterior_encoder
        self.decoder = shell.decoder

        # Conditioned flow & deterministic duration predictor use native global_conditioning (192-d)
        cond_cfg = copy.deepcopy(vits_cfg)
        cond_cfg.speaker_embedding_size = ninada_cfg.speaker_embedding_size
        cond_cfg.use_stochastic_duration_prediction = False
        self.flow = VitsResidualCouplingBlock(cond_cfg)
        self.dur_predictor = VitsDurationPredictor(cond_cfg)

        for layer in self.text_encoder.encoder.layers:
            _patch_vits_attention_for_onnx(layer.attention)

        H = ninada_cfg.hidden_size
        self.lang_embed = nn.Embedding(ninada_cfg.num_languages, H)
        nn.init.zeros_(self.lang_embed.weight)

        self.lang_text_gate = nn.Parameter(torch.tensor(0.0))
        self.lang_dec_proj = nn.Conv1d(H, vits_cfg.upsample_initial_channel, kernel_size=1)
        nn.init.zeros_(self.lang_dec_proj.weight)
        nn.init.zeros_(self.lang_dec_proj.bias)

        # Training-only multi-speaker disentanglement + persistent EMA per-language studio voice buffer
        self.utt_style_encoder = UtteranceStyleEncoder(n_mels=80, out_dim=H)
        self.register_buffer("lang_voice_embed", torch.zeros(ninada_cfg.num_languages, H), persistent=True)

        self.register_buffer("hann_1024", torch.hann_window(1024), persistent=False)
        self.register_buffer("hann_512", torch.hann_window(512), persistent=False)
        self.register_buffer("hann_2048", torch.hann_window(2048), persistent=False)
        self.register_buffer("mel_basis_80", _build_mel_basis(ninada_cfg.sampling_rate, 1024, 80), persistent=False)

    def _zero_init_conditioning_layers(self) -> None:
        with torch.no_grad():
            for flow_layer in self.flow.flows:
                if hasattr(flow_layer, "wavenet") and hasattr(flow_layer.wavenet, "cond_layer"):
                    nn.init.zeros_(flow_layer.wavenet.cond_layer.weight)
                    if flow_layer.wavenet.cond_layer.bias is not None:
                        nn.init.zeros_(flow_layer.wavenet.cond_layer.bias)
            if hasattr(self.dur_predictor, "cond"):
                nn.init.zeros_(self.dur_predictor.cond.weight)
                if self.dur_predictor.cond.bias is not None:
                    nn.init.zeros_(self.dur_predictor.cond.bias)

    @classmethod
    def from_mms_warmstart(
        cls,
        ninada_cfg: NinadaConfig,
        vocab_tokens: list[str],
        donor_map: dict[str, str],
    ) -> "NinadaModel":
        """Warm-start from facebook/mms-tts-hin with exact zero-init conditioning and 87-token donors."""
        base_vits = VitsModel.from_pretrained(ninada_cfg.base_checkpoint)
        strip_all_weight_norm(base_vits)

        vits_cfg = VitsConfig.from_pretrained(ninada_cfg.base_checkpoint)
        old_vocab_size = vits_cfg.vocab_size
        vits_cfg.vocab_size = ninada_cfg.vocab_size
        vits_cfg.speaker_embedding_size = 0
        vits_cfg.use_stochastic_duration_prediction = False
        vits_cfg.layerdrop = 0.0

        model = cls(ninada_cfg, vits_cfg)
        strip_all_weight_norm(model)

        with torch.no_grad():
            model.text_encoder.embed_tokens.weight[:old_vocab_size].copy_(
                base_vits.text_encoder.embed_tokens.weight[:old_vocab_size]
            )
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
            model.flow.load_state_dict(base_vits.flow.state_dict(), strict=False)
            model.posterior_encoder.load_state_dict(base_vits.posterior_encoder.state_dict(), strict=True)
            model.decoder.load_state_dict(base_vits.decoder.state_dict(), strict=True)

            model._zero_init_conditioning_layers()
            nn.init.normal_(model.lang_embed.weight, mean=0.0, std=0.02)
            nn.init.constant_(model.dur_predictor.proj.bias, 1.45)

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

    def compute_log_mel_from_spec(self, spec: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(self.mel_basis_80.float(), spec.float())
        return torch.log(torch.clamp_min(mel, 1e-5))

    def compute_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.compute_linear_spectrogram(wav)
        return self.compute_log_mel_from_spec(spec)

    def _encode_text_conditioned(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        g_lang: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_mask = attention_mask.unsqueeze(-1).to(self.text_encoder.embed_tokens.weight.dtype)
        attn_4d = (1.0 - attention_mask[:, None, None, :].to(pad_mask.dtype)) * -10000.0

        lang_bias = self.lang_text_gate * g_lang.unsqueeze(1)
        x = (self.text_encoder.embed_tokens(input_ids) * math.sqrt(self.ninada_cfg.hidden_size) + lang_bias) * pad_mask

        for layer in self.text_encoder.encoder.layers:
            x = layer(x, padding_mask=pad_mask, attention_mask=attn_4d, output_attentions=False)[0]
        x = x * pad_mask

        stats = self.text_encoder.project((x + lang_bias).transpose(1, 2)).transpose(1, 2)
        stats = stats * pad_mask
        prior_means, prior_log_vars = torch.split(stats, self.ninada_cfg.flow_size, dim=2)
        return x, prior_means, prior_log_vars, pad_mask

    def _decode_conditioned(self, z: torch.Tensor, g_vec: torch.Tensor) -> torch.Tensor:
        dec = self.decoder
        h = dec.conv_pre(z) + self.lang_dec_proj(g_vec.unsqueeze(-1))
        for i in range(dec.num_upsamples):
            h = F.leaky_relu(h, dec.config.leaky_relu_slope)
            h = dec.upsampler[i](h)
            res = dec.resblocks[i * dec.num_kernels](h)
            for j in range(1, dec.num_kernels):
                res = res + dec.resblocks[i * dec.num_kernels + j](h)
            h = res / dec.num_kernels
        h = F.leaky_relu(h)
        h = dec.conv_post(h)
        return torch.tanh(h).squeeze(1)

    def _consonant_weighted_stft_loss(self, y_hat: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT loss with 1.5x emphasis on 1.5-8.0 kHz consonant/sibilant bursts."""
        y_hat_f = y_hat.float()
        y_gt_f = y_gt.float()
        total = torch.tensor(0.0, device=y_hat.device)
        configs = [
            (512, 128, 512, self.hann_512.float()),
            (1024, 256, 1024, self.hann_1024.float()),
            (2048, 512, 2048, self.hann_2048.float()),
        ]
        for n_fft, hop, win, window in configs:
            s_hat = torch.stft(y_hat_f, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
            s_gt = torch.stft(y_gt_f, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
            m_hat = torch.sqrt(s_hat.real.pow(2) + s_hat.imag.pow(2) + 1e-6)
            m_gt = torch.sqrt(s_gt.real.pow(2) + s_gt.imag.pow(2) + 1e-6)

            sc_loss = torch.norm(m_gt - m_hat, p="fro") / (torch.norm(m_gt, p="fro") + 1e-6)
            mag_loss = F.l1_loss(torch.log(m_gt + 1e-5), torch.log(m_hat + 1e-5))

            hf_bin = max(1, int(round(1500.0 * n_fft / self.ninada_cfg.sampling_rate)))
            hf_mag = F.l1_loss(torch.log(m_gt[:, hf_bin:, :] + 1e-5), torch.log(m_hat[:, hf_bin:, :] + 1e-5))
            total = total + (sc_loss + 0.65 * mag_loss + 0.55 * hf_mag)
        return total / len(configs)

    def forward_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lang_id: torch.Tensor,
        wav: torch.Tensor,
        wav_lengths: torch.Tensor,
        is_primary_studio: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Disentangled Latent-Manifold VITS Training Step."""
        spec = self.compute_linear_spectrogram(wav)  # [B, 513, T_spec]
        T_spec = spec.shape[2]
        spec_lengths = torch.clamp(wav_lengths // self.ninada_cfg.hop_length, min=4, max=T_spec)
        spec_idx = torch.arange(T_spec, device=wav.device).unsqueeze(0)
        spec_mask_2d = (spec_idx < spec_lengths.unsqueeze(1)).to(spec.dtype)
        spec_mask = spec_mask_2d.unsqueeze(1)  # [B, 1, T_spec]

        mel_full = self.compute_log_mel_from_spec(spec)  # [B, 80, T_spec]
        utt_style = self.utt_style_encoder(mel_full, spec_mask)  # [B, 192]

        if self.training:
            with torch.no_grad():
                for c in range(self.ninada_cfg.num_languages):
                    c_mask = lang_id == c
                    if is_primary_studio is not None:
                        c_studio = c_mask & is_primary_studio.bool()
                        if c_studio.any():
                            c_mask = c_studio
                    if c_mask.any():
                        batch_mean = utt_style[c_mask].detach().mean(dim=0)
                        self.lang_voice_embed[c].mul_(0.96).add_(batch_mean, alpha=0.04)

        g_lang = self.lang_embed(lang_id)  # [B, 192]
        g_vec = g_lang + utt_style if self.training else (g_lang + F.embedding(lang_id, self.lang_voice_embed))
        g_cond = g_vec.unsqueeze(-1)  # [B, 192, 1]

        h_text, m_p, logs_p, text_pad_mask = self._encode_text_conditioned(input_ids, attention_mask, g_lang)

        # Frozen adversarial posterior encoder + conditioned normalizing flow forward (z -> z_p)
        with torch.no_grad():
            z, m_q, logs_q = self.posterior_encoder(spec, spec_mask)
        z_p = self.flow(z, spec_mask, global_conditioning=g_cond, reverse=False)

        # Exact Monotonic Alignment Search (MAS)
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

        # 1. Exact VITS Variational KL Divergence Loss (normalized per latent dimension & frame)
        kl = logs_p_aligned.float() - logs_q.float() - 0.5 + 0.5 * (
            (z_p.float() - m_p_aligned.float()).pow(2) * torch.exp(-2.0 * torch.clamp(logs_p_aligned.float(), -6.0, 3.0))
        )
        kl_denom = torch.clamp_min(spec_mask.float().sum() * float(self.ninada_cfg.flow_size), 1.0)
        kl_loss = torch.clamp_min((kl * spec_mask.float()).sum() / kl_denom, 0.0)

        # 2. Full-Utterance Latent Manifold Feature Matching: flow(m_p_aligned, reverse=True) -> m_q
        # Trains text_encoder & flow directly in the crisp HiFi-GAN latent space without L1 Mel blurring!
        z_hat_full = self.flow(m_p_aligned, spec_mask, global_conditioning=g_cond, reverse=True)
        latent_loss = (
            F.smooth_l1_loss(z_hat_full.float() * spec_mask.float(), m_q.detach().float() * spec_mask.float(), reduction="sum")
            / kl_denom
        )

        # 3. Token + Utterance-Level Duration Predictor Loss
        gt_dur = torch.sum(attn, dim=2, keepdim=True).transpose(1, 2)  # [B, 1, T_text]
        pm_t = text_pad_mask.transpose(1, 2)
        log_dur_pred = self.dur_predictor(h_text.detach().transpose(1, 2), pm_t, global_conditioning=g_cond.detach())
        log_dur_gt = torch.log(torch.clamp_min(gt_dur.float(), 1.0)) * pm_t.float()
        tok_dur_loss = ((log_dur_pred.float() - log_dur_gt).pow(2) * pm_t.float()).sum() / torch.clamp_min(
            pm_t.float().sum(), 1.0
        )

        frame_sec = self.ninada_cfg.hop_length / float(self.ninada_cfg.sampling_rate)
        pred_utt_sec = (torch.exp(torch.clamp(log_dur_pred.float(), -3.0, 5.0)) * pm_t.float()).sum(dim=[1, 2]) * frame_sec
        gt_utt_sec = (gt_dur.float() * pm_t.float()).sum(dim=[1, 2]) * frame_sec
        utt_dur_loss = F.smooth_l1_loss(pred_utt_sec, gt_utt_sec)
        dur_loss = tok_dur_loss + 0.85 * utt_dur_loss
        dur_mae = (
            ((torch.exp(torch.clamp(log_dur_pred.float(), -3.0, 5.0)) * pm_t.float() - gt_dur.float()).abs() * pm_t.float()).sum()
            / torch.clamp_min(pm_t.float().sum(), 1.0)
        )

        # 4. Windowed HiFi-GAN Acoustic Calibration (light weight to prevent over-smoothing)
        B = wav.shape[0]
        seg_f = self.ninada_cfg.segment_frames
        hop = self.ninada_cfg.hop_length
        seg_samples = seg_f * hop

        z_slices = []
        wav_slices = []
        eps_prior = torch.randn_like(m_p_aligned) * 0.30
        z_p_sample = (m_p_aligned + eps_prior * torch.exp(torch.clamp(logs_p_aligned, -6.0, 1.5))) * spec_mask
        z_hat_sample = self.flow(z_p_sample, spec_mask, global_conditioning=g_cond, reverse=True)

        for b in range(B):
            valid_f = int(spec_lengths[b].item())
            max_start = max(0, valid_f - seg_f)
            st = int(torch.randint(0, max_start + 1, (1,), device=wav.device).item()) if max_start > 0 else 0
            z_chunk = z_hat_sample[b, :, st : st + seg_f]
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
        wav_hat_slice = self._decode_conditioned(z_batch, g_vec)

        min_s = min(wav_gt_slice.shape[1], wav_hat_slice.shape[1])
        wav_gt_slice = wav_gt_slice[:, :min_s]
        wav_hat_slice = wav_hat_slice[:, :min_s]

        mel_gt = self.compute_log_mel(wav_gt_slice)
        mel_hat = self.compute_log_mel(wav_hat_slice)
        mel_l1 = F.l1_loss(mel_hat, mel_gt)
        mr_stft = self._consonant_weighted_stft_loss(wav_hat_slice, wav_gt_slice)

        # Balanced latent-first loss: avoids GAN-free Mel over-smoothing while calibrating spectral timbre
        total_loss = (
            3.0 * kl_loss
            + 14.0 * latent_loss
            + 2.5 * dur_loss
            + 2.2 * mel_l1
            + 1.0 * mr_stft
        )
        return {
            "loss": total_loss,
            "kl_loss": kl_loss.detach(),
            "latent_loss": latent_loss.detach(),
            "dur_loss": dur_loss.detach(),
            "mel_l1": mel_l1.detach(),
            "mr_stft": mr_stft.detach(),
            "dur_mae_frames": dur_mae.detach(),
            "utt_dur_mae_sec": (pred_utt_sec - gt_utt_sec).abs().mean().detach(),
        }

    def forward_infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lang_id: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure-tensor Inference Forward Pass (ONNX opset 17 compatible, zero UtteranceStyleEncoder overhead)."""
        g_lang = self.lang_embed(lang_id)
        g_voice = F.embedding(lang_id, self.lang_voice_embed)
        g_vec = g_lang + g_voice
        g_cond = g_vec.unsqueeze(-1)

        h_text, prior_means, prior_log_vars, pad_mask = self._encode_text_conditioned(
            input_ids, attention_mask, g_lang
        )
        pm_t = pad_mask.transpose(1, 2)
        log_dur = self.dur_predictor(h_text.transpose(1, 2), pm_t, global_conditioning=g_cond)

        length_scale = 1.0 / torch.clamp_min(scales[0], 0.25)
        dur = torch.floor(torch.exp(log_dur) * pm_t * length_scale + 0.5)
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

        eps = torch.randn_like(m_p) * scales[1]
        z_p = (m_p + eps * torch.exp(logs_p)) * out_mask
        z = self.flow(z_p, out_mask, global_conditioning=g_cond, reverse=True)
        waveform = self._decode_conditioned(z * out_mask, g_vec)
        return waveform, dur.squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lang_id: torch.Tensor,
        scales: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scales is None:
            scales = torch.tensor(
                [self.ninada_cfg.speaking_rate, self.ninada_cfg.noise_scale],
                dtype=torch.float32,
                device=input_ids.device,
            )
        return self.forward_infer(input_ids, attention_mask, lang_id, scales)

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
