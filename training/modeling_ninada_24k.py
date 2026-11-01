#!/usr/bin/env python3
"""
Ninada v3.0 — 24,000 Hz Pan-Indic & Classical Sanskrit Neural Speech Synthesis Architecture.

Self-contained Apache-2.0 PyTorch implementation of:
  1. Contextual Phoneme-Level BERT (`CustomAlbert` PL-BERT, 178-symbol IPA + Vedic/Dravidian `ɭ=79`)
  2. Convolutional + BiLSTM Acoustic Text Encoder (`TextEncoder`)
  3. Style-Conditioned Explicit Prosody Predictor (`ProsodyPredictor`: Ordinal Duration, 80 Hz F0 in Hz, 80 Hz Energy N)
  4. Harmonic-Source Neural Fourier Vocoder (`Decoder` + `SineGen` + real-valued `CustomSTFT` iSTFTNet @ 24,000 Hz)
  5. Smooth Length-Conditioned Per-Language Studio Voice Banks (`voices/{sa,hi,kn,ta,te,ml}.bin`, [510, 1, 256])
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertConfig, AlbertModel

SAMPLE_RATE = 24000
HOP_LENGTH = 600       # 40.0 Hz frame rate (25.0 ms per acoustic alignment frame)
F0_HOP_LENGTH = 300    # 80.0 Hz pitch/energy rate (12.5 ms per prosodic frame)
N_TOKENS = 178
VOICE_MAX_LEN = 510
RETROFLEX_LATERAL_ID = 79  # 'ɭ' (ळ / ಳ / ള / ள / ళ)
ALVEOLAR_LATERAL_ID = 54   # 'l'
RETROFLEX_STOP_ID = 80     # 'ɖ'

SUPPORTED_LANGS = ("sa", "hi", "kn", "ta", "te", "ml")


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class CustomSTFT(nn.Module):
    """ONNX-safe real-valued STFT / iSTFT implemented via Conv1d & ConvTranspose1d (no complex64 ops)."""

    def __init__(
        self,
        filter_length: int = 20,
        hop_length: int = 5,
        win_length: int = 20,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()
        assert window == "hann", window
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode
        self.freq_bins = self.n_fft // 2 + 1

        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if self.win_length < self.n_fft:
            window_tensor = F.pad(window_tensor, (0, self.n_fft - self.win_length))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
        self.register_buffer("window", window_tensor)

        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2.0 * np.pi * np.outer(k, n) / self.n_fft
        forward_win = window_tensor.numpy()
        forward_real = (np.cos(angle) * forward_win).astype(np.float32)
        forward_imag = (-np.sin(angle) * forward_win).astype(np.float32)
        self.register_buffer("weight_forward_real", torch.from_numpy(forward_real).unsqueeze(1))
        self.register_buffer("weight_forward_imag", torch.from_numpy(forward_imag).unsqueeze(1))

        inv_scale = 1.0 / self.n_fft
        angle_t = 2.0 * np.pi * np.outer(n, k) / self.n_fft
        inv_win = window_tensor.numpy() * inv_scale
        backward_real = (np.cos(angle_t).T * inv_win).astype(np.float32)
        backward_imag = (np.sin(angle_t).T * inv_win).astype(np.float32)
        self.register_buffer("weight_backward_real", torch.from_numpy(backward_real).unsqueeze(1))
        self.register_buffer("weight_backward_imag", torch.from_numpy(backward_imag).unsqueeze(1))

    def transform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        real_out = F.conv1d(x, self.weight_forward_real, bias=None, stride=self.hop_length, padding=0)
        imag_out = F.conv1d(x, self.weight_forward_imag, bias=None, stride=self.hop_length, padding=0)
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        # Guard ONNX opset-17 atan2 -> Atan(Div(imag_out, real_out)) against 0.0 / 0.0 -> NaN on silent/unvoiced bins
        safe_real = torch.where(
            real_out.abs() < 1e-12,
            torch.where(real_out < 0, torch.full_like(real_out, -1e-12), torch.full_like(real_out, 1e-12)),
            real_out,
        )
        phase = torch.atan2(imag_out, safe_real)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase = torch.where(correction_mask, torch.full_like(phase, math.pi), phase)
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length: int | None = None) -> torch.Tensor:
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        real_rec = F.conv_transpose1d(real_part, self.weight_backward_real, bias=None, stride=self.hop_length, padding=0)
        imag_rec = F.conv_transpose1d(imag_part, self.weight_backward_imag, bias=None, stride=self.hop_length, padding=0)
        waveform = real_rec - imag_rec
        if self.center:
            pad_len = self.n_fft // 2
            waveform = waveform[..., pad_len:-pad_len]
        if length is not None:
            waveform = waveform[..., :length]
        return waveform


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s).view(s.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1.0 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] | list[int] = (1, 3, 5),
        style_dim: int = 64,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])))
            for i in range(3)
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])
        self.convs2.apply(init_weights)
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)])

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1.0 / a1) * (torch.sin(a1 * xt) ** 2)
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1.0 / a2) * (torch.sin(a2 * xt) ** 2)
            xt = c2(xt)
            x = xt + x
        return x


class SineGen(nn.Module):
    """Harmonic sine + unvoiced excitation generator at 24,000 Hz."""

    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 10.0,
        deterministic_eval: bool = False,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        self.deterministic_eval = deterministic_eval
        self.register_buffer(
            "harmonic_mult",
            torch.arange(1, self.harmonic_num + 2, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).to(torch.float32)

    def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        rad_values = (f0_values / self.sampling_rate) % 1.0
        if not (self.deterministic_eval and not self.training):
            rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device, dtype=f0_values.dtype)
            rand_ini[:, 0] = 0.0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_down = F.interpolate(rad_values.transpose(1, 2), scale_factor=1.0 / self.upsample_scale, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_down, dim=1) * (2.0 * math.pi)
        phase_up = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=float(self.upsample_scale), mode="linear").transpose(1, 2)
        return torch.sin(phase_up)

    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = f0 * self.harmonic_mult.to(device=f0.device, dtype=f0.dtype)
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        if self.deterministic_eval and not self.training:
            # Tiny deterministic breath floor for clean ONNX reproducibility without harsh zero-energy nulls
            sine_waves = sine_waves * uv
            noise = torch.zeros_like(sine_waves)
        else:
            noise_amp = uv * self.noise_std + (1.0 - uv) * (self.sine_amp / 3.0)
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 8,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 10.0,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.zeros_like(uv)
        return sine_merge, noise, uv


class Generator(nn.Module):
    def __init__(
        self,
        style_dim: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=SAMPLE_RATE,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshod=10.0,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=float(math.prod(upsample_rates) * gen_istft_hop_size))
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        ch,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(ch, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, ch, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(ch, 11, [1, 3, 5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = CustomSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f0_up = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _, _ = self.m_source(f0_up)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = self.resblocks[i * self.num_kernels](x, s)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / float(self.num_kernels)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
    def __init__(self, layer_type: str | bool) -> None:
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "none":
            return x
        return F.interpolate(x, scale_factor=2.0, mode="nearest")


class AdainResBlk1d(nn.Module):
    INV_SQRT2 = 1.0 / math.sqrt(2.0)

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        actv: nn.Module = nn.LeakyReLU(0.2),
        upsample: str | bool = "none",
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))
        self.dropout = nn.Dropout(dropout_p)
        if upsample == "none":
            self.pool: nn.Module = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1)
            )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (self._residual(x, s) + self._shortcut(x)) * self.INV_SQRT2


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        dim_out: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
    ) -> None:
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList([
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
        ])
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        )

    def forward(
        self,
        asr: torch.Tensor,
        F0_curve: torch.Tensor,
        N: torch.Tensor,
        s: torch.Tensor,
        har_f0_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_feat = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_feat], dim=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_feat], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        gen_f0 = har_f0_override if har_f0_override is not None else F0_curve
        return self.generator(x, s, gen_f0)


class LinearNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear") -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int, actv: nn.Module = nn.LeakyReLU(0.2)) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            )
            for _ in range(depth)
        ])
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward_unpacked(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-1 unpadded forward path (used in both single-utterance GPU step & clean ONNX export)."""
        h = self.embedding(x).transpose(1, 2)
        for c in self.cnn:
            h = c(h)
        h = h.transpose(1, 2).contiguous()
        if self.training and h.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                h, _ = self.lstm(h)
        else:
            self.lstm.flatten_parameters()
            h, _ = self.lstm(h)
        return h.transpose(1, 2).contiguous()


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2).transpose(1, -1)
        h = self.fc(s).view(s.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1.0 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward_unpacked(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Batch-1 unpadded forward path."""
        x_p = x.permute(2, 0, 1)  # [T, B, C]
        s_exp = style.expand(x_p.shape[0], x_p.shape[1], -1)
        h = torch.cat([x_p, s_exp], dim=-1).transpose(0, 1).transpose(-1, -2)  # [B, C+S, T]
        s_cat = s_exp.permute(1, 2, 0)  # [B, S, T]
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                h = block(h.transpose(-1, -2), style).transpose(-1, -2)
                h = torch.cat([h, s_cat], dim=1)
            else:
                h_t = h.transpose(-1, -2).contiguous()
                if self.training and h_t.is_cuda:
                    with torch.backends.cudnn.flags(enabled=False):
                        h_t, _ = block(h_t)
                else:
                    block.flatten_parameters()
                    h_t, _ = block(h_t)
                if self.training and self.dropout > 0:
                    h_t = F.dropout(h_t, p=self.dropout, training=True)
                h = h_t.transpose(-1, -2).contiguous()
        return h.transpose(-1, -2).contiguous()


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50, dropout: float = 0.1) -> None:
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ])
        self.N = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ])
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = x.transpose(-1, -2).contiguous()
        if self.training and x_t.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                h, _ = self.shared(x_t)
        else:
            self.shared.flatten_parameters()
            h, _ = self.shared(x_t)
        F0 = h.transpose(-1, -2).contiguous()
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = h.transpose(-1, -2).contiguous()
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class CustomAlbert(AlbertModel):
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super().forward(*args, **kwargs).last_hidden_state


class Ninada24kModel(nn.Module):
    """Unified 6-language 24,000 Hz Neural Speech Synthesis Model with Smooth Studio Voice Banks."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config_dict = config
        self.vocab: dict[str, int] = dict(config["vocab"])
        self.vocab["ɭ"] = RETROFLEX_LATERAL_ID
        albert_cfg = AlbertConfig(vocab_size=config["n_token"], **config["plbert"])
        albert_cfg._attn_implementation = "eager"
        self.bert = CustomAlbert(albert_cfg)
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.context_length = int(self.bert.config.max_position_embeddings)
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            **config["istftnet"],
        )

        # Base Indian studio voice curve [510, 1, 256] + smooth per-language adaptation parameters
        self.register_buffer("voice_base_bank", torch.zeros(VOICE_MAX_LEN, 1, 256))
        t_axis = torch.linspace(-1.0, 1.0, VOICE_MAX_LEN, dtype=torch.float32).view(VOICE_MAX_LEN, 1, 1)
        basis = torch.cat([t_axis, 0.5 * (3.0 * t_axis**2 - 1.0)], dim=1)  # [510, 2, 1] Legendre P1, P2
        self.register_buffer("voice_len_basis", basis)
        self.voice_global_delta = nn.ParameterDict({
            lang: nn.Parameter(torch.zeros(1, 1, 256)) for lang in SUPPORTED_LANGS
        })
        self.voice_slope_delta = nn.ParameterDict({
            lang: nn.Parameter(torch.zeros(2, 1, 256)) for lang in SUPPORTED_LANGS
        })

    @classmethod
    def from_pretrained_warmstart(
        cls,
        base_repo: str = "hexgrad/Kokoro-82M",
        voice_repo: str = "onnx-community/Kokoro-82M-v1.0-ONNX",
    ) -> "Ninada24kModel":
        cfg_path = hf_hub_download(repo_id=base_repo, filename="config.json")
        ckpt_path = hf_hub_download(repo_id=base_repo, filename="kokoro-v1_0.pth")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = cls(cfg)
        try:
            raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            raw_state = torch.load(ckpt_path, map_location="cpu")
        for key, sd in raw_state.items():
            submod = getattr(model, key)
            cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
            submod.load_state_dict(cleaned, strict=False)

        # Initialize slot 79 ('ɭ' Vedic/Dravidian retroflex lateral) from 0.70 * 'l' + 0.30 * 'ɖ'
        with torch.no_grad():
            for emb_w in (
                model.bert.embeddings.word_embeddings.weight,
                model.text_encoder.embedding.weight,
            ):
                emb_w[RETROFLEX_LATERAL_ID].copy_(
                    0.70 * emb_w[ALVEOLAR_LATERAL_ID] + 0.30 * emb_w[RETROFLEX_STOP_ID]
                )

            # Initialize Indian Male studio voice prior from hm_omega & hm_psi
            omega_path = hf_hub_download(repo_id=voice_repo, filename="voices/hm_omega.bin")
            psi_path = hf_hub_download(repo_id=voice_repo, filename="voices/hm_psi.bin")
            omega = torch.from_numpy(np.fromfile(omega_path, dtype=np.float32).reshape(-1, 1, 256))
            psi = torch.from_numpy(np.fromfile(psi_path, dtype=np.float32).reshape(-1, 1, 256))
            n_len = min(VOICE_MAX_LEN, omega.shape[0], psi.shape[0])
            prior = 0.78 * omega[:n_len] + 0.22 * psi[:n_len]
            model.voice_base_bank[:n_len].copy_(prior)
            if n_len < VOICE_MAX_LEN:
                model.voice_base_bank[n_len:].copy_(prior[-1:])

        return model

    def get_voice_bank(self, lang: str) -> torch.Tensor:
        """Return full [510, 1, 256] studio style bank for `lang`."""
        lc = lang if lang in self.voice_global_delta else "sa"
        g_delta = self.voice_global_delta[lc]  # [1, 1, 256]
        s_delta = self.voice_slope_delta[lc]   # [2, 1, 256]
        # basis: [510, 2, 1] -> [510, 1, 256]
        smooth_curve = (
            self.voice_len_basis[:, 0:1, :] * s_delta[0:1]
            + self.voice_len_basis[:, 1:2, :] * s_delta[1:2]
        )
        return self.voice_base_bank + g_delta + smooth_curve

    def get_style_for_tokens(self, lang: str, num_core_tokens: int) -> torch.Tensor:
        bank = self.get_voice_bank(lang)
        idx = max(0, min(VOICE_MAX_LEN - 1, int(num_core_tokens)))
        return bank[idx]  # [1, 256]

    @staticmethod
    def build_alignment_matrix(durations: torch.Tensor) -> torch.Tensor:
        """Build [1, T, F] binary monotonic alignment matrix from 1D integer durations [T]."""
        cum_dur = torch.cumsum(durations, dim=0)
        total_len = cum_dur[-1]
        frame_idx = torch.arange(total_len, device=durations.device).unsqueeze(0)  # [1, F]
        starts = (cum_dur - durations).unsqueeze(1)                                # [T, 1]
        ends = cum_dur.unsqueeze(1)                                                # [T, 1]
        return ((frame_idx >= starts) & (frame_idx < ends)).to(torch.float32).unsqueeze(0)

    def forward_train_step(
        self,
        input_ids: torch.LongTensor,
        lang: str,
        gt_dur: torch.LongTensor,
        f0_gt: torch.FloatTensor,
    ) -> dict[str, torch.Tensor]:
        """
        Differentiable single-utterance studio training pass (`input_ids`: [1, T], `gt_dur`: [T], `f0_gt`: [1, 2*F]).

        Uses teacher-forced `gt_dur` alignment and feeds `f0_gt` to the frozen harmonic `SineGen` source so
        spectral harmonics match the studio recording's exact pitch comb without formant blurring, while `F0_pred`,
        `N_pred`, `duration_logits`, `text_encoder`, `bert`, and `voice_banks[lang]` receive crisp analytical gradients.
        """
        T = input_ids.shape[1]
        ref_s = self.get_style_for_tokens(lang, max(1, T - 2))  # [1, 256]
        s_aco = ref_s[:, :128]
        s_pro = ref_s[:, 128:]

        attn_mask = torch.ones_like(input_ids, dtype=torch.int32)
        bert_dur = self.bert(input_ids, attention_mask=attn_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2).contiguous()

        d = self.predictor.text_encoder.forward_unpacked(d_en, s_pro).contiguous()
        if self.training and d.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                x_dur, _ = self.predictor.lstm(d)
        else:
            self.predictor.lstm.flatten_parameters()
            x_dur, _ = self.predictor.lstm(d)
        dur_logits = self.predictor.duration_proj(x_dur).squeeze(0)  # [T, max_dur]
        dur_pred = torch.sigmoid(dur_logits).sum(dim=-1)             # [T]

        aln = self.build_alignment_matrix(gt_dur)                    # [1, T, F]
        en = d.transpose(-1, -2) @ aln                               # [1, d_hid, F]
        f0_pred, n_pred = self.predictor.F0Ntrain(en, s_pro)         # [1, 2*F], [1, 2*F]

        t_en = self.text_encoder.forward_unpacked(input_ids)         # [1, d_hid, T]
        asr = t_en @ aln                                             # [1, d_hid, F]

        # Feed f0_pred through F0_conv (smooth spectral conditioning) while locking harmonic SineGen carrier to f0_gt
        f0_cond = 0.5 * f0_gt + 0.5 * f0_pred
        audio_pred = self.decoder(asr, f0_cond, n_pred, s_aco, har_f0_override=f0_gt).squeeze(1)  # [1, 600*F]

        return {
            "audio_pred": audio_pred,
            "dur_logits": dur_logits,
            "dur_pred": dur_pred,
            "f0_pred": f0_pred,
            "n_pred": n_pred,
            "ref_s": ref_s,
        }

    @torch.no_grad()
    def infer_tokens(
        self,
        input_ids: torch.LongTensor,
        style: torch.FloatTensor,
        speed: torch.FloatTensor | float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic inference pass (`input_ids`: [1, T], `style`: [1, 256]). Returns (`waveform` [1, N], `durations` [T])."""
        if not isinstance(speed, torch.Tensor):
            speed = torch.tensor([float(speed)], dtype=torch.float32, device=input_ids.device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.int32)
        bert_dur = self.bert(input_ids, attention_mask=attn_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s_pro = style[:, 128:]
        d = self.predictor.text_encoder.forward_unpacked(d_en, s_pro)
        self.predictor.lstm.flatten_parameters()
        x_dur, _ = self.predictor.lstm(d)
        dur_logits = self.predictor.duration_proj(x_dur)
        dur = torch.sigmoid(dur_logits).sum(dim=-1) / speed.view(-1, 1)
        pred_dur = torch.round(dur).clamp(min=1).to(torch.int64).squeeze(0)
        aln = self.build_alignment_matrix(pred_dur)
        en = d.transpose(-1, -2) @ aln
        f0_pred, n_pred = self.predictor.F0Ntrain(en, s_pro)
        t_en = self.text_encoder.forward_unpacked(input_ids)
        asr = t_en @ aln
        audio = self.decoder(asr, f0_pred, n_pred, style[:, :128]).squeeze(1)
        return audio, pred_dur


class Ninada24kONNXWrapper(nn.Module):
    """Dynamic-length ONNX export wrapper (`input_ids`, `style`, `speed`) -> (`waveform`, `durations`)."""

    def __init__(self, model: Ninada24kModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor,
        style: torch.FloatTensor,
        speed: torch.FloatTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        attn_mask = torch.ones_like(input_ids, dtype=torch.int32)
        bert_dur = m.bert(input_ids, attention_mask=attn_mask)
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s_pro = style[:, 128:]
        d = m.predictor.text_encoder.forward_unpacked(d_en, s_pro)
        x_dur, _ = m.predictor.lstm(d)
        dur_logits = m.predictor.duration_proj(x_dur)
        dur = torch.sigmoid(dur_logits).sum(dim=-1) / speed.view(-1, 1)
        pred_dur = torch.round(dur).clamp(min=1).to(torch.int64).squeeze(0)
        aln = m.build_alignment_matrix(pred_dur)
        en = d.transpose(-1, -2) @ aln
        f0_pred, n_pred = m.predictor.F0Ntrain(en, s_pro)
        t_en = m.text_encoder.forward_unpacked(input_ids)
        asr = t_en @ aln
        audio = m.decoder(asr, f0_pred, n_pred, style[:, :128]).squeeze(1)
        return audio, pred_dur
