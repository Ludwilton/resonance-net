"""
rnet_model.py — Neural amplifier model core

Architecture follows arXiv:2403.08559 (Juvela et al.)
  - Single-layer LSTM, 32 hidden cells (LSTM-32)
  - Residual connection: network predicts correction to input
  - Optional conditioning: control knob values concatenated to input

Loss follows Wright et al. DAFx-2019 / Damskagg et al. ICASSP-2019:
  - ESR with high-pass pre-emphasis

Dataset:
  - 1-second chunks at 48 kHz (48 000 samples), as in Juvela 2403.08559
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AmpModel(nn.Module):
    """
    Single-layer LSTM 32 hidden cells as described in Juvela et al. arXiv:2403.08559

    No-controls variant:   AmpModel()
    With-controls variant: AmpModel(num_controls=5)


    Output: input signal + learned correction (residual).
    """

    def __init__(self, hidden_size: int = 32, num_controls: int = 0):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_controls = num_controls
        self.input_size   = 1 + num_controls

        self.lstm = nn.LSTM(
            input_size  = self.input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self._init_weights()


    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()
                n = param.shape[0]
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def forward(
        self,
        x:        torch.Tensor,
        controls: torch.Tensor | None = None,
        hidden:   tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        if controls is not None:
            if controls.dim() == 2:
                controls = controls.unsqueeze(1).expand(-1, x.shape[1], -1)
            lstm_in = torch.cat([x, controls], dim=-1)
        else: # controls datasets have yet to be implemented; use static version for now
            if self.num_controls > 0:
                raise ValueError(
                    f"Model expects {self.num_controls} controls but none were provided."
                )
            lstm_in = x

        lstm_out, hidden = self.lstm(lstm_in, hidden)
        correction = self.fc(lstm_out)
        return x + correction, hidden


    def detach_hidden(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if hidden is None:
            return None
        return tuple(h.detach() for h in hidden)


    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class ESRLoss(nn.Module):
    """
    Error-to-Signal Ratio high-pass pre-emphasis.

        H(z) = 1 - 0.95 * z^-1

    Pre-emphasis is applied to both prediction and target before computing
    the ratio, giving extra weight to high-frequency errors

        ESR = sum((y_pre - y_hat_pre)^2) / (sum(y_pre^2) + eps)

    Reference:
        Damskagg et al. ICASSP 2019
        Wright et al. DAFx 2019
        Juvela et al. arXiv:2403.08559
    """

    def __init__(self, pre_emphasis_coef: float = 0.95, eps: float = 1e-8):
        super().__init__()
        self.coef = pre_emphasis_coef
        self.eps  = eps

    def _pre_emphasis(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:, :] - self.coef * x[:, :-1, :]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_e   = self._pre_emphasis(pred)
        target_e = self._pre_emphasis(target)
        error    = target_e - pred_e
        esr = torch.sum(error ** 2) / (torch.sum(target_e ** 2) + self.eps)
        return esr


class AudioDataset(Dataset):
    """
    Paired (DI input, amp output) audio dataset.

    Splits the audio into non-overlapping 1-second chunks (CHUNK_SIZE samples).

    Normalisation:
        Input  — peak-normalised to [-1, 1]
        Target — stored as-is (model must learn absolute output level)

    Args:
        input_file:  path to dry / DI guitar recording
        output_file: path to corresponding amplifier output recording
        chunk_size:  samples per training segment (default: 48 000 = 1 s @ 48 kHz)
    """

    SAMPLE_RATE = 48_000
    CHUNK_SIZE  = 48_000

    def __init__(
        self,
        input_file:  str,
        output_file: str,
        chunk_size:  int = CHUNK_SIZE,
    ):
        x, sr_x = sf.read(input_file,  dtype="float32", always_2d=False)
        y, sr_y = sf.read(output_file, dtype="float32", always_2d=False)

        if sr_x != self.SAMPLE_RATE:
            raise ValueError(f"Input sample rate is {sr_x} Hz; expected {self.SAMPLE_RATE} Hz")
        if sr_y != self.SAMPLE_RATE:
            raise ValueError(f"Output sample rate is {sr_y} Hz; expected {self.SAMPLE_RATE} Hz")

        if x.ndim > 1:
            x = x[:, 0]
        if y.ndim > 1:
            y = y[:, 0]

        # Peak normalise
        x = x / (np.max(np.abs(x)) + 1e-8)

        n = min(len(x), len(y))
        n = (n // chunk_size) * chunk_size

        if n == 0:
            raise ValueError(
                f"Audio is shorter than one chunk ({chunk_size / self.SAMPLE_RATE:.1f} s). "
            )

        x = x[:n]
        y = y[:n]

        self.x = torch.from_numpy(x.reshape(-1, chunk_size, 1))
        self.y = torch.from_numpy(y.reshape(-1, chunk_size, 1))

        duration_s = n / self.SAMPLE_RATE
        print(
            f"AudioDataset: {len(self.x):,} chunks × {chunk_size / self.SAMPLE_RATE:.1f} s "
            f"= {duration_s:.1f} s of audio  ({input_file})"
        )


    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
