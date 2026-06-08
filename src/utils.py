"""
utils.py


Functions
---------
load_model(path, device)
    Load a saved checkpoint and reconstruct the AmpModel.

process_audio(model, input_file, output_file, buffer_size, device, controls)
    Run a trained model over an audio file in streaming fashion.

resume_training(checkpoint_path, input_file, output_file, **kwargs)
    Continue training from a saved checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from scipy import signal

from .rnet_model import AmpModel, AudioDataset


def load_model(
    checkpoint_path: str,
    device:          str | torch.device = "cpu",
) -> AmpModel:
    """
    Load a model from a checkpoint saved by train_model().

    Args:
        checkpoint_path: Path to the .pth file.
        device:          Torch device string or object.

    Returns:
        AmpModel in eval mode on the specified device.
    """
    device = torch.device(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)

    hidden_size  = ckpt.get("hidden_size",  32)
    num_controls = ckpt.get("num_controls",  0)

    model = AmpModel(hidden_size=hidden_size, num_controls=num_controls)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    sample_rate = ckpt.get("sample_rate", AudioDataset.SAMPLE_RATE)
    val_esr     = ckpt.get("val_esr",     float("nan"))
    iteration   = ckpt.get("iteration",   -1)

    print(
        f"Loaded '{checkpoint_path}'  "
        f"iter {iteration:,}  val ESR {val_esr:.5f}  "
        f"LSTM-{hidden_size}  controls {num_controls}  "
        f"{sample_rate} Hz"
    )
    return model



def process_audio(
    model:        AmpModel,
    input_file:   str,
    output_file:  str,
    buffer_size:  int  = 2048,
    device:       str | torch.device = "cuda",
    controls:     list[float] | np.ndarray | None = None,
) -> None:
    """
    Process an audio file through the model.

    Args:
        model:       Trained AmpModel
        input_file:  Path to dry/DI input .wav
        output_file: Destination path for the processed output .wav.
        buffer_size: Number of samples processed per forward pass.
        device:      Torch device to run inference on.
        controls:    List/array of C control values in [0, 1], one per control.
                     Pass None for models trained without controls.
    """
    device = torch.device(device)
    model.eval().to(device)

    x, sr = sf.read(input_file, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]

    assert sr == AudioDataset.SAMPLE_RATE, (
        f"Input is {sr} Hz; model expects {AudioDataset.SAMPLE_RATE} Hz"
    )

    x = x / (np.max(np.abs(x)) + 1e-8)

    ctrl_t: torch.Tensor | None = None
    if model.num_controls > 0:
        if controls is None:
            raise ValueError(
                f"Model has {model.num_controls} control(s); provide them via `controls`."
            )
        c = np.asarray(controls, dtype=np.float32)
        if c.shape != (model.num_controls,):
            raise ValueError(
                f"Expected controls of shape ({model.num_controls},), got {c.shape}."
            )
        ctrl_t = torch.from_numpy(c).unsqueeze(0).to(device)


    hidden:         tuple[torch.Tensor, torch.Tensor] | None = None
    output_chunks:  list[np.ndarray] = []

    total_samples   = len(x)
    n_full_buffers  = total_samples // buffer_size
    remainder       = total_samples  % buffer_size

    def _run_buffer(samples: np.ndarray) -> np.ndarray:
        nonlocal hidden
        x_t = (
            torch.from_numpy(samples)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(device)
        )
        with torch.no_grad():
            out, hidden = model(x_t, controls=ctrl_t, hidden=hidden)
        hidden = model.detach_hidden(hidden)
        return out.squeeze().cpu().numpy()

    for i in range(n_full_buffers):
        chunk = x[i * buffer_size : (i + 1) * buffer_size]
        output_chunks.append(_run_buffer(chunk))

    if remainder > 0:
        output_chunks.append(_run_buffer(x[n_full_buffers * buffer_size :]))

    output = np.concatenate(output_chunks).astype(np.float32)
    sf.write(output_file, output, sr, subtype="FLOAT")

    duration = len(output) / sr
    print(f"Output written: '{output_file}'  ({duration:.1f} s)")



def resume_training(
    checkpoint_path: str,
    input_file:      str,
    output_file:     str,
    max_iters:       int   = 2_000_000,
    **train_kwargs,
) -> AmpModel:
    """
    Resume training from a saved checkpoint.

    The model architecture (hidden_size, num_controls) and optimiser state
    are restored from the checkpoint.

    Args:
        checkpoint_path: Path to an existing .pth checkpoint.
        input_file:      Dry/DI input audio.
        output_file:     Amp output audio.
        max_iters:       Total iterations for the resumed run.
        **train_kwargs:  Any keyword argument accepted by train_model().

    Returns:
        Trained AmpModel.
    """
    from .train import train_model  # noqa: PLC0415

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    hidden_size  = ckpt.get("hidden_size", 32)
    num_controls = ckpt.get("num_controls", 0)
    start_iter   = ckpt.get("iteration", 0)

    print(
        f"Resuming from '{checkpoint_path}'"
        f"iter {start_iter:,} → {max_iters:,}"
    )

    return train_model(
        input_file      = input_file,
        output_file     = output_file,
        model_save_path = checkpoint_path,
        hidden_size     = hidden_size,
        num_controls    = num_controls,
        max_iters       = max_iters,
        **train_kwargs,
    )



def evaluate_model(target_file, pred_file, sample_rate=48000, snippet_start_sec=2.0, snippet_len_sec=0.05):
    y_target, _ = sf.read(target_file, dtype='float32')
    y_pred, _   = sf.read(pred_file,   dtype='float32')

    if y_target.ndim > 1: y_target = y_target[:, 0]
    if y_pred.ndim   > 1: y_pred   = y_pred[:,   0]

    n = min(len(y_target), len(y_pred))
    y_target, y_pred = y_target[:n], y_pred[:n]

    esr = np.sum((y_target - y_pred) ** 2) / (np.sum(y_target ** 2) + 1e-8)
    print(f"Overall ESR: {esr:.5f}")

    start_idx = int(snippet_start_sec * sample_rate)
    end_idx   = start_idx + int(snippet_len_sec * sample_rate)
    t_ms      = np.linspace(0, snippet_len_sec * 1000, end_idx - start_idx)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"LSTM Model Evaluation  —  Loss: {esr:.5f}", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(t_ms, y_target[start_idx:end_idx], label='Target',     linewidth=1.2)
    ax.plot(t_ms, y_pred[start_idx:end_idx],   label='Prediction', linewidth=1.2, linestyle='--')
    ax.set_title('Waveform Match')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    error = y_target[start_idx:end_idx] - y_pred[start_idx:end_idx]
    ax.plot(t_ms, error, color='tomato', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
    ax.set_title('Residual Error')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Error')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    f, Pxx_target = signal.welch(y_target, fs=sample_rate, nperseg=4096)
    _, Pxx_pred   = signal.welch(y_pred,   fs=sample_rate, nperseg=4096)

    ax.semilogx(f, 10 * np.log10(Pxx_target + 1e-10), label='Target',     linewidth=1.2)
    ax.semilogx(f, 10 * np.log10(Pxx_pred   + 1e-10), label='Prediction', linewidth=1.2, linestyle='--')
    ax.set_xlim(20, 20000)
    ax.set_title('Frequency Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB/Hz)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.show()

