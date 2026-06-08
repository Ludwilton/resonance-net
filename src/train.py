"""
train.py — Training loop for the neural amplifier model

Follows Juvela et al. arXiv:2403.08559:
  - LSTM-32, ESR loss, Adam, 1 M iterations
  - 48 kHz, 1-second segments
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .rnet_model import AmpModel, AudioDataset, ESRLoss


def train_model(
    input_file:         str,
    output_file:        str,
    model_save_path:    str   = "models/amp_model.pth",

    hidden_size:        int   = 32,
    num_controls:       int   = 0,

    learning_rate:      float = 2e-3,

    max_iters:          int   = 100000,
    val_every:          int   = 1000,
    print_every:        int   = 200,

    lr_patience:        int   = 20,      # ReduceLROnPlateau patience
    lr_factor:          float = 0.5,
    min_lr:             float = 1e-5,

    batch_size:         int   = 40,      # Wright DAFx-2019: mini-batches of 40
    val_split:          float = 0.1,
    num_workers:        int   = 2,

    grad_clip_norm:     float = 1.0,

    pre_emphasis_coef:  float = 0.95,
) -> AmpModel:
    """
    Train the LSTM-32 amplifier model.

    Args:
        input_file:        Path to dry/DI audio (.wav, 48 kHz).
        output_file:       Path to amplifier output audio (.wav, 48 kHz).
        model_save_path:   Where to save the best checkpoint (.pth).
        hidden_size:       LSTM hidden size (default 32).
        num_controls:      Number of conditioning controls (0 = no controls).
        learning_rate:     Initial Adam learning rate.
        max_iters:         Total training iterations.
        val_every:         Validate every N iterations.
        print_every:       Print train loss every N iterations.
        lr_patience:       ReduceLROnPlateau patience counted in validation steps.
        lr_factor:         LR reduction factor.
        min_lr:            Minimum learning rate; training stops if LR falls below.
        batch_size:        Mini-batch size (number of 1-second chunks per step).
        val_split:         Fraction of chunks held out for validation.
        num_workers:       DataLoader worker processes.
        grad_clip_norm:    Max gradient norm for clipping.
        pre_emphasis_coef: Pre-emphasis filter coefficient (default 0.95).

    Returns:
        Trained AmpModel.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # = Metal Performance Shaders for macOS
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")


    full_ds = AudioDataset(input_file, output_file)

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin,
        drop_last   = True,
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
        drop_last   = False,
        persistent_workers = (num_workers > 0),
    )

    print(
        f"Chunks — train: {n_train:,}  val: {n_val:,}  "
        f"steps/epoch ≈ {n_train // batch_size:,}"
    )


    model = AmpModel(hidden_size=hidden_size, num_controls=num_controls).to(device)
    print(f"Model params: {model.num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = lr_factor,
        patience = lr_patience,
    )
    loss_fn = ESRLoss(pre_emphasis_coef=pre_emphasis_coef)


    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)


    iteration    = 0
    best_val_esr = float("inf")
    train_iter   = iter(train_loader)
    t0           = time.time()

    header = f"{'iter':>10}  {'train ESR':>10}  {'val ESR':>10}  {'lr':>10}  {'elapsed':>10}"
    print("\n" + header)
    print("-" * len(header))

    while iteration < max_iters:
        model.train()

        try:
            x_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, y_batch = next(train_iter)

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Hidden state is zero-initialised each mini-batch, Wright DAFx-2019
        # "at the start of each mini-batch the recurrent unit's initial state is set to 0"

        optimizer.zero_grad()
        pred, _ = model(x_batch, hidden=None)
        loss    = loss_fn(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        iteration += 1


        if iteration % val_every == 0:
            val_esr = _evaluate(model, val_loader, loss_fn, device)
            lr_now  = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            print(
                f"{iteration:>10,}  {loss.item():>10.5f}  {val_esr:>10.5f}  "
                f"{lr_now:>10.2e}  {elapsed:>9.0f}s"
            )

            scheduler.step(val_esr)

            if optimizer.param_groups[0]["lr"] < min_lr:
                print(f"\nLR below {min_lr:.1e} — stopping early at iter {iteration:,}.")
                break

            if val_esr < best_val_esr:
                best_val_esr = val_esr
                _save_checkpoint(
                    path      = save_path,
                    model     = model,
                    optimizer = optimizer,
                    iteration = iteration,
                    val_esr   = val_esr,
                    meta      = {
                        "hidden_size":        hidden_size,
                        "num_controls":       num_controls,
                        "sample_rate":        AudioDataset.SAMPLE_RATE,
                        "chunk_size":         AudioDataset.CHUNK_SIZE,
                        "pre_emphasis_coef":  pre_emphasis_coef,
                    },
                )
                print(f"New best saved  (val ESR {val_esr:.5f})")

        elif iteration % print_every == 0:
            elapsed = time.time() - t0
            print(f"{iteration:>10,}  {loss.item():>10.5f}  {'':>10}  {'':>10}  {elapsed:>9.0f}s")

    print(f"\nDone. Best val ESR: {best_val_esr:.5f}  saved: {save_path}")

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def _evaluate(
    model:      AmpModel,
    loader:     DataLoader,
    loss_fn:    ESRLoss,
    device:     torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for x_v, y_v in loader:
        x_v = x_v.to(device, non_blocking=True)
        y_v = y_v.to(device, non_blocking=True)
        pred, _ = model(x_v, hidden=None)
        losses.append(loss_fn(pred, y_v).item())
    return float(np.mean(losses))


def _save_checkpoint(
    path:      Path,
    model:     AmpModel,
    optimizer: optim.Optimizer,
    iteration: int,
    val_esr:   float,
    meta:      dict,
) -> None:
    torch.save(
        {
            "iteration":            iteration,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_esr":              val_esr,
            **meta,
        },
        path,
    )



# CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM-32 neural amp model")
    p.add_argument("--input",        required=True,        help="DI / dry input .wav")
    p.add_argument("--output",       required=True,        help="Amp output .wav")
    p.add_argument("--save",         default="models/amp_model.pth")
    p.add_argument("--hidden-size",  type=int,   default=32)
    p.add_argument("--num-controls", type=int,   default=0,
                   help="Number of conditioning control inputs (0 = no controls)")
    p.add_argument("--lr",           type=float, default=2e-3)
    p.add_argument("--max-iters",    type=int,   default=1_000_000)
    p.add_argument("--val-every",    type=int,   default=1_000)
    p.add_argument("--batch-size",   type=int,   default=40)
    p.add_argument("--workers",      type=int,   default=2)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_model(
        input_file      = args.input,
        output_file     = args.output,
        model_save_path = args.save,
        hidden_size     = args.hidden_size,
        num_controls    = args.num_controls,
        learning_rate   = args.lr,
        max_iters       = args.max_iters,
        val_every       = args.val_every,
        batch_size      = args.batch_size,
        num_workers     = args.workers,
    )


if __name__ == "__main__":
    main()
