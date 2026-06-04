"""
bgsl/utils/progress.py
-----------------------
Cloud-safe progress reporting for PyTorch Lightning training.

Why not tqdm?
-------------
tqdm overwrites progress lines using ``\\r`` (carriage return).  This only
works in interactive terminals.  In piped / cloud environments —

    subprocess  →  pipe  →  executor  →  Colab / Slurm / nohup

— ``\\r`` does NOT cause visual line-overwriting.  Each tqdm refresh
becomes a separate line, flooding logs with hundreds of near-identical
lines per epoch.

Additionally, tqdm's ``sp()`` function captures the file handle in a
closure at bar-construction time; patching ``bar.fp`` afterwards has no
effect on actual writes.

Solution
--------
CloudProgressBar disables all tqdm bars and replaces them with plain
``print()`` calls at two granularities:

  * **Heartbeat** (every 30 s during an epoch): one line showing step
    progress and current loss so the user knows training is alive.
  * **Epoch summary** (after validation): one line with all key metrics.

Output example::

    Sanity check passed.
      [hb] Epoch 0:  37.6% [750/1993] | 30s | loss=0.2134
      [hb] Epoch 0:  75.2% [1500/1993] | 60s | loss=0.1876
    Epoch   0 |  82.3s | train_loss=0.2341 | val_loss=0.1987 | val_auprc=0.4521

YAML registration
-----------------
Add as the first callback entry.  ``enable_progress_bar`` must be true so
Lightning doesn't strip the callback::

    trainer:
      enable_progress_bar: true
      callbacks:
        - class_path: bgsl.utils.progress.CloudProgressBar
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          ...
"""

from __future__ import annotations

import sys
import time

from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.std import tqdm


class CloudProgressBar(TQDMProgressBar):
    """
    Cloud-safe progress callback for PyTorch Lightning.

    Disables all tqdm bars and replaces them with clean ``print()``-based
    epoch summaries and periodic heartbeats.

    Extends ``TQDMProgressBar`` (rather than ``Callback``) so that
    Lightning recognises it as *the* progress-bar callback and does not
    create a duplicate default bar.
    """

    def __init__(self, heartbeat_interval: float = 30.0) -> None:
        super().__init__()
        self._heartbeat_s = heartbeat_interval
        self._epoch_t0: float = 0.0
        self._last_hb: float = 0.0
        self._total_batches: int = 0

    # ------------------------------------------------------------------
    # Disable all tqdm bars
    # ------------------------------------------------------------------

    def _disabled_bar(self, bar: tqdm) -> tqdm:
        """Return *bar* with ``disable=True`` so all tqdm methods are no-ops."""
        bar.disable = True
        return bar

    def init_sanity_tqdm(self) -> tqdm:
        return self._disabled_bar(super().init_sanity_tqdm())

    def init_train_tqdm(self) -> tqdm:
        return self._disabled_bar(super().init_train_tqdm())

    def init_validation_tqdm(self) -> tqdm:
        return self._disabled_bar(super().init_validation_tqdm())

    def init_test_tqdm(self) -> tqdm:
        return self._disabled_bar(super().init_test_tqdm())

    def init_predict_tqdm(self) -> tqdm:
        return self._disabled_bar(super().init_predict_tqdm())

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self._epoch_t0 = time.monotonic()
        self._last_hb = self._epoch_t0
        n = trainer.num_training_batches
        self._total_batches = n if isinstance(n, int) else 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        now = time.monotonic()
        if now - self._last_hb >= self._heartbeat_s:
            step = batch_idx + 1
            total = self._total_batches
            elapsed = now - self._epoch_t0

            if total:
                pct = f"{step / total * 100:5.1f}%"
                progress = f"[{step}/{total}]"
            else:
                pct = "?"
                progress = f"[{step}]"

            loss_val = trainer.callback_metrics.get("train_loss_step")
            loss_str = f" | loss={float(loss_val):.4f}" if loss_val is not None else ""

            print(
                f"  [hb] Epoch {trainer.current_epoch}: "
                f"{pct} {progress} | {elapsed:.0f}s{loss_str}",
                flush=True,
            )
            self._last_hb = now

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        # Record elapsed time; the full summary is printed after validation.
        self._train_epoch_elapsed = time.monotonic() - self._epoch_t0

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        if trainer.sanity_checking:
            print("Sanity check passed.", flush=True)
            return

        elapsed = getattr(self, "_train_epoch_elapsed", 0.0)
        m = trainer.callback_metrics

        parts = [f"Epoch {trainer.current_epoch:>3d}", f"{elapsed:5.1f}s"]
        for key in [
            "train_loss", "val_loss",
            "train_auroc", "val_auroc",
            "train_auprc", "val_auprc",
        ]:
            val = m.get(key)
            if val is not None:
                parts.append(f"{key}={float(val):.4f}")

        print(" | ".join(parts), flush=True)

    # ------------------------------------------------------------------
    # Test hooks
    # ------------------------------------------------------------------

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        super().on_test_epoch_end(trainer, pl_module)
        m = trainer.callback_metrics
        parts = ["Test"]
        for key in sorted(m.keys()):
            if key.startswith("test_"):
                parts.append(f"{key}={float(m[key]):.4f}")
        if len(parts) > 1:
            print(" | ".join(parts), flush=True)
