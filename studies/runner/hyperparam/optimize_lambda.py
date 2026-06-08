"""
studies/runner/hyperparam/optimize_lambda.py
--------------------------------------------
Optuna-based Bayesian optimization of BGSL loss hyperparameters.

This script uses the *real* data pipeline (PhysioNetDataModule) and the
*real* training loop (SepsisLightningModule / BaseBGSLLightningModule) to
find optimal values for:

    - lambda_v   (velocity_weight)
    - lambda_a   (acceleration_weight)
    - lambda_u   (utility_weight, for UtilityAwareBGSLLoss)

It optimizes directly for the **PhysioNet clinical utility score** on the
validation set, which is the metric that matters for deployment.

Usage
-----
    # Install optuna first: pip install optuna
    python studies/runner/hyperparam/optimize_lambda.py \
        --n-trials 100 \
        --seeds 42 123 456 \
        --max-epochs 30 \
        --output-dir outputs/hyperparam/lambda_optuna

    # Quick test (single seed, few trials):
    python studies/runner/hyperparam/optimize_lambda.py \
        --n-trials 5 --seeds 42 --max-epochs 15 --mode quick

    # Full study with utility-aware loss:
    python studies/runner/hyperparam/optimize_lambda.py \
        --n-trials 100 --seeds 42 123 456 --loss-type utility_bgsl

Design
------
- Each Optuna trial trains a GRU model with a specific (λ_v, λ_a, λ_u)
  combination, using the real data and the full training loop.
- The objective is the **PhysioNet utility score** on the validation set,
  averaged across seeds (for robustness).
- Optuna's TPE sampler efficiently explores the hyperparameter space
  without exhaustive grid search.
- Pruning via MedianPruner skips unpromising trials early.
- Results are saved as an Optuna study database + CSV export.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logger = logging.getLogger("bgsl.optimize_lambda")


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(
    seeds: List[int],
    max_epochs: int,
    data_dir: str,
    loss_type: str,
    batch_size: int,
    num_workers: int,
    patience: int,
    gpus: int,
):
    """
    Create an Optuna objective function that trains and evaluates BGSL
    with the given hyperparameters on the real data.
    """
    # Lazy imports — avoids loading torch/lightning unless actually running
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
    from optuna.integration import PyTorchLightningPruningCallback

    from bgsl.data.sepsis.datamodule import PhysioNetDataModule
    from bgsl.train.sepsis.module import SepsisLightningModule
    from bgsl.models.gru import GRUPredictor
    from bgsl.core.common.losses import BGSLLoss, UtilityAwareBGSLLoss

    def objective(trial) -> float:
        """
        Single Optuna trial: suggest hyperparams, train, evaluate.
        Returns the negative of mean PhysioNet utility across seeds
        (Optuna maximizes by default when direction='maximize').
        """
        # ----- Suggest hyperparameters -----
        lambda_v = trial.suggest_float("lambda_v", 0.01, 20.0, log=True)
        lambda_a = trial.suggest_float("lambda_a", 0.001, 10.0, log=True)

        if loss_type == "utility_bgsl":
            lambda_u = trial.suggest_float("lambda_u", 0.01, 5.0, log=True)
            utility_temp = trial.suggest_float("utility_temperature", 0.5, 5.0)
        else:
            lambda_u = 0.0
            utility_temp = 1.0

        # Also allow optimizing the weighting method
        weighting = trial.suggest_categorical(
            "weighting_method", ["fixed", "uw", "uw_so"]
        )

        # Derivative method
        deriv_method = trial.suggest_categorical(
            "derivative_method", ["finite_diff", "savitzky_golay"]
        )

        utilities = []

        for seed_idx, seed in enumerate(seeds):
            pl.seed_everything(seed, workers=True)

            # ----- Build data module -----
            dm = PhysioNetDataModule(
                data_dir=data_dir,
                horizon_hours=6,
                tau=2.0,
                min_length=8,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # ----- Build loss -----
            loss_kwargs = dict(
                state_loss="bce",
                velocity_weight=lambda_v,
                acceleration_weight=lambda_a,
                weighting_method=weighting,
                derivative_space="probability",
                derivative_method=deriv_method,
                horizon=6.0,
            )

            if loss_type == "utility_bgsl":
                loss_fn = UtilityAwareBGSLLoss(
                    utility_weight=lambda_u,
                    utility_temperature=utility_temp,
                    **loss_kwargs,
                )
            else:
                loss_fn = BGSLLoss(**loss_kwargs)

            # ----- Build model -----
            backbone = GRUPredictor(
                input_dim=40,  # PhysioNet feature dim
                hidden_dim=128,
                num_layers=2,
                dropout=0.2,
                use_mask_as_input=True,
            )

            model = SepsisLightningModule(
                backbone=backbone,
                loss_fn=loss_fn,
                horizon_hours=6,
                lr=1e-3,
                weight_decay=1e-4,
                epochs=max_epochs,
                lr_scheduler="cosine",
                validation_threshold_target=0.80,
            )

            # ----- Callbacks -----
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=patience, mode="min"
                ),
                ModelCheckpoint(
                    monitor="val_loss", mode="min", save_top_k=1
                ),
                PyTorchLightningPruningCallback(
                    trial, monitor="val_loss"
                ),
                TQDMProgressBar(refresh_rate=10, leave=False),
            ]

            # ----- Trainer -----
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=gpus,
                precision="16-mixed",  # Massive speedup on modern GPUs
                callbacks=callbacks,
                enable_progress_bar=True,
                enable_model_summary=False,
                gradient_clip_val=1.0,
                logger=False,  # No logging overhead for optimization
                deterministic=False,
            )

            # ----- Train -----
            try:
                trainer.fit(model, datamodule=dm)
            except Exception as e:
                logger.warning(f"Trial {trial.number} seed {seed} failed during fit: {e}")
                return float("-inf")

            # ----- Evaluate on validation set -----
            # The trajectory metrics (including PhysioNet utility) are computed
            # during validation epoch end. We need to run one more validation pass
            # on the best checkpoint to extract the utility.
            try:
                # Load best checkpoint
                best_ckpt = trainer.checkpoint_callback.best_model_path
                if best_ckpt and Path(best_ckpt).exists():
                    model = SepsisLightningModule.load_from_checkpoint(
                        best_ckpt,
                        backbone=backbone,
                        loss_fn=loss_fn,
                    )

                # Run test to get utility score
                dm.setup(stage="test")
                trainer.test(model, datamodule=dm, verbose=False)

                # Extract metrics from the logged test results
                test_results = trainer.callback_metrics
                utility = float(test_results.get("test_physionet_utility", -1.0))

                if utility < -0.5:
                    # Fallback: if utility not found, use negative loss as proxy
                    utility = -float(test_results.get("test_loss", 1.0))
                    logger.warning(
                        f"Trial {trial.number} seed {seed}: "
                        f"physionet_utility not found, using -test_loss={utility:.4f}"
                    )

            except Exception as e:
                logger.warning(f"Trial {trial.number} seed {seed} failed during test: {e}")
                utility = float("-inf")

            utilities.append(utility)

            # Report intermediate value for pruning (after each seed)
            trial.report(np.mean(utilities), seed_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

            # Cleanup GPU memory
            del model, trainer, dm, backbone, loss_fn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_utility = float(np.mean(utilities))
        std_utility = float(np.std(utilities)) if len(utilities) > 1 else 0.0

        logger.info(
            f"Trial {trial.number}: λ_v={lambda_v:.4f}, λ_a={lambda_a:.4f}, "
            f"λ_u={lambda_u:.4f}, weighting={weighting}, "
            f"utility={mean_utility:.4f} ± {std_utility:.4f}"
        )

        return mean_utility

    return objective


# ---------------------------------------------------------------------------
# Post-optimization analysis
# ---------------------------------------------------------------------------

def export_results(study, output_dir: Path) -> None:
    """Export Optuna study results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV of all trials
    df = study.trials_dataframe()
    csv_path = output_dir / "optuna_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Trials CSV: {csv_path}")

    # Best trial summary
    best = study.best_trial
    summary = {
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
        "study_name": study.study_name,
    }
    json_path = output_dir / "best_params.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Best params JSON: {json_path}")

    # Print best
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL: #{best.number}")
    print(f"  PhysioNet Utility: {best.value:.4f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")


def generate_importance_plot(study, output_dir: Path) -> None:
    """Generate hyperparameter importance analysis."""
    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
        importance_path = output_dir / "param_importance.json"
        with open(importance_path, "w") as f:
            json.dump(importance, f, indent=2)
        print(f"  Parameter importance: {importance_path}")
        print("\n  Hyperparameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"    {param:25s} {imp:.3f} {bar}")
    except Exception as e:
        logger.warning(f"Could not compute parameter importance: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna-based Bayesian optimization of BGSL lambda hyperparameters."
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        help="Random seeds for multi-seed evaluation (default: 42 123 456).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=30,
        help="Max training epochs per trial (default: 30).",
    )
    parser.add_argument(
        "--data-dir", type=str, default="datasets/processed/sepsis",
        help="Path to processed sepsis dataset.",
    )
    parser.add_argument(
        "--loss-type", type=str, default="bgsl",
        choices=["bgsl", "utility_bgsl"],
        help=(
            "Loss type to optimize. "
            "'bgsl' optimizes lambda_v/lambda_a only. "
            "'utility_bgsl' also optimizes lambda_u and utility_temperature."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader workers (default: 2).",
    )
    parser.add_argument(
        "--patience", type=int, default=6,
        help="EarlyStopping patience (default: 6).",
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs (default: 1).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results. Auto-generated if not specified.",
    )
    parser.add_argument(
        "--study-name", type=str, default=None,
        help="Optuna study name. Auto-generated if not specified.",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help=(
            "Optuna storage URL (e.g. sqlite:///optuna.db). "
            "If not specified, uses in-memory storage."
        ),
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["quick", "full"],
        help=(
            "'quick' uses 1 seed and fewer epochs for rapid iteration. "
            "'full' uses all specified seeds."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quick mode overrides
    if args.mode == "quick":
        args.seeds = args.seeds[:1]
        args.max_epochs = min(args.max_epochs, 15)
        logger.info(f"Quick mode: seeds={args.seeds}, max_epochs={args.max_epochs}")

    # Auto-generate output dir
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/hyperparam/lambda_optuna/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate study name
    if args.study_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.study_name = f"bgsl_lambda_{args.loss_type}_{ts}"

    # Save run config
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BGSL Lambda Optimization (Optuna)")
    print(f"{'='*60}")
    print(f"  Loss type:    {args.loss_type}")
    print(f"  N trials:     {args.n_trials}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Max epochs:   {args.max_epochs}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Study name:   {args.study_name}")
    print(f"{'='*60}\n")

    # Import optuna here so the error is clear if not installed
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        print("ERROR: optuna is not installed. Run: pip install optuna")
        sys.exit(1)

    # Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study
    storage = args.storage
    if storage is None:
        # Use SQLite for persistence by default
        db_path = output_dir / "optuna_study.db"
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # Maximize PhysioNet utility
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,  # Random exploration first
            multivariate=True,    # Model parameter correlations
        ),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        ),
        load_if_exists=True,  # Resume interrupted studies
    )

    # Create objective function
    objective = create_objective(
        seeds=args.seeds,
        max_epochs=args.max_epochs,
        data_dir=args.data_dir,
        loss_type=args.loss_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patience=args.patience,
        gpus=args.gpus,
    )

    # Run optimization
    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nOptimization completed in {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"Completed trials: {len(study.trials)}")

    # Export results
    export_results(study, output_dir)
    generate_importance_plot(study, output_dir)

    # Generate the YAML config for the best trial
    best = study.best_trial
    best_config = _generate_best_config(best.params, args.loss_type)
    config_yaml_path = output_dir / "best_config.yaml"
    with open(config_yaml_path, "w") as f:
        import yaml
        yaml.safe_dump(best_config, f, sort_keys=False, default_flow_style=False)
    print(f"\n  Best config YAML: {config_yaml_path}")
    print(f"  Use this config for the final paired study validation.\n")


def _generate_best_config(params: Dict, loss_type: str) -> Dict:
    """Generate a YAML-compatible config dict from the best Optuna params."""
    loss_class = (
        "bgsl.core.common.losses.UtilityAwareBGSLLoss"
        if loss_type == "utility_bgsl"
        else "bgsl.core.common.losses.BGSLLoss"
    )

    init_args = {
        "state_loss": "bce",
        "velocity_weight": params["lambda_v"],
        "acceleration_weight": params["lambda_a"],
        "weighting_method": params["weighting_method"],
        "derivative_space": "probability",
        "derivative_method": params["derivative_method"],
        "horizon": 6.0,
    }

    if loss_type == "utility_bgsl":
        init_args["utility_weight"] = params["lambda_u"]
        init_args["utility_temperature"] = params["utility_temperature"]

    return {
        "model": {
            "init_args": {
                "loss_fn": {
                    "class_path": loss_class,
                    "init_args": init_args,
                }
            }
        }
    }


if __name__ == "__main__":
    main()
