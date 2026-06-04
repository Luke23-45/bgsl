"""
cli.py
------
LightningCLI entrypoint for BGSL.
This replaces custom train.py and evaluate.py scripts.
Instantiates all classes directly from YAML using jsonargparse.
"""

from typing import Any

from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl

from bgsl.models.lightning import BGSLLightningModule
from bgsl.data.datamodule import PhysioNetDataModule


class BGSLLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        """Add any custom CLI arguments if needed, otherwise jsonargparse handles it."""
        # By default, LightningCLI handles linking datamodule parameters to model if necessary.
        pass


def main():
    """
    Standard LightningCLI entrypoint.
    
    Usage:
      Training:
        bgsl-train fit --config experiments/configs/physionet_gru.yaml
        
      Testing:
        bgsl-train test --config experiments/configs/physionet_gru.yaml --ckpt_path best
    """
    # The CLI automatically handles parsing, instantiating the Trainer, DataModule, and LightningModule,
    # and then calls trainer.fit(), trainer.test(), etc., based on the subcommand.
    cli = BGSLLightningCLI(
        model_class=BGSLLightningModule,
        datamodule_class=PhysioNetDataModule,
        save_config_kwargs={"overwrite": True},
        run=True, # automatically runs fit/test
    )

if __name__ == "__main__":
    main()
