"""
cli.py
------
LightningCLI entrypoint for BGSL.
Instantiates all classes directly from YAML using jsonargparse.
Uses subclass mode to inject backbones and switch domain logic.
"""

from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl

from bgsl.train.common.module import BaseBGSLLightningModule


class BGSLCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.feature_dim", "model.init_args.backbone.init_args.input_dim", apply_on="instantiate")

def main():
    """
    Standard LightningCLI entrypoint using Subclass Mode.
    
    Usage:
      Training:
        bgsl-train fit --config experiments/configs/physionet_gru.yaml
        
      Testing:
        bgsl-train test --config experiments/configs/physionet_gru.yaml --ckpt_path best
    """
    cli = BGSLCLI(
        model_class=BaseBGSLLightningModule,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        run=True, # automatically runs fit/test
    )

if __name__ == "__main__":
    main()
