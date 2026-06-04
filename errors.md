--- Execution Mode: LOCAL_SEQUENTIAL | Total Jobs: 3 | Batch: 20260604-070735__5ca06212 ---
Outputs root: /content/bgsl/outputs/ablation/core_hypothesis/20260604-070735__5ca06212

=== [1/3] 01_BCE_Baseline_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/core_hypothesis/20260604-070735__5ca06212/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model         │ GRUPredictor     │  179 K │ train │     0 │
│ 1 │ loss_fn       │ WeightedBCELoss  │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 179 K                                                         
Non-trainable params: 0                                                         
Total params: 179 K                                                             
Total estimated model params size (MB): 0.716                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
