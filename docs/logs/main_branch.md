--- Execution Mode: LOCAL_SEQUENTIAL | Total Jobs: 6 | Batch: 20260606-160650__1ea45579 ---
Outputs root: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579

=== [1/6] 01_BCE_Baseline_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn       │ BCELoss          │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  62.9% [314/499] | 30s | loss=0.0958


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0039 | val_loss=0.0961
  [hb] Epoch 1:  66.1% [330/499] | 30s | loss=0.1607


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  51.2s | train_loss=0.0037 | val_loss=0.0930 | train_auroc=0.7520 | val_auroc=0.8410 | train_auprc=0.0922 | val_auprc=0.1698
  [hb] Epoch 2:  66.5% [332/499] | 30s | loss=0.1255


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  50.6s | train_loss=0.0047 | val_loss=0.0929 | train_auroc=0.7954 | val_auroc=0.8488 | train_auprc=0.1350 | val_auprc=0.1688
  [hb] Epoch 3:  64.1% [320/499] | 30s | loss=0.0106


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  49.5s | train_loss=0.0099 | val_loss=0.0902 | train_auroc=0.8120 | val_auroc=0.8531 | train_auprc=0.1471 | val_auprc=0.1702
  [hb] Epoch 4:  66.9% [334/499] | 30s | loss=0.0557


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  50.1s | train_loss=0.0027 | val_loss=0.0912 | train_auroc=0.8165 | val_auroc=0.8594 | train_auprc=0.1551 | val_auprc=0.1781
  [hb] Epoch 5:  67.3% [336/499] | 30s | loss=0.0911


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  50.2s | train_loss=0.0064 | val_loss=0.0911 | train_auroc=0.8257 | val_auroc=0.8588 | train_auprc=0.1649 | val_auprc=0.1760
  [hb] Epoch 6:  67.3% [336/499] | 30s | loss=0.0830


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  49.7s | train_loss=0.0241 | val_loss=0.0897 | train_auroc=0.8335 | val_auroc=0.8611 | train_auprc=0.1772 | val_auprc=0.1752
  [hb] Epoch 7:  66.3% [331/499] | 30s | loss=0.1493


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.7s | train_loss=0.0189 | val_loss=0.0905 | train_auroc=0.8391 | val_auroc=0.8604 | train_auprc=0.1860 | val_auprc=0.1783
  [hb] Epoch 8:  67.5% [337/499] | 30s | loss=0.0681


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  49.7s | train_loss=0.0206 | val_loss=0.0899 | train_auroc=0.8462 | val_auroc=0.8606 | train_auprc=0.1924 | val_auprc=0.1722
  [hb] Epoch 9:  63.9% [319/499] | 30s | loss=0.0962


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  49.9s | train_loss=0.0083 | val_loss=0.0916 | train_auroc=0.8513 | val_auroc=0.8609 | train_auprc=0.2055 | val_auprc=0.1841
  [hb] Epoch 10:  65.5% [327/499] | 30s | loss=0.0718


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  51.1s | train_loss=0.0045 | val_loss=0.0908 | train_auroc=0.8573 | val_auroc=0.8562 | train_auprc=0.2286 | val_auprc=0.1793
  [hb] Epoch 11:  70.5% [352/499] | 30s | loss=0.1261


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.3s | train_loss=0.4504 | val_loss=0.0910 | train_auroc=0.8661 | val_auroc=0.8570 | train_auprc=0.2416 | val_auprc=0.1784
  [hb] Epoch 12:  69.5% [347/499] | 30s | loss=0.0491


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  47.6s | train_loss=1.5446 | val_loss=0.0922 | train_auroc=0.8728 | val_auroc=0.8581 | train_auprc=0.2629 | val_auprc=0.1833
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/01_BCE_Baseline_seed42/checkpoints/epoch=06-val_loss=0.0897.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/01_BCE_Baseline_seed42/checkpoints/epoch=06-val_loss=0.0897.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/01_BCE_Baseline_seed42/checkpoints/epoch=06-val_loss=0.0897.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0002 | test_loss=0.0810 | test_state=0.0809 | test_velocity=0.0002
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      0.0001989491138374433      │
│            test_asf             │       0.03046925738453865       │
│        test_asf_ci_high         │       0.03173603117465973       │
│         test_asf_ci_low         │      0.029101422056555748       │
│           test_auprc            │       0.15010112524032593       │
│       test_auprc_ci_high        │       0.18015243113040924       │
│        test_auprc_ci_low        │       0.12317118048667908       │
│           test_auroc            │       0.8062106370925903        │
│       test_auroc_ci_high        │       0.8294516801834106        │
│        test_auroc_ci_low        │        0.781129777431488        │
│        test_brier_score         │      0.021812904626131058       │
│    test_brier_score_ci_high     │      0.024332251399755478       │
│     test_brier_score_ci_low     │       0.01939549669623375       │
│            test_ece             │      0.0035657393746078014      │
│        test_ece_ci_high         │      0.0064420681446790695      │
│         test_ece_ci_low         │      0.0016875131987035275      │
│             test_f1             │       0.10682038962841034       │
│         test_f1_ci_high         │       0.11792567372322083       │
│         test_f1_ci_low          │       0.09478483349084854       │
│          test_fa_rate           │      0.021590925753116608       │
│      test_fa_rate_ci_high       │      0.022348549216985703       │
│       test_fa_rate_ci_low       │      0.020721185952425003       │
│           test_fappd            │       0.5181822180747986        │
│       test_fappd_ci_high        │       0.5363651514053345        │
│        test_fappd_ci_low        │       0.4973084628582001        │
│     test_lead_time_iqr_high     │              91.0               │
│ test_lead_time_iqr_high_ci_high │              112.0              │
│ test_lead_time_iqr_high_ci_low  │              75.5               │
│     test_lead_time_iqr_low      │              18.0               │
│ test_lead_time_iqr_low_ci_high  │              25.5               │
│  test_lead_time_iqr_low_ci_low  │              12.0               │
│            test_loss            │       0.08098997920751572       │
│       test_mean_lead_time       │        66.48128509521484        │
│   test_mean_lead_time_ci_high   │        75.76036071777344        │
│   test_mean_lead_time_ci_low    │       57.808937072753906        │
│      test_median_lead_time      │              50.0               │
│  test_median_lead_time_ci_high  │              56.0               │
│  test_median_lead_time_ci_low   │              38.5               │
│            test_npv             │       0.9917664527893066        │
│        test_npv_ci_high         │       0.9935497641563416        │
│         test_npv_ci_low         │       0.9897969961166382        │
│     test_physionet_utility      │       0.05032606050372124       │
│ test_physionet_utility_ci_high  │       0.1660468429327011        │
│  test_physionet_utility_ci_low  │      -0.05214750021696091       │
│            test_poms            │        0.558230459690094        │
│        test_poms_ci_high        │       0.5931411385536194        │
│        test_poms_ci_low         │       0.5239482522010803        │
│            test_ppv             │       0.05742193013429642       │
│        test_ppv_ci_high         │       0.0638950914144516        │
│         test_ppv_ci_low         │       0.05049514397978783       │
│            test_rtv             │       0.11900772899389267       │
│        test_rtv_ci_high         │       0.12969930469989777       │
│         test_rtv_ci_low         │       0.1088876947760582        │
│     test_selected_threshold     │      0.015075377188622952       │
│        test_sensitivity         │       0.7644859552383423        │
│    test_sensitivity_ci_high     │       0.8081629872322083        │
│     test_sensitivity_ci_low     │       0.7210512161254883        │
│        test_specificity         │       0.6933116912841797        │
│    test_specificity_ci_high     │       0.7070202827453613        │
│     test_specificity_ci_low     │       0.6790537238121033        │
│            test_spj             │      0.0020511711481958628      │
│        test_spj_ci_high         │      0.0021414130460470915      │
│         test_spj_ci_low         │      0.001954797888174653       │
│           test_state            │       0.08094540983438492       │
│            test_tce             │       0.7101730704307556        │
│        test_tce_ci_high         │       0.7228183746337891        │
│         test_tce_ci_low         │        0.696524441242218        │
│          test_velocity          │      0.0002469233877491206      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [2/6] 02_TLS_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/02_TLS_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn       │ TLSLoss          │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  64.9% [324/499] | 30s | loss=0.0104


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0033 | val_loss=0.0801
  [hb] Epoch 1:  69.9% [349/499] | 30s | loss=0.0675


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  50.1s | train_loss=0.0022 | val_loss=0.0779 | train_auroc=0.7495 | val_auroc=0.8351 | train_auprc=0.0901 | val_auprc=0.1676
  [hb] Epoch 2:  70.9% [354/499] | 30s | loss=0.0820


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  48.2s | train_loss=0.0034 | val_loss=0.0774 | train_auroc=0.7939 | val_auroc=0.8446 | train_auprc=0.1349 | val_auprc=0.1714
  [hb] Epoch 3:  69.3% [346/499] | 30s | loss=0.0788


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  46.6s | train_loss=0.0074 | val_loss=0.0753 | train_auroc=0.8097 | val_auroc=0.8497 | train_auprc=0.1477 | val_auprc=0.1723
  [hb] Epoch 4:  70.5% [352/499] | 30s | loss=0.0773


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.1s | train_loss=0.0015 | val_loss=0.0764 | train_auroc=0.8146 | val_auroc=0.8557 | train_auprc=0.1547 | val_auprc=0.1790
  [hb] Epoch 5:  66.9% [334/499] | 30s | loss=0.0490


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  47.2s | train_loss=0.0043 | val_loss=0.0755 | train_auroc=0.8243 | val_auroc=0.8556 | train_auprc=0.1679 | val_auprc=0.1772
  [hb] Epoch 6:  69.7% [348/499] | 30s | loss=0.1204


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  48.3s | train_loss=0.0206 | val_loss=0.0756 | train_auroc=0.8318 | val_auroc=0.8591 | train_auprc=0.1740 | val_auprc=0.1752
  [hb] Epoch 7:  68.5% [342/499] | 30s | loss=0.0333


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.1s | train_loss=0.0166 | val_loss=0.0758 | train_auroc=0.8380 | val_auroc=0.8520 | train_auprc=0.1873 | val_auprc=0.1718
  [hb] Epoch 8:  66.7% [333/499] | 30s | loss=0.0810


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  48.0s | train_loss=0.0127 | val_loss=0.0755 | train_auroc=0.8446 | val_auroc=0.8571 | train_auprc=0.1939 | val_auprc=0.1617
  [hb] Epoch 9:  68.5% [342/499] | 30s | loss=0.0779


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  49.2s | train_loss=0.0075 | val_loss=0.0755 | train_auroc=0.8498 | val_auroc=0.8531 | train_auprc=0.2070 | val_auprc=0.1705
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/02_TLS_seed42/checkpoints/epoch=03-val_loss=0.0753.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/02_TLS_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/02_TLS_seed42/checkpoints/epoch=03-val_loss=0.0753.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/02_TLS_seed42/checkpoints/epoch=03-val_loss=0.0753.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0803 | test_state=0.0803 | test_velocity=0.0002
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │     0.00014424780965782702      │
│            test_asf             │      0.032366275787353516       │
│        test_asf_ci_high         │       0.03371787071228027       │
│         test_asf_ci_low         │      0.030949965119361877       │
│           test_auprc            │       0.15152935683727264       │
│       test_auprc_ci_high        │       0.18017838895320892       │
│        test_auprc_ci_low        │       0.12721219658851624       │
│           test_auroc            │       0.8031092286109924        │
│       test_auroc_ci_high        │       0.8263399004936218        │
│        test_auroc_ci_low        │       0.7786969542503357        │
│        test_brier_score         │      0.021795189008116722       │
│    test_brier_score_ci_high     │      0.024320531636476517       │
│     test_brier_score_ci_low     │      0.019284216687083244       │
│            test_ece             │      0.007985370233654976       │
│        test_ece_ci_high         │       0.01067312527447939       │
│         test_ece_ci_low         │      0.005289026070386171       │
│             test_f1             │       0.10488751530647278       │
│         test_f1_ci_high         │       0.11593828350305557       │
│         test_f1_ci_low          │       0.09346146881580353       │
│          test_fa_rate           │      0.022514715790748596       │
│      test_fa_rate_ci_high       │       0.02334868349134922       │
│       test_fa_rate_ci_low       │      0.021656427532434464       │
│           test_fappd            │       0.5403531789779663        │
│       test_fappd_ci_high        │       0.5603684186935425        │
│        test_fappd_ci_low        │       0.5197542905807495        │
│     test_lead_time_iqr_high     │              90.0               │
│ test_lead_time_iqr_high_ci_high │              108.0              │
│ test_lead_time_iqr_high_ci_low  │              77.0               │
│     test_lead_time_iqr_low      │              19.0               │
│ test_lead_time_iqr_low_ci_high  │       25.756250381469727        │
│  test_lead_time_iqr_low_ci_low  │       14.993749618530273        │
│            test_loss            │       0.08032350987195969       │
│       test_mean_lead_time       │        67.75138092041016        │
│   test_mean_lead_time_ci_high   │        76.82061767578125        │
│   test_mean_lead_time_ci_low    │        59.0876579284668         │
│      test_median_lead_time      │              51.0               │
│  test_median_lead_time_ci_high  │              63.5               │
│  test_median_lead_time_ci_low   │              37.0               │
│            test_npv             │       0.9915949106216431        │
│        test_npv_ci_high         │        0.993328332901001        │
│         test_npv_ci_low         │        0.989503800868988        │
│     test_physionet_utility      │       0.03393111005425453       │
│ test_physionet_utility_ci_high  │       0.1485038846731186        │
│  test_physionet_utility_ci_low  │      -0.06889098137617111       │
│            test_poms            │       0.5532236099243164        │
│        test_poms_ci_high        │       0.5918240547180176        │
│        test_poms_ci_low         │       0.5195591449737549        │
│            test_ppv             │      0.056323837488889694       │
│        test_ppv_ci_high         │       0.06266965717077255       │
│         test_ppv_ci_low         │       0.04986131936311722       │
│            test_rtv             │       0.08404912054538727       │
│        test_rtv_ci_high         │       0.0912671685218811        │
│         test_rtv_ci_low         │       0.07722851634025574       │
│     test_selected_threshold     │       0.01005025114864111       │
│        test_sensitivity         │       0.7612817287445068        │
│    test_sensitivity_ci_high     │       0.8018134236335754        │
│     test_sensitivity_ci_low     │       0.7153409123420715        │
│        test_specificity         │       0.6882802248001099        │
│    test_specificity_ci_high     │       0.7027914524078369        │
│     test_specificity_ci_low     │       0.6736042499542236        │
│            test_spj             │      0.0014452930772677064      │
│        test_spj_ci_high         │      0.0015079096192494035      │
│         test_spj_ci_low         │      0.0013846282381564379      │
│           test_state            │        0.080288365483284        │
│            test_tce             │       0.7507026195526123        │
│        test_tce_ci_high         │       0.7580018639564514        │
│         test_tce_ci_low         │       0.7432233691215515        │
│          test_velocity          │      0.0002071987692033872      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [3/6] 03_BCE_Smoothness_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/03_BCE_Smoothness_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn       │ SmoothnessLoss   │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  67.1% [335/499] | 30s | loss=0.1359


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0039 | val_loss=0.0961
  [hb] Epoch 1:  69.1% [345/499] | 30s | loss=0.0223


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  49.4s | train_loss=0.0037 | val_loss=0.0931 | train_auroc=0.7520 | val_auroc=0.8411 | train_auprc=0.0922 | val_auprc=0.1699
  [hb] Epoch 2:  66.7% [333/499] | 30s | loss=0.1343


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  47.9s | train_loss=0.0047 | val_loss=0.0930 | train_auroc=0.7955 | val_auroc=0.8488 | train_auprc=0.1352 | val_auprc=0.1685
  [hb] Epoch 3:  68.7% [343/499] | 30s | loss=0.0567


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  48.8s | train_loss=0.0099 | val_loss=0.0902 | train_auroc=0.8120 | val_auroc=0.8531 | train_auprc=0.1471 | val_auprc=0.1705
  [hb] Epoch 4:  68.9% [344/499] | 30s | loss=0.0990


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.8s | train_loss=0.0027 | val_loss=0.0914 | train_auroc=0.8165 | val_auroc=0.8592 | train_auprc=0.1550 | val_auprc=0.1782
  [hb] Epoch 5:  65.5% [327/499] | 30s | loss=0.0738


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  47.9s | train_loss=0.0068 | val_loss=0.0910 | train_auroc=0.8256 | val_auroc=0.8590 | train_auprc=0.1651 | val_auprc=0.1756
  [hb] Epoch 6:  68.7% [343/499] | 30s | loss=0.1098


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  49.4s | train_loss=0.0235 | val_loss=0.0896 | train_auroc=0.8335 | val_auroc=0.8610 | train_auprc=0.1767 | val_auprc=0.1763
  [hb] Epoch 7:  67.3% [336/499] | 30s | loss=0.0634


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.9s | train_loss=0.0195 | val_loss=0.0901 | train_auroc=0.8390 | val_auroc=0.8607 | train_auprc=0.1860 | val_auprc=0.1788
  [hb] Epoch 8:  65.3% [326/499] | 30s | loss=0.1308


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  48.1s | train_loss=0.0179 | val_loss=0.0900 | train_auroc=0.8470 | val_auroc=0.8618 | train_auprc=0.1920 | val_auprc=0.1754
  [hb] Epoch 9:  67.1% [335/499] | 30s | loss=0.0765


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  50.0s | train_loss=0.0078 | val_loss=0.0911 | train_auroc=0.8517 | val_auroc=0.8612 | train_auprc=0.2062 | val_auprc=0.1837
  [hb] Epoch 10:  66.7% [333/499] | 30s | loss=0.0592


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  50.2s | train_loss=0.0045 | val_loss=0.0905 | train_auroc=0.8577 | val_auroc=0.8568 | train_auprc=0.2282 | val_auprc=0.1789
  [hb] Epoch 11:  66.3% [331/499] | 30s | loss=0.1226


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.4s | train_loss=0.4936 | val_loss=0.0907 | train_auroc=0.8663 | val_auroc=0.8575 | train_auprc=0.2430 | val_auprc=0.1801
  [hb] Epoch 12:  67.7% [338/499] | 30s | loss=0.0901


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  48.9s | train_loss=1.5945 | val_loss=0.0905 | train_auroc=0.8731 | val_auroc=0.8596 | train_auprc=0.2652 | val_auprc=0.1850
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=06-val_loss=0.0896.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/03_BCE_Smoothness_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=06-val_loss=0.0896.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=06-val_loss=0.0896.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0002 | test_loss=0.0810 | test_state=0.0809 | test_velocity=0.0002
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      0.0001942797243827954      │
│            test_asf             │       0.03096504509449005       │
│        test_asf_ci_high         │       0.0323123000562191        │
│         test_asf_ci_low         │      0.029548268765211105       │
│           test_auprc            │       0.15043263137340546       │
│       test_auprc_ci_high        │       0.18119332194328308       │
│        test_auprc_ci_low        │       0.1238970085978508        │
│           test_auroc            │       0.8059117794036865        │
│       test_auroc_ci_high        │       0.8288847804069519        │
│        test_auroc_ci_low        │       0.7807155251502991        │
│        test_brier_score         │      0.021808011457324028       │
│    test_brier_score_ci_high     │      0.024326207116246223       │
│     test_brier_score_ci_low     │       0.01939140446484089       │
│            test_ece             │      0.003496034536510706       │
│        test_ece_ci_high         │      0.006357944570481777       │
│         test_ece_ci_low         │      0.0016834861598908901      │
│             test_f1             │       0.10594726353883743       │
│         test_f1_ci_high         │       0.11702217906713486       │
│         test_f1_ci_low          │       0.09408768266439438       │
│          test_fa_rate           │      0.021848727017641068       │
│      test_fa_rate_ci_high       │       0.02264406718313694       │
│       test_fa_rate_ci_low       │      0.020984606817364693       │
│           test_fappd            │        0.524369478225708        │
│       test_fappd_ci_high        │       0.5434576272964478        │
│        test_fappd_ci_low        │       0.5036305785179138        │
│     test_lead_time_iqr_high     │              91.0               │
│ test_lead_time_iqr_high_ci_high │              112.0              │
│ test_lead_time_iqr_high_ci_low  │              76.0               │
│     test_lead_time_iqr_low      │              18.0               │
│ test_lead_time_iqr_low_ci_high  │              26.0               │
│  test_lead_time_iqr_low_ci_low  │              12.5               │
│            test_loss            │       0.0809912160038948        │
│       test_mean_lead_time       │        66.59358215332031        │
│   test_mean_lead_time_ci_high   │        75.91221618652344        │
│   test_mean_lead_time_ci_low    │        57.98990249633789        │
│      test_median_lead_time      │              51.0               │
│  test_median_lead_time_ci_high  │              56.0               │
│  test_median_lead_time_ci_low   │              38.5               │
│            test_npv             │       0.9917165637016296        │
│        test_npv_ci_high         │       0.9935283064842224        │
│         test_npv_ci_low         │       0.9897304773330688        │
│     test_physionet_utility      │       0.04648858681321144       │
│ test_physionet_utility_ci_high  │       0.16187620162963867       │
│  test_physionet_utility_ci_low  │      -0.056235890835523605      │
│            test_poms            │       0.5623456835746765        │
│        test_poms_ci_high        │        0.597069501876831        │
│        test_poms_ci_low         │       0.5274279713630676        │
│            test_ppv             │      0.056920599192380905       │
│        test_ppv_ci_high         │       0.0633377954363823        │
│         test_ppv_ci_low         │      0.050151146948337555       │
│            test_rtv             │       0.11842866986989975       │
│        test_rtv_ci_high         │       0.12911291420459747       │
│         test_rtv_ci_low         │       0.10827361792325974       │
│     test_selected_threshold     │      0.015075377188622952       │
│        test_sensitivity         │       0.7639519572257996        │
│    test_sensitivity_ci_high     │       0.8075469136238098        │
│     test_sensitivity_ci_low     │       0.7204288244247437        │
│        test_specificity         │       0.6906622052192688        │
│    test_specificity_ci_high     │       0.7044176459312439        │
│     test_specificity_ci_low     │       0.6763116121292114        │
│            test_spj             │      0.0020459385123103857      │
│        test_spj_ci_high         │      0.002135632559657097       │
│         test_spj_ci_low         │      0.0019530956633388996      │
│           test_state            │       0.0809473842382431        │
│            test_tce             │       0.7099612355232239        │
│        test_tce_ci_high         │       0.7225765585899353        │
│         test_tce_ci_low         │       0.6960506439208984        │
│          test_velocity          │     0.00024402626149822026      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [4/6] 04_BCE_TotalVariation_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/04_BCE_TotalVariation_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type               ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ GRUPredictor       │  188 K │ train │     0 │
│ 1 │ loss_fn       │ TotalVariationLoss │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection   │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection   │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection   │      0 │ train │     0 │
└───┴───────────────┴────────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  63.3% [316/499] | 30s | loss=0.1076


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0040 | val_loss=0.0964
  [hb] Epoch 1:  69.9% [349/499] | 30s | loss=0.0834


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  49.3s | train_loss=0.0041 | val_loss=0.0933 | train_auroc=0.7527 | val_auroc=0.8415 | train_auprc=0.0925 | val_auprc=0.1696
  [hb] Epoch 2:  70.3% [351/499] | 30s | loss=0.0936


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  47.9s | train_loss=0.0048 | val_loss=0.0930 | train_auroc=0.7964 | val_auroc=0.8490 | train_auprc=0.1353 | val_auprc=0.1678
  [hb] Epoch 3:  66.7% [333/499] | 30s | loss=0.0488


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  47.1s | train_loss=0.0107 | val_loss=0.0904 | train_auroc=0.8127 | val_auroc=0.8531 | train_auprc=0.1468 | val_auprc=0.1714
  [hb] Epoch 4:  69.5% [347/499] | 30s | loss=0.1029


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.3s | train_loss=0.0027 | val_loss=0.0921 | train_auroc=0.8170 | val_auroc=0.8589 | train_auprc=0.1531 | val_auprc=0.1778
  [hb] Epoch 5:  69.7% [348/499] | 30s | loss=0.0809


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  48.2s | train_loss=0.0063 | val_loss=0.0909 | train_auroc=0.8255 | val_auroc=0.8592 | train_auprc=0.1642 | val_auprc=0.1722
  [hb] Epoch 6:  67.5% [337/499] | 30s | loss=0.0492


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  47.5s | train_loss=0.0259 | val_loss=0.0898 | train_auroc=0.8341 | val_auroc=0.8624 | train_auprc=0.1723 | val_auprc=0.1769
  [hb] Epoch 7:  68.7% [343/499] | 30s | loss=0.0816


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.0s | train_loss=0.0200 | val_loss=0.0905 | train_auroc=0.8396 | val_auroc=0.8599 | train_auprc=0.1846 | val_auprc=0.1791
  [hb] Epoch 8:  69.3% [346/499] | 30s | loss=0.1192


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  48.5s | train_loss=0.0181 | val_loss=0.0900 | train_auroc=0.8469 | val_auroc=0.8603 | train_auprc=0.1885 | val_auprc=0.1752
  [hb] Epoch 9:  68.1% [340/499] | 30s | loss=0.0671


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  48.0s | train_loss=0.0074 | val_loss=0.0906 | train_auroc=0.8524 | val_auroc=0.8613 | train_auprc=0.2020 | val_auprc=0.1811
  [hb] Epoch 10:  67.1% [335/499] | 30s | loss=0.0684


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  48.0s | train_loss=0.0048 | val_loss=0.0904 | train_auroc=0.8576 | val_auroc=0.8598 | train_auprc=0.2234 | val_auprc=0.1835
  [hb] Epoch 11:  68.1% [340/499] | 30s | loss=0.0618


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.3s | train_loss=0.4408 | val_loss=0.0908 | train_auroc=0.8651 | val_auroc=0.8584 | train_auprc=0.2358 | val_auprc=0.1853
  [hb] Epoch 12:  68.1% [340/499] | 30s | loss=0.0243


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  49.2s | train_loss=1.9492 | val_loss=0.0914 | train_auroc=0.8720 | val_auroc=0.8607 | train_auprc=0.2574 | val_auprc=0.1851
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=06-val_loss=0.0898.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/04_BCE_TotalVariation_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=06-val_loss=0.0898.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=06-val_loss=0.0898.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0002 | test_loss=0.0810 | test_state=0.0810 | test_velocity=0.0002
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │     0.00016041949857026339      │
│            test_asf             │      0.029911436140537262       │
│        test_asf_ci_high         │       0.03110921010375023       │
│         test_asf_ci_low         │      0.028659487143158913       │
│           test_auprc            │       0.15085718035697937       │
│       test_auprc_ci_high        │       0.1807190626859665        │
│        test_auprc_ci_low        │       0.12395773828029633       │
│           test_auroc            │       0.8055479526519775        │
│       test_auroc_ci_high        │       0.8289755582809448        │
│        test_auroc_ci_low        │       0.7803269624710083        │
│        test_brier_score         │      0.021790334954857826       │
│    test_brier_score_ci_high     │      0.024338120594620705       │
│     test_brier_score_ci_low     │      0.019378654658794403       │
│            test_ece             │      0.0031719922553747892      │
│        test_ece_ci_high         │      0.006129596848040819       │
│         test_ece_ci_low         │      0.0015303714899346232      │
│             test_f1             │       0.10616514831781387       │
│         test_f1_ci_high         │       0.11740335077047348       │
│         test_f1_ci_low          │       0.09426139295101166       │
│          test_fa_rate           │       0.02136176824569702       │
│      test_fa_rate_ci_high       │      0.022149335592985153       │
│       test_fa_rate_ci_low       │      0.020520171150565147       │
│           test_fappd            │       0.5126824378967285        │
│       test_fappd_ci_high        │       0.5315840840339661        │
│        test_fappd_ci_low        │       0.49248409271240234       │
│     test_lead_time_iqr_high     │              90.5               │
│ test_lead_time_iqr_high_ci_high │              112.0              │
│ test_lead_time_iqr_high_ci_low  │              75.5               │
│     test_lead_time_iqr_low      │              18.0               │
│ test_lead_time_iqr_low_ci_high  │              25.75              │
│  test_lead_time_iqr_low_ci_low  │              12.0               │
│            test_loss            │       0.08103200793266296       │
│       test_mean_lead_time       │              66.25              │
│   test_mean_lead_time_ci_high   │        75.75570678710938        │
│   test_mean_lead_time_ci_low    │        57.70690155029297        │
│      test_median_lead_time      │              50.0               │
│  test_median_lead_time_ci_high  │              55.0               │
│  test_median_lead_time_ci_low   │        38.98749923706055        │
│            test_npv             │       0.9917330741882324        │
│        test_npv_ci_high         │       0.9935091733932495        │
│         test_npv_ci_low         │       0.9897510409355164        │
│     test_physionet_utility      │       0.04706128314137459       │
│ test_physionet_utility_ci_high  │       0.16303297877311707       │
│  test_physionet_utility_ci_low  │      -0.05541886016726494       │
│            test_poms            │       0.5602880716323853        │
│        test_poms_ci_high        │       0.5961834192276001        │
│        test_poms_ci_low         │       0.5248031616210938        │
│            test_ppv             │       0.05704490467905998       │
│        test_ppv_ci_high         │       0.06359069794416428       │
│         test_ppv_ci_low         │       0.05038244649767876       │
│            test_rtv             │       0.10894560813903809       │
│        test_rtv_ci_high         │       0.11837080866098404       │
│         test_rtv_ci_low         │       0.09984572976827621       │
│     test_selected_threshold     │      0.015075377188622952       │
│        test_sensitivity         │       0.7642189860343933        │
│    test_sensitivity_ci_high     │       0.8066912889480591        │
│     test_sensitivity_ci_low     │       0.7188490033149719        │
│        test_specificity         │       0.6912690997123718        │
│    test_specificity_ci_high     │       0.7050589919090271        │
│     test_specificity_ci_low     │       0.6772263646125793        │
│            test_spj             │      0.001932614715769887       │
│        test_spj_ci_high         │      0.002015846548601985       │
│         test_spj_ci_low         │      0.001849086256697774       │
│           test_state            │       0.08099327236413956       │
│            test_tce             │       0.7115848064422607        │
│        test_tce_ci_high         │       0.7237682342529297        │
│         test_tce_ci_low         │       0.6984860897064209        │
│          test_velocity          │      0.0002270267577841878      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [5/6] 05_BGSL_StateOnly_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/05_BGSL_StateOnly_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn       │ BGSLLoss         │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  62.9% [314/499] | 30s | loss=0.0786


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0033 | val_loss=0.0812
  [hb] Epoch 1:  66.3% [331/499] | 30s | loss=0.0536


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  50.6s | train_loss=0.0022 | val_loss=0.0791 | train_auroc=0.7500 | val_auroc=0.8364 | train_auprc=0.0904 | val_auprc=0.1684
  [hb] Epoch 2:  68.5% [342/499] | 30s | loss=0.0850


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  49.2s | train_loss=0.0034 | val_loss=0.0785 | train_auroc=0.7945 | val_auroc=0.8455 | train_auprc=0.1354 | val_auprc=0.1714
  [hb] Epoch 3:  68.1% [340/499] | 30s | loss=0.0763


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  48.9s | train_loss=0.0077 | val_loss=0.0764 | train_auroc=0.8105 | val_auroc=0.8504 | train_auprc=0.1483 | val_auprc=0.1725
  [hb] Epoch 4:  67.9% [339/499] | 30s | loss=0.0807


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.9s | train_loss=0.0016 | val_loss=0.0776 | train_auroc=0.8151 | val_auroc=0.8562 | train_auprc=0.1548 | val_auprc=0.1794
  [hb] Epoch 5:  66.1% [330/499] | 30s | loss=0.0770


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  48.2s | train_loss=0.0044 | val_loss=0.0766 | train_auroc=0.8250 | val_auroc=0.8565 | train_auprc=0.1683 | val_auprc=0.1767
  [hb] Epoch 6:  67.5% [337/499] | 30s | loss=0.0420


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  49.1s | train_loss=0.0216 | val_loss=0.0767 | train_auroc=0.8326 | val_auroc=0.8601 | train_auprc=0.1751 | val_auprc=0.1757
  [hb] Epoch 7:  67.7% [338/499] | 30s | loss=0.1221


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  49.5s | train_loss=0.0176 | val_loss=0.0765 | train_auroc=0.8384 | val_auroc=0.8539 | train_auprc=0.1879 | val_auprc=0.1729
  [hb] Epoch 8:  68.1% [340/499] | 30s | loss=0.0501


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  49.4s | train_loss=0.0161 | val_loss=0.0764 | train_auroc=0.8459 | val_auroc=0.8594 | train_auprc=0.1971 | val_auprc=0.1632
  [hb] Epoch 9:  66.1% [330/499] | 30s | loss=0.0958


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  48.2s | train_loss=0.0070 | val_loss=0.0772 | train_auroc=0.8507 | val_auroc=0.8581 | train_auprc=0.2067 | val_auprc=0.1743
  [hb] Epoch 10:  66.9% [334/499] | 30s | loss=0.0431


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  49.3s | train_loss=0.0033 | val_loss=0.0772 | train_auroc=0.8563 | val_auroc=0.8561 | train_auprc=0.2267 | val_auprc=0.1661
  [hb] Epoch 11:  68.3% [341/499] | 30s | loss=0.0567


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.3s | train_loss=0.4263 | val_loss=0.0775 | train_auroc=0.8635 | val_auroc=0.8591 | train_auprc=0.2420 | val_auprc=0.1770
  [hb] Epoch 12:  68.9% [344/499] | 30s | loss=0.0751


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  48.9s | train_loss=1.5297 | val_loss=0.0772 | train_auroc=0.8695 | val_auroc=0.8577 | train_auprc=0.2601 | val_auprc=0.1716
  [hb] Epoch 13:  65.9% [329/499] | 30s | loss=0.0599


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  47.6s | train_loss=0.0069 | val_loss=0.0785 | train_auroc=0.8707 | val_auroc=0.8565 | train_auprc=0.2657 | val_auprc=0.1693
  [hb] Epoch 14:  68.3% [341/499] | 30s | loss=0.0454


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  14 |  48.9s | train_loss=0.0543 | val_loss=0.0783 | train_auroc=0.8740 | val_auroc=0.8544 | train_auprc=0.2751 | val_auprc=0.1650
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=08-val_loss=0.0764.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/05_BGSL_StateOnly_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=08-val_loss=0.0764.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=08-val_loss=0.0764.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0002 | test_loss=0.0807 | test_state=0.0807 | test_velocity=0.0002
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │     0.00015079845616128296      │
│            test_asf             │       0.03433820605278015       │
│        test_asf_ci_high         │      0.035723645240068436       │
│         test_asf_ci_low         │       0.03292651101946831       │
│           test_auprc            │       0.14969685673713684       │
│       test_auprc_ci_high        │       0.18010160326957703       │
│        test_auprc_ci_low        │       0.12362440675497055       │
│           test_auroc            │        0.807110071182251        │
│       test_auroc_ci_high        │       0.8282158374786377        │
│        test_auroc_ci_low        │        0.783918023109436        │
│        test_brier_score         │       0.02175075002014637       │
│    test_brier_score_ci_high     │      0.024290259927511215       │
│     test_brier_score_ci_low     │      0.019252454861998558       │
│            test_ece             │      0.007019111420959234       │
│        test_ece_ci_high         │      0.009896191768348217       │
│         test_ece_ci_low         │      0.004643163178116083       │
│             test_f1             │       0.10895603150129318       │
│         test_f1_ci_high         │       0.12099187076091766       │
│         test_f1_ci_low          │       0.09704528748989105       │
│          test_fa_rate           │      0.020939258858561516       │
│      test_fa_rate_ci_high       │      0.021823855116963387       │
│       test_fa_rate_ci_low       │      0.020064514130353928       │
│           test_fappd            │       0.5025421977043152        │
│       test_fappd_ci_high        │       0.5237725377082825        │
│        test_fappd_ci_low        │       0.4815483093261719        │
│     test_lead_time_iqr_high     │              93.0               │
│ test_lead_time_iqr_high_ci_high │              110.0              │
│ test_lead_time_iqr_high_ci_low  │              76.0               │
│     test_lead_time_iqr_low      │              18.0               │
│ test_lead_time_iqr_low_ci_high  │              24.0               │
│  test_lead_time_iqr_low_ci_low  │              14.0               │
│            test_loss            │       0.08070817589759827       │
│       test_mean_lead_time       │        65.86978912353516        │
│   test_mean_lead_time_ci_high   │        75.42582702636719        │
│   test_mean_lead_time_ci_low    │       57.403846740722656        │
│      test_median_lead_time      │              49.5               │
│  test_median_lead_time_ci_high  │              58.0               │
│  test_median_lead_time_ci_low   │              37.0               │
│            test_npv             │       0.9915108680725098        │
│        test_npv_ci_high         │        0.993195652961731        │
│         test_npv_ci_low         │       0.9897139668464661        │
│     test_physionet_utility      │       0.08089624345302582       │
│ test_physionet_utility_ci_high  │       0.19412381947040558       │
│  test_physionet_utility_ci_low  │      -0.021415747702121735      │
│            test_poms            │       0.5609053373336792        │
│        test_poms_ci_high        │       0.5985034108161926        │
│        test_poms_ci_low         │       0.5252577662467957        │
│            test_ppv             │      0.058726754039525986       │
│        test_ppv_ci_high         │       0.0656856819987297        │
│         test_ppv_ci_low         │       0.05183171108365059       │
│            test_rtv             │       0.08957308530807495       │
│        test_rtv_ci_high         │       0.09768276661634445       │
│         test_rtv_ci_low         │       0.08138246834278107       │
│     test_selected_threshold     │       0.01005025114864111       │
│        test_sensitivity         │       0.7530040144920349        │
│    test_sensitivity_ci_high     │       0.7936921119689941        │
│     test_sensitivity_ci_low     │       0.7113131880760193        │
│        test_specificity         │       0.7050386071205139        │
│    test_specificity_ci_high     │       0.7196897864341736        │
│     test_specificity_ci_low     │       0.6909903287887573        │
│            test_spj             │      0.0015621082857251167      │
│        test_spj_ci_high         │      0.001628889120183885       │
│         test_spj_ci_low         │      0.0014956044033169746      │
│           test_state            │       0.08067160099744797       │
│            test_tce             │       0.7405298352241516        │
│        test_tce_ci_high         │       0.7510345578193665        │
│         test_tce_ci_low         │       0.7294663190841675        │
│          test_velocity          │      0.0002150053041987121      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [6/6] 06_Full_BGSL_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/06_Full_BGSL_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name          ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone      │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn       │ BGSLLoss         │      0 │ train │     0 │
│ 2 │ train_metrics │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics   │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴───────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 18                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  65.1% [325/499] | 30s | loss=0.0577


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0033 | val_loss=0.0812
  [hb] Epoch 1:  66.7% [333/499] | 30s | loss=0.0510


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  50.1s | train_loss=0.0023 | val_loss=0.0790 | train_auroc=0.7501 | val_auroc=0.8365 | train_auprc=0.0905 | val_auprc=0.1683
  [hb] Epoch 2:  69.1% [345/499] | 30s | loss=0.0522


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  50.1s | train_loss=0.0034 | val_loss=0.0785 | train_auroc=0.7947 | val_auroc=0.8455 | train_auprc=0.1354 | val_auprc=0.1715
  [hb] Epoch 3:  69.3% [346/499] | 30s | loss=0.0789


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  48.4s | train_loss=0.0076 | val_loss=0.0765 | train_auroc=0.8106 | val_auroc=0.8503 | train_auprc=0.1482 | val_auprc=0.1732
  [hb] Epoch 4:  68.3% [341/499] | 30s | loss=0.1117


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  47.2s | train_loss=0.0017 | val_loss=0.0775 | train_auroc=0.8151 | val_auroc=0.8562 | train_auprc=0.1546 | val_auprc=0.1788
  [hb] Epoch 5:  70.7% [353/499] | 30s | loss=0.1230


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  48.2s | train_loss=0.0042 | val_loss=0.0769 | train_auroc=0.8251 | val_auroc=0.8567 | train_auprc=0.1676 | val_auprc=0.1768
  [hb] Epoch 6:  69.9% [349/499] | 30s | loss=0.0498


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  47.5s | train_loss=0.0228 | val_loss=0.0768 | train_auroc=0.8325 | val_auroc=0.8605 | train_auprc=0.1766 | val_auprc=0.1749
  [hb] Epoch 7:  67.7% [338/499] | 30s | loss=0.1224


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  47.0s | train_loss=0.0169 | val_loss=0.0766 | train_auroc=0.8384 | val_auroc=0.8538 | train_auprc=0.1884 | val_auprc=0.1725
  [hb] Epoch 8:  69.5% [347/499] | 30s | loss=0.0738


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  48.1s | train_loss=0.0163 | val_loss=0.0762 | train_auroc=0.8458 | val_auroc=0.8589 | train_auprc=0.1966 | val_auprc=0.1630
  [hb] Epoch 9:  69.7% [348/499] | 30s | loss=0.0609


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  48.3s | train_loss=0.0080 | val_loss=0.0773 | train_auroc=0.8505 | val_auroc=0.8601 | train_auprc=0.2069 | val_auprc=0.1758
  [hb] Epoch 10:  66.5% [332/499] | 30s | loss=0.0117


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  47.5s | train_loss=0.0033 | val_loss=0.0772 | train_auroc=0.8552 | val_auroc=0.8558 | train_auprc=0.2237 | val_auprc=0.1691
  [hb] Epoch 11:  68.7% [343/499] | 30s | loss=0.0677


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  48.7s | train_loss=0.4624 | val_loss=0.0783 | train_auroc=0.8630 | val_auroc=0.8590 | train_auprc=0.2389 | val_auprc=0.1761
  [hb] Epoch 12:  66.1% [330/499] | 30s | loss=0.0759


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  48.7s | train_loss=1.4478 | val_loss=0.0772 | train_auroc=0.8696 | val_auroc=0.8570 | train_auprc=0.2578 | val_auprc=0.1636
  [hb] Epoch 13:  67.7% [338/499] | 30s | loss=0.0748


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  49.7s | train_loss=0.0071 | val_loss=0.0780 | train_auroc=0.8698 | val_auroc=0.8568 | train_auprc=0.2636 | val_auprc=0.1692
  [hb] Epoch 14:  68.5% [342/499] | 30s | loss=0.0734


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  14 |  48.3s | train_loss=0.0698 | val_loss=0.0779 | train_auroc=0.8738 | val_auroc=0.8567 | train_auprc=0.2783 | val_auprc=0.1687
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/06_Full_BGSL_seed42/checkpoints/epoch=08-val_loss=0.0762.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/06_Full_BGSL_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/06_Full_BGSL_seed42/checkpoints/epoch=08-val_loss=0.0762.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/runs/06_Full_BGSL_seed42/checkpoints/epoch=08-val_loss=0.0762.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0803 | test_state=0.0802 | test_velocity=0.0002
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      0.0001385668438160792      │
│            test_asf             │       0.03261599689722061       │
│        test_asf_ci_high         │       0.03404081612825394       │
│         test_asf_ci_low         │      0.031229261308908463       │
│           test_auprc            │       0.1516709178686142        │
│       test_auprc_ci_high        │       0.18228168785572052       │
│        test_auprc_ci_low        │       0.12497042119503021       │
│           test_auroc            │       0.8104051351547241        │
│       test_auroc_ci_high        │       0.8314271569252014        │
│        test_auroc_ci_low        │       0.7868055701255798        │
│        test_brier_score         │       0.02172928862273693       │
│    test_brier_score_ci_high     │      0.024254854768514633       │
│     test_brier_score_ci_low     │      0.019243726506829262       │
│            test_ece             │      0.007269249297678471       │
│        test_ece_ci_high         │      0.010161375626921654       │
│         test_ece_ci_low         │      0.004947416018694639       │
│             test_f1             │       0.11004550755023956       │
│         test_f1_ci_high         │       0.12210901826620102       │
│         test_f1_ci_low          │       0.09804375469684601       │
│          test_fa_rate           │       0.02025894820690155       │
│      test_fa_rate_ci_high       │      0.021044518798589706       │
│       test_fa_rate_ci_low       │      0.019414709880948067       │
│           test_fappd            │       0.4862147569656372        │
│       test_fappd_ci_high        │       0.5050684809684753        │
│        test_fappd_ci_low        │       0.4659530222415924        │
│     test_lead_time_iqr_high     │              93.0               │
│ test_lead_time_iqr_high_ci_high │              111.0              │
│ test_lead_time_iqr_high_ci_low  │        75.9937515258789         │
│     test_lead_time_iqr_low      │              18.5               │
│ test_lead_time_iqr_low_ci_high  │              25.0               │
│  test_lead_time_iqr_low_ci_low  │              15.0               │
│            test_loss            │       0.08026609569787979       │
│       test_mean_lead_time       │        66.79679107666016        │
│   test_mean_lead_time_ci_high   │        76.48622131347656        │
│   test_mean_lead_time_ci_low    │       58.398677825927734        │
│      test_median_lead_time      │              50.0               │
│  test_median_lead_time_ci_high  │        60.01250076293945        │
│  test_median_lead_time_ci_low   │              38.0               │
│            test_npv             │       0.9915274381637573        │
│        test_npv_ci_high         │       0.9932053089141846        │
│         test_npv_ci_low         │       0.9896885752677917        │
│     test_physionet_utility      │       0.08125992864370346       │
│ test_physionet_utility_ci_high  │       0.1936301290988922        │
│  test_physionet_utility_ci_low  │      -0.021268079057335854      │
│            test_poms            │       0.5715363621711731        │
│        test_poms_ci_high        │        0.607744038105011        │
│        test_poms_ci_low         │       0.5367405414581299        │
│            test_ppv             │      0.059365253895521164       │
│        test_ppv_ci_high         │       0.06627758592367172       │
│         test_ppv_ci_low         │       0.05244472250342369       │
│            test_rtv             │       0.08404244482517242       │
│        test_rtv_ci_high         │       0.09180065989494324       │
│         test_rtv_ci_low         │       0.07642009109258652       │
│     test_selected_threshold     │       0.01005025114864111       │
│        test_sensitivity         │       0.7522029280662537        │
│    test_sensitivity_ci_high     │       0.7929890751838684        │
│     test_sensitivity_ci_low     │       0.7092394232749939        │
│        test_specificity         │       0.7087191939353943        │
│    test_specificity_ci_high     │       0.7235782742500305        │
│     test_specificity_ci_low     │        0.694444477558136        │
│            test_spj             │      0.0014535197988152504      │
│        test_spj_ci_high         │      0.001515773474238813       │
│         test_spj_ci_low         │      0.0013908256078138947      │
│           test_state            │       0.08023133128881454       │
│            test_tce             │       0.7411940693855286        │
│        test_tce_ci_high         │       0.7515994310379028        │
│         test_tce_ci_low         │       0.7300599217414856        │
│          test_velocity          │     0.00020904737175442278      │
└─────────────────────────────────┴─────────────────────────────────┘

Summary written: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/summary.csv
Manifest: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/batch_manifest.json
Overall status: success

========================================================================
  POST-PROCESSING
========================================================================

========================================================================
  AGGREGATED RESULTS — 20260606-160650__1ea45579
  Conditions: 6 | Total runs: 6
========================================================================

  --- 01_BCE_Baseline (n=1) ---
    test_auroc                                = 0.8062 ± 0.0000
    test_auprc                                = 0.1501 ± 0.0000
    test_physionet_utility                    = 0.0503 ± 0.0000
    test_median_lead_time                     = 50.0000 ± 0.0000
    test_mean_lead_time                       = 66.4813 ± 0.0000
    test_asf                                  = 0.0305 ± 0.0000
    test_rtv                                  = 0.1190 ± 0.0000
    test_spj                                  = 0.0021 ± 0.0000
    test_poms                                 = 0.5582 ± 0.0000
    test_tce                                  = 0.7102 ± 0.0000
    test_fa_rate                              = 0.0216 ± 0.0000
    test_fappd                                = 0.5182 ± 0.0000
    test_sensitivity                          = 0.7645 ± 0.0000
    test_specificity                          = 0.6933 ± 0.0000
    test_ppv                                  = 0.0574 ± 0.0000
    test_npv                                  = 0.9918 ± 0.0000
    test_f1                                   = 0.1068 ± 0.0000
    test_brier_score                          = 0.0218 ± 0.0000
    test_ece                                  = 0.0036 ± 0.0000
    test_selected_threshold                   = 0.0151 ± 0.0000

  --- 02_TLS (n=1) ---
    test_auroc                                = 0.8031 ± 0.0000
    test_auprc                                = 0.1515 ± 0.0000
    test_physionet_utility                    = 0.0339 ± 0.0000
    test_median_lead_time                     = 51.0000 ± 0.0000
    test_mean_lead_time                       = 67.7514 ± 0.0000
    test_asf                                  = 0.0324 ± 0.0000
    test_rtv                                  = 0.0840 ± 0.0000
    test_spj                                  = 0.0014 ± 0.0000
    test_poms                                 = 0.5532 ± 0.0000
    test_tce                                  = 0.7507 ± 0.0000
    test_fa_rate                              = 0.0225 ± 0.0000
    test_fappd                                = 0.5404 ± 0.0000
    test_sensitivity                          = 0.7613 ± 0.0000
    test_specificity                          = 0.6883 ± 0.0000
    test_ppv                                  = 0.0563 ± 0.0000
    test_npv                                  = 0.9916 ± 0.0000
    test_f1                                   = 0.1049 ± 0.0000
    test_brier_score                          = 0.0218 ± 0.0000
    test_ece                                  = 0.0080 ± 0.0000
    test_selected_threshold                   = 0.0101 ± 0.0000

  --- 03_BCE_Smoothness (n=1) ---
    test_auroc                                = 0.8059 ± 0.0000
    test_auprc                                = 0.1504 ± 0.0000
    test_physionet_utility                    = 0.0465 ± 0.0000
    test_median_lead_time                     = 51.0000 ± 0.0000
    test_mean_lead_time                       = 66.5936 ± 0.0000
    test_asf                                  = 0.0310 ± 0.0000
    test_rtv                                  = 0.1184 ± 0.0000
    test_spj                                  = 0.0020 ± 0.0000
    test_poms                                 = 0.5623 ± 0.0000
    test_tce                                  = 0.7100 ± 0.0000
    test_fa_rate                              = 0.0218 ± 0.0000
    test_fappd                                = 0.5244 ± 0.0000
    test_sensitivity                          = 0.7640 ± 0.0000
    test_specificity                          = 0.6907 ± 0.0000
    test_ppv                                  = 0.0569 ± 0.0000
    test_npv                                  = 0.9917 ± 0.0000
    test_f1                                   = 0.1059 ± 0.0000
    test_brier_score                          = 0.0218 ± 0.0000
    test_ece                                  = 0.0035 ± 0.0000
    test_selected_threshold                   = 0.0151 ± 0.0000

  --- 04_BCE_TotalVariation (n=1) ---
    test_auroc                                = 0.8055 ± 0.0000
    test_auprc                                = 0.1509 ± 0.0000
    test_physionet_utility                    = 0.0471 ± 0.0000
    test_median_lead_time                     = 50.0000 ± 0.0000
    test_mean_lead_time                       = 66.2500 ± 0.0000
    test_asf                                  = 0.0299 ± 0.0000
    test_rtv                                  = 0.1089 ± 0.0000
    test_spj                                  = 0.0019 ± 0.0000
    test_poms                                 = 0.5603 ± 0.0000
    test_tce                                  = 0.7116 ± 0.0000
    test_fa_rate                              = 0.0214 ± 0.0000
    test_fappd                                = 0.5127 ± 0.0000
    test_sensitivity                          = 0.7642 ± 0.0000
    test_specificity                          = 0.6913 ± 0.0000
    test_ppv                                  = 0.0570 ± 0.0000
    test_npv                                  = 0.9917 ± 0.0000
    test_f1                                   = 0.1062 ± 0.0000
    test_brier_score                          = 0.0218 ± 0.0000
    test_ece                                  = 0.0032 ± 0.0000
    test_selected_threshold                   = 0.0151 ± 0.0000

  --- 05_BGSL_StateOnly (n=1) ---
    test_auroc                                = 0.8071 ± 0.0000
    test_auprc                                = 0.1497 ± 0.0000
    test_physionet_utility                    = 0.0809 ± 0.0000
    test_median_lead_time                     = 49.5000 ± 0.0000
    test_mean_lead_time                       = 65.8698 ± 0.0000
    test_asf                                  = 0.0343 ± 0.0000
    test_rtv                                  = 0.0896 ± 0.0000
    test_spj                                  = 0.0016 ± 0.0000
    test_poms                                 = 0.5609 ± 0.0000
    test_tce                                  = 0.7405 ± 0.0000
    test_fa_rate                              = 0.0209 ± 0.0000
    test_fappd                                = 0.5025 ± 0.0000
    test_sensitivity                          = 0.7530 ± 0.0000
    test_specificity                          = 0.7050 ± 0.0000
    test_ppv                                  = 0.0587 ± 0.0000
    test_npv                                  = 0.9915 ± 0.0000
    test_f1                                   = 0.1090 ± 0.0000
    test_brier_score                          = 0.0218 ± 0.0000
    test_ece                                  = 0.0070 ± 0.0000
    test_selected_threshold                   = 0.0101 ± 0.0000

  --- 06_Full_BGSL (n=1) ---
    test_auroc                                = 0.8104 ± 0.0000
    test_auprc                                = 0.1517 ± 0.0000
    test_physionet_utility                    = 0.0813 ± 0.0000
    test_median_lead_time                     = 50.0000 ± 0.0000
    test_mean_lead_time                       = 66.7968 ± 0.0000
    test_asf                                  = 0.0326 ± 0.0000
    test_rtv                                  = 0.0840 ± 0.0000
    test_spj                                  = 0.0015 ± 0.0000
    test_poms                                 = 0.5715 ± 0.0000
    test_tce                                  = 0.7412 ± 0.0000
    test_fa_rate                              = 0.0203 ± 0.0000
    test_fappd                                = 0.4862 ± 0.0000
    test_sensitivity                          = 0.7522 ± 0.0000
    test_specificity                          = 0.7087 ± 0.0000
    test_ppv                                  = 0.0594 ± 0.0000
    test_npv                                  = 0.9915 ± 0.0000
    test_f1                                   = 0.1100 ± 0.0000
    test_brier_score                          = 0.0217 ± 0.0000
    test_ece                                  = 0.0073 ± 0.0000
    test_selected_threshold                   = 0.0101 ± 0.0000

  Full results saved to: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/aggregated_results.csv

[01_BCE_Baseline] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots/01_BCE_Baseline
[02_TLS] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots/02_TLS
[03_BCE_Smoothness] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots/03_BCE_Smoothness
[04_BCE_TotalVariation] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots/04_BCE_TotalVariation
[05_BGSL_StateOnly] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots/05_BGSL_StateOnly
[06_Full_BGSL] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots/06_Full_BGSL

All plots saved under /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260606-160650__1ea45579/plots