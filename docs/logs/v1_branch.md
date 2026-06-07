--- Execution Mode: LOCAL_SEQUENTIAL | Total Jobs: 6 | Batch: 20260607-073908__ae6401ae ---
Outputs root: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae

=== [1/6] 01_BCE_Baseline_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
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

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  65.1% [325/499] | 30s | loss=0.0685


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0039 | val_loss=0.0961
  [hb] Epoch 1:  66.7% [333/499] | 30s | loss=0.0607


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  50.3s | train_loss=0.0037 | val_loss=0.0930 | train_auroc=0.7539 | val_auroc=0.8418 | train_auprc=0.0918 | val_auprc=0.1684
  [hb] Epoch 2:  66.5% [332/499] | 30s | loss=0.1255


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  50.7s | train_loss=0.0047 | val_loss=0.0929 | train_auroc=0.7960 | val_auroc=0.8495 | train_auprc=0.1344 | val_auprc=0.1676
  [hb] Epoch 3:  63.9% [319/499] | 30s | loss=0.1187


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  49.3s | train_loss=0.0099 | val_loss=0.0902 | train_auroc=0.8126 | val_auroc=0.8540 | train_auprc=0.1462 | val_auprc=0.1691
  [hb] Epoch 4:  68.9% [344/499] | 30s | loss=0.0988


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  49.8s | train_loss=0.0027 | val_loss=0.0912 | train_auroc=0.8168 | val_auroc=0.8603 | train_auprc=0.1544 | val_auprc=0.1770
  [hb] Epoch 5:  67.9% [339/499] | 30s | loss=0.1205


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  49.1s | train_loss=0.0064 | val_loss=0.0911 | train_auroc=0.8259 | val_auroc=0.8595 | train_auprc=0.1640 | val_auprc=0.1748
  [hb] Epoch 6:  66.1% [330/499] | 30s | loss=0.0846


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  48.3s | train_loss=0.0241 | val_loss=0.0897 | train_auroc=0.8337 | val_auroc=0.8621 | train_auprc=0.1764 | val_auprc=0.1742
  [hb] Epoch 7:  68.3% [341/499] | 30s | loss=0.1289


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.9s | train_loss=0.0189 | val_loss=0.0905 | train_auroc=0.8397 | val_auroc=0.8612 | train_auprc=0.1854 | val_auprc=0.1773
  [hb] Epoch 8:  69.5% [347/499] | 30s | loss=0.0878


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  49.3s | train_loss=0.0206 | val_loss=0.0899 | train_auroc=0.8470 | val_auroc=0.8615 | train_auprc=0.1919 | val_auprc=0.1712
  [hb] Epoch 9:  67.5% [337/499] | 30s | loss=0.0868


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  47.4s | train_loss=0.0083 | val_loss=0.0916 | train_auroc=0.8516 | val_auroc=0.8616 | train_auprc=0.2047 | val_auprc=0.1829
  [hb] Epoch 10:  68.7% [343/499] | 30s | loss=0.0833


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  48.9s | train_loss=0.0045 | val_loss=0.0908 | train_auroc=0.8578 | val_auroc=0.8569 | train_auprc=0.2276 | val_auprc=0.1780
  [hb] Epoch 11:  68.9% [344/499] | 30s | loss=0.0939


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.0s | train_loss=0.4504 | val_loss=0.0910 | train_auroc=0.8661 | val_auroc=0.8577 | train_auprc=0.2402 | val_auprc=0.1770
  [hb] Epoch 12:  66.5% [332/499] | 30s | loss=0.0442


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  47.8s | train_loss=1.5446 | val_loss=0.0922 | train_auroc=0.8733 | val_auroc=0.8588 | train_auprc=0.2617 | val_auprc=0.1818
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/01_BCE_Baseline_seed42/checkpoints/epoch=06-val_loss=0.0897.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/01_BCE_Baseline_seed42/checkpoints/epoch=06-val_loss=0.0897.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/01_BCE_Baseline_seed42/checkpoints/epoch=06-val_loss=0.0897.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0796 | test_monotonicity=0.0000 | test_state=0.0796 | test_velocity=0.0009
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      7.348367216764018e-05      │
│            test_asf             │       0.03046925738453865       │
│        test_asf_ci_high         │       0.03173603117465973       │
│         test_asf_ci_low         │      0.029101422056555748       │
│           test_auprc            │        0.149944007396698        │
│       test_auprc_ci_high        │       0.18015243113040924       │
│        test_auprc_ci_low        │       0.12317118048667908       │
│           test_auroc            │       0.8060448169708252        │
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
│            test_loss            │       0.07964643090963364       │
│       test_mean_lead_time       │        66.48128509521484        │
│   test_mean_lead_time_ci_high   │        75.76036071777344        │
│   test_mean_lead_time_ci_low    │       57.808937072753906        │
│      test_median_lead_time      │              50.0               │
│  test_median_lead_time_ci_high  │              56.0               │
│  test_median_lead_time_ci_low   │              38.5               │
│        test_monotonicity        │      3.402131187613122e-05      │
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
│           test_state            │       0.07955222576856613       │
│            test_tce             │       0.6755558252334595        │
│        test_tce_ci_high         │       0.6882144808769226        │
│         test_tce_ci_low         │       0.6620106101036072        │
│          test_velocity          │      0.000865090754814446       │
└─────────────────────────────────┴─────────────────────────────────┘

=== [2/6] 02_TLS_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/02_TLS_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
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

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  64.1% [320/499] | 30s | loss=0.0856


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0033 | val_loss=0.0801
  [hb] Epoch 1:  70.1% [350/499] | 30s | loss=0.0879


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  48.7s | train_loss=0.0022 | val_loss=0.0779 | train_auroc=0.7514 | val_auroc=0.8359 | train_auprc=0.0897 | val_auprc=0.1662
  [hb] Epoch 2:  70.5% [352/499] | 30s | loss=0.1017


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  48.0s | train_loss=0.0034 | val_loss=0.0774 | train_auroc=0.7945 | val_auroc=0.8453 | train_auprc=0.1343 | val_auprc=0.1701
  [hb] Epoch 3:  67.9% [339/499] | 30s | loss=0.0702


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  47.2s | train_loss=0.0074 | val_loss=0.0753 | train_auroc=0.8103 | val_auroc=0.8505 | train_auprc=0.1467 | val_auprc=0.1710
  [hb] Epoch 4:  69.1% [345/499] | 30s | loss=0.0593


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.8s | train_loss=0.0015 | val_loss=0.0764 | train_auroc=0.8149 | val_auroc=0.8565 | train_auprc=0.1538 | val_auprc=0.1778
  [hb] Epoch 5:  68.7% [343/499] | 30s | loss=0.0300


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  48.7s | train_loss=0.0043 | val_loss=0.0755 | train_auroc=0.8245 | val_auroc=0.8563 | train_auprc=0.1669 | val_auprc=0.1760
  [hb] Epoch 6:  69.1% [345/499] | 30s | loss=0.0545


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  47.5s | train_loss=0.0206 | val_loss=0.0756 | train_auroc=0.8321 | val_auroc=0.8600 | train_auprc=0.1733 | val_auprc=0.1742
  [hb] Epoch 7:  69.3% [346/499] | 30s | loss=0.1110


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.8s | train_loss=0.0166 | val_loss=0.0758 | train_auroc=0.8386 | val_auroc=0.8528 | train_auprc=0.1866 | val_auprc=0.1707
  [hb] Epoch 8:  66.1% [330/499] | 30s | loss=0.0778


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  47.7s | train_loss=0.0127 | val_loss=0.0755 | train_auroc=0.8454 | val_auroc=0.8579 | train_auprc=0.1932 | val_auprc=0.1605
  [hb] Epoch 9:  68.7% [343/499] | 30s | loss=0.0855


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  49.0s | train_loss=0.0075 | val_loss=0.0755 | train_auroc=0.8501 | val_auroc=0.8538 | train_auprc=0.2059 | val_auprc=0.1693
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/02_TLS_seed42/checkpoints/epoch=03-val_loss=0.0753.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/02_TLS_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/02_TLS_seed42/checkpoints/epoch=03-val_loss=0.0753.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/02_TLS_seed42/checkpoints/epoch=03-val_loss=0.0753.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0787 | test_monotonicity=0.0000 | test_state=0.0786 | test_velocity=0.0009
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      7.803626067470759e-05      │
│            test_asf             │      0.032366275787353516       │
│        test_asf_ci_high         │       0.03371787071228027       │
│         test_asf_ci_low         │      0.030949965119361877       │
│           test_auprc            │       0.15141768753528595       │
│       test_auprc_ci_high        │       0.18017838895320892       │
│        test_auprc_ci_low        │       0.12721219658851624       │
│           test_auroc            │       0.8029822707176208        │
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
│            test_loss            │       0.07874274998903275       │
│       test_mean_lead_time       │        67.75138092041016        │
│   test_mean_lead_time_ci_high   │        76.82061767578125        │
│   test_mean_lead_time_ci_low    │        59.0876579284668         │
│      test_median_lead_time      │              51.0               │
│  test_median_lead_time_ci_high  │              63.5               │
│  test_median_lead_time_ci_low   │              37.0               │
│        test_monotonicity        │      1.771058487065602e-05      │
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
│           test_state            │       0.07864340394735336       │
│            test_tce             │       0.7160853743553162        │
│        test_tce_ci_high         │       0.7232306599617004        │
│         test_tce_ci_low         │       0.7085691690444946        │
│          test_velocity          │      0.0009136319858953357      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [3/6] 03_BCE_Smoothness_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/03_BCE_Smoothness_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
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

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  65.1% [325/499] | 30s | loss=0.0685


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0039 | val_loss=0.0961
  [hb] Epoch 1:  69.9% [349/499] | 30s | loss=0.0823


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  49.2s | train_loss=0.0037 | val_loss=0.0931 | train_auroc=0.7540 | val_auroc=0.8419 | train_auprc=0.0918 | val_auprc=0.1685
  [hb] Epoch 2:  69.5% [347/499] | 30s | loss=0.0710


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  48.3s | train_loss=0.0047 | val_loss=0.0930 | train_auroc=0.7961 | val_auroc=0.8495 | train_auprc=0.1345 | val_auprc=0.1673
  [hb] Epoch 3:  65.9% [329/499] | 30s | loss=0.0852


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  47.6s | train_loss=0.0099 | val_loss=0.0902 | train_auroc=0.8125 | val_auroc=0.8539 | train_auprc=0.1462 | val_auprc=0.1694
  [hb] Epoch 4:  68.7% [343/499] | 30s | loss=0.0834


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  49.7s | train_loss=0.0027 | val_loss=0.0914 | train_auroc=0.8168 | val_auroc=0.8601 | train_auprc=0.1542 | val_auprc=0.1771
  [hb] Epoch 5:  66.5% [332/499] | 30s | loss=0.1473


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  49.0s | train_loss=0.0068 | val_loss=0.0910 | train_auroc=0.8258 | val_auroc=0.8597 | train_auprc=0.1642 | val_auprc=0.1745
  [hb] Epoch 6:  68.7% [343/499] | 30s | loss=0.1098


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  48.6s | train_loss=0.0235 | val_loss=0.0896 | train_auroc=0.8337 | val_auroc=0.8619 | train_auprc=0.1759 | val_auprc=0.1753
  [hb] Epoch 7:  69.3% [346/499] | 30s | loss=0.1312


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  49.2s | train_loss=0.0195 | val_loss=0.0901 | train_auroc=0.8396 | val_auroc=0.8614 | train_auprc=0.1854 | val_auprc=0.1778
  [hb] Epoch 8:  68.3% [341/499] | 30s | loss=0.0715


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  48.8s | train_loss=0.0179 | val_loss=0.0900 | train_auroc=0.8477 | val_auroc=0.8628 | train_auprc=0.1915 | val_auprc=0.1744
  [hb] Epoch 9:  67.9% [339/499] | 30s | loss=0.0745


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  48.2s | train_loss=0.0078 | val_loss=0.0911 | train_auroc=0.8520 | val_auroc=0.8619 | train_auprc=0.2052 | val_auprc=0.1825
  [hb] Epoch 10:  68.3% [341/499] | 30s | loss=0.0851


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  49.1s | train_loss=0.0045 | val_loss=0.0905 | train_auroc=0.8582 | val_auroc=0.8575 | train_auprc=0.2271 | val_auprc=0.1776
  [hb] Epoch 11:  65.5% [327/499] | 30s | loss=0.1033


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  48.2s | train_loss=0.4936 | val_loss=0.0907 | train_auroc=0.8663 | val_auroc=0.8582 | train_auprc=0.2416 | val_auprc=0.1786
  [hb] Epoch 12:  68.5% [342/499] | 30s | loss=0.0967


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  49.2s | train_loss=1.5945 | val_loss=0.0905 | train_auroc=0.8736 | val_auroc=0.8602 | train_auprc=0.2638 | val_auprc=0.1836
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=06-val_loss=0.0896.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/03_BCE_Smoothness_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=06-val_loss=0.0896.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=06-val_loss=0.0896.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0797 | test_monotonicity=0.0000 | test_state=0.0796 | test_velocity=0.0009
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      7.377958536380902e-05      │
│            test_asf             │       0.03096504509449005       │
│        test_asf_ci_high         │       0.0323123000562191        │
│         test_asf_ci_low         │      0.029548268765211105       │
│           test_auprc            │       0.15027451515197754       │
│       test_auprc_ci_high        │       0.18119332194328308       │
│        test_auprc_ci_low        │       0.1238970085978508        │
│           test_auroc            │       0.8057432174682617        │
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
│            test_loss            │       0.0796516016125679        │
│       test_mean_lead_time       │        66.59358215332031        │
│   test_mean_lead_time_ci_high   │        75.91221618652344        │
│   test_mean_lead_time_ci_low    │        57.98990249633789        │
│      test_median_lead_time      │              51.0               │
│  test_median_lead_time_ci_high  │              56.0               │
│  test_median_lead_time_ci_low   │              38.5               │
│        test_monotonicity        │      3.210208524251357e-05      │
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
│           test_state            │       0.07955704629421234       │
│            test_tce             │       0.6753439903259277        │
│        test_tce_ci_high         │       0.6879831552505493        │
│         test_tce_ci_low         │       0.6615553498268127        │
│          test_velocity          │      0.0008686434011906385      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [4/6] 04_BCE_TotalVariation_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/04_BCE_TotalVariation_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
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

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  66.5% [332/499] | 30s | loss=0.1165


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0040 | val_loss=0.0964
  [hb] Epoch 1:  66.5% [332/499] | 30s | loss=0.0486


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  48.7s | train_loss=0.0041 | val_loss=0.0933 | train_auroc=0.7547 | val_auroc=0.8423 | train_auprc=0.0921 | val_auprc=0.1683
  [hb] Epoch 2:  66.9% [334/499] | 30s | loss=0.0496


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  50.0s | train_loss=0.0048 | val_loss=0.0930 | train_auroc=0.7969 | val_auroc=0.8498 | train_auprc=0.1347 | val_auprc=0.1666
  [hb] Epoch 3:  67.3% [336/499] | 30s | loss=0.0528


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  50.1s | train_loss=0.0107 | val_loss=0.0904 | train_auroc=0.8132 | val_auroc=0.8539 | train_auprc=0.1459 | val_auprc=0.1702
  [hb] Epoch 4:  65.1% [325/499] | 30s | loss=0.0896


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.3s | train_loss=0.0027 | val_loss=0.0921 | train_auroc=0.8173 | val_auroc=0.8598 | train_auprc=0.1524 | val_auprc=0.1768
  [hb] Epoch 5:  67.1% [335/499] | 30s | loss=0.0749


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  49.6s | train_loss=0.0063 | val_loss=0.0909 | train_auroc=0.8257 | val_auroc=0.8600 | train_auprc=0.1633 | val_auprc=0.1711
  [hb] Epoch 6:  67.5% [337/499] | 30s | loss=0.0492


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  49.8s | train_loss=0.0259 | val_loss=0.0898 | train_auroc=0.8344 | val_auroc=0.8633 | train_auprc=0.1716 | val_auprc=0.1760
  [hb] Epoch 7:  63.9% [319/499] | 30s | loss=0.0650


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  48.4s | train_loss=0.0200 | val_loss=0.0905 | train_auroc=0.8402 | val_auroc=0.8607 | train_auprc=0.1841 | val_auprc=0.1781
  [hb] Epoch 8:  67.7% [338/499] | 30s | loss=0.1414


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  50.1s | train_loss=0.0181 | val_loss=0.0900 | train_auroc=0.8477 | val_auroc=0.8612 | train_auprc=0.1879 | val_auprc=0.1742
  [hb] Epoch 9:  67.1% [335/499] | 30s | loss=0.0783


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  49.5s | train_loss=0.0074 | val_loss=0.0906 | train_auroc=0.8527 | val_auroc=0.8620 | train_auprc=0.2010 | val_auprc=0.1799
  [hb] Epoch 10:  64.1% [320/499] | 30s | loss=0.1000


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  48.5s | train_loss=0.0048 | val_loss=0.0904 | train_auroc=0.8580 | val_auroc=0.8606 | train_auprc=0.2224 | val_auprc=0.1821
  [hb] Epoch 11:  67.3% [336/499] | 30s | loss=0.1016


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.8s | train_loss=0.4408 | val_loss=0.0908 | train_auroc=0.8651 | val_auroc=0.8592 | train_auprc=0.2344 | val_auprc=0.1840
  [hb] Epoch 12:  67.7% [338/499] | 30s | loss=0.0924


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  49.7s | train_loss=1.9492 | val_loss=0.0914 | train_auroc=0.8725 | val_auroc=0.8614 | train_auprc=0.2559 | val_auprc=0.1838
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=06-val_loss=0.0898.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/04_BCE_TotalVariation_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=06-val_loss=0.0898.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=06-val_loss=0.0898.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0797 | test_monotonicity=0.0000 | test_state=0.0796 | test_velocity=0.0008
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      7.17876828275621e-05       │
│            test_asf             │      0.029911436140537262       │
│        test_asf_ci_high         │       0.03110921010375023       │
│         test_asf_ci_low         │      0.028659487143158913       │
│           test_auprc            │       0.15069729089736938       │
│       test_auprc_ci_high        │       0.1807190626859665        │
│        test_auprc_ci_low        │       0.12395773828029633       │
│           test_auroc            │       0.8053753972053528        │
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
│            test_loss            │       0.0796886533498764        │
│       test_mean_lead_time       │              66.25              │
│   test_mean_lead_time_ci_high   │        75.75570678710938        │
│   test_mean_lead_time_ci_low    │        57.70690155029297        │
│      test_median_lead_time      │              50.0               │
│  test_median_lead_time_ci_high  │              55.0               │
│  test_median_lead_time_ci_low   │        38.98749923706055        │
│        test_monotonicity        │     2.9802795324940234e-05      │
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
│           test_state            │       0.07959654927253723       │
│            test_tce             │       0.6769675612449646        │
│        test_tce_ci_high         │       0.6891003847122192        │
│         test_tce_ci_low         │       0.6638991832733154        │
│          test_velocity          │      0.0008463703561574221      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [5/6] 05_BGSL_StateOnly_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/05_BGSL_StateOnly_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
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

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  64.3% [321/499] | 30s | loss=0.0828


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0033 | val_loss=0.0796
  [hb] Epoch 1:  66.1% [330/499] | 30s | loss=0.1342


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  52.0s | train_loss=0.0022 | val_loss=0.0776 | train_auroc=0.7516 | val_auroc=0.8372 | train_auprc=0.0898 | val_auprc=0.1668
  [hb] Epoch 2:  64.5% [322/499] | 30s | loss=0.0934


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  49.8s | train_loss=0.0033 | val_loss=0.0770 | train_auroc=0.7949 | val_auroc=0.8463 | train_auprc=0.1347 | val_auprc=0.1701
  [hb] Epoch 3:  66.3% [331/499] | 30s | loss=0.0237


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  50.0s | train_loss=0.0075 | val_loss=0.0750 | train_auroc=0.8109 | val_auroc=0.8512 | train_auprc=0.1472 | val_auprc=0.1712
  [hb] Epoch 4:  65.9% [329/499] | 30s | loss=0.1034


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  50.3s | train_loss=0.0016 | val_loss=0.0761 | train_auroc=0.8151 | val_auroc=0.8569 | train_auprc=0.1537 | val_auprc=0.1776
  [hb] Epoch 5:  62.3% [311/499] | 30s | loss=0.1149


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  50.9s | train_loss=0.0041 | val_loss=0.0753 | train_auroc=0.8249 | val_auroc=0.8574 | train_auprc=0.1672 | val_auprc=0.1760
  [hb] Epoch 6:  62.7% [313/499] | 30s | loss=0.0484


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  53.1s | train_loss=0.0212 | val_loss=0.0752 | train_auroc=0.8326 | val_auroc=0.8609 | train_auprc=0.1742 | val_auprc=0.1744
  [hb] Epoch 7:  62.9% [314/499] | 30s | loss=0.0450


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  51.6s | train_loss=0.0166 | val_loss=0.0753 | train_auroc=0.8388 | val_auroc=0.8547 | train_auprc=0.1873 | val_auprc=0.1721
  [hb] Epoch 8:  64.9% [324/499] | 30s | loss=0.0941


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  50.8s | train_loss=0.0130 | val_loss=0.0748 | train_auroc=0.8462 | val_auroc=0.8596 | train_auprc=0.1959 | val_auprc=0.1604
  [hb] Epoch 9:  67.1% [335/499] | 30s | loss=0.0639


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  50.7s | train_loss=0.0074 | val_loss=0.0760 | train_auroc=0.8502 | val_auroc=0.8589 | train_auprc=0.2011 | val_auprc=0.1719
  [hb] Epoch 10:  66.1% [330/499] | 30s | loss=0.0200


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  50.2s | train_loss=0.0033 | val_loss=0.0757 | train_auroc=0.8561 | val_auroc=0.8551 | train_auprc=0.2234 | val_auprc=0.1647
  [hb] Epoch 11:  66.5% [332/499] | 30s | loss=0.0908


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  50.4s | train_loss=0.4311 | val_loss=0.0759 | train_auroc=0.8627 | val_auroc=0.8600 | train_auprc=0.2353 | val_auprc=0.1749
  [hb] Epoch 12:  64.9% [324/499] | 30s | loss=0.1154


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  49.2s | train_loss=1.4144 | val_loss=0.0767 | train_auroc=0.8688 | val_auroc=0.8585 | train_auprc=0.2569 | val_auprc=0.1705
  [hb] Epoch 13:  65.1% [325/499] | 30s | loss=0.0366


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  50.7s | train_loss=0.0069 | val_loss=0.0765 | train_auroc=0.8707 | val_auroc=0.8530 | train_auprc=0.2674 | val_auprc=0.1587
  [hb] Epoch 14:  64.3% [321/499] | 30s | loss=0.0378


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  14 |  51.2s | train_loss=0.0543 | val_loss=0.0764 | train_auroc=0.8735 | val_auroc=0.8570 | train_auprc=0.2717 | val_auprc=0.1659
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=08-val_loss=0.0748.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/05_BGSL_StateOnly_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=08-val_loss=0.0748.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=08-val_loss=0.0748.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0792 | test_monotonicity=0.0000 | test_state=0.0791 | test_velocity=0.0008
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      6.832555664004758e-05      │
│            test_asf             │      0.036345940083265305       │
│        test_asf_ci_high         │       0.03776872158050537       │
│         test_asf_ci_low         │       0.0348864309489727        │
│           test_auprc            │       0.14568978548049927       │
│       test_auprc_ci_high        │       0.1741805076599121        │
│        test_auprc_ci_low        │       0.12030553072690964       │
│           test_auroc            │       0.8052347898483276        │
│       test_auroc_ci_high        │       0.8270618319511414        │
│        test_auroc_ci_low        │       0.7819582223892212        │
│        test_brier_score         │      0.021788179874420166       │
│    test_brier_score_ci_high     │       0.02432798407971859       │
│     test_brier_score_ci_low     │      0.019319595769047737       │
│            test_ece             │      0.006861014757305384       │
│        test_ece_ci_high         │      0.009773114696145058       │
│         test_ece_ci_low         │      0.004509091842919588       │
│             test_f1             │       0.10651110112667084       │
│         test_f1_ci_high         │       0.1181306391954422        │
│         test_f1_ci_low          │       0.09483613818883896       │
│          test_fa_rate           │      0.021605247631669044       │
│      test_fa_rate_ci_high       │      0.022438058629631996       │
│       test_fa_rate_ci_low       │       0.02076820842921734       │
│           test_fappd            │       0.5185259580612183        │
│       test_fappd_ci_high        │       0.5385134220123291        │
│        test_fappd_ci_low        │       0.4984370172023773        │
│     test_lead_time_iqr_high     │              93.0               │
│ test_lead_time_iqr_high_ci_high │              109.0              │
│ test_lead_time_iqr_high_ci_low  │              75.0               │
│     test_lead_time_iqr_low      │              17.0               │
│ test_lead_time_iqr_low_ci_high  │              23.0               │
│  test_lead_time_iqr_low_ci_low  │              13.75              │
│            test_loss            │        0.079231396317482        │
│       test_mean_lead_time       │        65.12245178222656        │
│   test_mean_lead_time_ci_high   │         74.464599609375         │
│   test_mean_lead_time_ci_low    │        56.47641372680664        │
│      test_median_lead_time      │              48.5               │
│  test_median_lead_time_ci_high  │              58.0               │
│  test_median_lead_time_ci_low   │       36.974998474121094        │
│        test_monotonicity        │     2.4999382731039077e-05      │
│            test_npv             │       0.9914952516555786        │
│        test_npv_ci_high         │       0.9931238889694214        │
│         test_npv_ci_low         │       0.9896675944328308        │
│     test_physionet_utility      │       0.07087618112564087       │
│ test_physionet_utility_ci_high  │       0.1825292557477951        │
│  test_physionet_utility_ci_low  │      -0.03296421840786934       │
│            test_poms            │       0.5531550049781799        │
│        test_poms_ci_high        │        0.591316819190979        │
│        test_poms_ci_low         │       0.5165383815765381        │
│            test_ppv             │       0.05729324743151665       │
│        test_ppv_ci_high         │       0.06394670158624649       │
│         test_ppv_ci_low         │       0.05059739202260971       │
│            test_rtv             │       0.09057223051786423       │
│        test_rtv_ci_high         │       0.09909269958734512       │
│         test_rtv_ci_low         │       0.08239764720201492       │
│     test_selected_threshold     │       0.01005025114864111       │
│        test_sensitivity         │       0.7556742429733276        │
│    test_sensitivity_ci_high     │       0.7949895262718201        │
│     test_sensitivity_ci_low     │       0.7138551473617554        │
│        test_specificity         │       0.6961243152618408        │
│    test_specificity_ci_high     │       0.7102245688438416        │
│     test_specificity_ci_low     │       0.6818621158599854        │
│            test_spj             │      0.0015923160826787353      │
│        test_spj_ci_high         │      0.0016577178612351418      │
│         test_spj_ci_low         │      0.001525563420727849       │
│           test_state            │       0.0791444405913353        │
│            test_tce             │       0.7070766091346741        │
│        test_tce_ci_high         │        0.717106819152832        │
│         test_tce_ci_low         │       0.6965024471282959        │
│          test_velocity          │      0.0007988543366082013      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [6/6] 06_Full_BGSL_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/06_Full_BGSL_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
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

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  65.1% [325/499] | 30s | loss=0.0567


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0033 | val_loss=0.0797
  [hb] Epoch 1:  68.1% [340/499] | 30s | loss=0.0776


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  49.4s | train_loss=0.0022 | val_loss=0.0777 | train_auroc=0.7516 | val_auroc=0.8372 | train_auprc=0.0898 | val_auprc=0.1668
  [hb] Epoch 2:  69.9% [349/499] | 30s | loss=0.0342


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  48.6s | train_loss=0.0033 | val_loss=0.0770 | train_auroc=0.7949 | val_auroc=0.8463 | train_auprc=0.1347 | val_auprc=0.1701
  [hb] Epoch 3:  68.1% [340/499] | 30s | loss=0.0749


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  48.3s | train_loss=0.0075 | val_loss=0.0751 | train_auroc=0.8109 | val_auroc=0.8512 | train_auprc=0.1472 | val_auprc=0.1712
  [hb] Epoch 4:  68.5% [342/499] | 30s | loss=0.1176


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  48.4s | train_loss=0.0016 | val_loss=0.0762 | train_auroc=0.8151 | val_auroc=0.8569 | train_auprc=0.1537 | val_auprc=0.1776
  [hb] Epoch 5:  66.5% [332/499] | 30s | loss=0.1233


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  48.8s | train_loss=0.0041 | val_loss=0.0754 | train_auroc=0.8249 | val_auroc=0.8574 | train_auprc=0.1672 | val_auprc=0.1760
  [hb] Epoch 6:  66.9% [334/499] | 30s | loss=0.1202


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  50.5s | train_loss=0.0212 | val_loss=0.0753 | train_auroc=0.8326 | val_auroc=0.8609 | train_auprc=0.1742 | val_auprc=0.1744
  [hb] Epoch 7:  64.9% [324/499] | 30s | loss=0.1266


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  49.2s | train_loss=0.0166 | val_loss=0.0754 | train_auroc=0.8388 | val_auroc=0.8547 | train_auprc=0.1873 | val_auprc=0.1721
  [hb] Epoch 8:  66.7% [333/499] | 30s | loss=0.0799


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  50.7s | train_loss=0.0130 | val_loss=0.0749 | train_auroc=0.8462 | val_auroc=0.8596 | train_auprc=0.1959 | val_auprc=0.1604
  [hb] Epoch 9:  67.1% [335/499] | 30s | loss=0.0640


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  50.1s | train_loss=0.0074 | val_loss=0.0761 | train_auroc=0.8502 | val_auroc=0.8589 | train_auprc=0.2011 | val_auprc=0.1719
  [hb] Epoch 10:  66.5% [332/499] | 30s | loss=0.0112


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  48.7s | train_loss=0.0033 | val_loss=0.0758 | train_auroc=0.8561 | val_auroc=0.8551 | train_auprc=0.2234 | val_auprc=0.1647
  [hb] Epoch 11:  63.1% [315/499] | 30s | loss=0.1219


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  49.0s | train_loss=0.4317 | val_loss=0.0760 | train_auroc=0.8627 | val_auroc=0.8600 | train_auprc=0.2353 | val_auprc=0.1749
  [hb] Epoch 12:  66.1% [330/499] | 30s | loss=0.0748


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  52.3s | train_loss=1.4151 | val_loss=0.0768 | train_auroc=0.8688 | val_auroc=0.8585 | train_auprc=0.2569 | val_auprc=0.1705
  [hb] Epoch 13:  67.9% [339/499] | 30s | loss=0.0663


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  50.6s | train_loss=0.0069 | val_loss=0.0766 | train_auroc=0.8707 | val_auroc=0.8530 | train_auprc=0.2674 | val_auprc=0.1587
  [hb] Epoch 14:  67.1% [335/499] | 30s | loss=0.0447


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  14 |  48.8s | train_loss=0.0543 | val_loss=0.0764 | train_auroc=0.8735 | val_auroc=0.8570 | train_auprc=0.2717 | val_auprc=0.1659
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/06_Full_BGSL_seed42/checkpoints/epoch=08-val_loss=0.0749.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/06_Full_BGSL_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/06_Full_BGSL_seed42/checkpoints/epoch=08-val_loss=0.0749.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/runs/06_Full_BGSL_seed42/checkpoints/epoch=08-val_loss=0.0749.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0792 | test_monotonicity=0.0000 | test_state=0.0791 | test_velocity=0.0008
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      6.832555664004758e-05      │
│            test_asf             │      0.036345940083265305       │
│        test_asf_ci_high         │       0.03776872158050537       │
│         test_asf_ci_low         │       0.0348864309489727        │
│           test_auprc            │       0.14568978548049927       │
│       test_auprc_ci_high        │       0.1741805076599121        │
│        test_auprc_ci_low        │       0.12030553072690964       │
│           test_auroc            │       0.8052347898483276        │
│       test_auroc_ci_high        │       0.8270618319511414        │
│        test_auroc_ci_low        │       0.7819582223892212        │
│        test_brier_score         │      0.021788179874420166       │
│    test_brier_score_ci_high     │       0.02432798407971859       │
│     test_brier_score_ci_low     │      0.019319595769047737       │
│            test_ece             │      0.006861014757305384       │
│        test_ece_ci_high         │      0.009773114696145058       │
│         test_ece_ci_low         │      0.004509091842919588       │
│             test_f1             │       0.10651110112667084       │
│         test_f1_ci_high         │       0.1181306391954422        │
│         test_f1_ci_low          │       0.09483613818883896       │
│          test_fa_rate           │      0.021605247631669044       │
│      test_fa_rate_ci_high       │      0.022438058629631996       │
│       test_fa_rate_ci_low       │       0.02076820842921734       │
│           test_fappd            │       0.5185259580612183        │
│       test_fappd_ci_high        │       0.5385134220123291        │
│        test_fappd_ci_low        │       0.4984370172023773        │
│     test_lead_time_iqr_high     │              93.0               │
│ test_lead_time_iqr_high_ci_high │              109.0              │
│ test_lead_time_iqr_high_ci_low  │              75.0               │
│     test_lead_time_iqr_low      │              17.0               │
│ test_lead_time_iqr_low_ci_high  │              23.0               │
│  test_lead_time_iqr_low_ci_low  │              13.75              │
│            test_loss            │        0.079231396317482        │
│       test_mean_lead_time       │        65.12245178222656        │
│   test_mean_lead_time_ci_high   │         74.464599609375         │
│   test_mean_lead_time_ci_low    │        56.47641372680664        │
│      test_median_lead_time      │              48.5               │
│  test_median_lead_time_ci_high  │              58.0               │
│  test_median_lead_time_ci_low   │       36.974998474121094        │
│        test_monotonicity        │     2.4999382731039077e-05      │
│            test_npv             │       0.9914952516555786        │
│        test_npv_ci_high         │       0.9931238889694214        │
│         test_npv_ci_low         │       0.9896675944328308        │
│     test_physionet_utility      │       0.07087618112564087       │
│ test_physionet_utility_ci_high  │       0.1825292557477951        │
│  test_physionet_utility_ci_low  │      -0.03296421840786934       │
│            test_poms            │       0.5531550049781799        │
│        test_poms_ci_high        │        0.591316819190979        │
│        test_poms_ci_low         │       0.5165383815765381        │
│            test_ppv             │       0.05729324743151665       │
│        test_ppv_ci_high         │       0.06394670158624649       │
│         test_ppv_ci_low         │       0.05059739202260971       │
│            test_rtv             │       0.09057223051786423       │
│        test_rtv_ci_high         │       0.09909269958734512       │
│         test_rtv_ci_low         │       0.08239764720201492       │
│     test_selected_threshold     │       0.01005025114864111       │
│        test_sensitivity         │       0.7556742429733276        │
│    test_sensitivity_ci_high     │       0.7949895262718201        │
│     test_sensitivity_ci_low     │       0.7138551473617554        │
│        test_specificity         │       0.6961243152618408        │
│    test_specificity_ci_high     │       0.7102245688438416        │
│     test_specificity_ci_low     │       0.6818621158599854        │
│            test_spj             │      0.0015923160826787353      │
│        test_spj_ci_high         │      0.0016577178612351418      │
│         test_spj_ci_low         │      0.001525563420727849       │
│           test_state            │       0.0791444405913353        │
│            test_tce             │       0.7070766091346741        │
│        test_tce_ci_high         │        0.717106819152832        │
│         test_tce_ci_low         │       0.6965024471282959        │
│          test_velocity          │      0.0007988543366082013      │
└─────────────────────────────────┴─────────────────────────────────┘

Summary written: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/summary.csv
Manifest: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/batch_manifest.json
Overall status: success

========================================================================
  POST-PROCESSING
========================================================================

========================================================================
  AGGREGATED RESULTS — 20260607-073908__ae6401ae
  Conditions: 6 | Total runs: 6
  Baseline for t-tests: 01_BCE_Baseline
========================================================================

  --- 01_BCE_Baseline (n=1) ---
    test_auroc                                = 0.8060 ± 0.0000 (95% CI)
    test_auprc                                = 0.1499 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.0503 ± 0.0000 (95% CI)
    test_median_lead_time                     = 50.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 66.4813 ± 0.0000 (95% CI)
    test_asf                                  = 0.0305 ± 0.0000 (95% CI)
    test_rtv                                  = 0.1190 ± 0.0000 (95% CI)
    test_spj                                  = 0.0021 ± 0.0000 (95% CI)
    test_poms                                 = 0.5582 ± 0.0000 (95% CI)
    test_tce                                  = 0.6756 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0216 ± 0.0000 (95% CI)
    test_fappd                                = 0.5182 ± 0.0000 (95% CI)

  --- 02_TLS (n=1) ---
    test_auroc                                = 0.8030 ± 0.0000 (95% CI)
    test_auprc                                = 0.1514 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.0339 ± 0.0000 (95% CI)
    test_median_lead_time                     = 51.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 67.7514 ± 0.0000 (95% CI)
    test_asf                                  = 0.0324 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0840 ± 0.0000 (95% CI)
    test_spj                                  = 0.0014 ± 0.0000 (95% CI)
    test_poms                                 = 0.5532 ± 0.0000 (95% CI)
    test_tce                                  = 0.7161 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0225 ± 0.0000 (95% CI)
    test_fappd                                = 0.5404 ± 0.0000 (95% CI)

  --- 03_BCE_Smoothness (n=1) ---
    test_auroc                                = 0.8057 ± 0.0000 (95% CI)
    test_auprc                                = 0.1503 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.0465 ± 0.0000 (95% CI)
    test_median_lead_time                     = 51.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 66.5936 ± 0.0000 (95% CI)
    test_asf                                  = 0.0310 ± 0.0000 (95% CI)
    test_rtv                                  = 0.1184 ± 0.0000 (95% CI)
    test_spj                                  = 0.0020 ± 0.0000 (95% CI)
    test_poms                                 = 0.5623 ± 0.0000 (95% CI)
    test_tce                                  = 0.6753 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0218 ± 0.0000 (95% CI)
    test_fappd                                = 0.5244 ± 0.0000 (95% CI)

  --- 04_BCE_TotalVariation (n=1) ---
    test_auroc                                = 0.8054 ± 0.0000 (95% CI)
    test_auprc                                = 0.1507 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.0471 ± 0.0000 (95% CI)
    test_median_lead_time                     = 50.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 66.2500 ± 0.0000 (95% CI)
    test_asf                                  = 0.0299 ± 0.0000 (95% CI)
    test_rtv                                  = 0.1089 ± 0.0000 (95% CI)
    test_spj                                  = 0.0019 ± 0.0000 (95% CI)
    test_poms                                 = 0.5603 ± 0.0000 (95% CI)
    test_tce                                  = 0.6770 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0214 ± 0.0000 (95% CI)
    test_fappd                                = 0.5127 ± 0.0000 (95% CI)

  --- 05_BGSL_StateOnly (n=1) ---
    test_auroc                                = 0.8052 ± 0.0000 (95% CI)
    test_auprc                                = 0.1457 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.0709 ± 0.0000 (95% CI)
    test_median_lead_time                     = 48.5000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 65.1225 ± 0.0000 (95% CI)
    test_asf                                  = 0.0363 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0906 ± 0.0000 (95% CI)
    test_spj                                  = 0.0016 ± 0.0000 (95% CI)
    test_poms                                 = 0.5532 ± 0.0000 (95% CI)
    test_tce                                  = 0.7071 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0216 ± 0.0000 (95% CI)
    test_fappd                                = 0.5185 ± 0.0000 (95% CI)

  --- 06_Full_BGSL (n=1) ---
    test_auroc                                = 0.8052 ± 0.0000 (95% CI)
    test_auprc                                = 0.1457 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.0709 ± 0.0000 (95% CI)
    test_median_lead_time                     = 48.5000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 65.1225 ± 0.0000 (95% CI)
    test_asf                                  = 0.0363 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0906 ± 0.0000 (95% CI)
    test_spj                                  = 0.0016 ± 0.0000 (95% CI)
    test_poms                                 = 0.5532 ± 0.0000 (95% CI)
    test_tce                                  = 0.7071 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0216 ± 0.0000 (95% CI)
    test_fappd                                = 0.5185 ± 0.0000 (95% CI)

  Full results saved to: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/aggregated_results.csv

========================================================================
  LaTeX TABLE (95% CI)
========================================================================

\begin{table}[h]
\centering
\caption{Aggregated Test Results (Mean $\pm$ 95\% CI)}
\begin{tabular}{lccccc}
\toprule
Condition & AUPRC & Utility & Lead Time & ASF & POMS \\
\midrule
01\_BCE\_Baseline & 0.150 $\pm$ 0.000 & 0.050 $\pm$ 0.000 & 50.000 $\pm$ 0.000 & 0.030 $\pm$ 0.000 & 0.558 $\pm$ 0.000 \\
02\_TLS & \textbf{0.151 $\pm$ 0.000} & 0.034 $\pm$ 0.000 & \textbf{51.000 $\pm$ 0.000} & 0.032 $\pm$ 0.000 & 0.553 $\pm$ 0.000 \\
03\_BCE\_Smoothness & 0.150 $\pm$ 0.000 & 0.046 $\pm$ 0.000 & \textbf{51.000 $\pm$ 0.000} & 0.031 $\pm$ 0.000 & \textbf{0.562 $\pm$ 0.000} \\
04\_BCE\_TotalVariation & 0.151 $\pm$ 0.000 & 0.047 $\pm$ 0.000 & 50.000 $\pm$ 0.000 & \textbf{0.030 $\pm$ 0.000} & 0.560 $\pm$ 0.000 \\
05\_BGSL\_StateOnly & 0.146 $\pm$ 0.000 & \textbf{0.071 $\pm$ 0.000} & 48.500 $\pm$ 0.000 & 0.036 $\pm$ 0.000 & 0.553 $\pm$ 0.000 \\
06\_Full\_BGSL & 0.146 $\pm$ 0.000 & \textbf{0.071 $\pm$ 0.000} & 48.500 $\pm$ 0.000 & 0.036 $\pm$ 0.000 & 0.553 $\pm$ 0.000 \\
\bottomrule
\end{tabular}
\end{table}


[01_BCE_Baseline] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots/01_BCE_Baseline
[02_TLS] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots/02_TLS
[03_BCE_Smoothness] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots/03_BCE_Smoothness
[04_BCE_TotalVariation] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots/04_BCE_TotalVariation
[05_BGSL_StateOnly] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots/05_BGSL_StateOnly
[06_Full_BGSL] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots/06_Full_BGSL

All plots saved under /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260607-073908__ae6401ae/plots