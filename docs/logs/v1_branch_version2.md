--- Execution Mode: LOCAL_SEQUENTIAL | Total Jobs: 6 | Batch: 20260608-044438__0c2bc87d ---
Outputs root: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d

=== [1/6] 01_BCE_Baseline_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name                 ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone             │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn              │ BCELoss          │      0 │ train │     0 │
│ 2 │ train_metrics        │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics          │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics         │ MetricCollection │      0 │ train │     0 │
│ 5 │ train_target_metrics │ MetricCollection │      0 │ train │     0 │
│ 6 │ val_target_metrics   │ MetricCollection │      0 │ train │     0 │
│ 7 │ test_target_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴──────────────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 27                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  55.6% [277/498] | 30s | loss=0.1185


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0321 | val_loss=0.0983
  [hb] Epoch 1:  58.0% [289/498] | 30s | loss=0.0558


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  59.7s | train_loss=0.0434 | val_loss=0.0953 | train_auroc=0.7733 | val_auroc=0.8577 | train_auprc=0.0638 | val_auprc=0.1141
  [hb] Epoch 2:  57.4% [286/498] | 30s | loss=0.1998


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  57.1s | train_loss=0.0669 | val_loss=0.0931 | train_auroc=0.8142 | val_auroc=0.8567 | train_auprc=0.0913 | val_auprc=0.1171
  [hb] Epoch 3:  57.0% [284/498] | 30s | loss=0.0481


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  57.3s | train_loss=0.1048 | val_loss=0.0898 | train_auroc=0.8251 | val_auroc=0.8635 | train_auprc=0.1032 | val_auprc=0.1189
  [hb] Epoch 4:  54.8% [273/498] | 30s | loss=0.0834


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  58.2s | train_loss=0.0798 | val_loss=0.0906 | train_auroc=0.8351 | val_auroc=0.8676 | train_auprc=0.1160 | val_auprc=0.1156
  [hb] Epoch 5:  56.6% [282/498] | 30s | loss=0.0696


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  58.8s | train_loss=0.1189 | val_loss=0.0917 | train_auroc=0.8464 | val_auroc=0.8613 | train_auprc=0.1189 | val_auprc=0.1139
  [hb] Epoch 6:  58.0% [289/498] | 30s | loss=0.0696


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  57.8s | train_loss=0.0855 | val_loss=0.0891 | train_auroc=0.8518 | val_auroc=0.8704 | train_auprc=0.1297 | val_auprc=0.1096
  [hb] Epoch 7:  57.0% [284/498] | 30s | loss=0.0695


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  57.3s | train_loss=0.1010 | val_loss=0.0891 | train_auroc=0.8539 | val_auroc=0.8735 | train_auprc=0.1322 | val_auprc=0.1184
  [hb] Epoch 8:  57.4% [286/498] | 30s | loss=0.0763


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  57.7s | train_loss=0.0464 | val_loss=0.0890 | train_auroc=0.8633 | val_auroc=0.8711 | train_auprc=0.1440 | val_auprc=0.1099
  [hb] Epoch 9:  57.8% [288/498] | 30s | loss=0.0363


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  57.5s | train_loss=0.0895 | val_loss=0.0896 | train_auroc=0.8684 | val_auroc=0.8715 | train_auprc=0.1515 | val_auprc=0.1161
  [hb] Epoch 10:  57.0% [284/498] | 30s | loss=0.0910


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  57.4s | train_loss=0.0685 | val_loss=0.0894 | train_auroc=0.8728 | val_auroc=0.8705 | train_auprc=0.1508 | val_auprc=0.1128
  [hb] Epoch 11:  56.6% [282/498] | 30s | loss=0.0102


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  57.7s | train_loss=0.0610 | val_loss=0.0922 | train_auroc=0.8805 | val_auroc=0.8725 | train_auprc=0.1696 | val_auprc=0.1185
  [hb] Epoch 12:  56.2% [280/498] | 30s | loss=0.0401


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  57.9s | train_loss=0.0470 | val_loss=0.0924 | train_auroc=0.8881 | val_auroc=0.8632 | train_auprc=0.1897 | val_auprc=0.1049
  [hb] Epoch 13:  56.2% [280/498] | 30s | loss=0.0849


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  57.8s | train_loss=0.0551 | val_loss=0.0915 | train_auroc=0.8939 | val_auroc=0.8613 | train_auprc=0.2038 | val_auprc=0.0972
  [hb] Epoch 14:  56.6% [282/498] | 30s | loss=0.0506


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  14 |  58.6s | train_loss=0.0823 | val_loss=0.0951 | train_auroc=0.9010 | val_auroc=0.8659 | train_auprc=0.2237 | val_auprc=0.1029
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/01_BCE_Baseline_seed42/checkpoints/epoch=08-val_loss=0.0890.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/01_BCE_Baseline_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/01_BCE_Baseline_seed42/checkpoints/epoch=08-val_loss=0.0890.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/01_BCE_Baseline_seed42/checkpoints/epoch=08-val_loss=0.0890.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0001 | test_loss=0.0787 | test_monotonicity=0.0000 | test_state=0.0787 | test_velocity=0.0001
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      8.814050670480356e-05      │
│            test_asf             │       0.02937408536672592       │
│        test_asf_ci_high         │      0.030746471136808395       │
│         test_asf_ci_low         │      0.027830500155687332       │
│           test_auprc            │       0.1300858110189438        │
│       test_auprc_ci_high        │        0.16107077896595         │
│        test_auprc_ci_low        │       0.10439357906579971       │
│           test_auroc            │       0.8284213542938232        │
│       test_auroc_ci_high        │       0.8538293242454529        │
│        test_auroc_ci_low        │        0.802020788192749        │
│        test_brier_score         │       0.01430758461356163       │
│    test_brier_score_ci_high     │       0.01588846929371357       │
│     test_brier_score_ci_low     │      0.012723524123430252       │
│            test_ece             │      0.007117815315723419       │
│        test_ece_ci_high         │       0.00918085128068924       │
│         test_ece_ci_low         │      0.005032193381339312       │
│             test_f1             │       0.08903523534536362       │
│         test_f1_ci_high         │       0.09877989441156387       │
│         test_f1_ci_low          │       0.07935138046741486       │
│          test_fa_rate           │      0.018615897744894028       │
│      test_fa_rate_ci_high       │       0.01945084147155285       │
│       test_fa_rate_ci_low       │      0.017685621976852417       │
│           test_fappd            │       0.44678154587745667       │
│       test_fappd_ci_high        │       0.46682021021842957       │
│        test_fappd_ci_low        │        0.424454927444458        │
│     test_lead_time_iqr_high     │              85.0               │
│ test_lead_time_iqr_high_ci_high │              104.0              │
│ test_lead_time_iqr_high_ci_low  │        71.9937515258789         │
│     test_lead_time_iqr_low      │              16.0               │
│ test_lead_time_iqr_low_ci_high  │              22.0               │
│  test_lead_time_iqr_low_ci_low  │              13.0               │
│            test_loss            │       0.07867664843797684       │
│       test_mean_lead_time       │        64.5049057006836         │
│   test_mean_lead_time_ci_high   │        73.15656280517578        │
│   test_mean_lead_time_ci_low    │        55.82514572143555        │
│      test_median_lead_time      │              46.0               │
│  test_median_lead_time_ci_high  │              56.0               │
│  test_median_lead_time_ci_low   │              37.0               │
│        test_monotonicity        │               0.0               │
│            test_npv             │       0.9947471618652344        │
│        test_npv_ci_high         │       0.9958439469337463        │
│         test_npv_ci_low         │        0.993493378162384        │
│     test_physionet_utility      │       0.31311747431755066       │
│ test_physionet_utility_ci_high  │       0.37315672636032104       │
│  test_physionet_utility_ci_low  │       0.2498941570520401        │
│            test_poms            │       0.5330578684806824        │
│        test_poms_ci_high        │       0.5642181038856506        │
│        test_poms_ci_low         │       0.5018864274024963        │
│            test_ppv             │       0.04739166423678398       │
│        test_ppv_ci_high         │       0.05293184146285057       │
│         test_ppv_ci_low         │       0.04210495576262474       │
│            test_rtv             │       0.12408240884542465       │
│        test_rtv_ci_high         │       0.13464565575122833       │
│         test_rtv_ci_low         │       0.11370068788528442       │
│     test_selected_threshold     │       0.02010050229728222       │
│        test_sensitivity         │       0.7340744137763977        │
│    test_sensitivity_ci_high     │       0.7826671600341797        │
│     test_sensitivity_ci_low     │       0.6825258135795593        │
│        test_specificity         │       0.7733924984931946        │
│    test_specificity_ci_high     │       0.7886328101158142        │
│     test_specificity_ci_low     │       0.7583351731300354        │
│            test_spj             │      0.002314933342859149       │
│        test_spj_ci_high         │       0.00241594179533422       │
│         test_spj_ci_low         │      0.002214849693700671       │
│           test_state            │       0.07865691930055618       │
│        test_target_auprc        │       0.15946654975414276       │
│        test_target_auroc        │       0.8141050934791565        │
│            test_tce             │       0.8538075089454651        │
│        test_tce_ci_high         │       0.8686256408691406        │
│         test_tce_ci_low         │       0.8374662399291992        │
│          test_velocity          │     0.00010925015521934256      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [2/6] 02_TLS_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/02_TLS_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name                 ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone             │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn              │ TLSLoss          │      0 │ train │     0 │
│ 2 │ train_metrics        │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics          │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics         │ MetricCollection │      0 │ train │     0 │
│ 5 │ train_target_metrics │ MetricCollection │      0 │ train │     0 │
│ 6 │ val_target_metrics   │ MetricCollection │      0 │ train │     0 │
│ 7 │ test_target_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴──────────────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 27                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  55.0% [274/498] | 30s | loss=0.1262


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0251 | val_loss=0.0818
  [hb] Epoch 1:  56.4% [281/498] | 30s | loss=0.0804


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  58.9s | train_loss=0.0354 | val_loss=0.0794 | train_auroc=0.7726 | val_auroc=0.8587 | train_auprc=0.0625 | val_auprc=0.1145
  [hb] Epoch 2:  56.2% [280/498] | 30s | loss=0.0332


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  58.3s | train_loss=0.0550 | val_loss=0.0777 | train_auroc=0.8165 | val_auroc=0.8592 | train_auprc=0.0934 | val_auprc=0.1195
  [hb] Epoch 3:  55.6% [277/498] | 30s | loss=0.0175


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  58.2s | train_loss=0.0935 | val_loss=0.0745 | train_auroc=0.8286 | val_auroc=0.8650 | train_auprc=0.1076 | val_auprc=0.1205
  [hb] Epoch 4:  57.0% [284/498] | 30s | loss=0.0675


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  59.1s | train_loss=0.0633 | val_loss=0.0749 | train_auroc=0.8384 | val_auroc=0.8700 | train_auprc=0.1206 | val_auprc=0.1209
  [hb] Epoch 5:  57.2% [285/498] | 30s | loss=0.0555


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  57.8s | train_loss=0.0947 | val_loss=0.0758 | train_auroc=0.8502 | val_auroc=0.8688 | train_auprc=0.1274 | val_auprc=0.1151
  [hb] Epoch 6:  57.2% [285/498] | 30s | loss=0.0520


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  57.8s | train_loss=0.0681 | val_loss=0.0744 | train_auroc=0.8549 | val_auroc=0.8722 | train_auprc=0.1336 | val_auprc=0.1134
  [hb] Epoch 7:  56.8% [283/498] | 30s | loss=0.0200


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  57.7s | train_loss=0.0800 | val_loss=0.0738 | train_auroc=0.8576 | val_auroc=0.8736 | train_auprc=0.1378 | val_auprc=0.1180
  [hb] Epoch 8:  57.0% [284/498] | 30s | loss=0.1254


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  57.9s | train_loss=0.0365 | val_loss=0.0745 | train_auroc=0.8670 | val_auroc=0.8765 | train_auprc=0.1522 | val_auprc=0.1155
  [hb] Epoch 9:  56.4% [281/498] | 30s | loss=0.0536


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  58.1s | train_loss=0.0753 | val_loss=0.0756 | train_auroc=0.8716 | val_auroc=0.8745 | train_auprc=0.1622 | val_auprc=0.1186
  [hb] Epoch 10:  55.6% [277/498] | 30s | loss=0.0493


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  58.4s | train_loss=0.0582 | val_loss=0.0756 | train_auroc=0.8775 | val_auroc=0.8710 | train_auprc=0.1737 | val_auprc=0.1035
  [hb] Epoch 11:  55.6% [277/498] | 30s | loss=0.0603


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  58.5s | train_loss=0.0495 | val_loss=0.0797 | train_auroc=0.8864 | val_auroc=0.8706 | train_auprc=0.1936 | val_auprc=0.1174
  [hb] Epoch 12:  56.8% [283/498] | 30s | loss=0.0709


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  58.5s | train_loss=0.0380 | val_loss=0.0774 | train_auroc=0.8950 | val_auroc=0.8626 | train_auprc=0.2124 | val_auprc=0.1083
  [hb] Epoch 13:  56.8% [283/498] | 30s | loss=0.0674


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  58.0s | train_loss=0.0420 | val_loss=0.0783 | train_auroc=0.9006 | val_auroc=0.8617 | train_auprc=0.2320 | val_auprc=0.0962
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/02_TLS_seed42/checkpoints/epoch=07-val_loss=0.0738.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/02_TLS_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/02_TLS_seed42/checkpoints/epoch=07-val_loss=0.0738.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/02_TLS_seed42/checkpoints/epoch=07-val_loss=0.0738.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0000 | test_loss=0.0780 | test_monotonicity=0.0000 | test_state=0.0780 | test_velocity=0.0001
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      4.423856807989068e-05      │
│            test_asf             │      0.029563888907432556       │
│        test_asf_ci_high         │      0.030989667400717735       │
│         test_asf_ci_low         │      0.028051747009158134       │
│           test_auprc            │       0.13584184646606445       │
│       test_auprc_ci_high        │       0.16958539187908173       │
│        test_auprc_ci_low        │       0.10761763900518417       │
│           test_auroc            │       0.8384032249450684        │
│       test_auroc_ci_high        │       0.8610809445381165        │
│        test_auroc_ci_low        │       0.8142303824424744        │
│        test_brier_score         │      0.014136254787445068       │
│    test_brier_score_ci_high     │       0.01572478748857975       │
│     test_brier_score_ci_low     │      0.012550079263746738       │
│            test_ece             │      0.0040525575168430805      │
│        test_ece_ci_high         │      0.006180701777338982       │
│         test_ece_ci_low         │       0.0022631143219769        │
│             test_f1             │       0.08902136981487274       │
│         test_f1_ci_high         │       0.09893092513084412       │
│         test_f1_ci_low          │       0.07994604855775833       │
│          test_fa_rate           │       0.01817162148654461       │
│      test_fa_rate_ci_high       │       0.01895156316459179       │
│       test_fa_rate_ci_low       │      0.017294738441705704       │
│           test_fappd            │       0.4361189305782318        │
│       test_fappd_ci_high        │       0.45483753085136414       │
│        test_fappd_ci_low        │       0.4150736927986145        │
│     test_lead_time_iqr_high     │              81.5               │
│ test_lead_time_iqr_high_ci_high │        103.0374984741211        │
│ test_lead_time_iqr_high_ci_low  │              72.25              │
│     test_lead_time_iqr_low      │              15.0               │
│ test_lead_time_iqr_low_ci_high  │              21.0               │
│  test_lead_time_iqr_low_ci_low  │              10.75              │
│            test_loss            │       0.07799698412418365       │
│       test_mean_lead_time       │        62.45283126831055        │
│   test_mean_lead_time_ci_high   │        71.34910583496094        │
│   test_mean_lead_time_ci_low    │       54.217037200927734        │
│      test_median_lead_time      │              45.0               │
│  test_median_lead_time_ci_high  │              56.0               │
│  test_median_lead_time_ci_low   │              35.0               │
│        test_monotonicity        │               0.0               │
│            test_npv             │       0.9951989650726318        │
│        test_npv_ci_high         │       0.9962643384933472        │
│         test_npv_ci_low         │       0.9940503835678101        │
│     test_physionet_utility      │       0.32507240772247314       │
│ test_physionet_utility_ci_high  │       0.3862494230270386        │
│  test_physionet_utility_ci_low  │       0.26307305693626404       │
│            test_poms            │       0.5344352722167969        │
│        test_poms_ci_high        │       0.5692657828330994        │
│        test_poms_ci_low         │       0.49998319149017334       │
│            test_ppv             │       0.04728075861930847       │
│        test_ppv_ci_high         │       0.0528285875916481        │
│         test_ppv_ci_low         │       0.04226778447628021       │
│            test_rtv             │       0.09984767436981201       │
│        test_rtv_ci_high         │       0.10857617110013962       │
│         test_rtv_ci_low         │       0.09145927429199219       │
│     test_selected_threshold     │      0.015075377188622952       │
│        test_sensitivity         │       0.7597264051437378        │
│    test_sensitivity_ci_high     │       0.8039483428001404        │
│     test_sensitivity_ci_low     │        0.711580753326416        │
│        test_specificity         │       0.7648962736129761        │
│    test_specificity_ci_high     │       0.7801684141159058        │
│     test_specificity_ci_low     │       0.7496464848518372        │
│            test_spj             │      0.0018364106072112918      │
│        test_spj_ci_high         │      0.001909721060656011       │
│         test_spj_ci_low         │      0.0017572678625583649      │
│           test_state            │       0.07798431813716888       │
│        test_target_auprc        │       0.16520465910434723       │
│        test_target_auroc        │       0.8184373378753662        │
│            test_tce             │       0.8701581954956055        │
│        test_tce_ci_high         │        0.883350670337677        │
│         test_tce_ci_low         │       0.8568339943885803        │
│          test_velocity          │      8.264673670055345e-05      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [3/6] 03_BCE_Smoothness_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/03_BCE_Smoothness_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name                 ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone             │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn              │ SmoothnessLoss   │      0 │ train │     0 │
│ 2 │ train_metrics        │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics          │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics         │ MetricCollection │      0 │ train │     0 │
│ 5 │ train_target_metrics │ MetricCollection │      0 │ train │     0 │
│ 6 │ val_target_metrics   │ MetricCollection │      0 │ train │     0 │
│ 7 │ test_target_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴──────────────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 27                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  52.4% [261/498] | 30s | loss=0.0500


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0321 | val_loss=0.0983
  [hb] Epoch 1:  56.4% [281/498] | 30s | loss=0.1006


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  60.9s | train_loss=0.0433 | val_loss=0.0953 | train_auroc=0.7734 | val_auroc=0.8577 | train_auprc=0.0638 | val_auprc=0.1141
  [hb] Epoch 2:  57.4% [286/498] | 30s | loss=0.2002


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  58.4s | train_loss=0.0669 | val_loss=0.0931 | train_auroc=0.8142 | val_auroc=0.8568 | train_auprc=0.0914 | val_auprc=0.1173
  [hb] Epoch 3:  56.4% [281/498] | 30s | loss=0.1705


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  57.6s | train_loss=0.1051 | val_loss=0.0898 | train_auroc=0.8253 | val_auroc=0.8634 | train_auprc=0.1033 | val_auprc=0.1189
  [hb] Epoch 4:  56.2% [280/498] | 30s | loss=0.0903


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  58.4s | train_loss=0.0804 | val_loss=0.0906 | train_auroc=0.8351 | val_auroc=0.8676 | train_auprc=0.1155 | val_auprc=0.1164
  [hb] Epoch 5:  55.8% [278/498] | 30s | loss=0.1360


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  58.2s | train_loss=0.1193 | val_loss=0.0917 | train_auroc=0.8461 | val_auroc=0.8610 | train_auprc=0.1181 | val_auprc=0.1133
  [hb] Epoch 6:  56.0% [279/498] | 30s | loss=0.1201


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  59.3s | train_loss=0.0862 | val_loss=0.0892 | train_auroc=0.8517 | val_auroc=0.8697 | train_auprc=0.1292 | val_auprc=0.1110
  [hb] Epoch 7:  56.4% [281/498] | 30s | loss=0.1290


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  59.0s | train_loss=0.0999 | val_loss=0.0890 | train_auroc=0.8538 | val_auroc=0.8725 | train_auprc=0.1307 | val_auprc=0.1189
  [hb] Epoch 8:  56.6% [282/498] | 30s | loss=0.1473


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  58.6s | train_loss=0.0467 | val_loss=0.0891 | train_auroc=0.8631 | val_auroc=0.8713 | train_auprc=0.1454 | val_auprc=0.1112
  [hb] Epoch 9:  56.2% [280/498] | 30s | loss=0.1094


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  58.3s | train_loss=0.0890 | val_loss=0.0891 | train_auroc=0.8685 | val_auroc=0.8706 | train_auprc=0.1492 | val_auprc=0.1160
  [hb] Epoch 10:  56.6% [282/498] | 30s | loss=0.1017


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  58.6s | train_loss=0.0678 | val_loss=0.0897 | train_auroc=0.8743 | val_auroc=0.8718 | train_auprc=0.1581 | val_auprc=0.1127
  [hb] Epoch 11:  55.2% [275/498] | 30s | loss=0.0283


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  58.7s | train_loss=0.0590 | val_loss=0.0927 | train_auroc=0.8824 | val_auroc=0.8703 | train_auprc=0.1764 | val_auprc=0.1144
  [hb] Epoch 12:  58.0% [289/498] | 30s | loss=0.0691


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  57.3s | train_loss=0.0466 | val_loss=0.0922 | train_auroc=0.8905 | val_auroc=0.8625 | train_auprc=0.1961 | val_auprc=0.1032
  [hb] Epoch 13:  58.8% [293/498] | 30s | loss=0.0814


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  54.9s | train_loss=0.0540 | val_loss=0.0922 | train_auroc=0.8957 | val_auroc=0.8602 | train_auprc=0.2091 | val_auprc=0.0948
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=07-val_loss=0.0890.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/03_BCE_Smoothness_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=07-val_loss=0.0890.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/03_BCE_Smoothness_seed42/checkpoints/epoch=07-val_loss=0.0890.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0000 | test_loss=0.0789 | test_monotonicity=0.0000 | test_state=0.0788 | test_velocity=0.0001
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      4.882733628619462e-05      │
│            test_asf             │       0.02573934756219387       │
│        test_asf_ci_high         │      0.027102254331111908       │
│         test_asf_ci_low         │      0.024414923042058945       │
│           test_auprc            │       0.12607483565807343       │
│       test_auprc_ci_high        │       0.15619519352912903       │
│        test_auprc_ci_low        │       0.10165689140558243       │
│           test_auroc            │       0.8324635624885559        │
│       test_auroc_ci_high        │       0.8562632203102112        │
│        test_auroc_ci_low        │       0.8076746463775635        │
│        test_brier_score         │      0.014569924212992191       │
│    test_brier_score_ci_high     │       0.01621364615857601       │
│     test_brier_score_ci_low     │       0.01294913049787283       │
│            test_ece             │      0.009642940014600754       │
│        test_ece_ci_high         │      0.012057020328938961       │
│         test_ece_ci_low         │      0.0073614418506622314      │
│             test_f1             │       0.08865310996770859       │
│         test_f1_ci_high         │       0.0988122820854187        │
│         test_f1_ci_low          │       0.07960879802703857       │
│          test_fa_rate           │      0.016321685165166855       │
│      test_fa_rate_ci_high       │       0.01708926446735859       │
│       test_fa_rate_ci_low       │      0.015499998815357685       │
│           test_fappd            │       0.3917204439640045        │
│       test_fappd_ci_high        │       0.41014236211776733       │
│        test_fappd_ci_low        │       0.37199997901916504       │
│     test_lead_time_iqr_high     │              85.0               │
│ test_lead_time_iqr_high_ci_high │        104.7562484741211        │
│ test_lead_time_iqr_high_ci_low  │              74.0               │
│     test_lead_time_iqr_low      │              16.0               │
│ test_lead_time_iqr_low_ci_high  │              22.0               │
│  test_lead_time_iqr_low_ci_low  │              12.0               │
│            test_loss            │       0.07885534316301346       │
│       test_mean_lead_time       │        64.1268310546875         │
│   test_mean_lead_time_ci_high   │        72.5699691772461         │
│   test_mean_lead_time_ci_low    │        55.61144256591797        │
│      test_median_lead_time      │              45.0               │
│  test_median_lead_time_ci_high  │              57.0               │
│  test_median_lead_time_ci_low   │              36.0               │
│        test_monotonicity        │               0.0               │
│            test_npv             │       0.9950856566429138        │
│        test_npv_ci_high         │       0.9961819052696228        │
│         test_npv_ci_low         │       0.9938355088233948        │
│     test_physionet_utility      │       0.3193415403366089        │
│ test_physionet_utility_ci_high  │       0.38033658266067505       │
│  test_physionet_utility_ci_low  │       0.2552871108055115        │
│            test_poms            │        0.544765830039978        │
│        test_poms_ci_high        │       0.5784764885902405        │
│        test_poms_ci_low         │       0.5122122168540955        │
│            test_ppv             │       0.04709622263908386       │
│        test_ppv_ci_high         │       0.05282843858003616       │
│         test_ppv_ci_low         │       0.04210326075553894       │
│            test_rtv             │       0.10726431757211685       │
│        test_rtv_ci_high         │       0.11610577255487442       │
│         test_rtv_ci_low         │       0.0989856868982315        │
│     test_selected_threshold     │       0.02010050229728222       │
│        test_sensitivity         │       0.7537409067153931        │
│    test_sensitivity_ci_high     │       0.8013298511505127        │
│     test_sensitivity_ci_low     │        0.70326167345047         │
│        test_specificity         │        0.765789270401001        │
│    test_specificity_ci_high     │       0.7812042832374573        │
│     test_specificity_ci_low     │       0.7502220273017883        │
│            test_spj             │      0.002038744278252125       │
│        test_spj_ci_high         │      0.0021230182610452175      │
│         test_spj_ci_low         │      0.0019528691191226244      │
│           test_state            │        0.078841932117939        │
│        test_target_auprc        │       0.16081102192401886       │
│        test_target_auroc        │        0.818419337272644        │
│            test_tce             │        0.849729597568512        │
│        test_tce_ci_high         │       0.8645839095115662        │
│         test_tce_ci_low         │       0.8349003791809082        │
│          test_velocity          │      8.525946032023057e-05      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [4/6] 04_BCE_TotalVariation_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/04_BCE_TotalVariation_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name                 ┃ Type               ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone             │ GRUPredictor       │  188 K │ train │     0 │
│ 1 │ loss_fn              │ TotalVariationLoss │      0 │ train │     0 │
│ 2 │ train_metrics        │ MetricCollection   │      0 │ train │     0 │
│ 3 │ val_metrics          │ MetricCollection   │      0 │ train │     0 │
│ 4 │ test_metrics         │ MetricCollection   │      0 │ train │     0 │
│ 5 │ train_target_metrics │ MetricCollection   │      0 │ train │     0 │
│ 6 │ val_target_metrics   │ MetricCollection   │      0 │ train │     0 │
│ 7 │ test_target_metrics  │ MetricCollection   │      0 │ train │     0 │
└───┴──────────────────────┴────────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 27                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  62.2% [310/498] | 30s | loss=0.0822


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0326 | val_loss=0.0986
  [hb] Epoch 1:  61.6% [307/498] | 30s | loss=0.1010


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  52.0s | train_loss=0.0436 | val_loss=0.0955 | train_auroc=0.7738 | val_auroc=0.8577 | train_auprc=0.0642 | val_auprc=0.1140
  [hb] Epoch 2:  64.1% [319/498] | 30s | loss=0.1486


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  52.0s | train_loss=0.0676 | val_loss=0.0934 | train_auroc=0.8147 | val_auroc=0.8567 | train_auprc=0.0915 | val_auprc=0.1164
  [hb] Epoch 3:  66.1% [329/498] | 30s | loss=0.0677


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  52.0s | train_loss=0.1058 | val_loss=0.0901 | train_auroc=0.8262 | val_auroc=0.8632 | train_auprc=0.1043 | val_auprc=0.1191
  [hb] Epoch 4:  64.3% [320/498] | 30s | loss=0.1046


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  51.3s | train_loss=0.0837 | val_loss=0.0910 | train_auroc=0.8353 | val_auroc=0.8676 | train_auprc=0.1158 | val_auprc=0.1149
  [hb] Epoch 5:  64.7% [322/498] | 30s | loss=0.0361


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  51.4s | train_loss=0.1216 | val_loss=0.0920 | train_auroc=0.8465 | val_auroc=0.8605 | train_auprc=0.1174 | val_auprc=0.1143
  [hb] Epoch 6:  62.7% [312/498] | 30s | loss=0.0524


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  50.3s | train_loss=0.0864 | val_loss=0.0893 | train_auroc=0.8520 | val_auroc=0.8694 | train_auprc=0.1284 | val_auprc=0.1117
  [hb] Epoch 7:  63.7% [317/498] | 30s | loss=0.1074


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  52.4s | train_loss=0.1015 | val_loss=0.0888 | train_auroc=0.8539 | val_auroc=0.8726 | train_auprc=0.1298 | val_auprc=0.1173
  [hb] Epoch 8:  63.7% [317/498] | 30s | loss=0.1255


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  52.4s | train_loss=0.0469 | val_loss=0.0895 | train_auroc=0.8633 | val_auroc=0.8719 | train_auprc=0.1435 | val_auprc=0.1164
  [hb] Epoch 9:  63.3% [315/498] | 30s | loss=0.1235


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  52.4s | train_loss=0.0901 | val_loss=0.0897 | train_auroc=0.8685 | val_auroc=0.8693 | train_auprc=0.1479 | val_auprc=0.1132
  [hb] Epoch 10:  63.9% [318/498] | 30s | loss=0.0737


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  52.3s | train_loss=0.0677 | val_loss=0.0902 | train_auroc=0.8746 | val_auroc=0.8702 | train_auprc=0.1577 | val_auprc=0.1090
  [hb] Epoch 11:  62.9% [313/498] | 30s | loss=0.0710


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  51.3s | train_loss=0.0568 | val_loss=0.0930 | train_auroc=0.8830 | val_auroc=0.8685 | train_auprc=0.1751 | val_auprc=0.1168
  [hb] Epoch 12:  63.5% [316/498] | 30s | loss=0.1085


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  51.8s | train_loss=0.0496 | val_loss=0.0937 | train_auroc=0.8898 | val_auroc=0.8617 | train_auprc=0.1963 | val_auprc=0.1050
  [hb] Epoch 13:  64.5% [321/498] | 30s | loss=0.0903


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  52.3s | train_loss=0.0547 | val_loss=0.0938 | train_auroc=0.8949 | val_auroc=0.8559 | train_auprc=0.2037 | val_auprc=0.0951
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=07-val_loss=0.0888.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/04_BCE_TotalVariation_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=07-val_loss=0.0888.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/04_BCE_TotalVariation_seed42/checkpoints/epoch=07-val_loss=0.0888.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0000 | test_loss=0.0789 | test_monotonicity=0.0000 | test_state=0.0789 | test_velocity=0.0001
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      3.708951408043504e-05      │
│            test_asf             │       0.02810933254659176       │
│        test_asf_ci_high         │      0.029579652473330498       │
│         test_asf_ci_low         │       0.02678225375711918       │
│           test_auprc            │       0.12347789853811264       │
│       test_auprc_ci_high        │       0.15399296581745148       │
│        test_auprc_ci_low        │       0.10007026046514511       │
│           test_auroc            │       0.8328994512557983        │
│       test_auroc_ci_high        │       0.8564261198043823        │
│        test_auroc_ci_low        │        0.808285117149353        │
│        test_brier_score         │       0.01459193043410778       │
│    test_brier_score_ci_high     │      0.016203293576836586       │
│     test_brier_score_ci_low     │      0.012947799637913704       │
│            test_ece             │      0.008964091539382935       │
│        test_ece_ci_high         │      0.011418445967137814       │
│         test_ece_ci_low         │      0.006625037174671888       │
│             test_f1             │       0.08243411034345627       │
│         test_f1_ci_high         │       0.09177707135677338       │
│         test_f1_ci_low          │       0.07400049269199371       │
│          test_fa_rate           │       0.01809879019856453       │
│      test_fa_rate_ci_high       │      0.018884964287281036       │
│       test_fa_rate_ci_low       │      0.017278488725423813       │
│           test_fappd            │       0.4343709349632263        │
│       test_fappd_ci_high        │       0.45323917269706726       │
│        test_fappd_ci_low        │       0.4146837592124939        │
│     test_lead_time_iqr_high     │              84.0               │
│ test_lead_time_iqr_high_ci_high │             105.75              │
│ test_lead_time_iqr_high_ci_low  │              75.5               │
│     test_lead_time_iqr_low      │              17.5               │
│ test_lead_time_iqr_low_ci_high  │       22.512500762939453        │
│  test_lead_time_iqr_low_ci_low  │              13.0               │
│            test_loss            │       0.07891161739826202       │
│       test_mean_lead_time       │        64.49762725830078        │
│   test_mean_lead_time_ci_high   │        73.24073028564453        │
│   test_mean_lead_time_ci_low    │       56.223670959472656        │
│      test_median_lead_time      │              45.0               │
│  test_median_lead_time_ci_high  │        55.01250076293945        │
│  test_median_lead_time_ci_low   │        36.48749923706055        │
│        test_monotonicity        │               0.0               │
│            test_npv             │       0.9954156875610352        │
│        test_npv_ci_high         │       0.9964389204978943        │
│         test_npv_ci_low         │       0.9942228198051453        │
│     test_physionet_utility      │       0.29012158513069153       │
│ test_physionet_utility_ci_high  │       0.35391128063201904       │
│  test_physionet_utility_ci_low  │       0.22456727921962738       │
│            test_poms            │       0.5619834661483765        │
│        test_poms_ci_high        │       0.5989434719085693        │
│        test_poms_ci_low         │       0.5269498229026794        │
│            test_ppv             │       0.04351980239152908       │
│        test_ppv_ci_high         │       0.0486970990896225        │
│         test_ppv_ci_low         │       0.03883913531899452       │
│            test_rtv             │       0.09563006460666656       │
│        test_rtv_ci_high         │       0.10373770445585251       │
│         test_rtv_ci_low         │       0.08803430199623108       │
│     test_selected_threshold     │      0.015075377188622952       │
│        test_sensitivity         │       0.7789653539657593        │
│    test_sensitivity_ci_high     │        0.824604868888855        │
│     test_sensitivity_ci_low     │       0.7306579947471619        │
│        test_specificity         │       0.7370767593383789        │
│    test_specificity_ci_high     │       0.7527483105659485        │
│     test_specificity_ci_low     │        0.721444308757782        │
│            test_spj             │      0.0017937248339876533      │
│        test_spj_ci_high         │      0.0018691568402573466      │
│         test_spj_ci_low         │      0.0017158426344394684      │
│           test_state            │       0.07889989018440247       │
│        test_target_auprc        │       0.15828029811382294       │
│        test_target_auroc        │       0.8191714286804199        │
│            test_tce             │       0.8497945666313171        │
│        test_tce_ci_high         │       0.8649527430534363        │
│         test_tce_ci_low         │       0.8347396850585938        │
│          test_velocity          │      8.022735710255802e-05      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [5/6] 05_BGSL_StateOnly_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/05_BGSL_StateOnly_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name                 ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone             │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn              │ BGSLLoss         │      0 │ train │     0 │
│ 2 │ train_metrics        │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics          │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics         │ MetricCollection │      0 │ train │     0 │
│ 5 │ train_target_metrics │ MetricCollection │      0 │ train │     0 │
│ 6 │ val_target_metrics   │ MetricCollection │      0 │ train │     0 │
│ 7 │ test_target_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴──────────────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 27                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  56.4% [281/498] | 30s | loss=0.1531


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0255 | val_loss=0.0814
  [hb] Epoch 1:  60.6% [302/498] | 30s | loss=0.0315


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  55.9s | train_loss=0.0355 | val_loss=0.0790 | train_auroc=0.7721 | val_auroc=0.8589 | train_auprc=0.0625 | val_auprc=0.1145
  [hb] Epoch 2:  60.6% [302/498] | 30s | loss=0.1639


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  53.6s | train_loss=0.0554 | val_loss=0.0773 | train_auroc=0.8162 | val_auroc=0.8597 | train_auprc=0.0935 | val_auprc=0.1201
  [hb] Epoch 3:  60.0% [299/498] | 30s | loss=0.0805


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  53.2s | train_loss=0.0922 | val_loss=0.0743 | train_auroc=0.8282 | val_auroc=0.8651 | train_auprc=0.1080 | val_auprc=0.1197
  [hb] Epoch 4:  61.8% [308/498] | 30s | loss=0.0680


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  53.1s | train_loss=0.0633 | val_loss=0.0747 | train_auroc=0.8377 | val_auroc=0.8701 | train_auprc=0.1206 | val_auprc=0.1198
  [hb] Epoch 5:  62.9% [313/498] | 30s | loss=0.0877


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  52.5s | train_loss=0.0953 | val_loss=0.0754 | train_auroc=0.8493 | val_auroc=0.8674 | train_auprc=0.1266 | val_auprc=0.1151
  [hb] Epoch 6:  63.5% [316/498] | 30s | loss=0.0753


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  52.7s | train_loss=0.0684 | val_loss=0.0745 | train_auroc=0.8543 | val_auroc=0.8722 | train_auprc=0.1349 | val_auprc=0.1131
  [hb] Epoch 7:  62.9% [313/498] | 30s | loss=0.0609


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  52.7s | train_loss=0.0818 | val_loss=0.0737 | train_auroc=0.8565 | val_auroc=0.8738 | train_auprc=0.1388 | val_auprc=0.1163
  [hb] Epoch 8:  63.9% [318/498] | 30s | loss=0.0387


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  52.7s | train_loss=0.0373 | val_loss=0.0740 | train_auroc=0.8659 | val_auroc=0.8754 | train_auprc=0.1524 | val_auprc=0.1163
  [hb] Epoch 9:  63.3% [315/498] | 30s | loss=0.1002


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  52.4s | train_loss=0.0762 | val_loss=0.0754 | train_auroc=0.8707 | val_auroc=0.8736 | train_auprc=0.1627 | val_auprc=0.1187
  [hb] Epoch 10:  61.6% [307/498] | 30s | loss=0.0358


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  52.7s | train_loss=0.0576 | val_loss=0.0744 | train_auroc=0.8744 | val_auroc=0.8685 | train_auprc=0.1580 | val_auprc=0.1099
  [hb] Epoch 11:  60.0% [299/498] | 30s | loss=0.0849


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  54.6s | train_loss=0.0492 | val_loss=0.0780 | train_auroc=0.8818 | val_auroc=0.8742 | train_auprc=0.1819 | val_auprc=0.1195
  [hb] Epoch 12:  61.8% [308/498] | 30s | loss=0.0667


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  54.4s | train_loss=0.0405 | val_loss=0.0770 | train_auroc=0.8908 | val_auroc=0.8646 | train_auprc=0.1989 | val_auprc=0.1032
  [hb] Epoch 13:  62.9% [313/498] | 30s | loss=0.0771


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  53.1s | train_loss=0.0454 | val_loss=0.0760 | train_auroc=0.8968 | val_auroc=0.8649 | train_auprc=0.2223 | val_auprc=0.0968
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=07-val_loss=0.0737.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/05_BGSL_StateOnly_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=07-val_loss=0.0737.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/05_BGSL_StateOnly_seed42/checkpoints/epoch=07-val_loss=0.0737.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0000 | test_loss=0.0780 | test_monotonicity=0.0000 | test_state=0.0780 | test_velocity=0.0001
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │     4.4250686187297106e-05      │
│            test_asf             │       0.03755508363246918       │
│        test_asf_ci_high         │       0.03918440267443657       │
│         test_asf_ci_low         │       0.03590932860970497       │
│           test_auprc            │       0.13438110053539276       │
│       test_auprc_ci_high        │       0.16702419519424438       │
│        test_auprc_ci_low        │       0.10768326371908188       │
│           test_auroc            │       0.8380885124206543        │
│       test_auroc_ci_high        │       0.8611435294151306        │
│        test_auroc_ci_low        │        0.813226044178009        │
│        test_brier_score         │      0.014146005734801292       │
│    test_brier_score_ci_high     │      0.015743572264909744       │
│     test_brier_score_ci_low     │      0.012574609369039536       │
│            test_ece             │      0.004111734684556723       │
│        test_ece_ci_high         │      0.006258551497012377       │
│         test_ece_ci_low         │      0.002316484460607171       │
│             test_f1             │       0.07674084603786469       │
│         test_f1_ci_high         │       0.08505412191152573       │
│         test_f1_ci_low          │       0.06874116510152817       │
│          test_fa_rate           │      0.022891145199537277       │
│      test_fa_rate_ci_high       │       0.02371419034898281       │
│       test_fa_rate_ci_low       │      0.021953821182250977       │
│           test_fappd            │       0.5493874549865723        │
│       test_fappd_ci_high        │       0.5691405534744263        │
│        test_fappd_ci_low        │       0.5268917083740234        │
│     test_lead_time_iqr_high     │              85.0               │
│ test_lead_time_iqr_high_ci_high │              106.0              │
│ test_lead_time_iqr_high_ci_low  │              77.0               │
│     test_lead_time_iqr_low      │              16.0               │
│ test_lead_time_iqr_low_ci_high  │              22.0               │
│  test_lead_time_iqr_low_ci_low  │       11.243749618530273        │
│            test_loss            │       0.07801910489797592       │
│       test_mean_lead_time       │        63.72398376464844        │
│   test_mean_lead_time_ci_high   │        72.06462860107422        │
│   test_mean_lead_time_ci_low    │        55.89776611328125        │
│      test_median_lead_time      │              45.0               │
│  test_median_lead_time_ci_high  │              55.0               │
│  test_median_lead_time_ci_low   │              36.0               │
│        test_monotonicity        │               0.0               │
│            test_npv             │       0.9958733916282654        │
│        test_npv_ci_high         │       0.9968676567077637        │
│         test_npv_ci_low         │       0.9947417378425598        │
│     test_physionet_utility      │       0.25623634457588196       │
│ test_physionet_utility_ci_high  │       0.32218268513679504       │
│  test_physionet_utility_ci_low  │       0.18161813914775848       │
│            test_poms            │        0.53925621509552         │
│        test_poms_ci_high        │       0.5707767009735107        │
│        test_poms_ci_low         │       0.5084008574485779        │
│            test_ppv             │       0.04027801379561424       │
│        test_ppv_ci_high         │       0.04484429210424423       │
│         test_ppv_ci_low         │       0.03593215346336365       │
│            test_rtv             │       0.09911303222179413       │
│        test_rtv_ci_high         │       0.10790806263685226       │
│         test_rtv_ci_low         │       0.09065794944763184       │
│     test_selected_threshold     │       0.01005025114864111       │
│        test_sensitivity         │       0.8101752996444702        │
│    test_sensitivity_ci_high     │       0.8534704446792603        │
│     test_sensitivity_ci_low     │       0.7662215828895569        │
│        test_specificity         │        0.703531801700592        │
│    test_specificity_ci_high     │       0.7190847396850586        │
│     test_specificity_ci_low     │       0.6883581280708313        │
│            test_spj             │      0.001809611334465444       │
│        test_spj_ci_high         │      0.0018852852517738938      │
│         test_spj_ci_low         │      0.001730405492708087       │
│           test_state            │       0.07800642400979996       │
│        test_target_auprc        │       0.16407713294029236       │
│        test_target_auroc        │       0.8188542127609253        │
│            test_tce             │       0.8689616918563843        │
│        test_tce_ci_high         │       0.8824642300605774        │
│         test_tce_ci_low         │       0.8553473353385925        │
│          test_velocity          │      8.255628199549392e-05      │
└─────────────────────────────────┴─────────────────────────────────┘

=== [6/6] 06_Full_BGSL_seed42 ===
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/06_Full_BGSL_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name                 ┃ Type             ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ backbone             │ GRUPredictor     │  188 K │ train │     0 │
│ 1 │ loss_fn              │ BGSLLoss         │      0 │ train │     0 │
│ 2 │ train_metrics        │ MetricCollection │      0 │ train │     0 │
│ 3 │ val_metrics          │ MetricCollection │      0 │ train │     0 │
│ 4 │ test_metrics         │ MetricCollection │      0 │ train │     0 │
│ 5 │ train_target_metrics │ MetricCollection │      0 │ train │     0 │
│ 6 │ val_target_metrics   │ MetricCollection │      0 │ train │     0 │
│ 7 │ test_target_metrics  │ MetricCollection │      0 │ train │     0 │
└───┴──────────────────────┴──────────────────┴────────┴───────┴───────┘
Trainable params: 188 K                                                         
Non-trainable params: 0                                                         
Total params: 188 K                                                             
Total estimated model params size (MB): 0.753                                   
Modules in train mode: 27                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  

Sanity Checking: |          | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Sanity check passed.

Training: |          | 0/? [00:00<?, ?it/s]  [hb] Epoch 0:  61.2% [305/498] | 30s | loss=0.0482


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   0 |   0.0s | train_loss=0.0255 | val_loss=0.0814
  [hb] Epoch 1:  63.9% [318/498] | 30s | loss=0.1569


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   1 |  52.8s | train_loss=0.0356 | val_loss=0.0790 | train_auroc=0.7723 | val_auroc=0.8590 | train_auprc=0.0626 | val_auprc=0.1145
  [hb] Epoch 2:  64.3% [320/498] | 30s | loss=0.0938


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   2 |  52.0s | train_loss=0.0554 | val_loss=0.0773 | train_auroc=0.8163 | val_auroc=0.8597 | train_auprc=0.0936 | val_auprc=0.1198
  [hb] Epoch 3:  63.1% [314/498] | 30s | loss=0.0286


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   3 |  50.4s | train_loss=0.0924 | val_loss=0.0743 | train_auroc=0.8283 | val_auroc=0.8650 | train_auprc=0.1084 | val_auprc=0.1196
  [hb] Epoch 4:  65.5% [326/498] | 30s | loss=0.1261


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   4 |  51.6s | train_loss=0.0631 | val_loss=0.0747 | train_auroc=0.8378 | val_auroc=0.8702 | train_auprc=0.1207 | val_auprc=0.1202
  [hb] Epoch 5:  63.9% [318/498] | 30s | loss=0.1598


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   5 |  51.4s | train_loss=0.0955 | val_loss=0.0755 | train_auroc=0.8494 | val_auroc=0.8680 | train_auprc=0.1268 | val_auprc=0.1150
  [hb] Epoch 6:  64.5% [321/498] | 30s | loss=0.1090


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   6 |  52.5s | train_loss=0.0684 | val_loss=0.0746 | train_auroc=0.8543 | val_auroc=0.8722 | train_auprc=0.1353 | val_auprc=0.1132
  [hb] Epoch 7:  64.5% [321/498] | 30s | loss=0.0200


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   7 |  51.8s | train_loss=0.0822 | val_loss=0.0737 | train_auroc=0.8567 | val_auroc=0.8732 | train_auprc=0.1393 | val_auprc=0.1159
  [hb] Epoch 8:  63.5% [316/498] | 30s | loss=0.0784


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   8 |  51.2s | train_loss=0.0373 | val_loss=0.0739 | train_auroc=0.8659 | val_auroc=0.8754 | train_auprc=0.1519 | val_auprc=0.1169
  [hb] Epoch 9:  62.2% [310/498] | 30s | loss=0.0799


Validation: |          | 0/? [00:00<?, ?it/s]Epoch   9 |  51.1s | train_loss=0.0786 | val_loss=0.0752 | train_auroc=0.8703 | val_auroc=0.8750 | train_auprc=0.1606 | val_auprc=0.1186
  [hb] Epoch 10:  63.7% [317/498] | 30s | loss=0.0652


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  10 |  52.7s | train_loss=0.0576 | val_loss=0.0749 | train_auroc=0.8761 | val_auroc=0.8703 | train_auprc=0.1690 | val_auprc=0.1044
  [hb] Epoch 11:  63.7% [317/498] | 30s | loss=0.0622


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  11 |  52.3s | train_loss=0.0481 | val_loss=0.0781 | train_auroc=0.8840 | val_auroc=0.8726 | train_auprc=0.1896 | val_auprc=0.1176
  [hb] Epoch 12:  64.3% [320/498] | 30s | loss=0.0638


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  12 |  52.3s | train_loss=0.0390 | val_loss=0.0770 | train_auroc=0.8919 | val_auroc=0.8675 | train_auprc=0.2100 | val_auprc=0.1026
  [hb] Epoch 13:  63.7% [317/498] | 30s | loss=0.0662


Validation: |          | 0/? [00:00<?, ?it/s]Epoch  13 |  52.3s | train_loss=0.0449 | val_loss=0.0771 | train_auroc=0.8987 | val_auroc=0.8642 | train_auprc=0.2289 | val_auprc=0.0973
  [TEST] Running test on /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/06_Full_BGSL_seed42/checkpoints/epoch=07-val_loss=0.0737.ckpt
Seed set to 42
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
/usr/local/lib/python3.12/dist-packages/lightning_fabric/loggers/csv_logs.py:268: Experiment logs directory /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/06_Full_BGSL_seed42/logs/ exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!
Restoring states from the checkpoint path at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/06_Full_BGSL_seed42/checkpoints/epoch=07-val_loss=0.0737.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/runs/06_Full_BGSL_seed42/checkpoints/epoch=07-val_loss=0.0737.ckpt
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()

Testing: |          | 0/? [00:00<?, ?it/s]Test | test_acceleration=0.0000 | test_loss=0.0780 | test_monotonicity=0.0000 | test_state=0.0780 | test_velocity=0.0001
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Test metric           ┃          DataLoader 0           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        test_acceleration        │      4.265449024387635e-05      │
│            test_asf             │      0.028686869889497757       │
│        test_asf_ci_high         │       0.03014093078672886       │
│         test_asf_ci_low         │       0.02720138244330883       │
│           test_auprc            │       0.13378065824508667       │
│       test_auprc_ci_high        │       0.16642220318317413       │
│        test_auprc_ci_low        │       0.10689272731542587       │
│           test_auroc            │       0.8381925821304321        │
│       test_auroc_ci_high        │       0.8611743450164795        │
│        test_auroc_ci_low        │       0.8132524490356445        │
│        test_brier_score         │      0.014143560081720352       │
│    test_brier_score_ci_high     │      0.015742206946015358       │
│     test_brier_score_ci_low     │       0.01256585493683815       │
│            test_ece             │      0.004060809034854174       │
│        test_ece_ci_high         │      0.006195131689310074       │
│         test_ece_ci_low         │      0.002286513103172183       │
│             test_f1             │       0.08987341821193695       │
│         test_f1_ci_high         │       0.0999142974615097        │
│         test_f1_ci_low          │       0.0806560069322586        │
│          test_fa_rate           │      0.017741911113262177       │
│      test_fa_rate_ci_high       │      0.018580079078674316       │
│       test_fa_rate_ci_low       │      0.016821028664708138       │
│           test_fappd            │       0.4258058965206146        │
│       test_fappd_ci_high        │       0.4459218978881836        │
│        test_fappd_ci_low        │       0.4037046730518341        │
│     test_lead_time_iqr_high     │              82.0               │
│ test_lead_time_iqr_high_ci_high │              104.5              │
│ test_lead_time_iqr_high_ci_low  │              72.0               │
│     test_lead_time_iqr_low      │              14.5               │
│ test_lead_time_iqr_low_ci_high  │              19.5               │
│  test_lead_time_iqr_low_ci_low  │              12.0               │
│            test_loss            │       0.07797367870807648       │
│       test_mean_lead_time       │        62.53080749511719        │
│   test_mean_lead_time_ci_high   │        71.6010971069336         │
│   test_mean_lead_time_ci_low    │        54.08906173706055        │
│      test_median_lead_time      │              44.0               │
│  test_median_lead_time_ci_high  │              54.5               │
│  test_median_lead_time_ci_low   │       34.974998474121094        │
│        test_monotonicity        │               0.0               │
│            test_npv             │       0.9951992034912109        │
│        test_npv_ci_high         │       0.9962686896324158        │
│         test_npv_ci_low         │        0.994067370891571        │
│     test_physionet_utility      │       0.32823359966278076       │
│ test_physionet_utility_ci_high  │       0.38919368386268616       │
│  test_physionet_utility_ci_low  │       0.2650963366031647        │
│            test_poms            │       0.5344352722167969        │
│        test_poms_ci_high        │       0.5652346014976501        │
│        test_poms_ci_low         │        0.501506507396698        │
│            test_ppv             │       0.04776512831449509       │
│        test_ppv_ci_high         │      0.053403522819280624       │
│         test_ppv_ci_low         │       0.04270116239786148       │
│            test_rtv             │       0.09830136597156525       │
│        test_rtv_ci_high         │       0.10704219341278076       │
│         test_rtv_ci_low         │       0.08991182595491409       │
│     test_selected_threshold     │      0.015075377188622952       │
│        test_sensitivity         │        0.75887131690979         │
│    test_sensitivity_ci_high     │       0.8055206537246704        │
│     test_sensitivity_ci_low     │        0.710267961025238        │
│        test_specificity         │       0.7676604986190796        │
│    test_specificity_ci_high     │       0.7830487489700317        │
│     test_specificity_ci_low     │       0.7522352337837219        │
│            test_spj             │      0.0017938814125955105      │
│        test_spj_ci_high         │      0.001868995139375329       │
│         test_spj_ci_low         │      0.001714373123832047       │
│           test_state            │       0.07796121388673782       │
│        test_target_auprc        │       0.16435018181800842       │
│        test_target_auroc        │       0.8190918564796448        │
│            test_tce             │       0.8684713840484619        │
│        test_tce_ci_high         │       0.8819119930267334        │
│         test_tce_ci_low         │       0.8546415567398071        │
│          test_velocity          │      8.184038597391918e-05      │
└─────────────────────────────────┴─────────────────────────────────┘

Summary written: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/summary.csv
Manifest: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/batch_manifest.json
Overall status: success

========================================================================
  POST-PROCESSING
========================================================================

========================================================================
  AGGREGATED RESULTS — 20260608-044438__0c2bc87d
  Conditions: 6 | Total runs: 6
  Baseline for t-tests: 01_BCE_Baseline
========================================================================

  --- 01_BCE_Baseline (n=1) ---
    test_auroc                                = 0.8284 ± 0.0000 (95% CI)
    test_auprc                                = 0.1301 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.3131 ± 0.0000 (95% CI)
    test_median_lead_time                     = 46.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 64.5049 ± 0.0000 (95% CI)
    test_asf                                  = 0.0294 ± 0.0000 (95% CI)
    test_rtv                                  = 0.1241 ± 0.0000 (95% CI)
    test_spj                                  = 0.0023 ± 0.0000 (95% CI)
    test_poms                                 = 0.5331 ± 0.0000 (95% CI)
    test_tce                                  = 0.8538 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0186 ± 0.0000 (95% CI)
    test_fappd                                = 0.4468 ± 0.0000 (95% CI)

  --- 02_TLS (n=1) ---
    test_auroc                                = 0.8384 ± 0.0000 (95% CI)
    test_auprc                                = 0.1358 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.3251 ± 0.0000 (95% CI)
    test_median_lead_time                     = 45.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 62.4528 ± 0.0000 (95% CI)
    test_asf                                  = 0.0296 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0998 ± 0.0000 (95% CI)
    test_spj                                  = 0.0018 ± 0.0000 (95% CI)
    test_poms                                 = 0.5344 ± 0.0000 (95% CI)
    test_tce                                  = 0.8702 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0182 ± 0.0000 (95% CI)
    test_fappd                                = 0.4361 ± 0.0000 (95% CI)

  --- 03_BCE_Smoothness (n=1) ---
    test_auroc                                = 0.8325 ± 0.0000 (95% CI)
    test_auprc                                = 0.1261 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.3193 ± 0.0000 (95% CI)
    test_median_lead_time                     = 45.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 64.1268 ± 0.0000 (95% CI)
    test_asf                                  = 0.0257 ± 0.0000 (95% CI)
    test_rtv                                  = 0.1073 ± 0.0000 (95% CI)
    test_spj                                  = 0.0020 ± 0.0000 (95% CI)
    test_poms                                 = 0.5448 ± 0.0000 (95% CI)
    test_tce                                  = 0.8497 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0163 ± 0.0000 (95% CI)
    test_fappd                                = 0.3917 ± 0.0000 (95% CI)

  --- 04_BCE_TotalVariation (n=1) ---
    test_auroc                                = 0.8329 ± 0.0000 (95% CI)
    test_auprc                                = 0.1235 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.2901 ± 0.0000 (95% CI)
    test_median_lead_time                     = 45.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 64.4976 ± 0.0000 (95% CI)
    test_asf                                  = 0.0281 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0956 ± 0.0000 (95% CI)
    test_spj                                  = 0.0018 ± 0.0000 (95% CI)
    test_poms                                 = 0.5620 ± 0.0000 (95% CI)
    test_tce                                  = 0.8498 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0181 ± 0.0000 (95% CI)
    test_fappd                                = 0.4344 ± 0.0000 (95% CI)

  --- 05_BGSL_StateOnly (n=1) ---
    test_auroc                                = 0.8381 ± 0.0000 (95% CI)
    test_auprc                                = 0.1344 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.2562 ± 0.0000 (95% CI)
    test_median_lead_time                     = 45.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 63.7240 ± 0.0000 (95% CI)
    test_asf                                  = 0.0376 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0991 ± 0.0000 (95% CI)
    test_spj                                  = 0.0018 ± 0.0000 (95% CI)
    test_poms                                 = 0.5393 ± 0.0000 (95% CI)
    test_tce                                  = 0.8690 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0229 ± 0.0000 (95% CI)
    test_fappd                                = 0.5494 ± 0.0000 (95% CI)

  --- 06_Full_BGSL (n=1) ---
    test_auroc                                = 0.8382 ± 0.0000 (95% CI)
    test_auprc                                = 0.1338 ± 0.0000 (95% CI)
    test_physionet_utility                    = 0.3282 ± 0.0000 (95% CI)
    test_median_lead_time                     = 44.0000 ± 0.0000 (95% CI)
    test_mean_lead_time                       = 62.5308 ± 0.0000 (95% CI)
    test_asf                                  = 0.0287 ± 0.0000 (95% CI)
    test_rtv                                  = 0.0983 ± 0.0000 (95% CI)
    test_spj                                  = 0.0018 ± 0.0000 (95% CI)
    test_poms                                 = 0.5344 ± 0.0000 (95% CI)
    test_tce                                  = 0.8685 ± 0.0000 (95% CI)
    test_fa_rate                              = 0.0177 ± 0.0000 (95% CI)
    test_fappd                                = 0.4258 ± 0.0000 (95% CI)

  Full results saved to: /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/aggregated_results.csv

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
01\_BCE\_Baseline & 0.130 $\pm$ 0.000 & 0.313 $\pm$ 0.000 & \textbf{46.000 $\pm$ 0.000} & 0.029 $\pm$ 0.000 & 0.533 $\pm$ 0.000 \\
02\_TLS & \textbf{0.136 $\pm$ 0.000} & 0.325 $\pm$ 0.000 & 45.000 $\pm$ 0.000 & 0.030 $\pm$ 0.000 & 0.534 $\pm$ 0.000 \\
03\_BCE\_Smoothness & 0.126 $\pm$ 0.000 & 0.319 $\pm$ 0.000 & 45.000 $\pm$ 0.000 & \textbf{0.026 $\pm$ 0.000} & 0.545 $\pm$ 0.000 \\
04\_BCE\_TotalVariation & 0.123 $\pm$ 0.000 & 0.290 $\pm$ 0.000 & 45.000 $\pm$ 0.000 & 0.028 $\pm$ 0.000 & \textbf{0.562 $\pm$ 0.000} \\
05\_BGSL\_StateOnly & 0.134 $\pm$ 0.000 & 0.256 $\pm$ 0.000 & 45.000 $\pm$ 0.000 & 0.038 $\pm$ 0.000 & 0.539 $\pm$ 0.000 \\
06\_Full\_BGSL & 0.134 $\pm$ 0.000 & \textbf{0.328 $\pm$ 0.000} & 44.000 $\pm$ 0.000 & 0.029 $\pm$ 0.000 & 0.534 $\pm$ 0.000 \\
\bottomrule
\end{tabular}
\end{table}


[01_BCE_Baseline] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots/01_BCE_Baseline
[02_TLS] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots/02_TLS
[03_BCE_Smoothness] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots/03_BCE_Smoothness
[04_BCE_TotalVariation] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots/04_BCE_TotalVariation
[05_BGSL_StateOnly] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots/05_BGSL_StateOnly
[06_Full_BGSL] seed=42: loading predictions...
  -> 6 plots saved to /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots/06_Full_BGSL

All plots saved under /content/bgsl/outputs/ablation/sepsis/core_hypothesis/20260608-044438__0c2bc87d/plots