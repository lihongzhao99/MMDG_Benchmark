# Evaluation Utilities

Run commands from the action-recognition folder:

```bash
cd "Action recognition"
```

These scripts evaluate HAC checkpoints on four robustness and reliability
settings:

- `test_untils/test_HAC_corruption.py`: corruption robustness
- `test_untils/test_HAC_missing.py`: missing-modality generalization
- `test_untils/test_HAC_misd.py`: misclassification detection
- `test_untils/test_HAC_ood.py`: OOD detection

## Requirements

Use the same environment, pretrained backbones, and HAC data layout described
in [../README_train.md](../README_train.md).

All scripts expect a trained HAC checkpoint. If `--resumef` is omitted, the
default checkpoint path is inferred as:

```text
models/log_ERM{source}2{target}_{modalities}_seed_{seed}_best.pt
```

## 1. Corruption Robustness

Evaluate robustness when one observed modality is corrupted at test time.

```bash
python test_untils/test_HAC_corruption.py \
  -s human animal \
  -t cartoon \
  --use_video --use_audio \
  --use_audio_corruption \
  --datapath /path/to/HAC_DATA_ROOT \
  --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt
```

Useful flags:

```text
--use_video_corruption
--use_audio_corruption
--use_video --use_audio --use_flow
--datapath /path/to/HAC_DATA_ROOT
--resumef /path/to/checkpoint.pt
```

The script reports classification loss and test accuracy on corrupted HAC test
samples.

## 2. Missing-Modality Generalization

Evaluate a model when one or more enabled modalities are missing at test time.
Missing modalities are replaced with zeros before fusion.

```bash
python test_untils/test_HAC_missing.py \
  -s human animal \
  -t cartoon \
  --use_video --use_audio \
  --zero_audio \
  --datapath /path/to/HAC_DATA_ROOT \
  --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt
```

Useful flags:

```text
--zero_video
--zero_audio
--zero_flow
--use_video --use_audio --use_flow
```

The script reports classification loss and test accuracy under the selected
missing-modality condition.

## 3. Misclassification Detection

Evaluate confidence-based error detection on HAC test samples. The script uses
maximum softmax probability and reports selective-classification metrics.

```bash
python test_untils/test_HAC_misd.py \
  -s human animal \
  -t cartoon \
  --use_video --use_audio \
  --datapath /path/to/HAC_DATA_ROOT \
  --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt
```

Reported metrics:

```text
TestAcc
AURC
AUROC
FPR95
```

The last summary line starts with `METRIC_MISD` for easy log parsing.

## 4. OOD Detection

Evaluate out-of-distribution detection with HAC as in-distribution data and
EPIC as out-of-distribution data. The score is the maximum softmax
probability.

```bash
python test_untils/test_HAC_ood.py \
  -s human animal \
  -t cartoon \
  --use_video --use_audio \
  --datapath /path/to/HAC_DATA_ROOT \
  --datapath_epic /path/to/EPIC_SPLIT_ROOT \
  --resumef models/log_ERM_human_animal2cartoon_video_audio_seed_0_best.pt
```

Reported metrics:

```text
TestAcc
FPR95
AUROC
```

The last summary line starts with `METRIC_OOD` for easy log parsing.

## Notes

- Enable at least one modality with `--use_video`, `--use_audio`, or
  `--use_flow`.
- `test_HAC_misd.py` and `test_HAC_ood.py` normalize user-provided HAC paths
  before loading data.
- `test_HAC_ood.py` also requires an EPIC path through `--datapath_epic`.
  It accepts either `DATA_ROOT/` or `DATA_ROOT/MM-SADA_Domain_Adaptation_Splits/`.
