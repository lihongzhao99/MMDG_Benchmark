# 🎬 Action Recognition Training

Run commands from this folder:

```bash
cd "Action recognition"
```

Dataset downloads for both EPIC-Kitchens and HAC:

[Hugging Face](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main)

## 📌 Methods

```text
train_ERM.py
train_RNA.py
train_SimMMDG.py
train_MOOSA.py
train_CMRF.py
train_NEL.py
train_JAT.py
train_MBCD.py
train_GMP.py
```

## ▶️ Single Run

```bash
python train_ERM.py \
  --dataset epic \
  --num_class 8 \
  -s D2 D3 \
  -t D1 \
  --use_video --use_audio \
  --datapath /path/to/DATA_ROOT
```

Use `--dataset hac --num_class 7` for HAC. Modalities are enabled with
`--use_video`, `--use_audio`, and `--use_flow`.

For EPIC, `--datapath` can point to either `DATA_ROOT/` or
`DATA_ROOT/MM-SADA_Domain_Adaptation_Splits/`. The EPIC layout is:
`video/{train,test}/D*/`, `flow/{train,test}/D*/`, and
`audio/{train,test}/D*/`.

## ⚙️ Batch Runner

`run_all_cross_domain.sh` runs one selected method across benchmark
cross-domain settings.

```bash
./run_all_cross_domain.sh --method MBCD --dataset epic --setting all --modality all --datapath /path/to/DATA_ROOT
```

Useful options:

```text
--method ERM|RNA|SimMMDG|MOOSA|CMRF|NEL|JAT|MBCD|GMP
--dataset epic|hac|all
--setting multi|single|all
--modality va|vf|af|vaf|all
--datapath /path/to/DATA_ROOT
--dry-run
```

Pass extra training options after `--`:

```bash
./run_all_cross_domain.sh -m JAT -d hac -s multi -M vaf -- --nepochs 10 --seed 1
```

Logs and checkpoints are saved under `outputs/logs/` and `outputs/models/`.
