# 💬 MMSA Training

Run commands from this folder:

```bash
cd MMSA
```

By default, the dataloader looks for:

```text
../data/mosi.pkl
../data/mosei.pkl
../data/sims.pkl
```

You can pass a dataset directory or a concrete `.pkl` file with `--datapath`.

## 📌 Methods

```text
train_MMSA_ERM.py
train_MMSA_RNA.py
train_MMSA_SimMMDG.py
train_MMSA_MOOSA.py
train_MMSA_CMRF.py
train_MMSA_NEL.py
train_MMSA_JAT.py
train_MMSA_MBCD.py
train_MMSA_GMP.py
```

## ▶️ Single Run

```bash
python train_MMSA_ERM.py \
  --source_datasets mosi mosei \
  --target_dataset sims \
  --datapath /path/to/mmsa_data
```

Valid dataset names are `mosi`, `mosei`, and `sims`.

## ⚙️ Batch Runner

`run_all_cross_domain.sh` runs one selected method across the MMSA benchmark
cross-domain settings.

```bash
./run_all_cross_domain.sh --method CMRF --setting all --datapath /path/to/mmsa_data
```

Useful options:

```text
--method ERM|RNA|SimMMDG|MOOSA|CMRF|NEL|JAT|MBCD|GMP
--setting multi|single|all
--datapath /path/to/mmsa_data
--dry-run
```

Pass extra training options after `--`:

```bash
./run_all_cross_domain.sh -m MBCD -s all --datapath ../data -- --num_epochs 5 --seed 1
```

Logs are saved under `outputs/logs/`. ERM logs are stored under `BASE`.
