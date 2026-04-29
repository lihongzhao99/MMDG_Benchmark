# ⚙️ HUST Motor Training

Run commands from this folder:

```bash
cd HUSTmotor
```

Training scripts read:

```text
data/Motor_Vib.mat
data/Motor_Aud.mat
```

## 📌 Methods

```text
train_HUST_EMR.py
train_HUST_RNA.py
train_HUST_SimMMDG.py
train_HUST_MOOSA.py
train_HUST_CMRF.py
train_HUST_NEL.py
train_HUST_JAT.py
train_HUST_MBCD.py
train_HUST_GMP.py
```

## ▶️ Single Run

```bash
python train_HUST_EMR.py -s D2 D3 D4 -t D1
```

Valid domains are `D1`, `D2`, `D3`, and `D4`.

## ⚙️ Batch Runner

`run_all_cross_domain.sh` runs one selected method across the HUST
cross-domain settings.

```bash
./run_all_cross_domain.sh --method GMP --setting all
```

Useful options:

```text
--method ERM|RNA|SimMMDG|MOOSA|CMRF|NEL|JAT|MBCD|GMP
--setting multi|single|all
--dry-run
```

Pass extra training options after `--`:

```bash
./run_all_cross_domain.sh -m MOOSA -s all -- --iteration 2000 --seed 1
```

Logs and checkpoints are saved under `outputs/logs/` and `outputs/models/`.
