import csv
import json
import os
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.environ.get(
    "MMSA_LOG_ROOT",
    os.path.join(PROJECT_DIR, "outputs", "logs"),
)
DATASET_NAME = "MMSA"

METHOD_LOG_ROOTS = {
    "gmp": os.path.join(LOG_ROOT, "GMP"),
    "mbcd": os.path.join(LOG_ROOT, "MBCD"),
    "nel": os.path.join(LOG_ROOT, "NEL"),
    "cmrf": os.path.join(LOG_ROOT, "CMRF"),
    "moosa": os.path.join(LOG_ROOT, "MOOSA"),
    "rna": os.path.join(LOG_ROOT, "RNA"),
    "simmmdg": os.path.join(LOG_ROOT, "SimMMDG"),
    "mdja": os.path.join(LOG_ROOT, "JAT"),
    "jat": os.path.join(LOG_ROOT, "JAT"),
    "main": os.path.join(LOG_ROOT, "BASE"),
    "base": os.path.join(LOG_ROOT, "BASE"),
}


def _normalize_domains(domains):
    if domains is None:
        return []
    if isinstance(domains, (list, tuple)):
        return [str(x).lower().strip() for x in domains if str(x).strip()]
    return [str(domains).lower().strip()]


def _infer_dg_mode(source_domains):
    return "single_source_dg" if len(source_domains) == 1 else "multi_source_dg"


def _get_source_domains(args):
    source_domains = _normalize_domains(getattr(args, "source_datasets", None))
    if len(source_domains) == 0:
        fallback = getattr(args, "dataset", None)
        source_domains = _normalize_domains(fallback)
    return source_domains


def _get_target_domains(args):
    target_dataset = getattr(args, "target_dataset", None)
    target_domains = _normalize_domains(target_dataset)
    if len(target_domains) == 0:
        fallback = getattr(args, "dataset", None)
        target_domains = _normalize_domains(fallback)
    return target_domains


def _sanitize_for_json(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return str(obj)


def _get_method_log_root(method_name):
    key = str(method_name).lower().strip()
    if key not in METHOD_LOG_ROOTS:
        key = "base"
    return METHOD_LOG_ROOTS[key]


def build_run_log_path(args, method_name):
    source_domains = _get_source_domains(args)
    target_domains = _get_target_domains(args)
    dg_mode = _infer_dg_mode(source_domains)
    method_root = _get_method_log_root(method_name)
    log_dir = os.path.join(method_root, dg_mode)
    os.makedirs(log_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_tag = "-".join(source_domains)
    target_tag = "-".join(target_domains)
    filename = (
        f"{str(method_name).upper()}_{DATASET_NAME}_"
        f"{source_tag}_to_{target_tag}_t_a_v_seed{getattr(args, 'seed', '')}_{run_id}.csv"
    )
    return os.path.join(log_dir, filename), dg_mode, source_domains, target_domains


def append_run_log(args, method_name, run_result, run_started_at=None):
    log_path, dg_mode, source_domains, target_domains = build_run_log_path(args, method_name)

    if run_started_at is None:
        run_started_at = datetime.now().isoformat()

    payload = {
        "run_start": run_started_at,
        "run_end": datetime.now().isoformat(),
        "method": method_name,
        "dg_mode": dg_mode,
        "source_domains": source_domains,
        "target_domains": target_domains,
        "hyperparameters": _sanitize_for_json(vars(args)),
        "results": _sanitize_for_json(run_result),
    }

    # 仅记录随机超参数（由自动化脚本传入）+ seed，避免输出大量默认参数。
    random_hparams = {}
    random_hparams_raw = getattr(args, "random_hparams_json", "")
    if random_hparams_raw:
        try:
            loaded = json.loads(random_hparams_raw)
            if isinstance(loaded, dict):
                random_hparams = loaded
        except Exception:
            random_hparams = {}
    random_hparams["seed"] = getattr(args, "seed", "")

    results = payload["results"] if isinstance(payload["results"], dict) else {}
    best_val = results.get("best_val", {}) if isinstance(results.get("best_val", {}), dict) else {}
    best_test = results.get("best_test", {}) if isinstance(results.get("best_test", {}), dict) else {}

    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow([])
        writer.writerow(["run_start", payload["run_start"]])
        writer.writerow(["run_end", payload["run_end"]])
        writer.writerow(["method", payload["method"]])
        writer.writerow(["dg_mode", payload["dg_mode"]])
        writer.writerow(["source_domains", str(payload["source_domains"])])
        writer.writerow(["target_domains", str(payload["target_domains"])])

        writer.writerow(["hparams_begin", ""])
        for k in sorted(random_hparams.keys()):
            writer.writerow([k, random_hparams[k]])
        writer.writerow(["hparams_end", ""])

        writer.writerow(["best_epoch", results.get("best_epoch", "")])
        writer.writerow(["best_val_loss", results.get("best_val_loss", "")])
        writer.writerow(["best_test_loss", results.get("best_test_loss", "")])
        writer.writerow(["best_val_mae", best_val.get("mae", "")])
        writer.writerow(["best_val_f1", best_val.get("f1", "")])
        writer.writerow(["best_val_acc2", best_val.get("acc2", "")])
        writer.writerow(["best_test_mae", best_test.get("mae", "")])
        writer.writerow(["best_test_f1", best_test.get("f1", "")])
        writer.writerow(["best_test_acc2", best_test.get("acc2", "")])

    print(f"Run log saved to: {log_path}")
    return log_path
