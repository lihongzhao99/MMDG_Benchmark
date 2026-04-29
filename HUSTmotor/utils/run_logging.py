import csv
import os
from datetime import datetime


def normalize_domains(domains):
    if domains is None:
        return []
    if hasattr(domains, "tolist"):
        domains = domains.tolist()
    if isinstance(domains, (list, tuple)):
        return [_normalize_domain_value(x) for x in domains]
    return [_normalize_domain_value(domains)]


def _normalize_domain_value(domain):
    if isinstance(domain, str):
        domain = domain.strip()
        if domain.upper().startswith("D"):
            domain = domain[1:]
    return int(domain)


def parse_hust_domain_args(source_domains, target_domains, valid_domains=(1, 2, 3, 4)):
    source_domains = normalize_domains(source_domains)
    target_domains = normalize_domains(target_domains)
    valid_set = set(valid_domains)

    if not source_domains:
        raise ValueError("At least one source domain is required.")
    if not target_domains:
        raise ValueError("At least one target domain is required.")

    invalid = sorted((set(source_domains) | set(target_domains)) - valid_set)
    if invalid:
        valid_text = ", ".join(f"D{domain}" for domain in valid_domains)
        invalid_text = ", ".join(f"D{domain}" for domain in invalid)
        raise ValueError(f"Invalid HUST domain(s): {invalid_text}. Valid domains are: {valid_text}.")

    overlap = sorted(set(source_domains) & set(target_domains))
    if overlap:
        overlap_text = ", ".join(f"D{domain}" for domain in overlap)
        raise ValueError(f"Source and target domains must be disjoint; overlap: {overlap_text}.")

    return source_domains, target_domains


def infer_dg_mode(source_domains):
    return "single_source_dg" if len(normalize_domains(source_domains)) == 1 else "multi_source_dg"


def format_domain_tag(domains):
    return "-".join(str(domain) for domain in normalize_domains(domains))


def build_task_name(source_domains, target_domains):
    source_tag = "-".join(f"D{domain}" for domain in normalize_domains(source_domains))
    target_tag = "-".join(f"D{domain}" for domain in normalize_domains(target_domains))
    return f"{source_tag}_to_{target_tag}"


def build_run_name(method_name, dataset_name, source_domains, target_domains, seed, run_name=None):
    if run_name:
        return run_name
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{method_name}_{dataset_name}_"
        f"{format_domain_tag(source_domains)}_to_{format_domain_tag(target_domains)}_"
        f"vib_aud_seed{seed}_{run_id}"
    )


def build_output_paths(script_file, dataset_name, method_name, dg_mode, run_name):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    output_root = os.path.join(script_dir, "outputs")
    log_dir = os.path.join(output_root, "logs", method_name, dg_mode)
    model_dir = os.path.join(output_root, "models", method_name, dg_mode)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir, os.path.join(log_dir, run_name + ".csv")


def write_run_header(log_path, args, method_name, dg_mode, source_domains, target_domains):
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow([])
        writer.writerow(["run_start", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["method", method_name])
        writer.writerow(["dg_mode", dg_mode])
        writer.writerow(["source_domain", normalize_domains(source_domains)])
        writer.writerow(["target_domain", normalize_domains(target_domains)])
        writer.writerow(["hparams_begin", ""])
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])
        writer.writerow(["hparams_end", ""])
        writer.writerow(["epoch", "split", "loss", "acc"])


def append_best_result(log_path, best_epoch, val_acc, test_acc, per_target_results, mean_target_acc):
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["best_val_acc", f"{val_acc:.6f}"])
        writer.writerow(["test_acc_at_best_val", f"{test_acc:.6f}"])
        for domain, acc in per_target_results.items():
            writer.writerow([best_epoch, f"test D{domain}", "", f"{acc:.6f}"])
        writer.writerow([best_epoch, "test mean", "", f"{mean_target_acc:.6f}"])
