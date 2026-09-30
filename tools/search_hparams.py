#!/usr/bin/env python3
"""Run the paper's per-task default + random-search + two-seed protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import secrets
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD_CONFIG_DIR = ROOT / "configs" / "hparam_search"
TASK_DIRS = {"action": ROOT / "Action recognition", "hust": ROOT / "HUSTmotor", "mmsa": ROOT / "MMSA"}
FORBIDDEN_EXTRA = {"--seed", "--run_name", "--random_hparams_json"}


def load_method_specs():
    specs = {}
    for path in sorted(METHOD_CONFIG_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            spec = json.load(handle)
        name = spec.get("name")
        if not name or name in specs:
            raise ValueError(f"invalid or duplicate method name in {path}: {name!r}")
        missing_tasks = set(TASK_DIRS) - set(spec.get("scripts", {}))
        if missing_tasks:
            raise ValueError(f"{path} has no scripts for: {sorted(missing_tasks)}")
        if not isinstance(spec.get("space"), dict):
            raise ValueError(f"{path} must contain an object named space")
        specs[name] = spec
    if not specs:
        raise ValueError(f"no method configurations found in {METHOD_CONFIG_DIR}")
    return specs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Per-task hyperparameter search used by MMDG-Bench")
    parser.add_argument("--task", required=True, choices=sorted(TASK_DIRS))
    parser.add_argument("--method", required=True, choices=sorted(load_method_specs()))
    parser.add_argument("--source", nargs="+", required=True, help="source domain(s) or MMSA dataset(s)")
    parser.add_argument("--target", required=True, help="one target domain or MMSA dataset")
    parser.add_argument("--dataset", choices=("epic", "hac"), help="required for action recognition")
    parser.add_argument("--modality", choices=("va", "vf", "af", "vaf"), help="required for action recognition")
    parser.add_argument("--datapath", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--search-seed", type=int, default=None,
                        help="seed for reproducibly generating hparams and random training seeds")
    parser.add_argument("--split-seed", type=int, default=8,
                        help="fixed HUST source train/validation split seed")
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=2_147_483_647)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="extra training arguments after --")
    args = parser.parse_args(argv)
    if args.task == "action" and (not args.dataset or not args.modality):
        parser.error("--dataset and --modality are required when --task action")
    if args.task != "action" and (args.dataset or args.modality):
        parser.error("--dataset/--modality are only valid for --task action")
    if args.random_trials < 0:
        parser.error("--random-trials must be non-negative")
    if args.seed_min < 0 or args.seed_max < args.seed_min:
        parser.error("invalid seed range")
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    for token in args.extra:
        if token.split("=", 1)[0] in FORBIDDEN_EXTRA:
            parser.error(f"{token.split('=', 1)[0]} is controlled by the search runner")
    return args



def default_hparams(space):
    return {name: spec["default"] for name, spec in space.items()}


def sample_hparams(space, rng):
    sampled = {}
    for name, spec in space.items():
        if spec["type"] == "choice":
            sampled[name] = rng.choice(spec["values"])
        else:
            value = rng.uniform(spec["low"], spec["high"])
            digits = spec.get("digits")
            sampled[name] = round(value, digits) if digits is not None else value
    return sampled


def unique_seeds(rng, count, low, high):
    if high - low + 1 < count:
        raise ValueError("seed range is smaller than the number of required unique seeds")
    seeds = []
    seen = set()
    while len(seeds) < count:
        value = rng.randint(low, high)
        if value not in seen:
            seeds.append(value)
            seen.add(value)
    return seeds


def task_slug(args):
    source = "-".join(args.source)
    base = f"{source}_to_{args.target}"
    if args.task == "action":
        return f"{args.dataset}_{base}_{args.modality}"
    return base


def output_dir(args):
    if args.output_dir:
        return args.output_dir.resolve()
    dataset = args.dataset if args.task == "action" else args.task
    return ROOT / "outputs" / "search" / args.task / dataset / args.method / task_slug(args)


def hparam_cli(task, method_spec, params):
    aliases = method_spec.get("aliases", {}).get(task, {})
    result = []
    for canonical, value in params.items():
        if value == "1/batch_size":
            continue
        for actual in aliases.get(canonical, (canonical,)):
            result.extend((f"--{actual}", str(value)))
    return result


def build_command(args, run_name, seed, params):
    task_dir = TASK_DIRS[args.task]
    method_spec = load_method_specs()[args.method]
    cmd = [args.python, method_spec["scripts"][args.task]]
    if args.task == "action":
        modality_flags = {
            "va": ["--use_video", "--use_audio"],
            "vf": ["--use_video", "--use_flow"],
            "af": ["--use_audio", "--use_flow"],
            "vaf": ["--use_video", "--use_audio", "--use_flow"],
        }
        cmd += ["--dataset", args.dataset, "--num_class", "8" if args.dataset == "epic" else "7"]
        cmd += ["-s", *args.source, "-t", args.target, *modality_flags[args.modality]]
        if method_spec.get("uses_num_modals", False):
            cmd += ["--num_modals", "3" if args.modality == "vaf" else "2"]
    elif args.task == "hust":
        cmd += ["-s", *args.source, "-t", args.target, "--split_seed", str(args.split_seed)]
    else:
        cmd += ["--source_datasets", *args.source, "--target_dataset", args.target]
    if args.datapath:
        if args.task == "hust":
            raise ValueError("HUST scripts do not expose --datapath; place data under HUSTmotor/data")
        cmd += ["--datapath", args.datapath]
    cmd += ["--seed", str(seed), "--run_name", run_name]
    enable_flags = method_spec.get("enable_flags", {}).get(args.task, [])
    if enable_flags:
        cmd.extend(enable_flags)
    cmd += hparam_cli(args.task, method_spec, params)
    if args.task == "mmsa":
        cmd += ["--random_hparams_json", json.dumps(params, sort_keys=True, separators=(",", ":"))]
    cmd += args.extra
    return task_dir, cmd


def find_log(run_name):
    matches = list((ROOT / "Action recognition" / "outputs" / "logs").glob(f"**/{run_name}.csv"))
    matches += list((ROOT / "HUSTmotor" / "outputs" / "logs").glob(f"**/{run_name}.csv"))
    matches += list((ROOT / "MMSA" / "outputs" / "logs").glob(f"**/{run_name}.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one log for {run_name}, found {len(matches)}: {matches}")
    return matches[0]


def _float(value):
    try:
        parsed = float(value.strip())
        return parsed if math.isfinite(parsed) else None
    except (AttributeError, TypeError, ValueError):
        return None


def parse_result(log_path):
    values = {}
    with log_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            for index, cell in enumerate(row):
                text = cell.strip()
                if "=" in text:
                    key, raw = text.split("=", 1)
                    parsed = _float(raw)
                    if parsed is not None:
                        values[key.strip().lower()] = parsed
                parsed = _float(row[index + 1]) if index + 1 < len(row) else None
                if parsed is not None:
                    values[text.lower()] = parsed
    val = next((values[key] for key in ("best_val_acc2", "best_val_acc", "bestvalacc") if key in values), None)
    test = next((values[key] for key in ("best_test_acc2", "test_acc_at_best_val", "besttestacc") if key in values), None)
    val_loss = next((values[key] for key in ("best_val_loss", "bestloss") if key in values), None)
    if val is None or test is None:
        raise RuntimeError(f"could not parse validation/test metrics from {log_path}")
    return {"val_score": val, "test_score": test, "val_loss": val_loss, "log_path": str(log_path)}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def trial_rank(trial):
    loss = trial["result"].get("val_loss")
    return (trial["result"]["val_score"], -(loss if loss is not None else math.inf), -trial["index"])


def execute_trial(args, state, state_path, phase, index, seed, params):
    trial_key = f"{phase}-{index:02d}"
    existing = state["trials"].get(trial_key)
    if args.resume and existing and existing.get("status") == "completed":
        return existing
    run_name = f"hsearch_{state['search_id']}_{trial_key}_seed{seed}"
    task_dir, cmd = build_command(args, run_name, seed, params)
    record = {
        "phase": phase, "index": index, "seed": seed, "hparams": params,
        "run_name": run_name, "command": cmd, "status": "dry-run" if args.dry_run else "running",
    }
    state["trials"][trial_key] = record
    write_json(state_path, state)
    print("+", shlex.join(cmd), flush=True)
    if args.dry_run:
        return record
    stdout_path = state_path.parent / f"{trial_key}.stdout.log"
    with stdout_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(cmd, cwd=task_dir, stdout=output, stderr=subprocess.STDOUT, text=True)
    record["stdout_path"] = str(stdout_path)
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        record["status"] = "failed"
        write_json(state_path, state)
        raise RuntimeError(f"{trial_key} failed; see {stdout_path}")
    record["result"] = parse_result(find_log(run_name))
    record["status"] = "completed"
    write_json(state_path, state)
    print(f"{trial_key}: val={record['result']['val_score']:.6f}, test={record['result']['test_score']:.6f}")
    return record


def create_state(args, space, state_path):
    generator_seed = args.search_seed if args.search_seed is not None else secrets.randbits(63)
    rng = random.Random(generator_seed)
    total_seeds = 1 + args.random_trials + 2
    seeds = unique_seeds(rng, total_seeds, args.seed_min, args.seed_max)
    search_params = [default_hparams(space)] + [sample_hparams(space, rng) for _ in range(args.random_trials)]
    search_id = f"{task_slug(args)}_{generator_seed}"
    return {
        "version": 1, "search_id": search_id, "task": args.task, "method": args.method,
        "dataset": args.dataset, "source": args.source, "target": args.target,
        "modality": args.modality, "generator_seed": generator_seed,
        "protocol": {"default_trials": 1, "random_trials": args.random_trials, "additional_seeds": 2},
        "search_plan": [{"index": i, "seed": seeds[i], "hparams": params} for i, params in enumerate(search_params)],
        "retrain_seeds": seeds[len(search_params):], "trials": {},
    }


def main(argv=None):
    args = parse_args(argv)
    method_specs = load_method_specs()
    space = method_specs[args.method]["space"]
    destination = output_dir(args)
    state_path = destination / "search_state.json"
    if state_path.exists():
        if not args.resume:
            raise SystemExit(f"search state already exists: {state_path}; use --resume or choose --output-dir")
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
    else:
        state = create_state(args, space, state_path)
        write_json(state_path, state)

    search_trials = []
    for planned in state["search_plan"]:
        search_trials.append(execute_trial(
            args, state, state_path, "search", planned["index"], planned["seed"], planned["hparams"]
        ))
    if args.dry_run:
        print(f"Dry run plan saved to {state_path}")
        return 0

    best = max(search_trials, key=trial_rank)
    state["best_search_trial"] = f"search-{best['index']:02d}"
    state["best_hparams"] = best["hparams"]
    write_json(state_path, state)
    final_trials = [best]
    for index, seed in enumerate(state["retrain_seeds"], start=1):
        final_trials.append(execute_trial(args, state, state_path, "retrain", index, seed, best["hparams"]))

    test_scores = [trial["result"]["test_score"] for trial in final_trials]
    state["final"] = {
        "trial_keys": [state["best_search_trial"], "retrain-01", "retrain-02"],
        "seeds": [trial["seed"] for trial in final_trials],
        "test_scores": test_scores,
        "mean": statistics.fmean(test_scores),
        "std_population": statistics.pstdev(test_scores),
        "std_sample": statistics.stdev(test_scores),
    }
    write_json(state_path, state)
    print(json.dumps(state["final"], ensure_ascii=False, indent=2))
    print(f"Search state saved to {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
