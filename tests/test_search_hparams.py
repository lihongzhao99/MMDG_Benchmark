import importlib.util
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("search_hparams", ROOT / "tools" / "search_hparams.py")
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)


def args_for(task, method):
    return SimpleNamespace(
        task=task,
        method=method,
        source=["D2", "D3"] if task != "mmsa" else ["mosi", "mosei"],
        target="D1" if task != "mmsa" else "sims",
        dataset="epic" if task == "action" else None,
        modality="vaf" if task == "action" else None,
        datapath="/data" if task != "hust" else "",
        python="python",
        random_trials=10,
        search_seed=123,
        split_seed=8,
        seed_min=0,
        seed_max=2_147_483_647,
        output_dir=None,
        resume=False,
        dry_run=True,
        extra=[],
    )


class SearchProtocolTests(unittest.TestCase):
    def test_plan_has_default_plus_ten_and_two_new_seeds(self):
        args = args_for("action", "JAT")
        state = SEARCH.create_state(args, SEARCH.load_method_specs()["JAT"]["space"], Path("unused"))
        self.assertEqual(len(state["search_plan"]), 11)
        self.assertEqual(len(state["retrain_seeds"]), 2)
        seeds = [trial["seed"] for trial in state["search_plan"]] + state["retrain_seeds"]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(state["search_plan"][0]["hparams"]["cls_loss"], 3.0)

    def test_ordinary_uniform_values_are_rounded_to_one_decimal(self):
        spaces = {name: spec["space"] for name, spec in SEARCH.load_method_specs().items()}
        sampled = SEARCH.sample_hparams(spaces["JAT"], random.Random(9))
        for value in sampled.values():
            self.assertAlmostEqual(value * 10, round(value * 10))
        self.assertIsNone(spaces["MBCD"]["ema_beta"]["digits"])
        self.assertIsNone(spaces["MOOSA"]["entropy_min_weight"]["digits"])
        self.assertIsNone(spaces["NEL"]["beta"]["digits"])

    def test_action_jat_maps_one_domain_weight_to_both_losses(self):
        args = args_for("action", "JAT")
        _, command = SEARCH.build_command(
            args, "trial", 17,
            {"alpha_rev": 0.1, "alpha_rev2": 0.3, "domain_adv_loss": 0.7,
             "modal_adv_loss": 0.2, "cls_loss": 4.1},
        )
        self.assertIn("--domain_adv_loss_global", command)
        self.assertIn("--domain_adv_loss_local", command)
        self.assertEqual(command[command.index("--num_modals") + 1], "3")

    def test_hust_jat_uses_aliases_and_fixed_split(self):
        args = args_for("hust", "JAT")
        _, command = SEARCH.build_command(args, "trial", 23, {"domain_adv_loss": 0.4, "cls_loss": 2.0})
        self.assertIn("--lambda_domain_global", command)
        self.assertIn("--lambda_domain_local", command)
        self.assertIn("--lambda_cls", command)
        self.assertEqual(command[command.index("--split_seed") + 1], "8")

    def test_mmsa_nel_is_enabled_and_has_stable_run_name(self):
        args = args_for("mmsa", "NEL")
        _, command = SEARCH.build_command(args, "trial", 31, {"beta": "1/batch_size", "k": 8})
        self.assertIn("--enable_nel", command)
        self.assertNotIn("--beta", command)
        self.assertEqual(command[command.index("--run_name") + 1], "trial")

    def test_parses_all_three_log_styles(self):
        samples = (
            "BestEpoch,3,BestLoss,0.2,BestValAcc,0.8,BestTestAcc,0.7\n",
            "best_val_acc,81.2\ntest_acc_at_best_val,72.3\n",
            "best_val_loss,0.4\nbest_val_acc2,0.82\nbest_test_acc2,0.71\n",
        )
        for content in samples:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "result.csv"
                path.write_text(content, encoding="utf-8")
                result = SEARCH.parse_result(path)
                self.assertIsInstance(result["val_score"], float)
                self.assertIsInstance(result["test_score"], float)


if __name__ == "__main__":
    unittest.main()
