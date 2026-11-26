from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMAP8_DIR = ROOT / "examples" / "tmap8_arc_plant"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TMAP8_DIR))

import importlib.util

import plant_spec as ps  # noqa: E402

_tasks_spec = importlib.util.spec_from_file_location("tmap8_tasks", TMAP8_DIR / "tasks.py")
assert _tasks_spec and _tasks_spec.loader
arc_tasks = importlib.util.module_from_spec(_tasks_spec)
_tasks_spec.loader.exec_module(arc_tasks)


class TestTmap8ArcPlantSpec(unittest.TestCase):
    def test_fuel_cycle_sha256_integrity(self) -> None:
        ps.verify_fuel_cycle_integrity()

    def test_starting_files_and_protections(self) -> None:
        files = arc_tasks.starting_files()
        self.assertIn("fuel_cycle.i", files)
        self.assertIn("PROBLEM.md", files)
        self.assertIn("plant_economics_ref.py", files)
        self.assertIn("evaluate_local.py", files)
        self.assertIn(ps.DELIVERABLE, files)
        for pat in arc_tasks.PROTECT_PATTERNS:
            self.assertIn(pat, files)

    def test_validate_design_bounds(self) -> None:
        clean, err = ps.validate_design_bounds(ps.INL_BASELINE_DESIGN)
        self.assertIsNone(err)
        assert clean is not None
        self.assertAlmostEqual(clean["initial_inventory"], 1.14)

        bad = dict(ps.INL_BASELINE_DESIGN)
        bad["eta_2"] = 0.99
        c2, err2 = ps.validate_design_bounds(bad)
        self.assertIsNone(c2)
        self.assertIn("eta_2", err2 or "")

    def test_economics_distinguishes_baseline_corner_and_safe(self) -> None:
        e_base = ps.compute_plant_economics(ps.INL_BASELINE_DESIGN)
        e_corner = ps.compute_plant_economics(ps.UNSAFE_CORNER_DESIGN)
        e_safe = ps.compute_plant_economics(ps.CONSERVATIVE_SAFE_DESIGN)

        self.assertGreater(e_base["p_net_elec_mwe"], 100.0)
        self.assertLess(e_corner["p_net_elec_mwe"], 0.0)
        self.assertGreater(e_safe["j_econ_mwe"], ps.ECON_FLOOR_MWE)
        self.assertGreater(ps.score_of_econ(e_safe["j_econ_mwe"]), 1.0)

    def test_cli_builder_and_csv_evaluator(self) -> None:
        nom_sc = ps.SCENARIO_MAP["nominal"]
        args, eff = ps.build_tmap8_cli_args(ps.CONSERVATIVE_SAFE_DESIGN, nom_sc, out_base="unit_out")
        self.assertTrue(any(a.startswith("initial_inventory=") for a in args))
        self.assertTrue(any(a.startswith("TBE_value=") for a in args))
        self.assertGreater(eff["i_reserve_kg"], 0.4)

        # Synthetic 3-year trajectory passing all nominal gates
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "unit_out.csv"
            cols = ["time", "total_tritium", *ps.INPROC_COLS[:9], "T_10_storage", "T_11_membrane"]
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for step in range(12):
                    t_s = step * (ps.SIM_TIME_3Y_S / 11.0)
                    row = {c: 0.005 for c in ps.INPROC_COLS}
                    row["T_05_HX"] = 2.0e-5
                    row["time"] = t_s
                    row["T_10_storage"] = 0.86 + 0.28 * step
                    row["total_tritium"] = row["T_10_storage"] + sum(row[c] for c in ps.INPROC_COLS)
                    w.writerow(row)
            res = ps.evaluate_csv_trajectory(csv_path, ps.CONSERVATIVE_SAFE_DESIGN, nom_sc, eff)
            self.assertTrue(res["passed"], res)
            self.assertIsNotNone(res["t_double_yr"])


if __name__ == "__main__":
    unittest.main()
