import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkHarnessTests(unittest.TestCase):
    def make_fixture(self, tmp):
        genes = [f"g{i}" for i in range(8)]
        emb = np.eye(8, 4, dtype=np.float32)

        emb_path = tmp / "emb.csv"
        genes_path = tmp / "genes.txt"
        gmt_path = tmp / "sets.gmt"
        ranked_path = tmp / "ranked.tsv"

        np.savetxt(emb_path, emb, delimiter=",")
        genes_path.write_text("\n".join(genes) + "\n")
        gmt_path.write_text(
            "t1\tname\tg0\tg1\n"
            "t2\tname\tg2\tg3\tg4\n"
            "t3\tname\tg4\tg5\tg6\n"
        )
        ranked_path.write_text("\n".join(f"{g}\t{8 - i}" for i, g in enumerate(genes)) + "\n")
        return emb_path, genes_path, gmt_path, ranked_path

    def test_benchmark_writes_json_report(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            emb, genes, gmt, _ = self.make_fixture(tmp)
            json_out = tmp / "andes.json"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "bench_end_to_end.py"),
                    "andes",
                    "--emb",
                    str(emb),
                    "--genelist",
                    str(genes),
                    "--geneset1",
                    str(gmt),
                    "--geneset2",
                    str(gmt),
                    "--min",
                    "2",
                    "--max",
                    "3",
                    "--skip-cache-build",
                    "--json-out",
                    str(json_out),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            report = json.loads(json_out.read_text())
            self.assertEqual(report["mode"], "andes")
            self.assertEqual(report["n_terms1"], 3)
            self.assertEqual(report["n_terms2"], 3)
            self.assertEqual(report["query_mode"], "bestmatch")
            self.assertEqual(report["null_mode"], "prefix")
            self.assertIn("query_scoring", report["timing"]["stages"])

    def test_benchmark_runs_bma_prototype_modes(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            emb, genes, gmt, _ = self.make_fixture(tmp)
            json_out = tmp / "andes_bestmatch.json"
            cache = tmp / "andes_prefix.pkl"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "bench_end_to_end.py"),
                    "andes",
                    "--emb",
                    str(emb),
                    "--genelist",
                    str(genes),
                    "--geneset1",
                    str(gmt),
                    "--geneset2",
                    str(gmt),
                    "--min",
                    "2",
                    "--max",
                    "3",
                    "--ite",
                    "3",
                    "--workers",
                    "1",
                    "--query-mode",
                    "bestmatch",
                    "--null-mode",
                    "prefix",
                    "--query-memory-mb",
                    "0.001",
                    "--cache",
                    str(cache),
                    "--json-out",
                    str(json_out),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            report = json.loads(json_out.read_text())
            self.assertEqual(report["mode"], "andes")
            self.assertEqual(report["query_mode"], "bestmatch")
            self.assertEqual(report["null_mode"], "prefix")
            self.assertGreater(report["cache_entries"], 0)
            self.assertIn("query_scoring", report["timing"]["stages"])

    def test_benchmark_runs_gsea_bestmatch_mode(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            emb, genes, gmt, ranked = self.make_fixture(tmp)
            json_out = tmp / "gsea_bestmatch.json"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "bench_end_to_end.py"),
                    "gsea",
                    "--emb",
                    str(emb),
                    "--genelist",
                    str(genes),
                    "--geneset",
                    str(gmt),
                    "--rankedlist",
                    str(ranked),
                    "--min",
                    "2",
                    "--max",
                    "3",
                    "--skip-cache-build",
                    "--score-mode",
                    "bestmatch",
                    "--query-memory-mb",
                    "0.001",
                    "--json-out",
                    str(json_out),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            report = json.loads(json_out.read_text())
            self.assertEqual(report["mode"], "gsea")
            self.assertEqual(report["score_mode"], "bestmatch")
            self.assertGreater(report["cache_entries"], 0)
            self.assertIn("query_scoring", report["timing"]["stages"])


class SpeedupBenchmarkTests(unittest.TestCase):
    def test_speedup_bench_runs_and_produces_json(self):
        with tempfile.TemporaryDirectory() as d:
            json_out = Path(d) / "speedup.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "bench_speedup.py"),
                    "--n-genes", "200",
                    "--dim", "16",
                    "--n-terms", "8",
                    "--min-size", "5",
                    "--max-size", "20",
                    "--n-pairs", "10",
                    "--ite", "10",
                    "--workers", "2",
                    "--json-out", str(json_out),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(result.returncode, 0,
                             msg=f"bench_speedup.py failed:\n{result.stderr}")
            report = json.loads(json_out.read_text())
            self.assertIn("results", report)
            names = {r["name"] for r in report["results"]}
            self.assertIn("ANDES (BMA)", names)
            self.assertIn("GSEA-ANDES", names)
            for r in report["results"]:
                self.assertGreater(r["speedup"], 0)
                self.assertGreaterEqual(r["rho"], -1.0)
                self.assertLessEqual(r["rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
