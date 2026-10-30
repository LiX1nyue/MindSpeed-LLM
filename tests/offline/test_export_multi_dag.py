"""CPU-only integration checks; run directly, outside training pytest hooks."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class MultiDagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="multi-dag-", dir="/tmp/opencode")
        cls.binary = Path(cls.work.name) / "export_multi_dag"
        source = Path(__file__).resolve().parents[2] / "export_multi_dag.cpp"
        subprocess.run(["g++", "-std=c++11", "-O1", "-Wall", "-Wextra", "-Werror",
                        str(source), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.work.name)
        self.root = Path(self.case.name)
        for rank in range(8):
            directory = self.root / f"rank{rank}"
            directory.mkdir()
            metadata = {"distributed_args": {"rank": rank, "world_size": 8},
                        "parallel_group_info": {"dp": {"group_name": f"group_{rank}",
                                                         "global_ranks": list(range(8))}}}
            (directory / "profiler_metadata.json").write_text(json.dumps(metadata))
            events = [
                {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Ascend Hardware"}},
                {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Communication"}},
                self.event("计算", 1, "1000", 5, 1, {"Task Type": "AI_CORE"}),
                self.event("hcom_allReduce__123_1", 2, 1010, 10, 1,
                           {"group_name": f"group_{rank}", "count": 100, "data_type": "BF16"}),
                self.event("after", 1, 1030, 5, 1, {"Task Type": "AI_CORE"}),
                self.event("independent", 1, 1002, 35, 7, {"Task Type": "AI_CORE"}),
            ]
            (directory / "view_trace.json").write_text(json.dumps(events))

    def tearDown(self):
        self.case.cleanup()

    @staticmethod
    def event(name, pid, ts, dur, stream, args):
        return {"name": name, "ph": "X", "pid": pid, "tid": stream, "ts": ts,
                "dur": dur, "args": {"Physic Stream Id": stream, **args}}

    def run_export(self):
        return subprocess.run([str(self.binary), str(self.root), str(self.root / "dag.txt")],
                              text=True, capture_output=True)

    def test_merge_and_rank_scoped_edges(self):
        result = self.run_export()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = (self.root / "dag.txt").read_text().splitlines()
        self.assertEqual(lines[0], "25 16")
        comm = [line for line in lines if "type=COLLECTIVE" in line]
        self.assertEqual(len(comm), 1)
        self.assertIn("participants=[0,1,2,3,4,5,6,7]", comm[0])
        self.assertIn("estimated_bandwidth_GBps=0.020", comm[0])
        self.assertTrue(all("children=[]" in line for line in lines if 'name="independent"' in line))

    def test_duplicate_name_is_rejected(self):
        path = self.root / "rank0/view_trace.json"
        events = json.loads(path.read_text())
        events.append(events[3])
        path.write_text(json.dumps(events))
        self.assertIn("duplicate collective name", self.run_export().stderr)

    def test_group_mismatch_is_rejected(self):
        path = self.root / "rank0/profiler_metadata.json"
        meta = json.loads(path.read_text())
        meta["parallel_group_info"]["dp"]["global_ranks"] = [0, 1]
        path.write_text(json.dumps(meta))
        self.assertIn("participants conflict", self.run_export().stderr)

    def test_missing_metadata_identity_is_rejected(self):
        (self.root / "rank0/profiler_metadata.json").write_text("{}")
        self.assertIn("requires distributed_args", self.run_export().stderr)

    def test_cycle_after_merge_is_rejected(self):
        for rank in range(8):
            path = self.root / f"rank{rank}/view_trace.json"
            events = json.loads(path.read_text())
            first = events[3]
            second = {**first, "name": "hcom_allReduce__123_2", "ts": 1021, "dur": 2}
            if rank == 1:
                first["ts"], second["ts"] = second["ts"], first["ts"]
                first["dur"] = second["dur"] = 2
            events.append(second)
            path.write_text(json.dumps(events))
        self.assertIn("contains a cycle", self.run_export().stderr)


if __name__ == "__main__":
    unittest.main()
