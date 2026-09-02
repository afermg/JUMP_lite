#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("synthesis_analyze",HERE/"analyze.py")
assert SPEC and SPEC.loader
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)

class SynthesisTests(unittest.TestCase):
    def test_pinned_inputs_and_counts(self):
        paths,records=m.verify_inputs(m.DEFAULT_INPUT_ROOT)
        self.assertEqual(len(paths),16); self.assertEqual(len(records),16)
        d=m.load_data(paths)
        self.assertEqual(len(d["paired"]),240)
        self.assertEqual(len(d["fixed"]),5)
        self.assertEqual(int(d["signs"].unanimous_sign.sum()),1)
        figure_data=m.build_figure_data(d)
        self.assertEqual(set(figure_data.panel),{"B"})
        self.assertEqual(len(figure_data),5)

    def test_input_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for _key,(rel,_size,_digest) in m.INPUTS.items():
                dst=root/rel; dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(m.DEFAULT_INPUT_ROOT/rel,dst)
            rel=m.INPUTS["pooled"][0]
            with (root/rel).open("ab") as handle: handle.write(b"drift")
            with self.assertRaises(m.AnalysisError): m.verify_inputs(root)

    def test_release_verifies(self):
        m.verify_release(m.DEFAULT_OUTPUT)

    def test_deterministic_generation(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/"a"; b=Path(td)/"b"
            m.generate(m.DEFAULT_INPUT_ROOT,a); m.generate(m.DEFAULT_INPUT_ROOT,b)
            for name in ("mq_d2e8_synthesis.png","mq_d2e8_synthesis.pdf","figure_data.csv","CAPTION.md","REPORT.md"):
                ha=hashlib.sha256((a/name).read_bytes()).hexdigest(); hb=hashlib.sha256((b/name).read_bytes()).hexdigest()
                self.assertEqual(ha,hb,name)
            m.verify_release(a); m.verify_release(b)

if __name__=="__main__": unittest.main()
