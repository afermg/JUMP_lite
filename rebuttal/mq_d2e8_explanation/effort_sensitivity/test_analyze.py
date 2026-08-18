#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import pandas as pd

MODULE=Path(__file__).with_name("analyze.py")
spec=importlib.util.spec_from_file_location("effort_analysis",MODULE); assert spec and spec.loader
analysis=importlib.util.module_from_spec(spec); sys.modules[spec.name]=analysis; spec.loader.exec_module(analysis)

class Tests(unittest.TestCase):
    def test_selection_is_zstd_only_and_lexical_on_tie(self):
        rows=[]
        for fam in analysis.FAMILIES:
            model=analysis.model_name(fam,"zstd")
            for i in range(48):
                rows.append({"model":model,"config":f"c{i:02d}","PA":1,"PC":1,"PA_mean_nap":.1,"PC_mean_nap":.1})
            rows[-1].update(config="zeta",PA=10,PC=10); rows[-2].update(config="alpha",PA=10,PC=10)
            rows.append({"model":analysis.model_name(fam,"jpegxl_lossy_effort_3"),"config":"leaky","PA":99,"PC":99,"PA_mean_nap":.9,"PC_mean_nap":.9})
        self.assertEqual(set(analysis.select_recipes(pd.DataFrame(rows)).values()),{"alpha"})
    def test_alignment_rejects_duplicates_and_key_drift(self):
        left=pd.DataFrame({"k":["a","b"],"x":[1,2]})
        with self.assertRaises(analysis.AnalysisError): analysis.align([left,pd.DataFrame({"k":["a","a"],"y":[3,4]})],"k",2)
        with self.assertRaises(analysis.AnalysisError): analysis.align([left,pd.DataFrame({"k":["a","c"],"y":[3,4]})],"k",2)
    def test_holm(self):
        np.testing.assert_allclose(analysis.holm(np.array([.01,.04,.03])),[.03,.06,.06])
    def test_bootstrap_deterministic_and_shared_weights(self):
        cols=[f"{f}__{c}" for f in analysis.FAMILIES for c in ("Zstd","HQ","E3")]
        rng=np.random.default_rng(3); a=rng.normal(.4,.01,(20,len(cols))); c=rng.normal(.08,.01,(15,len(cols)))
        for f in analysis.FAMILIES:
            a[:,cols.index(f+"__E3")]=a[:,cols.index(f+"__HQ")]+.02
            c[:,cols.index(f+"__E3")]=c[:,cols.index(f+"__HQ")]
        pa=pd.DataFrame(a,columns=cols); pa.insert(0,"Metadata_broad_sample",[f"p{i}" for i in range(20)])
        pc=pd.DataFrame(c,columns=cols); pc.insert(0,"Metadata_target",[f"t{i}" for i in range(15)])
        _,one=analysis.bootstrap(pa,pc,replicates=500,seed=9); _,two=analysis.bootstrap(pa,pc,replicates=500,seed=9)
        pd.testing.assert_frame_equal(one,two); self.assertTrue((one.product_delta_e3_minus_hq>0).all())
    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.txt"; analysis.atomic_text(p,"ok\n"); self.assertEqual(p.read_text(),"ok\n")
    def test_production_sources_match_frozen_hashes(self):
        for path,expected in analysis.EXPECTED.items():
            record=analysis.verify(Path(path),expected); self.assertEqual((record["size_bytes"],record["sha256"]),expected)

    def test_frozen_manifest_rejects_input_identity_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); payload=root/"payload.csv"; payload.write_text("a\n1\n")
            record=analysis.verify(payload); producer=analysis.verify(analysis.COMPRESS_SOURCE)
            manifest={
                "protocol":"effort_sensitivity_inputs_v1",
                "selected_recipes":{"x":"recipe"},
                "producer_source":{
                    "repository_commit":analysis.PRODUCER_REPOSITORY_COMMIT,
                    "path":str(analysis.COMPRESS_SOURCE.relative_to(analysis.REPO)),
                    "size_bytes":producer["size_bytes"],"sha256":producer["sha256"],
                },
                "inputs":analysis.normalized_records([record]),
            }
            frozen=root/"frozen.json"; frozen.write_text(json.dumps(manifest))
            with mock.patch.object(analysis,"FROZEN_INPUTS",frozen):
                analysis.verify_frozen_inputs([record],{"x":"recipe"},producer)
                drifted=dict(record); drifted["sha256"]="0"*64
                with self.assertRaises(analysis.AnalysisError):
                    analysis.verify_frozen_inputs([drifted],{"x":"recipe"},producer)

    def test_release_verifier_rejects_artifact_drift(self):
        with tempfile.TemporaryDirectory() as d:
            release=Path(d)/"release"; shutil.copytree(analysis.OUTPUT_DIR,release)
            with mock.patch.object(analysis,"OUTPUT_DIR",release):
                analysis.verify_release(release)
                with (release/"e3_vs_hq_bootstrap.csv").open("a") as f: f.write("drift\n")
                with self.assertRaises(analysis.AnalysisError): analysis.verify_release(release)

    def test_clean_promotion_replaces_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); old=root/"outputs"; staged=root/"staged"
            old.mkdir(); staged.mkdir(); (old/"old").write_text("old"); (staged/"new").write_text("new")
            analysis.promote_clean_release(staged,old)
            self.assertEqual({p.name for p in old.iterdir()},{"new"})
            self.assertFalse(staged.exists())

if __name__=="__main__": unittest.main(verbosity=2)
