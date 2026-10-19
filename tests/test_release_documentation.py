"""Offline regression checks for the published data's documented access paths."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_docs', ROOT / 'scripts/validate_release_documentation.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseDocumentationTests(unittest.TestCase):
    def test_current_release(self):
        result = MODULE.validate()
        self.assertEqual(result['fields'], 58)
        self.assertEqual(result['record_sets'], 5)

    def test_rejects_stale_or_unverified_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['croissant.json', 'README.md', 'DATA_CARD.md',
                         'cpg_upload/PUBLIC_PATHS.md', 'cpg_upload/PUBLIC_RELEASE_AUDIT.md',
                         'cpg_upload/mq_repaired_sites_20261001.csv',
                         'reproducibility/validation/cpg-public-20261019.json']:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, root / name)
            original = (root / 'croissant.json').read_text()
            for field, wrong_value in [('contentUrl', 'https://example.org/stale.parquet'),
                                       ('sha256', '0' * 64), ('contentSize', '1 B')]:
                with self.subTest(field=field):
                    data = json.loads(original)
                    data['distribution'][0][field] = wrong_value
                    (root / 'croissant.json').write_text(json.dumps(data))
                    with self.assertRaises(AssertionError):
                        MODULE.validate(root)
            data = json.loads(original)
            data['creator'][0]['name'] = 'Incorrect author'
            (root / 'croissant.json').write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                MODULE.validate(root)


if __name__ == '__main__':
    unittest.main()
