#!/usr/bin/env python3
"""Validate release documentation without changing or uploading any dataset.

Optionally verify cached public metadata with --metadata-dir. This script does
not download files, execute uploaders, or modify canonical artifacts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METADATA_URL = ('https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/'
                'source_all/workspace/publication_data/2026_jump_lite/metadata/v1.0/')
AUTHORS = ['Alán F. Muñoz', 'Johan Fredin Haslum', 'Runxi Shen',
           'Anne E. Carpenter', 'Shantanu Singh']


def validate(repo: Path = REPO, metadata_dir: Path | None = None) -> dict:
    spec = json.loads((repo / 'croissant.json').read_text())
    assert spec['conformsTo'] == 'http://mlcommons.org/croissant/1.0'
    assert [p['name'] for p in spec['creator']] == AUTHORS
    files = {d['@id']: d for d in spec['distribution']}
    assert len(files) == len(spec['distribution']) == 6
    sets = spec['recordSet']
    assert len(sets) == 5
    fields = [f for rs in sets for f in rs['field']]
    field_ids = {f['@id'] for f in fields}
    assert len(field_ids) == len(fields)
    public = json.loads((repo / 'reproducibility/validation/cpg-public-20261019.json').read_text())
    expected = {m['name']: m for m in public['metadata']}
    for item in files.values():
        assert item['contentUrl'] == METADATA_URL + item['name']
        match = expected[item['name']]
        assert item['sha256'] == match['sha256']
        assert item['contentSize'] == f"{match['bytes']} B"
        if metadata_dir is not None:
            path = metadata_dir / item['name']
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == match['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == match['sha256']
    for rs in sets:
        keys = rs.get('key', [])
        if isinstance(keys, dict):
            keys = [keys]
        assert all(k['@id'] in field_ids for k in keys)
        for field in rs['field']:
            source = field['source']
            assert source['fileObject']['@id'] in files
            assert field['dataType'] in {'sc:Text', 'sc:Integer', 'sc:Boolean', 'sc:URL'}
            if 'references' in field:
                assert field['references']['field']['@id'] in field_ids
    if metadata_dir is not None:
        import duckdb
        connection = duckdb.connect()
        connection.execute('SET threads=2')
        for rs in sets:
            file_id = rs['field'][0]['source']['fileObject']['@id']
            path = metadata_dir / files[file_id]['name']
            columns = {r[0]: r[1] for r in connection.execute(
                'DESCRIBE SELECT * FROM read_parquet(?)', [str(path)]).fetchall()}
            for field in rs['field']:
                assert field['source']['extract']['column'] in columns
        counts = json.loads((repo / 'reproducibility/validation/dataset-counts-20261019.json').read_text())
        connection.read_parquet(str(metadata_dir / 'jump_lite_perturbation_metadata.parquet')).create_view('m')
        for item in counts['queries'].values():
            cursor = connection.execute(item['sql'])
            names = [x[0] for x in cursor.description]
            assert [dict(zip(names, row)) for row in cursor.fetchall()] == item['rows']
    assert public['repair_matches'] == len(public['repair_payloads']) == 119
    assert all(r['matches'] for r in public['repair_payloads'])
    with (repo / 'cpg_upload/mq_repaired_sites_20261001.csv').open() as handle:
        affected = list(csv.DictReader(handle))
    by_key = {row['public_key']: row for row in affected}
    assert len(by_key) == len(affected) == 119
    for payload in public['repair_payloads']:
        row = by_key[payload['key']]
        assert row['site_key'] == payload['key'].split('/')[-2]
        assert row['sha256'] == payload['sha256']
        assert int(row['bytes']) == payload['bytes']
    assert sum(row['mask_notice'] == 'No released masks' for row in affected) == 6
    assert sum(row['mask_notice'] == 'MQ cell and nuclei counts zero; lossless counts nonzero'
               for row in affected) == 113
    for name in ['DATA_CARD.md', 'cpg_upload/PUBLIC_PATHS.md', 'cpg_upload/PUBLIC_RELEASE_AUDIT.md',
                 'cpg_upload/mq_repaired_sites_20261001.csv']:
        assert (repo / name).is_file()
    readme = (repo / 'README.md').read_text()
    assert 'images/jump_lite_v1.0/images_compressed/<codec>.zarr/' in readme
    assert 'workspace_dl/embeddings/<model>-<codec>/jump_lite/v1.0/' in readme
    assert 'awaiting gallery promotion' not in readme
    return {'status': 'PASS', 'distributions': len(files), 'record_sets': len(sets),
            'fields': len(fields), 'repaired_chunks': 119,
            'cached_metadata_and_count_queries_checked': metadata_dir is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(metadata_dir=args.metadata_dir), indent=2))
