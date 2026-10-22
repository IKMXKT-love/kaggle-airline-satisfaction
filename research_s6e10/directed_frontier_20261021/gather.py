"""Bounded read-only Kaggle source/output metadata audit. Never executes remote code."""
import concurrent.futures
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('discovery', ROOT.parent / 'upstream_generalization_20261020/discover.py')
discovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(discovery)
discovery.ROOT = ROOT
sys.stdout.reconfigure(encoding='utf-8')


def gather(ref):
    owner, slug = ref.split('/')
    tag = owner + '__' + slug
    receipt = discovery.fetch(tag + '_pull', 'https://www.kaggle.com/api/v1/kernels/pull/' + ref)
    result = {'ref': ref, 'pull': receipt}
    if 'error' in receipt:
        return result
    data = json.loads((ROOT / (tag + '_pull.json')).read_text('utf-8'))
    meta, blob = data.get('metadata', {}), data.get('blob', {})
    source = blob.get('source', '')
    try:
        notebook = json.loads(source)
        cells = notebook.get('cells', [])
        source_text = '\n\n'.join(f'### CELL {i} {c.get("cell_type")}\n' + ''.join(c.get('source', [])) for i, c in enumerate(cells))
        output_text = '\n\n'.join(f'### CELL {i}\n' + '\n'.join(''.join(o.get('text', [])) if 'text' in o else str(o.get('data', {}).get('text/plain', '')) for o in c.get('outputs', [])) for i, c in enumerate(cells) if c.get('outputs'))
    except (json.JSONDecodeError, AttributeError):
        source_text, output_text = source, ''
    (ROOT / (tag + '_source.txt')).write_text(source_text, encoding='utf-8')
    (ROOT / (tag + '_cell_outputs.txt')).write_text(output_text, encoding='utf-8')
    version = meta.get('currentVersionNumber')
    result.update({'version': version, 'metadata': meta, 'source_chars': len(source_text), 'cell_output_chars': len(output_text)})
    if version is not None:
        result['outputs'] = discovery.fetch(tag + f'_outputs_v{version}', 'https://api.kaggle.com/v1/kernels.KernelsApiService/ListKernelSessionOutput', {'userName': owner, 'kernelSlug': slug, 'pageSize': 200, 'versionLabel': f'v{version}'})
    return result


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(gather, sys.argv[1:]))
    for result in results:
        tag = result['ref'].replace('/', '__')
        (ROOT / (tag + '_receipt.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'metadata'}, ensure_ascii=False))
