"""Assemble research receipts and verify preserved files. No model fitting or inference."""
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text('utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


dispositions = {
    'zekaizhan': 'reported historical OOF tables; no new paired output obtained',
    'kiza123123': 'non-nested target encoding and in-sample meta scoring; no paired output obtained',
    'sai10py': 'original-data teacher mechanism covered; test submission only',
    'denpugovkin': 'v47 nonlinear meta comparison failed its own gate; no new base signal',
    'anhadmahajan06': 'pseudo-label validation dependency; outputs are test submissions',
    'sarunpm': 'route mechanism covered; encoding boundary issue; no current paired output obtained',
    'goodpjw2008': 'public selection-noise research; final test file is LB-selected',
}
sources = []
for path in sorted(ROOT.glob('*_receipt.json')):
    value = read(path)
    if not value.get('ref'):
        continue
    ref = value['ref']
    item = {'ref': ref, 'version': value.get('version'), 'receipt': path.name}
    item['newly_reviewed'] = ref.split('/')[0] in dispositions
    item['disposition'] = dispositions.get(ref.split('/')[0], 'retained notebook version unchanged')
    output_path = ROOT / (ref.replace('/', '__') + f'_outputs_v{value.get("version")}.json')
    if output_path.exists():
        output = read(output_path)
        item['current_output_file_count'] = len(output.get('files', []))
        item['output_next_page_token'] = output.get('nextPageToken')
    sources.append(item)
assert len(sources) == 12
assert sum(x['newly_reviewed'] for x in sources) == 7

dataset_versions = {}
for row in csv.DictReader((PROJECT / 'public_pool_sources_20261020.csv').open(encoding='utf-8-sig')):
    if '/datasets/' in row['source_url']:
        dataset_versions[row['source_ref']] = int(row['source_version'])
refreshed_datasets = []
for path in sorted(ROOT.glob('refresh_*.json')):
    value = read(path)
    if not isinstance(value, dict) or 'ref' not in value:
        continue
    ref = value['ref']
    item = {'ref': ref, 'current_version': value.get('currentVersionNumber'), 'retained_version': dataset_versions.get(ref)}
    item['version_changed'] = item['current_version'] != item['retained_version']
    refreshed_datasets.append(item)
assert len(refreshed_datasets) == 11
assert not any(x['version_changed'] for x in refreshed_datasets)

previous = read(PROJECT / '.artifact_runtime/joint_upstream_20261021/independent_verification.json')['protected_sha256']
protected = {}
for name, expected in previous.items():
    digest = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
    protected[name] = {'sha256': digest, 'matches_previous_verification': digest == expected}
assert all(x['matches_previous_verification'] for x in protected.values())

write(ROOT / 'candidate_ledger.json', {
    'date': '2026-10-21',
    'sources': sources,
    'refreshed_retained_datasets': refreshed_datasets,
    'source_code_executed': False,
    'new_training_runs': 0,
    'new_base_inferences': 0,
    'new_submissions': 0,
    'candidate_for_training': None,
    'research_lead_only': {
        'mechanism': 'Q-FT discriminative feature-token Transformer',
        'status': 'awaiting paired original OOF/test, training code, fold/ID evidence and incremental evidence',
        'author_reported_oof': 0.959795,
        'increment_over_current_pool': None,
        'local_reproduction_recommended': False,
    },
    'evidence_limits': [
        'author OOF and public claims are not private generalization proofs',
        'Den saved metrics arithmetic checked; AUC not independently reconstructed',
        'empty current output list is not proof of a failed run',
        'negative searches are not proof of global nonexistence',
    ],
    'protected_files': protected,
})
state = read(ROOT / 'research_state.json')
state.update({
    'status': 'completed_no_training_candidate',
    'report': 'research_s6e10/DIRECTED_FRONTIER_RESEARCH_20261021.md',
    'newly_reviewed_notebooks': 7,
    'retained_notebooks_refreshed': 5,
    'retained_datasets_refreshed': 11,
    'next_research_lead': 'Q-FT paired raw outputs and producer code; no local training scheduled',
    'protected_files_match_prior_verification': True,
})
write(ROOT / 'research_state.json', state)
print(json.dumps(state, ensure_ascii=False, indent=2))
