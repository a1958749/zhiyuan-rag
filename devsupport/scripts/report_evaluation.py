"""Generate offline JSON/Markdown summaries and optional blinded human-review30 packets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'devsupport/src'))


def main():
    """Use only recorded results; never infer human/AI-judge quality labels."""
    from devsupport.data import load_questions, sha256_file
    from devsupport.evaluate import digest, make_record, read_json, write_new
    from devsupport.report import markdown, paired, summarize
    from devsupport.review import blind_review
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', action='store_true')
    args = parser.parse_args()
    run = read_json(args.run / 'run.json')
    profiles = run['profiles']
    if (not isinstance(profiles, list) or not profiles
            or any(not isinstance(name, str) or not name.replace('-', '').replace('_', '').isalnum()
                   for name in profiles) or len(set(profiles)) != len(profiles)):
        raise ValueError('profiles must be unique valid names')
    variants, summaries, contract = {}, {}, None
    for name in profiles:
        path = args.run / (name + '.jsonl')
        manifest_path = path.with_suffix('.manifest.json')
        if not path.exists() and not manifest_path.exists():
            summaries[name] = summarize([], run['query_ids'], allow_partial=True)
            continue
        meta = read_json(manifest_path)
        this_contract = meta['evaluation_contract']
        if digest(this_contract) != run['contract_sha256'] or any(meta[k] != v for k, v in run.items()):
            raise ValueError('mixed evaluation artifact versions')
        contract = this_contract
        frozen_profiles = contract.get('profiles', {})
        settings = frozen_profiles.get(name)
        if (not set(profiles) <= set(frozen_profiles) or not isinstance(settings, dict)
                or not settings.get('retrieval') or meta.get('profile') != name
                or meta.get('settings') != settings
                or meta.get('variant') != (settings.get('reranker') or settings['retrieval'])):
            raise ValueError('profile provenance mismatch')
        if not path.exists():
            summaries[name] = summarize([], run['query_ids'], allow_partial=True)
            continue
        if sha256_file(contract['paths']['queries']) != contract['artifacts']['queries']:
            raise ValueError('offline query annotations changed')
        questions = {q.query_id: q for q in load_questions(contract['paths']['queries'])}
        if run.get('controls'):
            from devsupport.review import load_controls
            from devsupport.types import Question
            if sha256_file(contract['paths']['controls']) != contract['artifacts']['controls']:
                raise ValueError('control annotations changed')
            controls = load_controls(contract['paths']['controls'])
            questions = {r['query_id']: Question(r['query_id'], '', r['query'], '', r.get('nuggets', []), {})
                         for r in controls}
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        for row in rows:
            # Historical producers omit variant, including operational-error results.
            # When recorded, it must agree with the frozen method, even on error rows.
            if 'variant' in row['result'] and row['result']['variant'] != meta['variant']:
                raise ValueError('result variant provenance mismatch')
            if row != make_record(questions[row['query_id']], row['result'], row['elapsed_seconds'], row['error']):
                raise ValueError('result integrity mismatch')
        variants[name] = rows
        summaries[name] = summarize(rows, run['query_ids'], allow_partial=True)
        summaries[name]['latency_scope'] = run['phase']
    comparisons = {}
    if variants:
        baseline = next(iter(variants))
        for name, rows in variants.items():
            if name != baseline:
                ids = [r['query_id'] for r in rows]
                same = ids == [r['query_id'] for r in variants[baseline]]
                comparisons[f'{name}-minus-{baseline}'] = paired(variants[baseline], rows, contract['seeds'][0]) \
                    if same else {'unavailable': 'different attempted cohorts; no successful-only selection'}
    args.output.mkdir(parents=True, exist_ok=True)
    write_new(args.output / 'summary.json', {'run': run, 'methods': summaries, 'paired_bootstrap': comparisons})
    with (args.output / 'summary.md').open('x', encoding='utf-8') as handle:
        handle.write(markdown(summaries))
    if args.review:
        if run['phase'] != 'pipeline' or len(run['query_ids']) != 30 or not all(s['complete'] for s in summaries.values()):
            raise ValueError('human review requires complete preregistered30 pipeline outputs')
        from devsupport.index import CorpusIndex
        if sha256_file(Path(contract['paths']['index']) / 'manifest.json') != contract['artifacts']['index']:
            raise ValueError('review corpus version changed')
        index = CorpusIndex.load(contract['paths']['index'])
        packets, mapping = blind_review([questions[i] for i in run['query_ids']], variants,
                                         index.corpus, contract['seeds'][0])
        write_new(args.output / 'human-review30.json', {'instructions': 'Human ratings only; leave unreviewed fields null. '
                  'Assess correctness, citation support, completeness and appropriate refusal. '
                  'Any AI-assisted inspection must be stored separately.', 'questions': packets})
        write_new(args.output / 'sealed-method-mapping.json', mapping)
    print('Report written:', args.output)


if __name__ == '__main__':
    main()
