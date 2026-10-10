"""Prepare an offline, unscored human-review handoff from a verified generation run."""
import argparse
import hashlib
import json
from pathlib import Path

from answer_judge import build_packets, load_run, source_contract
from human_review import render_html
from devsupport.data import sha256_file
from devsupport.evaluate import digest, read_json, write_new


def prepare(run: Path, output: Path, split: str) -> dict:
    """Validate the complete run before publishing separate share/private artifacts.

    Args:
        run: Existing four-profile generation directory with frozen provenance.
        output: New handoff directory. Existing paths are never overwritten.
        split: Explicit dev rehearsal or test acceptance cohort.
    """
    if output.exists():
        raise FileExistsError('human review output must be a new directory')
    if split not in ('dev', 'test'):
        raise ValueError('explicit dev or test split required')
    questions, variants, corpus, contract, inputs = load_run(run, split)
    if split == 'test':
        frozen_test_ids = read_json(contract['paths']['split'])['test_ids']
        expected = frozen_test_ids[:30]
        if (len(questions) != 30 or contract['split_ids']['test'] != frozen_test_ids
                or contract['generation_ids'] != expected or [q.query_id for q in questions] != expected):
            raise ValueError('TEST human review requires the frozen first 30 TEST questions in order')
    seed = contract.get('seeds', [20261008])[0]
    packets, mapping = build_packets(questions, variants, corpus, seed)
    raw = (json.dumps(packets, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    packet_hash = hashlib.sha256(raw).hexdigest()
    page = render_html(packets, packet_hash)
    scope = 'TEST acceptance cohort' if split == 'test' else 'DEV rehearsal; not TEST acceptance'
    manifest = {
        'schema': 'devsupport.human-review-handoff.v1', 'split': split, 'scope': scope,
        'query_ids': [q.query_id for q in questions], 'seed': seed,
        'packet_sha256': packet_hash, 'mapping_sha256': digest(mapping),
        'generation_contract_sha256': digest(contract), 'inputs': inputs,
        'sources': {**source_contract(),
                    'analysis/prepare_human_review.py': sha256_file(Path(__file__)),
                    'analysis/human_review.py': sha256_file(Path(__file__).with_name('human_review.py'))},
        'human_reviews_completed': 0,
        'operational_failures': sum(r['operational_failure'] for p in packets for r in p['responses']),
        'review_response_count': sum(len(p['responses']) for p in packets),
        'note': 'Preparation is not human review. Share only the share directory; keep private files sealed.',
    }
    output.mkdir(parents=True, exist_ok=False)
    share, private = output / 'share', output / 'private'
    share.mkdir()
    private.mkdir()
    (share / 'packets.json').write_bytes(raw)
    (share / 'review.html').write_text(page, encoding='utf-8')
    (share / 'README.txt').write_text(
        f'DevSupport 人工盲评 / {scope}\n'
        '打开 review.html。评分初始为空；请亲自阅读答案和证据，再填写评分、署名和人工声明。\n'
        '服务失败条目单独计数，不填写答案质量分。请定期导出 JSON，并返回最终导出文件。\n'
        '这份材料尚无人工评分。完成表单不代表生成成功率或项目整体通过验收。\n'
        f'题数：{len(questions)}；匿名回答数：{manifest["review_response_count"]}。\n'
        f'评审包 SHA-256：{packet_hash}\n', encoding='utf-8')
    write_new(private / 'sealed-mapping.json', mapping)
    # Last file is the completion marker for artifact preparation, never for human review.
    write_new(private / 'manifest.json', manifest)
    return manifest


def main() -> None:
    """Prepare review files only; never access provider credentials or run an AI judge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    args = parser.parse_args()
    manifest = prepare(args.run, args.output, args.split)
    print(f'Prepared {len(manifest["query_ids"])} questions; human reviews completed: 0.')
    print(f'Share only: {args.output / "share"}')


if __name__ == '__main__':
    main()
