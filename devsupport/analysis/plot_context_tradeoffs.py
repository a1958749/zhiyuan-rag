"""Plot observed DEV context metrics and explicitly selected diagnostic examples."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    """Export a figure and its input hashes without touching runtime or TEST data."""
    base = Path(__file__).resolve().parents[1] / 'reports'
    summary_path = base / 'context-quality-dev-v1/summary.json'
    cases_path = base / 'context-failure-cases-dev-20261009.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    case_hashes = {name.replace('\\', '/'): value for name, value in cases['source_sha256'].items()}
    if case_hashes['reports/context-quality-dev-v1/summary.json'] != digest:
        raise ValueError('case report and context summary are from different audits')
    methods = ['B0', 'H0', 'H1', 'C', 'R1', 'R2']
    keys = ['context_nugget_coverage', 'packed_context_doc_precision']
    if len(summary['query_ids']) != 53 or any(
            summary['methods'][method]['metrics'][key]['n'] != 53
            or summary['methods'][method]['metrics'][key]['missing'] != 0
            for method in methods for key in keys):
        raise ValueError('expected complete DEV53 macro metrics')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#fafbfc')
    x = np.arange(len(methods))
    for offset, key, label, color in [
            (-0.18, keys[0], 'Nugget coverage', '#137b91'),
            (0.18, keys[1], 'Positive-doc fraction', '#e8a34b')]:
        values = [summary['methods'][m]['metrics'][key]['mean'] * 100 for m in methods]
        bars = axes[0].bar(x + offset, values, width=0.34, color=color, label=label)
        axes[0].bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    axes[0].set_xticks(x, methods)
    axes[0].set_ylim(0, 50)
    axes[0].set_title('Coverage and evidence density differ\nAll 53 DEV questions', loc='left', pad=16)
    axes[0].set_ylabel('Macro mean (%)')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')

    selected = ['77561864', '77844073', '78023380', '78485621']
    case_by_id = {c['query_id']: c for c in cases['cases']}
    if set(case_by_id) != set(selected):
        raise ValueError('diagnostic example cohort changed')
    x = np.arange(len(selected))
    for index, (method, color) in enumerate([
            ('H0', '#8ca3b8'), ('H1', '#3e6a91'), ('C', '#137b91'), ('R1', '#b06954')]):
        values = [case_by_id[q]['profiles'][method]['quality'][keys[0]] * 100 for q in selected]
        axes[1].bar(x + (index - 1.5) * 0.2, values, width=0.18, color=color, label=method)
    axes[1].set_xticks(x, ['PDF to agent', 'Chroma metadata', 'Parser import', 'chain.invoke'])
    axes[1].tick_params(axis='x', labelsize=9)
    axes[1].set_ylim(0, 118)
    axes[1].set_ylabel('Packed nugget coverage (%)')
    axes[1].set_title('A retrieved document can be lost during packing\nFour post-hoc diagnostic examples',
                      loc='left', pad=16)
    axes[1].legend(frameon=False, ncol=4, fontsize=9, loc='upper left')
    for axis in axes:
        axis.set_facecolor('#fafbfc')
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)

    fig.suptitle('DevSupport Copilot | What actually fits in the context', x=0.065, y=0.97,
                 ha='left', fontsize=18, fontweight='bold')
    fig.text(0.065, 0.13, 'Same canonical corpus, ranking artifacts and 4,000 cl100k proxy-token budget.',
             fontsize=10)
    fig.text(0.065, 0.085, 'Qrels-based diagnostics. Unjudged documents may still be relevant. '
             'Bars show point estimates, not significance.', fontsize=10)
    fig.text(0.065, 0.04, 'DEV only; selected examples are not representative. '
             'These are not answer accuracy, human ratings or TEST results.', fontsize=10, color='#88451f')
    fig.subplots_adjust(left=0.065, right=0.98, top=0.79, bottom=0.24, wspace=0.25)
    output = base / 'figures'
    output.mkdir(exist_ok=True)
    stem = output / 'dev-context-tradeoffs'
    for suffix in ('.png', '.svg'):
        fig.savefig(stem.with_suffix(suffix), dpi=160)
    plt.close(fig)
    proof = {'scope': 'DEV53 context metrics plus four post-hoc diagnostic examples',
             'input_sha256': {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (summary_path, cases_path)},
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (stem.with_suffix('.png'), stem.with_suffix('.svg'))}}
    stem.with_suffix('.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
