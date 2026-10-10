"""Plot immutable DEV53 adaptive comparisons without implying answer-quality gains."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def sha256(path: Path) -> str:
    """Bind the exact input and exported bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Validate audited artifacts and export the recorded paired intervals."""
    base = Path(__file__).resolve().parents[1] / 'reports'
    run = base / 'dev-adaptive-h0-v2'
    audit_path = base / 'dev-adaptive-h0-v2-audit/audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    names = ['H0', 'Q', 'QAlways']
    if (audit['run']['split'] != 'dev' or audit['run']['phase'] != 'retrieve-only'
            or len(audit['run']['query_ids']) != 53 or set(audit['methods']) != set(names)
            or sha256(run / 'run.json') != audit['bindings']['run_sha256']):
        raise ValueError('expected complete audited DEV53 retrieve-only run')
    for name in names:
        binding = audit['bindings']['profiles'][name]
        if (sha256(run / (name + '.jsonl')) != binding['rows_sha256']
                or sha256(run / (name + '.manifest.json')) != binding['manifest_sha256']
                or not audit['methods'][name]['complete']
                or audit['methods'][name]['attempted'] != 53):
            raise ValueError('audited input changed or incomplete cohort')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.patch.set_facecolor('#fafbfc')
    styles = [('packed', 'coverage@50', 'Packed nugget coverage', '#137b91', -0.12),
              ('prepack', 'coverage@10', 'Top-10 nugget coverage', '#bc7956', 0.12)]
    for section, metric, label, color, offset in styles:
        for index, name in enumerate(['Q', 'QAlways']):
            row = audit['comparisons'][name + '-minus-H0']['paired_bootstrap_all53'][section][metric]
            mean, low, high = [100 * row[k] for k in ('mean_delta', 'ci_low', 'ci_high')]
            if not low <= mean <= high or row['n'] != 53:
                raise ValueError('invalid paired interval')
            axes[0].errorbar(mean, index + offset, xerr=[[mean - low], [high - mean]],
                             fmt='o', color=color, capsize=5, label=label if index == 0 else None)
    axes[0].axvline(0, color='#566477', linestyle='--', linewidth=1)
    axes[0].set_yticks([0, 1], ['Selective Q', 'Always QAlways'])
    axes[0].set_xlabel('Difference from H0 (percentage points)')
    axes[0].set_title('No stable coverage improvement established\nPaired 95% bootstrap intervals, 53 questions',
                      loc='left', pad=18)
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    axes[0].set_ylim(-0.5, 1.5)

    calls = [audit['methods'][name]['provider_calls'] for name in names]
    attempts = [audit['methods'][name]['provider_attempts_recorded'] for name in names]
    x = list(range(3))
    axes[1].bar([i - 0.18 for i in x], calls, width=0.34, color='#137b91', label='Logical provider stages')
    axes[1].bar([i + 0.18 for i in x], attempts, width=0.34, color='#8ca3b8', label='Recorded transport attempts')
    for bars in axes[1].containers:
        axes[1].bar_label(bars, padding=3)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel('Recorded counts across DEV53')
    axes[1].set_ylim(0, max(calls + attempts) * 1.3 + 1)
    axes[1].set_title('Expansion adds provider work\nRetrieval only; no gate or answer generation', loc='left', pad=18)
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    for axis in axes:
        axis.set_facecolor('#fafbfc')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x' if axis == axes[0] else 'y', alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle('DevSupport Copilot | More retrieval is not automatically better',
                 x=0.075, y=0.96, ha='left', fontsize=17, fontweight='bold')
    fig.text(0.075, 0.12, 'DEV53, not TEST or answer accuracy. All four plotted intervals include zero.', fontsize=10)
    fig.text(0.075, 0.075, '4,000 proxy-token whole-chunk packing. Counts include failures; billing and currency unverified.',
             fontsize=10)
    fig.text(0.075, 0.03, 'Selective and always use separate model calls; their contrast does not isolate trigger-policy causality.',
             fontsize=9, color='#88451f')
    fig.subplots_adjust(left=0.12, right=0.98, top=0.75, bottom=0.26, wspace=0.32)
    output = base / 'figures'
    output.mkdir(exist_ok=True)
    stem = output / 'dev-adaptive-tradeoffs'
    for suffix in ('.png', '.svg'):
        fig.savefig(stem.with_suffix(suffix), dpi=160)
    plt.close(fig)
    proof = {'scope': 'complete audited DEV53 retrieve-only; no new measurements or quality judgments',
             'input': str(audit_path), 'input_sha256': sha256(audit_path),
             'verified_bindings': audit['bindings'], 'script_sha256': sha256(Path(__file__)),
             'outputs': {stem.with_suffix(s).name: sha256(stem.with_suffix(s)) for s in ('.png', '.svg')}}
    stem.with_suffix('.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
