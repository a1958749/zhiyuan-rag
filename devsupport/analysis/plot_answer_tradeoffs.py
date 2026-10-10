"""Plot complete DEV12 AI ratings without hiding generation failures."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    """Export observed scores and input hashes; make no new quality judgments."""
    base = Path(__file__).resolve().parents[1] / 'reports'
    directory = base / 'dev-generation-v2-ai-judge-v4'
    comparison_path = base / 'dev-ai-quality-comparison-v4.json'
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    for name, expected in comparison['input_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('judge artifacts differ from verified paired comparison')
    summary_path = directory / 'summary-012.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    methods = ['B0', 'H0', 'H1', 'C']
    if not summary['complete'] or summary['judged_questions'] != 12 or set(summary['methods']) != set(methods):
        raise ValueError('expected complete four-method DEV12 AI assessment')
    for name in methods:
        row = summary['methods'][name]
        if (row['attempted'] != 12 or row['judge_missing'] != 0
                or row['conditional_quality_n'] + row['operational_failures'] != 12
                or not math.isclose(row['attempted_utility_mean'],
                                    row['conditional_quality_mean'] * row['conditional_quality_n'] / 12)
                or not all(0 <= row[key] <= 3 for key in ('conditional_quality_mean', 'attempted_utility_mean'))):
            raise ValueError('score scale or denominator identity changed')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#fafbfc')
    x = np.arange(len(methods))
    valid = [summary['methods'][name]['conditional_quality_n'] for name in methods]
    failures = [summary['methods'][name]['operational_failures'] for name in methods]
    bars = axes[0].bar(x, valid, width=0.6, color='#137b91', label='Validated output')
    axes[0].bar_label(bars, labels=[f'{n}/12' for n in valid], label_type='center', color='white')
    bars = axes[0].bar(x, failures, bottom=valid, width=0.6, color='#d68a62', label='Operational failure')
    axes[0].bar_label(bars, labels=[str(n) if n else '' for n in failures], label_type='center')
    axes[0].set_ylim(0, 15.5)
    axes[0].set_yticks([0, 3, 6, 9, 12])
    axes[0].set_ylabel('Recorded attempts')
    axes[0].set_title('Output validity is not correctness\nAll 12 attempted questions per method', loc='left', pad=16)
    axes[0].legend(frameon=False, ncol=2, fontsize=9, loc='upper left')

    for offset, key, label, color in [
            (-0.18, 'conditional_quality_mean', 'Successful outputs only', '#8ca3b8'),
            (0.18, 'attempted_utility_mean', 'All attempts; failures = 0', '#137b91')]:
        values = [summary['methods'][name][key] for name in methods]
        bars = axes[1].bar(x + offset, values, width=0.34, color=color, label=label)
        axes[1].bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
    axes[1].set_ylim(0, 3.55)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_ylabel('Mean AI score (0 to 3)')
    axes[1].set_title('Including failures changes the ranking\nFive rubric dimensions, equal weight', loc='left', pad=16)
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    for axis in axes:
        axis.set_xticks(x, methods)
        axis.set_facecolor('#fafbfc')
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('DevSupport Copilot | Reliability changes the choice', x=0.065, y=0.97,
                 ha='left', fontsize=18, fontweight='bold')
    fig.text(0.065, 0.135, 'DEV12 enriched diagnostic sample. One generation run per method; '
             'AI assessment, not human or TEST scores.', fontsize=10)
    fig.text(0.065, 0.09, 'Same gateway route; underlying model identity unverified. '
             'Historical provider/schema failures remain included.', fontsize=10)
    fig.text(0.065, 0.045, 'Original scores retained, including a flagged reference concern. '
             'Point estimates do not establish causal or population gains.', fontsize=10, color='#88451f')
    fig.subplots_adjust(left=0.065, right=0.98, top=0.79, bottom=0.25, wspace=0.24)
    output = base / 'figures'
    output.mkdir(exist_ok=True)
    stem = output / 'dev-answer-reliability'
    for suffix in ('.png', '.svg'):
        fig.savefig(stem.with_suffix(suffix), dpi=160)
    plt.close(fig)
    proof = {'scope': 'Complete DEV12 v4-only AI assessment; original operational failures included',
             'input_sha256': {p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (summary_path, comparison_path)},
             'validated_judge_inputs': comparison['input_sha256'],
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (stem.with_suffix('.png'), stem.with_suffix('.svg'))}}
    stem.with_suffix('.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
