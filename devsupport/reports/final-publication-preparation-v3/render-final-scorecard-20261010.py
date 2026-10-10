"""Render complete corrective TEST reports offline; never select methods from TEST."""
import importlib.util
from itertools import combinations
import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('final_publication', WORK / 'publish-final-evidence-20261010.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)
from devsupport.report import paired, summarize  # noqa: E402
import control_decisions  # noqa: E402

ROOT, REPORTS = publication.ROOT, publication.REPORTS
OUTPUT = REPORTS / 'public-final-scorecard-v2'
METHODS = ['B0', 'B1', 'H0', 'H1', 'C', 'U', 'R1', 'R2', 'Q', 'QAlways']
NOTES = [
    'H0 was selected from DEV before the first TEST; TEST never changes the default method.',
    'Full=H0 and FullMinusHybridBranch=B0 reuse the same records, not independent generations.',
    'Corrective parser-source v2 retains the original v1 lock and 750 records; repeated local TEST is not independent.',
    'Ranking metrics measure qrels/nugget retrieval; packed-context qrel proxies are not answer accuracy.',
    'AI answer ratings use the fixed first TEST30 and the same three-method blinded queue; human ratings=0.',
    'Control labels are provisional; unsupported-control answer rate is not measured hallucination rate.',
    'Single runs, exploratory paired 2000-sample percentile bootstrap; no multiplicity correction.',
    'All failures stay in denominators. Timing scopes differ by manifest phase, and shared pacing is included.',
    'Response usage may omit retries/unknown transport attempts; configured-currency cost may remain null.',
    'U uses the shared Qwen embedding and native 45000 proxy-token candidate budget, not stock Granite.',
    '4000 cl100k proxy-token whole-chunk context budget is not the gateway model exact token count.',
    'Requested/reported gateway names differ; underlying model identity is unverified.',
]


def read_report(relative, bindings, required_inputs=()):
    """Verify every consumed input hash from a previously completed audit."""
    report = publication.read(REPORTS / relative / 'summary.json', bindings)
    recorded_inputs = report.get('input_sha256')
    if not isinstance(recorded_inputs, dict) or not recorded_inputs:
        raise ValueError('report requires nonempty byte bindings')
    if not {str(Path(path).resolve()) for path in required_inputs} <= set(recorded_inputs):
        raise ValueError('report input coverage omits required original records')
    for path, expected in recorded_inputs.items():
        if (not isinstance(path, str) or not path or not isinstance(expected, str) or len(expected) != 64
                or any(char not in '0123456789abcdef' for char in expected)):
            raise ValueError('report bindings require canonical SHA256 strings')
        publication.bind(path, bindings, expected=expected)
    return report


def validate_observations(report, queue, locked, ids, methods):
    """Reject a complete report substituted from another same-contract queue."""
    phase = 'retrieve-only' if queue == 'adaptive' else 'pipeline'
    if (report.get('complete') is not True or report.get('contract_sha256') != locked
            or report.get('split') != 'test' or report.get('phase') != phase
            or report.get('controls') is not (queue == 'controls')
            or report.get('query_ids') != ids or not isinstance(report.get('methods'), dict)
            or set(report['methods']) != set(methods)):
        raise ValueError('provider observations queue identity differs')


def number(value, multiplier=1):
    """Display missing values explicitly and keep scores distinct from percentages."""
    return '不可用' if value is None else f'{value * multiplier:.4f}'


def render(scorecard):
    """Render a concise scorecard from complete recorded aggregates."""
    lines = ['# DevSupport Copilot 最终留出评测', '',
             '正式默认方案：**H0（Dense + 普通 BM25 + RRF）**。所有 TEST 结果用于验证预先选择，未用于重新调参。',
             '', f"冻结合同：`{scorecard['final_contract_sha256']}`。人工核验/评分：**0**。", '',
             '## 检索 TEST150', '',
             '| 方法 | 尝试/错误 | Coverage@10 % | α-nDCG@10 | Recall@50 % | MRR | 装箱 nugget 覆盖 % | p50 / p95 秒 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        row = scorecard['retrieval']['methods'][method]
        metrics = row['metrics_prepack']
        context = scorecard['context']['methods'][method]['metrics']['context_nugget_coverage']['mean']
        timing = row['elapsed_seconds']
        lines.append(f"| {method} | {row['attempted']}/{row['errors']} | "
                     f"{number(metrics['coverage@10'], 100)} | {number(metrics['alpha_ndcg@10'])} | "
                     f"{number(metrics['recall@50'], 100)} | {number(metrics['mrr'])} | "
                     f"{number(context, 100)} | {number(timing['p50'])} / {number(timing['p95'])} |")
    lines += ['', '## 增量与消融', '',
              '| 条件差值 | Coverage@10 差值 pp [95% 区间] | α-nDCG@10 差值 [95% 区间] |',
              '|---|---:|---:|']
    for candidate, baseline in [('H0', 'B0'), ('H1', 'H0'), ('R1', 'H1'), ('R2', 'H1'),
                                ('C', 'B0'), ('Q', 'H0'), ('QAlways', 'H0')]:
        comparison = scorecard['retrieval']['paired'][f'{candidate}-minus-{baseline}']['prepack']
        values = []
        for key, multiplier in [('coverage@10', 100), ('alpha_ndcg@10', 1)]:
            interval = comparison[key]
            values.append(f"{number(interval['mean_delta'], multiplier)} "
                          f"[{number(interval['ci_low'], multiplier)}, {number(interval['ci_high'], multiplier)}]")
        lines.append(f"| {candidate} − {baseline} | {' | '.join(values)} |")
    lines += ['', '## 回答 TEST30：AI 评分', '',
              '| 方法 | 尝试 | 有效回答 | 运行失败 | 含失败效用 / 3 | 回答覆盖 % | 回答子集 Faithfulness / 3 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method, row in scorecard['answer_quality']['methods'].items():
        lines.append(f"| {method} | {row['attempted']} | {row['answers']} | {row['operational_failures']} | "
                     f"{number(row['attempted_utility_mean'])} | {number(row['answer_coverage'], 100)} | "
                     f"{number(row['answered_dimensions']['faithfulness'])} |")
    lines += ['', '## 产品控制 TEST30：暂定标签代理指标', '',
              '| 方法 | 尝试 | 运行失败 | 决策匹配率 | 不支持题回答 | 可回答题误拒答 |',
              '|---|---:|---:|---:|---:|---:|']
    for method, row in scorecard['controls']['methods'].items():
        values = []
        for key in ['decision_accuracy', 'unsupported_control_answer_rate', 'false_refusal_insufficient_rate']:
            item = row['metrics'][key]
            values.append(f"{item['numerator']}/{item['denominator']} ({number(item['rate'], 100)}%)")
        lines.append(f"| {method} | {row['attempted']} | {row['operational_errors']} | {' | '.join(values)} |")
    lines += ['', '## 调用与费用观测', '',
              '| 队列 / 方法 | 阶段调用 | 已知传输尝试 | 未知尝试数的调用 | 响应用量完整 | 估算费用 |',
              '|---|---:|---:|---:|---|---:|']
    for queue, report in scorecard['provider_observations'].items():
        for method, row in report['methods'].items():
            item = row['accounting']
            lines.append(f"| {queue} / {method} | {item['stage_invocations']} | {item['known_transport_attempts']} | "
                         f"{item['invocations_missing_attempts']} | {item['usage_complete_for_recorded_attempts']} | "
                         f"{number(item['estimated_provider_cost_total'])} |")
    lines += ['', '费用不可用时不按共享余额反推，也不将平台额度单位称为实际货币。完整逐题响应 token、阶段延迟、'
              '失败分类和配对区间保存在相邻 JSON 与公开逐题视图中。AI judge 调用的 trace 保存在公开 assessment 队列，'
              '上表仅为三种产品执行队列，不合并成账户总用量。', '', '## 解释限制', '']
    lines += [f'- {note}' for note in NOTES]
    return '\n'.join(lines) + '\n'


def main():
    """Validate complete scopes and publish new byte-scanned reports without model calls."""
    if OUTPUT.exists():
        raise FileExistsError('scorecard output exists; never overwrite')
    contract, locked, items, bindings = publication.collect()
    rows_by_method = {}
    for label, path, method, ids in items:
        if not label.startswith(('test-local-', 'test-native-', 'test-rerank-', 'test-adaptive-')):
            continue
        raw = publication.bind(path, bindings)
        rows = [json.loads(line) for line in raw.splitlines()]
        if [row['query_id'] for row in rows] != ids:
            raise ValueError('scorecard ordered denominator mismatch')
        for row in rows:
            if row['record_sha256'] != publication.digest({k: v for k, v in row.items() if k != 'record_sha256'}):
                raise ValueError('scorecard original digest mismatch')
        rows_by_method[method] = rows
    if set(rows_by_method) != set(METHODS):
        raise ValueError('all ten retrieval conditions required')
    retrieval_paths = [path for label, path, _, _ in items
                       if label.startswith(('test-local-', 'test-native-', 'test-rerank-', 'test-adaptive-'))]
    context = read_report('test-context-v2', bindings, required_inputs=retrieval_paths)
    if (context.get('complete') is not True or context['contract_sha256'] != locked
            or context['query_ids'] != contract['split_ids']['test'] or set(context['methods']) != set(METHODS)):
        raise ValueError('complete same-contract TEST150 context required')
    quality = read_report('test-answer-quality-v2', bindings, required_inputs=[
        path for label, path, _, _ in items if label.startswith('test-generation-')])
    controls = publication.read(REPORTS / 'test-controls-report-v2/summary.json', bindings)
    if controls != control_decisions.audit(REPORTS / 'test-controls-v2'):
        raise ValueError('control scorecard differs from fresh complete audit')
    for kind in ('inputs', 'implementation'):
        for path, expected in controls['bindings'][kind].items():
            publication.bind(path, bindings, expected=expected)
    observations = {}
    for queue in ('generation', 'controls', 'adaptive'):
        members = [(path, method, ids) for label, path, method, ids in items if label.startswith(f'test-{queue}-')]
        report = read_report(f'test-{queue}-accounting-v2', bindings,
                             required_inputs=[path for path, _, _ in members])
        validate_observations(report, queue, locked, members[0][2], [method for _, method, _ in members])
        observations[queue] = report
    scorecard = {'schema': 'devsupport.final-scorecard.v2', 'complete': True, 'final_contract_sha256': locked,
                 'selected_method': 'H0', 'aliases': contract['config']['final_method_aliases'],
                 'retrieval': {'methods': {m: summarize(rows_by_method[m], contract['split_ids']['test']) for m in METHODS},
                               'paired': {f'{c}-minus-{b}': paired(rows_by_method[b], rows_by_method[c], contract['seeds'][0])
                                          for b, c in combinations(METHODS, 2)}},
                 'context': context, 'answer_quality': quality, 'controls': controls,
                 'provider_observations': observations, 'human_ratings': 0, 'paid_calls': 0, 'notes': NOTES,
                 'input_sha256': bindings, 'script_sha256': publication.checksum(Path(__file__))}
    serialized = json.dumps(scorecard, ensure_ascii=False, allow_nan=False).encode('utf-8')
    rendered = render(scorecard).encode('utf-8')
    credentials = tuple(publication.load_credential(ROOT / '.secrets' / name)
                        for name in ('deepseek.dpapi', 'internal.dpapi'))
    for raw in (serialized, rendered):
        if any(value.encode() in raw for value in credentials) or publication.public_module.KEY_PATTERN.search(raw):
            raise ValueError('credential literal or pattern in final scorecard')
    for path, expected in bindings.items():
        if publication.checksum(path) != expected:
            raise ValueError('scorecard consumed source changed')
    publication.verify_lock(publication.LOCK, publication.build_contract(publication.CONFIG, ROOT))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, raw in [('summary.json', serialized), ('summary.md', rendered)]:
        with (OUTPUT / name).open('xb') as handle:
            handle.write(raw)
        if (OUTPUT / name).read_bytes() != raw:
            raise ValueError('scorecard saved bytes differ')
    print(json.dumps({'output': str(OUTPUT), 'retrieval_methods': len(METHODS), 'questions': 150,
                      'generation_questions': 30, 'control_questions': 30, 'paid_calls': 0}))


if __name__ == '__main__':
    main()
