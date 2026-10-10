"""Generate an offline human blind-review form; never generate human ratings."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PACKET_FIELDS = {'query_id', 'question', 'reference_answer', 'nuggets', 'gold_context', 'responses'}
RESPONSE_FIELDS = {'review_id', 'answer', 'decision', 'citations', 'context',
                   'missing_information'}
OPTIONAL_RESPONSE_FIELDS = {'ratings', 'reviewer_notes', 'operational_failure'}
FORBIDDEN_FIELDS = {'method', 'methods', 'method_mapping', 'sealed', 'sealed_mapping', 'variant', 'model'}
# Reserved metadata terms, not a filter on evidence/answer string contents.
SCORING_TERMS = {'score', 'scores', 'confidence', 'confidences', 'probability', 'probabilities',
                 'logprob', 'logprobs'}


def validate_packets(packets: Any) -> None:
    """Reject malformed, pre-scored, or explicitly unblinded input packets.

    Args:
        packets: JSON value from the caller's explicit packet file.
    """
    def check_json(value: Any) -> None:
        """Reject identifying/scoring metadata recursively, preserving string values."""
        if isinstance(value, dict):
            if any(not isinstance(k, str) or k.lower() in FORBIDDEN_FIELDS for k in value):
                raise ValueError('Packet contains forbidden identifying metadata')
            for key, item in value.items():
                # Split camelCase/acronyms and separators without rewriting the input.
                words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', key)
                words = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', words)
                if SCORING_TERMS.intersection(re.split(r'[^a-z0-9]+', words.lower())):
                    raise ValueError('Packet contains forbidden scoring metadata')
                check_json(item)
        elif isinstance(value, list):
            for item in value:
                check_json(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite JSON numbers are not supported')
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError('Packet must contain JSON values only')

    if not isinstance(packets, list) or not packets:
        raise ValueError('Packets must be a non-empty JSON list')
    check_json(packets)
    queries, responses = set(), set()
    for packet in packets:
        if not isinstance(packet, dict) or set(packet) != PACKET_FIELDS:
            raise ValueError('Packet fields must match the blind-review schema exactly')
        for key in ('query_id', 'question', 'reference_answer'):
            if not isinstance(packet[key], str) or (key != 'reference_answer' and not packet[key].strip()):
                raise ValueError(f'{key} must be text')
        if packet['query_id'] in queries:
            raise ValueError('Duplicate query_id')
        queries.add(packet['query_id'])
        for key in ('nuggets', 'gold_context', 'responses'):
            if not isinstance(packet[key], list):
                raise ValueError(f'{key} must be a list')
        if not packet['responses']:
            raise ValueError('Every question must have responses')
        for response in packet['responses']:
            if (not isinstance(response, dict) or not RESPONSE_FIELDS <= set(response)
                    or set(response) - RESPONSE_FIELDS - OPTIONAL_RESPONSE_FIELDS):
                raise ValueError('Response fields must match the anonymous schema exactly')
            identifier = response['review_id']
            if not isinstance(identifier, str) or not re.fullmatch(r'Q[0-9]+-A[0-9]+', identifier):
                raise ValueError('review_id must be an anonymous Q-number/A-number identifier')
            if identifier in responses:
                raise ValueError('Duplicate review_id')
            responses.add(identifier)
            if not isinstance(response['answer'], str):
                raise ValueError('answer must be exact plain text')
            if response['decision'] is not None and not isinstance(response['decision'], str):
                raise ValueError('decision must be text or null')
            for key in ('citations', 'context', 'missing_information'):
                if not isinstance(response[key], list):
                    raise ValueError(f'{key} must be a list')
            if response.get('ratings') is not None or response.get('reviewer_notes') is not None:
                raise ValueError('Input human ratings and reviewer_notes must be null')
            if 'operational_failure' in response and not isinstance(response['operational_failure'], bool):
                raise ValueError('operational_failure must be a boolean')


STATE_JS = r'''
const DIMENSIONS = ['correctness', 'completeness', 'faithfulness', 'citation_support', 'relevance'];
const SCHEMA = 'devsupport.human-review.v1';
const rows = PACKETS.flatMap(p => p.responses.map(r => ({query_id: p.query_id, review_id: r.review_id,
  skip_reason: r.operational_failure === true ? 'operational_failure' : null})));
function freshState() {
  return {reviewer: '', human_declaration: false, ratings: rows.map(r => ({...r,
    ratings: Object.fromEntries(DIMENSIONS.map(d => [d, null])), reviewer_notes: ''}))};
}
function scored(row) { return DIMENSIONS.every(d => Number.isInteger(row.ratings[d]) &&
  row.ratings[d] >= 0 && row.ratings[d] <= 3); }
function questionDone(s, p) {
  return s.ratings.filter(r => r.query_id === p.query_id).every(r => r.skip_reason !== null || scored(r));
}
function summary(s) {
  const eligible = s.ratings.filter(r => r.skip_reason === null), count = eligible.filter(scored).length;
  return {status: count === eligible.length && s.reviewer.trim() && s.human_declaration ? 'complete' : 'partial',
    scored_responses: count, total_responses: eligible.length,
    skipped_operational_failures: rows.length - eligible.length,
    completed_questions: PACKETS.filter(p => questionDone(s, p)).length,
    total_questions: PACKETS.length};
}
function exportState(s) {
  return {schema: SCHEMA, packet_sha256: PACKET_HASH, timestamp: new Date().toISOString(),
    reviewer: s.reviewer, human_declaration: s.human_declaration,
    ratings: structuredClone(s.ratings), completion: summary(s)};
}
function exactKeys(value, keys) {
  return value !== null && typeof value === 'object' && !Array.isArray(value) &&
    Object.keys(value).sort().join('|') === [...keys].sort().join('|');
}
function validateState(x) {
  const fail = () => { throw new Error('文件格式、评分范围、完成状态或评审包指纹不匹配。'); };
  if (!exactKeys(x, ['schema','packet_sha256','timestamp','reviewer','human_declaration','ratings','completion']) ||
      x.schema !== SCHEMA || x.packet_sha256 !== PACKET_HASH || typeof x.reviewer !== 'string' ||
      typeof x.human_declaration !== 'boolean' || typeof x.timestamp !== 'string' ||
      !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(x.timestamp) ||
      !Number.isFinite(Date.parse(x.timestamp)) || !Array.isArray(x.ratings) || x.ratings.length !== rows.length) fail();
  x.ratings.forEach((r, i) => {
    if (!exactKeys(r, ['query_id','review_id','skip_reason','ratings','reviewer_notes']) ||
        r.query_id !== rows[i].query_id || r.review_id !== rows[i].review_id ||
        r.skip_reason !== rows[i].skip_reason ||
        typeof r.reviewer_notes !== 'string' || !exactKeys(r.ratings, DIMENSIONS) ||
        !DIMENSIONS.every(d => r.ratings[d] === null ||
          (r.skip_reason === null && Number.isInteger(r.ratings[d]) && r.ratings[d] >= 0 && r.ratings[d] <= 3))) fail();
  });
  const s = {reviewer: x.reviewer, human_declaration: x.human_declaration, ratings: structuredClone(x.ratings)};
  const expected = summary(s);
  if (!exactKeys(x.completion, Object.keys(expected)) ||
      !Object.keys(expected).every(k => x.completion[k] === expected[k])) fail();
  return s;
}
'''

PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline';
style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>人工盲评 · DevSupport</title>
<style>
:root{font-family:system-ui,sans-serif;color:#192838;background:#f0f3f6;line-height:1.65}
body{margin:0}main{max-width:1050px;margin:auto;padding:24px}h1,h2,h3{line-height:1.3}
header,section,article,details.rubric{background:white;border:1px solid #ccd5df;border-radius:12px;padding:20px;margin:16px 0}
button,select,input,textarea{font:inherit;border:1px solid #718398;border-radius:6px;padding:7px;background:white;color:inherit}
button{cursor:pointer}button:hover{background:#e8f0fa}button:focus-visible,select:focus-visible{outline:3px solid #207ac2}
input[type=text]{width:min(300px,80%)}textarea{width:100%;box-sizing:border-box;min-height:80px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f7f9fb;padding:12px;border-radius:6px}
.toolbar,.ratings{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.ratings label{display:grid;gap:4px}
nav{position:sticky;top:0;background:#f0f3f6;padding:12px 0;z-index:1}summary{cursor:pointer;font-weight:600}
table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd5df;padding:8px;text-align:left;vertical-align:top}
.scroll{overflow-x:auto}.muted{color:#495f73}.status{font-weight:700;color:#14568a}#notice{color:#82410c}
</style></head><body><main>
<header><h1>人工盲评</h1>
<p>请独立阅读问题、参考要点及证据，再逐一评价匿名回答。全部分数初始为空；0 是有效分数，不代表未评分。
请勿让模型代填，也不要猜测方法身份。本页只收集人工评审，生成页面本身不产生任何人工评分。</p>
<p>评审范围为此包的全部题目。每个有效回答需填写五项 0–3 整数评分；如不确定，请留空并在备注说明。
标记为生成/服务失败的回答会单独记录跳过原因，不参与质量评分，也不会自动赋分。
空回答或拒答按实际正确性、完整性与证据质量评价，不因措辞流畅而加分。</p>
<div><label>评审者姓名或化名 <input id="reviewer" type="text" autocomplete="off"></label></div>
<p><label><input id="declaration" type="checkbox"> 我声明：这些评分由我本人独立阅读并作出，未由 AI 代填。</label></p>
<p class="muted">数据仅保存在本浏览器；浏览器可能限制本地文件存储，请及时导出备份。导入只能校验格式与指纹，无法核实人员身份。</p>
<p id="fingerprint" class="muted"></p><p id="notice" role="status" aria-live="polite"></p>
</header>
<details class="rubric"><summary>评分标尺（点击展开）</summary><div class="scroll"><table>
<thead><tr><th>维度</th><th>0</th><th>1</th><th>2</th><th>3</th></tr></thead><tbody>
<tr><th>正确性</th><td>核心结论错误或误导</td><td>有关键错误，仅少量正确</td><td>核心正确，存在次要错误</td><td>结论与细节正确，适当表达不确定性</td></tr>
<tr><th>完整性</th><td>未覆盖必要要点</td><td>遗漏多数重要要点</td><td>覆盖主要要点，少量遗漏</td><td>覆盖全部必要要点与限制</td></tr>
<tr><th>忠实性</th><td>关键说法与提供证据冲突或无依据</td><td>多个重要说法无证据支持</td><td>主要说法有依据，仅轻微外推</td><td>所有实质说法均受提供证据支持</td></tr>
<tr><th>引用支持</th><td>需要引用但缺失、无效或不支持核心说法</td><td>少数引用正确，大量错配或遗漏</td><td>主要说法有正确引用，少量遗漏</td><td>所需引用齐全、可定位且支持对应说法</td></tr>
<tr><th>相关性</th><td>未回应问题</td><td>部分相关，大量偏题</td><td>直接回应，少量冗余</td><td>紧扣问题，必要且清晰</td></tr>
</tbody></table></div><p>对于合理拒答或澄清：依据问题和证据判断是否应拒答、缺失信息是否说明充分。
若确实没有需要外部证据支持的实质断言，引用支持可评 3，并在备注说明依据。参考答案可能不完备，发现冲突请记录。</p></details>
<nav><div class="toolbar"><button id="prev" type="button">上一题</button>
<label>题目 <select id="question-nav"></select></label><button id="next" type="button">下一题</button>
<span id="progress" class="status" aria-live="polite"></span></div></nav>
<section id="question"></section><div id="responses"></div>
<section><div class="toolbar"><button id="export" type="button">导出部分评审 JSON</button>
<label>导入本人备份 <input id="import" type="file" accept="application/json,.json"></label>
<button id="clear" type="button">清除此评审包的本地评分</button></div>
<p>未完成的文件会明确标为 partial。只有全部有效回答的五项评分齐全、评审者非空且人工声明勾选时，才标为 complete。
服务失败的跳过数量会单独导出；complete 仅表示本表评审流程完成，不代表所有回答均生成成功。</p></section>
<noscript>此页面需要 JavaScript 才能填写和导出评分。</noscript>
<script id="packets" type="application/json">__PACKETS__</script>
<script>
'use strict';
const PACKETS = JSON.parse(document.getElementById('packets').textContent);
const PACKET_HASH = '__HASH__';
__STATE_JS__
const STORAGE_KEY = 'devsupport-human-review-v1:' + PACKET_HASH;
const names = ['正确性', '完整性', '忠实性', '引用支持', '相关性'];
const byId = id => document.getElementById(id);
function element(tag, text) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; return el; }
function pretty(value) { return typeof value === 'string' ? value : JSON.stringify(value, null, 2); }
function detail(parent, title, value) {
  const block = element('details'); block.append(element('summary', title), element('pre', pretty(value)));
  parent.append(block);
}
function notify(text) { byId('notice').textContent = text; }
let state = freshState(), index = 0;
try { const saved = localStorage.getItem(STORAGE_KEY); if (saved) state = validateState(JSON.parse(saved)); }
catch { notify('本地记录不可用或校验失败；当前显示空白评分。可导入之前导出的本人备份。'); }
function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(exportState(state))); }
  catch { notify('本地存储不可用，当前评分仅在内存中。请导出 JSON 保存。'); }
}
function updateProgress() {
  const s = summary(state);
  byId('progress').textContent = `已评分 ${s.scored_responses}/${s.total_responses} · 已处理题目 ${s.completed_questions}/${s.total_questions} · 服务失败跳过 ${s.skipped_operational_failures}`;
  byId('export').textContent = s.status === 'complete' ? '导出完整人工评审 JSON' : '导出部分评审 JSON';
  PACKETS.forEach((p, i) => {
    const done = questionDone(state, p);
    byId('question-nav').options[i].textContent = `${i + 1}. ${p.query_id}${done ? ' ✓' : ''}`;
  });
}
function render() {
  byId('reviewer').value = state.reviewer; byId('declaration').checked = state.human_declaration;
  byId('question-nav').value = String(index);
  byId('prev').disabled = index === 0; byId('next').disabled = index === PACKETS.length - 1;
  const p = PACKETS[index], q = byId('question'); q.replaceChildren();
  q.append(element('h2', `第 ${index + 1} 题 · ${p.query_id}`), element('pre', p.question));
  detail(q, '参考答案', p.reference_answer); detail(q, '参考要点（nuggets）', p.nuggets);
  detail(q, '参考证据（gold context）', p.gold_context);
  const answers = byId('responses'); answers.replaceChildren();
  p.responses.forEach(response => {
    const row = state.ratings.find(r => r.review_id === response.review_id);
    const article = element('article'); article.append(element('h3', `匿名回答 ${response.review_id}`));
    if (row.skip_reason !== null) {
      article.append(element('p', '生成/服务失败，不是知识性拒答。此回答跳过质量评分；导出保留 operational_failure 原因。'));
    }
    article.append(element('pre', response.answer), element('p', `决策：${response.decision ?? '未提供'}`));
    detail(article, '引用', response.citations); detail(article, '回答可用证据', response.context);
    detail(article, '报告的缺失信息', response.missing_information);
    const fields = element('div'); fields.className = 'ratings';
    (row.skip_reason === null ? DIMENSIONS : []).forEach((dimension, i) => {
      const label = element('label', names[i]); const select = element('select');
      select.setAttribute('aria-label', `${response.review_id} ${names[i]}`);
      ['', '0', '1', '2', '3'].forEach(value => {
        const option = element('option', value === '' ? '未评分' : value); option.value = value; select.append(option);
      });
      select.value = row.ratings[dimension] === null ? '' : String(row.ratings[dimension]);
      select.addEventListener('change', () => {
        row.ratings[dimension] = select.value === '' ? null : Number(select.value); persist(); updateProgress();
      });
      label.append(select); fields.append(label);
    });
    article.append(fields);
    const notesLabel = element('label', '备注（可选）'); const notes = element('textarea');
    notes.setAttribute('aria-label', `${response.review_id} 备注`); notes.value = row.reviewer_notes;
    notes.addEventListener('input', () => { row.reviewer_notes = notes.value; persist(); });
    notesLabel.append(notes); article.append(notesLabel); answers.append(article);
  });
  updateProgress();
}
PACKETS.forEach((p, i) => { const option = element('option'); option.value = String(i); byId('question-nav').append(option); });
byId('fingerprint').textContent = '评审包 SHA-256：' + PACKET_HASH;
byId('reviewer').addEventListener('input', event => { state.reviewer = event.target.value; persist(); updateProgress(); });
byId('declaration').addEventListener('change', event => {
  state.human_declaration = event.target.checked; persist(); updateProgress();
});
byId('question-nav').addEventListener('change', event => { index = Number(event.target.value); render(); });
byId('prev').addEventListener('click', () => { if (index > 0) { index--; render(); } });
byId('next').addEventListener('click', () => { if (index < PACKETS.length - 1) { index++; render(); } });
byId('export').addEventListener('click', () => {
  const exported = exportState(state);
  const url = URL.createObjectURL(new Blob([JSON.stringify(exported, null, 2)], {type: 'application/json'}));
  const link = element('a'); link.href = url;
  link.download = `human-review-${PACKET_HASH.slice(0, 12)}-${exported.completion.status}.json`;
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify(exported.completion.status === 'complete' ? '已导出完整人工评审。' : '已导出部分评审；仍有评分或人工声明待完成。');
});
byId('import').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    const imported = validateState(JSON.parse(await file.text()));
    if (!confirm('确认这是您本人此前导出的评审备份，并替换此包当前评分？')) return;
    state = imported; persist(); render(); notify('已导入本人备份；请核对评分与人工声明。');
  } catch { notify('导入失败：文件格式、评分范围、完成状态或评审包指纹不匹配。当前评分保留。'); }
  finally { event.target.value = ''; }
});
byId('clear').addEventListener('click', () => {
  if (!confirm('清除此评审包的姓名、声明和全部评分？已导出的文件不受影响。')) return;
  try { localStorage.removeItem(STORAGE_KEY); }
  catch { notify('浏览器拒绝清除本地记录；已重置内存，请检查浏览器设置。'); }
  state = freshState(); render();
});
render();
</script></main></body></html>'''


def render_html(packets: Any, packet_sha256: str) -> str:
    """Return a self-contained form with inert JSON and exact plain-text answers.

    Args:
        packets: Unscored, anonymous packets from blind_review.
        packet_sha256: SHA-256 of the exact input file bytes.
    """
    validate_packets(packets)
    if not re.fullmatch(r'[0-9a-f]{64}', packet_sha256):
        raise ValueError('packet_sha256 must be a lowercase SHA-256 digest')
    # ASCII JSON escapes U+2028/U+2029; escaping angle brackets prevents script termination.
    data = json.dumps(packets, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e')
    data = data.replace('&', '\\u0026')
    # Substitute packet data last so user strings cannot act as template tokens.
    return PAGE.replace('__STATE_JS__', STATE_JS).replace('__HASH__', packet_sha256).replace('__PACKETS__', data)


def main(argv: list[str] | None = None) -> None:
    """Read only the supplied packet file and write a standalone local HTML artifact.

    Args:
        argv: CLI arguments, defaulting to the current process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.packets.resolve() == args.output.resolve():
        raise ValueError('Output must not overwrite the packet input')
    if args.output.suffix.lower() != '.html':
        raise ValueError('Output must have an .html extension')
    raw = args.packets.read_bytes()
    packets = json.loads(raw.decode('utf-8-sig'))
    page = render_html(packets, hashlib.sha256(raw).hexdigest())
    args.output.write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
