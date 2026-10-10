"""Synthetic-only tests for the isolated human review artifact."""
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def review():
    """Load the standalone generator without importing runtime code."""
    path = Path(__file__).parents[1] / 'analysis/human_review.py'
    assert path.exists(), 'human review generator must exist'
    spec = importlib.util.spec_from_file_location('human_review', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packets():
    """Return anonymous packets with deliberately hostile plain text."""
    return [{'query_id': 'synthetic-1', 'question': 'Question?',
             'reference_answer': 'Reference', 'nuggets': [{'text': 'Nugget'}],
             'gold_context': [{'doc_id': 'd1', 'text': 'Gold evidence'}],
             'responses': [{'review_id': 'Q001-A01',
                            'answer': '</script><img src=x onerror=alert(1)>\u2028\u2029 & < exact',
                            'decision': 'answer', 'citations': ['d1'],
                            'context': [{'doc_id': 'd1', 'text': 'Evidence', 'citation_number': 1}],
                            'missing_information': [], 'ratings': None, 'reviewer_notes': None}]}]


def test_safe_embedded_data_and_blank_ratings(review, packets):
    """The HTML round-trips exact answers without executable packet strings."""
    page = review.render_html(packets, 'a' * 64)
    embedded = re.search(r'<script id="packets" type="application/json">(.*?)</script>', page, re.S)[1]
    assert json.loads(embedded) == packets
    assert '<img src=x' not in page
    assert '\u2028' not in page and '\u2029' not in page
    assert 'innerHTML' not in page and 'localStorage.clear' not in page
    assert 'textContent' in page and 'localStorage.removeItem(STORAGE_KEY)' in page
    assert 'fetch(' not in page and '<script src=' not in page
    assert 'sealed' not in page and 'method_mapping' not in page
    assert packets[0]['responses'][0]['ratings'] is None


@pytest.mark.parametrize('change', ['empty', 'duplicate_query', 'duplicate_response', 'method',
                                  'nested_method', 'ratings', 'notes', 'bad_id', 'answer_type', 'no_response'])
def test_rejects_invalid_or_unblinded_packets(review, packets, change):
    """Fail closed for unblinded, pre-scored, malformed, or ambiguous packets."""
    response = packets[0]['responses'][0]
    if change == 'empty':
        packets = []
    elif change == 'duplicate_query':
        packets.append(copy.deepcopy(packets[0]))
    elif change == 'duplicate_response':
        packets[0]['responses'].append(copy.deepcopy(response))
    elif change == 'method':
        response['method'] = 'DO_NOT_EMBED'
    elif change == 'nested_method':
        response['context'][0]['metadata'] = {'method': 'DO_NOT_EMBED'}
    elif change == 'ratings':
        response['ratings'] = {'correctness': 3}
    elif change == 'notes':
        response['reviewer_notes'] = 'Machine filled'
    elif change == 'bad_id':
        response['review_id'] = 'method-C'
    elif change == 'answer_type':
        response['answer'] = 42
    elif change == 'no_response':
        packets[0]['responses'] = []
    with pytest.raises(ValueError):
        review.render_html(packets, 'a' * 64)


def test_cli_hashes_only_explicit_packet_file(review, packets, tmp_path):
    """Generate from explicit bytes and refuse to overwrite the packet input."""
    source = tmp_path / 'packets.json'
    source.write_text(json.dumps(packets), encoding='utf-8')
    output = tmp_path / 'review.html'
    review.main(['--packets', str(source), '--output', str(output)])
    assert hashlib.sha256(source.read_bytes()).hexdigest() in output.read_text(encoding='utf-8')
    with pytest.raises(ValueError):
        review.main(['--packets', str(source), '--output', str(source)])


@pytest.mark.parametrize('field', ['context', 'citations', 'missing_information', 'gold_context', 'nuggets'])
@pytest.mark.parametrize('key', ['scores', 'retrieval_score', 'rerank_score', 'confidence',
                               'judge_scores', 'similarityScore', 'Relevance-Score', 'logprobs'])
def test_rejects_nested_score_metadata(review, packets, field, key):
    """Machine scores must never reach expandable evidence, at any nesting depth."""
    owner = packets[0] if field in ('gold_context', 'nuggets') else packets[0]['responses'][0]
    owner[field] = [{'doc_id': 'd1', 'metadata': {'nested': [{key: {'correctness': 3}}]}}]
    with pytest.raises(ValueError, match='scoring metadata'):
        review.render_html(packets, 'a' * 64)


def test_cli_rejects_scored_evidence_before_writing(review, packets, tmp_path):
    """The reported leak fails closed without producing a biased review artifact."""
    packets[0]['responses'][0]['context'][0]['metadata'] = {
        'scores': {'correctness': 3}, 'retrieval_score': 0.91,
    }
    source, output = tmp_path / 'packets.json', tmp_path / 'review.html'
    source.write_text(json.dumps(packets), encoding='utf-8')
    with pytest.raises(ValueError, match='scoring metadata'):
        review.main(['--packets', str(source), '--output', str(output)])
    assert not output.exists()


def test_preserves_corpus_and_gold_metadata_and_score_prose(review, packets):
    """Source code/prose and legitimate gold annotations remain byte-for-byte content."""
    text = 'def score(x): return {"scores": x, "confidence": 0.9}\nretrieval_score = score(3)'
    packets[0]['responses'][0]['answer'] = text
    metadata = {'url': 'https://example.invalid/score', 'start_byte': 7, 'end_byte': 33,
                'nested': [{'source': 'scores and confidence are ordinary documentation words'}]}
    packets[0]['responses'][0]['context'][0].update(text=text, metadata=metadata)
    packets[0]['gold_context'][0].update(text=text, metadata=metadata)
    packets[0]['nuggets'] = [{'text': 'Explain score(x)', 'relevant_corpus_ids': ['d1'],
                              'relevant': {'d1': 2}, 'metadata': {'source': 'gold annotation'}}]
    original = copy.deepcopy(packets)
    page = review.render_html(packets, 'a' * 64)
    embedded = re.search(r'<script id="packets" type="application/json">(.*?)</script>', page, re.S)[1]
    assert json.loads(embedded) == original
    assert packets == original


def test_operational_failures_and_optional_empty_fields(review, packets):
    """Service failures are explicit skips, never fabricated human zero ratings."""
    response = packets[0]['responses'][0]
    del response['ratings']
    del response['reviewer_notes']
    response['operational_failure'] = True
    assert '生成/服务失败，不是知识性拒答' in review.render_html(packets, 'a' * 64)
    response['operational_failure'] = 'true'
    with pytest.raises(ValueError):
        review.render_html(packets, 'a' * 64)


def test_browser_state_validation_and_completion(review, packets, tmp_path):
    """Execute the same pure JS state logic used by the page with Node."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to verify the browser state contract')
    failed = copy.deepcopy(packets[0]['responses'][0])
    failed['review_id'] = 'Q001-A02'
    failed['operational_failure'] = True
    packets[0]['responses'].append(failed)
    program = ('const PACKETS = ' + json.dumps(packets) + ';\n'
               'const PACKET_HASH = "' + 'a' * 64 + '";\n' + review.STATE_JS + r'''
const assert = require('node:assert/strict');
let state = freshState();
assert.equal(summary(state).status, 'partial');
assert.equal(summary(state).scored_responses, 0);
assert.equal(summary(state).skipped_operational_failures, 1);
assert.equal(summary(state).total_responses, 1);
assert.equal(state.ratings[1].skip_reason, 'operational_failure');
assert.ok(Object.values(state.ratings[1].ratings).every(x => x === null));
assert.ok(Object.values(state.ratings[0].ratings).every(x => x === null));
for (const key of DIMENSIONS) state.ratings[0].ratings[key] = 0;
assert.equal(summary(state).scored_responses, 1);
assert.equal(summary(state).status, 'partial');
state.reviewer = 'Human'; state.human_declaration = true;
assert.equal(summary(state).status, 'complete');
const exported = exportState(state);
assert.deepEqual(validateState(exported), state);
exported.completion.status = 'partial';
assert.throws(() => validateState(exported));
for (const value of [true, 0.5, -1, 4, '3']) {
  const altered = exportState(state); altered.ratings[0].ratings.correctness = value;
  assert.throws(() => validateState(altered));
}
for (const mutate of [
  x => x.packet_sha256 = 'wrong', x => x.schema = 'wrong',
  x => x.ratings.push(x.ratings[0]), x => x.ratings[0].review_id = 'unknown',
  x => x.ratings[0].method = 'leak', x => x.reviewer = 2,
  x => x.ratings[1].ratings.correctness = 0,
  x => x.human_declaration = 'true', x => x.timestamp = 'invalid'
]) { const altered = exportState(state); mutate(altered); assert.throws(() => validateState(altered)); }
state.ratings[0].ratings.correctness = null;
assert.equal(summary(state).status, 'partial');
assert.equal(summary(state).completed_questions, 0);
''')
    script = tmp_path / 'state-contract.cjs'
    script.write_text(program, encoding='utf-8')
    result = subprocess.run([node, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_generated_browser_script_syntax(review, packets, tmp_path):
    """Parse the complete embedded script, including navigation and import handlers."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the browser script syntax check')
    page = review.render_html(packets, 'a' * 64)
    script = tmp_path / 'browser.cjs'
    script.write_text(re.search(r'<script>\n(.*?)</script>', page, re.S)[1], encoding='utf-8')
    result = subprocess.run([node, '--check', str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
