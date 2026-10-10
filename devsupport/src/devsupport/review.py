"""Blinded human-review artifacts; no invented ratings or online annotation exposure."""
import random

from .data import relevant_doc_ids


def blind_review(questions: list, variants: dict, corpus: dict, seed: int = 20261008) -> tuple:
    """Return question/reference packets and a separate sealed method mapping."""
    ids = [q.query_id for q in questions]
    tables = {}
    for method, rows in variants.items():
        if [r['query_id'] for r in rows] != ids:
            raise ValueError('review cohort must be complete and identically ordered')
        tables[method] = {r['query_id']: r for r in rows}
    rng, packets, sealed = random.Random(seed), [], {}
    for index, q in enumerate(questions):
        methods = list(tables)
        rng.shuffle(methods)
        responses = []
        for n, method in enumerate(methods):
            identifier = f'Q{index + 1:03d}-A{n + 1:02d}'
            result = tables[method][q.query_id]['result']
            responses.append({'review_id': identifier, 'answer': result.get('answer_text', ''),
                              'decision': result.get('decision'), 'citations': result.get('citation_ids', []),
                              'context': [{k: c[k] for k in ('doc_id', 'text', 'metadata', 'citation_number') if k in c}
                                          for c in result.get('context', [])],
                              'missing_information': result.get('missing_information', []),
                              'ratings': None, 'reviewer_notes': None})
            sealed[identifier] = {'query_id': q.query_id, 'method': method}
        gold_ids = sorted(set().union(*(relevant_doc_ids(n) for n in q.nuggets))) if q.nuggets else []
        context = [corpus[i].to_corpus_dict() if hasattr(corpus[i], 'to_corpus_dict') else corpus[i]
                   for i in gold_ids if i in corpus]
        packets.append({'query_id': q.query_id, 'question': q.query, 'reference_answer': q.answer_text,
                        'nuggets': q.nuggets, 'gold_context': context, 'responses': responses})
    return packets, sealed


def load_controls(path) -> list[dict]:
    """Validate externally curated40 controls; never synthesize truth labels."""
    from .evaluate import read_json
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError('controls require40 curated rows')
    if len({r['query_id'] for r in rows}) != 40:
        raise ValueError('duplicate control IDs')
    if {label: sum(r['expected_decision'] == label for r in rows)
            for label in ('answer', 'insufficient', 'clarify')} != {'answer': 20, 'insufficient': 10, 'clarify': 10}:
        raise ValueError('controls must be20 supported/10 insufficient/10 clarify')
    if sum(r['split'] == 'dev' for r in rows) != 10 or sum(r['split'] == 'test' for r in rows) != 30:
        raise ValueError('controls must be dev10/test30')
    if any(not isinstance(r['query'], str) or not r['query'].strip() for r in rows):
        raise ValueError('control question missing')
    return rows
