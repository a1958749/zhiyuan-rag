"""Fixed gateway experiments retain the native fallback title without hidden paid calls."""
from types import SimpleNamespace
from unittest.mock import Mock


def test_fixed_gateway_does_not_issue_untracked_title_call():
    from docsgpt.api.answer.services.conversation_service import ConversationService
    llm = SimpleNamespace(model_id='deepseek-v4-pro', gen=Mock(return_value='title'))
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = ('conversation', 'question')
    service = ConversationService.__new__(ConversationService)
    service._maybe_generate_title(conn, 'message', {'llm': llm, 'question': 'question',
                                                   'model_id': 'devsupport-opus'})
    llm.gen.assert_not_called()
