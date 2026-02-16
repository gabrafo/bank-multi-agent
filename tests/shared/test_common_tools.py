"""Testes para ferramentas comuns (compartilhadas entre agentes)."""

from src.tools.common import end_conversation
from src.tools.routing import (
    transfer_to_credit,
    transfer_to_exchange,
    transfer_to_interview,
    transfer_to_triage,
)


class TestEndConversation:
    """Testes para a ferramenta end_conversation."""

    def test_returns_confirmation(self):
        """end_conversation deve retornar mensagem de encerramento."""
        result = end_conversation.invoke({})
        assert result.startswith("ENCERRAMENTO")
        assert "finalizado" in result


class TestTransferTools:
    """Testes para as ferramentas de transferência entre agentes."""

    def test_transfer_to_credit(self):
        result = transfer_to_credit.invoke({})
        assert result.startswith("TRANSFERÊNCIA")
        assert "crédito" in result

    def test_transfer_to_interview(self):
        result = transfer_to_interview.invoke({})
        assert result.startswith("TRANSFERÊNCIA")
        assert "entrevista" in result

    def test_transfer_to_triage(self):
        result = transfer_to_triage.invoke({})
        assert result.startswith("TRANSFERÊNCIA")
        assert "triagem" in result

    def test_transfer_to_exchange(self):
        result = transfer_to_exchange.invoke({})
        assert result.startswith("TRANSFERÊNCIA")
        assert "câmbio" in result
