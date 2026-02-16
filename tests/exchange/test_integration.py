"""Testes de integração para o fluxo do agente de câmbio (LLM mockado)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import build_graph, _get_initial_state


def _ai_text(content: str) -> AIMessage:
    return AIMessage(content=content)


def _ai_tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": name, "args": args, "id": call_id}]
    return msg


def _make_exchange_state(**overrides):
    """Estado de um cliente já autenticado e no agente de câmbio."""
    state = {
        **_get_initial_state(),
        "authenticated": True,
        "current_agent": "exchange",
        "client_data": {
            "nome": "João Silva",
            "cpf": "12345678901",
            "limite_credito": 5000.0,
            "score": 650,
        },
        "messages": [],
    }
    state.update(overrides)
    return state


class TestExchangeFlowIntegration:
    def test_query_exchange_rate_flow(self):
        """Fluxo: cliente pergunta cotação → agente consulta → responde."""
        query_call = _ai_tool_call(
            "get_exchange_rate", {"currency_code": "USD"}
        )
        response = _ai_text(
            "A cotação atual do Dólar é R$ 5,05 para compra."
        )
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return query_call if call_count == 1 else response

        # Mock both the LLM and the exchange API
        with patch("src.graph.get_llm") as mock_llm, \
             patch("src.tools.exchange.requests.get") as mock_get:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "USDBRL": {
                    "bid": "5.0500", "ask": "5.0600",
                    "high": "5.1234", "low": "5.0012",
                    "pctChange": "0.45",
                }
            }
            mock_get.return_value = mock_resp

            graph = build_graph()
            state = _make_exchange_state(messages=[
                HumanMessage(content="Qual a cotação do dólar?"),
            ])
            result = graph.invoke(state)

        ai_msgs = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert any(
            "dólar" in m.content.lower() or "5,05" in m.content
            for m in ai_msgs
        )

    def test_transfer_to_triage_flow(self):
        """Após câmbio, cliente pode voltar à triagem."""
        transfer_call = _ai_tool_call("transfer_to_triage", {})
        triage_msg = _ai_text("Claro! Como posso ajudá-lo?")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return transfer_call if call_count == 1 else triage_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_exchange_state(messages=[
                HumanMessage(content="Quero ver meu limite de crédito"),
            ])
            result = graph.invoke(state)

        assert result["current_agent"] == "triage"

    def test_end_conversation_during_exchange(self):
        """Cliente pode encerrar durante o câmbio."""
        end_call = _ai_tool_call("end_conversation", {})
        goodbye = _ai_text("Até logo! Tenha um ótimo dia.")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return end_call if call_count == 1 else goodbye

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_exchange_state(messages=[
                HumanMessage(content="Quero encerrar"),
            ])
            result = graph.invoke(state)

        assert result["should_end"] is True

    def test_llm_error_in_exchange(self):
        """LLM falhando no agente de câmbio retorna mensagem amigável."""
        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = Exception("API error")
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_exchange_state(messages=[
                HumanMessage(content="Cotação do euro"),
            ])
            result = graph.invoke(state)

        ai_msgs = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert any("dificuldades técnicas" in m.content for m in ai_msgs)

    def test_triage_transfers_to_exchange(self):
        """Triagem → transfer_to_exchange → câmbio assume."""
        transfer_call = _ai_tool_call("transfer_to_exchange", {})
        exchange_msg = _ai_text("Qual moeda deseja consultar?")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return transfer_call if call_count == 1 else exchange_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = {
                **_get_initial_state(),
                "authenticated": True,
                "current_agent": "triage",
                "messages": [
                    HumanMessage(content="Quero ver a cotação do dólar"),
                ],
            }
            result = graph.invoke(state)

        assert result["current_agent"] == "exchange"
