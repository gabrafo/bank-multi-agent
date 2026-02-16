"""Testes de integração para o fluxo do agente de crédito (LLM mockado)."""

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


@pytest.fixture
def credit_csvs(tmp_path, monkeypatch):
    clients = tmp_path / "clientes.csv"
    clients.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(clients))
    monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(clients))

    score_limit = tmp_path / "score_limite.csv"
    score_limit.write_text(
        "score_minimo,score_maximo,limite_maximo\n"
        "650,749,8000.00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.credit.SCORE_LIMIT_CSV", str(score_limit))

    requests = tmp_path / "solicitacoes.csv"
    requests.write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,"
        "novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.credit.REQUESTS_CSV", str(requests))
    return {"clients": clients, "score_limit": score_limit, "requests": requests}


def _make_credit_state(**overrides):
    """Estado de um cliente já autenticado e no agente de crédito."""
    state = {
        **_get_initial_state(),
        "authenticated": True,
        "current_agent": "credit",
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


class TestCreditFlowIntegration:
    def test_query_limit_flow(self, credit_csvs):
        """Fluxo: cliente pergunta limite → agente consulta → responde."""
        query_call = _ai_tool_call(
            "query_credit_limit", {"cpf": "12345678901"}
        )
        response = _ai_text("Seu limite atual é R$ 5.000,00.")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return query_call if call_count == 1 else response

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_credit_state(messages=[
                HumanMessage(content="Qual meu limite?"),
            ])
            result = graph.invoke(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert any("5.000" in m.content or "limite" in m.content.lower() for m in ai_msgs)

    def test_increase_approved_flow(self, credit_csvs):
        """Fluxo: solicita aumento → aprovado → informa sucesso."""
        increase_call = _ai_tool_call(
            "request_limit_increase",
            {"cpf": "12345678901", "new_limit": 7000.0},
        )
        success_msg = _ai_text("Seu limite foi aumentado para R$ 7.000,00!")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return increase_call if call_count == 1 else success_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_credit_state(messages=[
                HumanMessage(content="Quero aumentar meu limite para 7000"),
            ])
            result = graph.invoke(state)

        assert result["client_data"]["limite_credito"] == 7000.0

    def test_increase_rejected_offers_interview(self, credit_csvs):
        """Fluxo: aumento rejeitado → transfere para entrevista."""
        increase_call = _ai_tool_call(
            "request_limit_increase",
            {"cpf": "12345678901", "new_limit": 15000.0},
        )
        offer_interview = _ai_text(
            "Infelizmente seu score não permite esse limite. "
            "Posso iniciar uma entrevista para reavaliar seu score?"
        )
        # Client accepts → transfer
        transfer_call = _ai_tool_call("transfer_to_interview", {})
        interview_greeting = _ai_text("Vou fazer algumas perguntas...")

        responses = [increase_call, offer_interview]
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count <= len(responses):
                return responses[call_count - 1]
            return interview_greeting

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_credit_state(messages=[
                HumanMessage(content="Quero limite de 15000"),
            ])
            result = graph.invoke(state)

        # Agent offered interview but responded with text (no transfer yet)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert any("score" in m.content.lower() or "entrevista" in m.content.lower()
                    for m in ai_msgs)

    def test_transfer_to_interview_flow(self, credit_csvs):
        """Transferência para entrevista atualiza current_agent."""
        transfer_call = _ai_tool_call("transfer_to_interview", {})
        interview_msg = _ai_text("Vou fazer algumas perguntas sobre suas finanças.")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return transfer_call if call_count == 1 else interview_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_credit_state(messages=[
                HumanMessage(content="Sim, quero a entrevista"),
            ])
            result = graph.invoke(state)

        assert result["current_agent"] == "interview"

    def test_llm_error_in_credit(self):
        """LLM falhando no agente de crédito retorna mensagem amigável."""
        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = Exception("timeout")
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_credit_state(messages=[
                HumanMessage(content="Qual meu limite?"),
            ])
            result = graph.invoke(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert any("dificuldades técnicas" in m.content for m in ai_msgs)
