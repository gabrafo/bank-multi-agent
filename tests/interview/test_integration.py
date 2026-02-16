"""Testes de integração para o fluxo do agente de entrevista de crédito (LLM mockado)."""

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
def interview_csvs(tmp_path, monkeypatch):
    clients = tmp_path / "clientes.csv"
    clients.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(clients))
    monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(clients))
    monkeypatch.setattr("src.tools.interview.CLIENTS_CSV", str(clients))
    return clients


def _make_interview_state(**overrides):
    """Estado de cliente na entrevista de crédito."""
    state = {
        **_get_initial_state(),
        "authenticated": True,
        "current_agent": "interview",
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


class TestInterviewFlowIntegration:
    def test_calculate_score_flow(self, interview_csvs):
        """Fluxo: entrevista → calcula score → informa ao cliente."""
        calc_call = _ai_tool_call(
            "calculate_credit_score",
            {
                "renda_mensal": 5000.0,
                "tipo_emprego": "formal",
                "despesas_fixas": 2000.0,
                "num_dependentes": 0,
                "tem_dividas": "não",
            },
        )
        inform = _ai_text("Seu novo score é 575!")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return calc_call if call_count == 1 else inform

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_interview_state(messages=[
                HumanMessage(content="Minha renda é 5000"),
            ])
            result = graph.invoke(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert any("575" in m.content or "score" in m.content.lower() for m in ai_msgs)

    def test_update_score_and_transfer_flow(self, interview_csvs):
        """Fluxo: atualiza score → transfere de volta para crédito."""
        update_call = _ai_tool_call(
            "update_client_score",
            {"cpf": "12345678901", "new_score": 800},
        )
        transfer_call = _ai_tool_call(
            "transfer_to_credit", {}, call_id="call_2"
        )
        credit_msg = _ai_text("Score atualizado! Vamos reavaliar seu limite.")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return update_call
            elif call_count == 2:
                return transfer_call
            return credit_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_interview_state(messages=[
                HumanMessage(content="Pronto, terminei a entrevista"),
            ])
            result = graph.invoke(state)

        assert result["current_agent"] == "credit"
        assert result["client_data"]["score"] == 800

    def test_llm_error_in_interview(self):
        """LLM falhando na entrevista retorna mensagem amigável."""
        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = Exception("API error")
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _make_interview_state(messages=[
                HumanMessage(content="Olá"),
            ])
            result = graph.invoke(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert any("dificuldades técnicas" in m.content for m in ai_msgs)

    def test_end_conversation_during_interview(self):
        """Cliente pode encerrar durante a entrevista."""
        end_call = _ai_tool_call("end_conversation", {})
        goodbye = _ai_text("Até logo!")
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
            state = _make_interview_state(messages=[
                HumanMessage(content="Quero encerrar"),
            ])
            result = graph.invoke(state)

        assert result["should_end"] is True
