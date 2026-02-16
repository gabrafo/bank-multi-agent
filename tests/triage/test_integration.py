"""Testes de integração para o fluxo do agente de triagem (LLM mockado)."""

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
def fake_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "clientes.csv"
    csv_path.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n"
        "98765432100,22/07/1985,Maria Santos,8000.00,780\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))
    return csv_path


class TestTriageFlowIntegration:
    def test_greeting_flow(self):
        greeting = _ai_text("Olá! Bem-vindo ao Banco Ágil. Qual é o seu CPF?")

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.return_value = greeting
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = {**_get_initial_state(), "messages": []}
            result = graph.invoke(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        assert "Banco Ágil" in ai_msgs[0].content

    def test_auth_success_flow(self, fake_csv):
        auth_call = _ai_tool_call(
            "authenticate_client",
            {"cpf": "12345678901", "birth_date": "15/03/1990"},
        )
        welcome = _ai_text("Olá, João! Como posso ajudá-lo hoje?")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return auth_call if call_count == 1 else welcome

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = {**_get_initial_state(), "messages": [
                HumanMessage(content="CPF 12345678901, nascimento 15/03/1990")
            ]}
            result = graph.invoke(state)

        assert result["authenticated"] is True
        assert result["client_data"]["nome"] == "João Silva"

    def test_auth_failure_increments_attempts(self, fake_csv):
        auth_call = _ai_tool_call(
            "authenticate_client",
            {"cpf": "00000000000", "birth_date": "01/01/2000"},
        )
        retry_msg = _ai_text("Dados incorretos. Tente novamente.")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return auth_call if call_count == 1 else retry_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = {**_get_initial_state(), "messages": [
                HumanMessage(content="CPF 00000000000, nascimento 01/01/2000")
            ]}
            result = graph.invoke(state)

        assert result["authenticated"] is False
        assert result["auth_attempts"] == 1

    def test_end_conversation_flow(self):
        end_call = _ai_tool_call("end_conversation", {})
        goodbye = _ai_text("Obrigado pelo contato! Até logo.")
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
            state = {**_get_initial_state(), "messages": [
                HumanMessage(content="Quero encerrar.")
            ]}
            result = graph.invoke(state)

        assert result["should_end"] is True
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert any("Até logo" in m.content for m in ai_msgs)

    def test_llm_error_returns_friendly_message(self):
        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = Exception("API timeout")
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = {**_get_initial_state(), "messages": [
                HumanMessage(content="Olá")
            ]}
            result = graph.invoke(state)

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        assert len(ai_msgs) >= 1
        assert "dificuldades técnicas" in ai_msgs[0].content

    def test_transfer_to_credit_flow(self, fake_csv):
        """Triagem autentica e transfere para crédito."""
        transfer_call = _ai_tool_call("transfer_to_credit", {})
        credit_greeting = _ai_text("Certo! Como posso ajudá-lo com seu crédito?")
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return transfer_call if call_count == 1 else credit_greeting

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
                "messages": [HumanMessage(content="Quero ver meu limite")],
            }
            result = graph.invoke(state)

        assert result["current_agent"] == "credit"
