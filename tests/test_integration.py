"""Testes de integração para o fluxo do agente de triagem.

Usam mock do LLM para simular respostas sem chamar a API do Groq.
"""

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import build_graph, _get_initial_state


# --- Helpers ---


def _ai_text(content: str) -> AIMessage:
    """Cria uma AIMessage com texto puro (sem tool calls)."""
    return AIMessage(content=content)


def _ai_tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Cria uma AIMessage com um tool_call."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"name": name, "args": args, "id": call_id}]
    return msg


# --- Fixtures ---


@pytest.fixture
def fake_csv(tmp_path, monkeypatch):
    """Cria um CSV de clientes temporário e aponta a tool para ele."""
    csv_path = tmp_path / "clientes.csv"
    csv_path.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n"
        "98765432100,22/07/1985,Maria Santos,8000.00,780\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))
    return csv_path


# --- Testes de integração ---


class TestTriageFlowIntegration:
    """Testa fluxos completos do agente de triagem com LLM mockado."""

    def test_greeting_flow(self):
        """O grafo deve retornar uma saudação na primeira interação."""
        greeting = _ai_text("Olá! Bem-vindo ao Banco Ágil. Qual é o seu CPF?")

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.return_value = greeting
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _get_initial_state()
            state["messages"] = []

            result = graph.invoke(state)

        # Deve ter pelo menos a mensagem de saudação
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1
        assert "Banco Ágil" in ai_messages[0].content

    def test_auth_success_flow(self, fake_csv):
        """Fluxo: usuário fornece CPF/data → auth sucesso → saudação pelo nome."""

        # Sequência simulada de respostas do LLM:
        # 1. LLM chama authenticate_client
        # 2. Após receber resultado SUCESSO, responde com saudação
        auth_call = _ai_tool_call(
            "authenticate_client",
            {"cpf": "12345678901", "birth_date": "15/03/1990"},
        )
        welcome = _ai_text("Olá, João! Como posso ajudá-lo hoje?")

        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return auth_call
            return welcome

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _get_initial_state()
            state["messages"] = [
                HumanMessage(content="Meu CPF é 12345678901 e nasci em 15/03/1990")
            ]

            result = graph.invoke(state)

        assert result["authenticated"] is True
        assert result["client_data"]["nome"] == "João Silva"
        assert result["should_end"] is False

    def test_auth_failure_increments_attempts(self, fake_csv):
        """Falha de autenticação deve incrementar auth_attempts no estado."""
        auth_call = _ai_tool_call(
            "authenticate_client",
            {"cpf": "00000000000", "birth_date": "01/01/2000"},
        )
        retry_msg = _ai_text("Dados incorretos. Tente novamente.")

        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return auth_call
            return retry_msg

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _get_initial_state()
            state["messages"] = [
                HumanMessage(content="CPF 00000000000, nascimento 01/01/2000")
            ]

            result = graph.invoke(state)

        assert result["authenticated"] is False
        assert result["auth_attempts"] == 1

    def test_end_conversation_flow(self, fake_csv):
        """Quando o LLM chama end_conversation, o grafo deve encerrar após a resposta."""
        end_call = _ai_tool_call("end_conversation", {})
        goodbye = _ai_text("Obrigado pelo contato! Até logo.")

        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return end_call
            return goodbye

        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = mock_invoke
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _get_initial_state()
            state["messages"] = [
                HumanMessage(content="Quero encerrar o atendimento.")
            ]

            result = graph.invoke(state)

        assert result["should_end"] is True
        # A última AI message deve ser a despedida
        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert any("Até logo" in m.content for m in ai_messages)

    def test_llm_error_returns_friendly_message(self):
        """Se o LLM falhar, deve retornar mensagem amigável sem crashar."""
        with patch("src.graph.get_llm") as mock_llm:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            mock_model.invoke.side_effect = Exception("API timeout")
            mock_llm.return_value = mock_model

            graph = build_graph()
            state = _get_initial_state()
            state["messages"] = [HumanMessage(content="Olá")]

            result = graph.invoke(state)

        ai_messages = [
            m for m in result["messages"]
            if isinstance(m, AIMessage) and m.content
        ]
        assert len(ai_messages) >= 1
        assert "dificuldades técnicas" in ai_messages[-1].content
