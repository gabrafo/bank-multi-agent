"""Testes para o grafo LangGraph: compilação, nós e roteamento."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import (
    _get_initial_state,
    build_graph,
    should_continue,
    after_tools,
    tool_node,
)
from langgraph.graph import END


# --- Testes: Estado inicial ---


class TestInitialState:
    """Testes para o estado inicial da conversa."""

    def test_initial_state_values(self):
        state = _get_initial_state()
        assert state["authenticated"] is False
        assert state["client_data"] is None
        assert state["auth_attempts"] == 0
        assert state["current_agent"] == "triage"
        assert state["should_end"] is False

    def test_initial_state_is_new_instance(self):
        """Cada chamada deve retornar uma nova instância."""
        s1 = _get_initial_state()
        s2 = _get_initial_state()
        assert s1 is not s2


# --- Testes: Compilação do grafo ---


class TestBuildGraph:
    """Testes para a construção do grafo."""

    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = list(graph.nodes.keys())
        assert "triage" in node_names
        assert "tools" in node_names


# --- Testes: Roteamento condicional ---


class TestShouldContinue:
    """Testes para a função de roteamento should_continue."""

    def test_routes_to_tools_when_tool_calls_present(self):
        """Se a última mensagem tem tool_calls, deve ir para 'tools'."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": "authenticate_client", "args": {}, "id": "call_1"}
        ]
        state = {"messages": [ai_msg]}
        assert should_continue(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        """Se a última mensagem é texto puro, deve ir para END."""
        ai_msg = AIMessage(content="Olá, como posso ajudar?")
        state = {"messages": [ai_msg]}
        assert should_continue(state) == END

    def test_routes_to_end_when_tool_calls_empty(self):
        """Se tool_calls existe mas está vazio, deve ir para END."""
        ai_msg = AIMessage(content="Resposta")
        ai_msg.tool_calls = []
        state = {"messages": [ai_msg]}
        assert should_continue(state) == END


class TestAfterTools:
    """Testes para a função de roteamento after_tools."""

    def test_always_returns_triage(self):
        """Após ferramentas, sempre volta ao agente de triagem."""
        state = {"should_end": False, "messages": []}
        assert after_tools(state) == "triage"

    def test_returns_triage_even_when_should_end(self):
        """Mesmo com should_end=True, volta ao agente para a mensagem final."""
        state = {"should_end": True, "messages": []}
        assert after_tools(state) == "triage"


# --- Testes: Nó de ferramentas (tool_node) ---


class TestToolNode:
    """Testes para o nó de execução de ferramentas."""

    def _make_state_with_tool_call(self, tool_name, tool_args, tool_id="call_1"):
        """Helper: cria um estado com uma mensagem AI contendo um tool_call."""
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": tool_name, "args": tool_args, "id": tool_id}
        ]
        return {
            "messages": [ai_msg],
            "authenticated": False,
            "client_data": None,
            "auth_attempts": 0,
            "current_agent": "triage",
            "should_end": False,
        }

    def test_end_conversation_sets_should_end(self):
        """Chamar end_conversation deve setar should_end=True."""
        state = self._make_state_with_tool_call("end_conversation", {})
        result = tool_node(state)
        assert result["should_end"] is True
        assert len(result["messages"]) == 1
        assert "ENCERRAMENTO" in result["messages"][0].content

    def test_auth_success_updates_state(self, tmp_path, monkeypatch):
        """Autenticação bem-sucedida deve setar authenticated e client_data."""
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))

        state = self._make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "12345678901", "birth_date": "15/03/1990"},
        )
        result = tool_node(state)
        assert result["authenticated"] is True
        assert result["client_data"]["nome"] == "João Silva"
        assert result["client_data"]["score"] == 650
        assert result["client_data"]["limite_credito"] == 5000.00

    def test_auth_failure_increments_attempts(self, tmp_path, monkeypatch):
        """Falha de autenticação deve incrementar auth_attempts."""
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))

        state = self._make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "00000000000", "birth_date": "01/01/2000"},
        )
        result = tool_node(state)
        assert result["auth_attempts"] == 1

    def test_auth_failure_increments_from_existing(self, tmp_path, monkeypatch):
        """Incrementa auth_attempts a partir do valor atual do estado."""
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))

        state = self._make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "00000000000", "birth_date": "01/01/2000"},
        )
        state["auth_attempts"] = 1  # Já teve 1 falha
        result = tool_node(state)
        assert result["auth_attempts"] == 2

    def test_unknown_tool_returns_error_message(self):
        """Chamar ferramenta desconhecida deve retornar mensagem de erro."""
        state = self._make_state_with_tool_call(
            "ferramenta_inexistente", {"arg": "value"}
        )
        result = tool_node(state)
        assert len(result["messages"]) == 1
        assert "ERRO" in result["messages"][0].content
        assert "não disponível" in result["messages"][0].content

    def test_tool_invoke_exception_returns_erro_sistema(self, monkeypatch):
        """Quando tool.invoke() lança exceção, deve retornar ERRO_SISTEMA."""
        from src.graph import TOOL_MAP

        # Cria uma tool fake que lança exceção ao ser invocada
        fake_tool = MagicMock()
        fake_tool.name = "end_conversation"
        fake_tool.invoke.side_effect = RuntimeError("falha inesperada")
        monkeypatch.setitem(TOOL_MAP, "end_conversation", fake_tool)

        state = self._make_state_with_tool_call("end_conversation", {})
        result = tool_node(state)

        assert len(result["messages"]) == 1
        assert "ERRO_SISTEMA" in result["messages"][0].content

    def test_auth_success_parse_error_falls_back_to_raw(self, monkeypatch):
        """Quando o parse dos dados do cliente falha, deve gravar raw."""
        from src.graph import TOOL_MAP

        # Retorna SUCESSO com "Score:" contendo valor não-numérico para forçar ValueError
        fake_tool = MagicMock()
        fake_tool.name = "authenticate_client"
        fake_tool.invoke.return_value = (
            "SUCESSO: Cliente autenticado. "
            "Nome: Teste, "
            "CPF: 12345678901, "
            "Limite de crédito: R$ abc, "
            "Score: xyz"
        )
        monkeypatch.setitem(TOOL_MAP, "authenticate_client", fake_tool)

        state = self._make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "12345678901", "birth_date": "15/03/1990"},
        )
        result = tool_node(state)

        assert result["authenticated"] is True
        # Parse falhou → fallback para raw
        assert result["client_data"] == {"raw": fake_tool.invoke.return_value}
