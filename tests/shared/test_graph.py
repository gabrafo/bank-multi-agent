"""Testes para o grafo LangGraph: estrutura, roteamento e tool_node compartilhado."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.graph import (
    TOOL_MAP,
    TRANSFER_MAP,
    _get_initial_state,
    build_graph,
    route_entry,
    route_after_tools,
    should_continue,
    tool_node,
    _handle_auth_result,
    _handle_score_update,
    _handle_limit_increase,
)
from langgraph.graph import END


# --- Helpers ---


def _make_state_with_tool_call(tool_name, tool_args, tool_id="call_1", **extra):
    """Cria um estado com uma mensagem AI contendo um tool_call."""
    ai_msg = AIMessage(content="")
    ai_msg.tool_calls = [{"name": tool_name, "args": tool_args, "id": tool_id}]
    return {
        "messages": [ai_msg],
        "authenticated": False,
        "client_data": None,
        "auth_attempts": 0,
        "current_agent": "triage",
        "should_end": False,
        **extra,
    }


# --- Estado inicial ---


class TestInitialState:
    def test_initial_state_values(self):
        state = _get_initial_state()
        assert state["authenticated"] is False
        assert state["client_data"] is None
        assert state["auth_attempts"] == 0
        assert state["current_agent"] == "triage"
        assert state["should_end"] is False

    def test_initial_state_is_new_instance(self):
        s1 = _get_initial_state()
        s2 = _get_initial_state()
        assert s1 is not s2


# --- Compilação do grafo ---


class TestBuildGraph:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_agent_nodes(self):
        graph = build_graph()
        node_names = list(graph.nodes.keys())
        assert "triage" in node_names
        assert "credit" in node_names
        assert "interview" in node_names
        assert "exchange" in node_names
        assert "tools" in node_names


# --- Roteamento ---


class TestRouteEntry:
    def test_routes_to_triage_by_default(self):
        state = {"current_agent": "triage"}
        assert route_entry(state) == "triage"

    def test_routes_to_credit(self):
        state = {"current_agent": "credit"}
        assert route_entry(state) == "credit"

    def test_routes_to_interview(self):
        state = {"current_agent": "interview"}
        assert route_entry(state) == "interview"

    def test_routes_to_exchange(self):
        state = {"current_agent": "exchange"}
        assert route_entry(state) == "exchange"

    def test_defaults_to_triage_when_missing(self):
        assert route_entry({}) == "triage"


class TestShouldContinue:
    def test_routes_to_tools_when_tool_calls_present(self):
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": "authenticate_client", "args": {}, "id": "call_1"}
        ]
        state = {"messages": [ai_msg]}
        assert should_continue(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        ai_msg = AIMessage(content="Olá!")
        state = {"messages": [ai_msg]}
        assert should_continue(state) == END

    def test_routes_to_end_when_tool_calls_empty(self):
        ai_msg = AIMessage(content="Resposta")
        ai_msg.tool_calls = []
        state = {"messages": [ai_msg]}
        assert should_continue(state) == END


class TestRouteAfterTools:
    def test_returns_current_agent(self):
        assert route_after_tools({"current_agent": "triage"}) == "triage"
        assert route_after_tools({"current_agent": "credit"}) == "credit"
        assert route_after_tools({"current_agent": "interview"}) == "interview"
        assert route_after_tools({"current_agent": "exchange"}) == "exchange"

    def test_defaults_to_triage_when_missing(self):
        assert route_after_tools({}) == "triage"


# --- tool_node: comportamento compartilhado ---


class TestToolNodeCommon:
    def test_end_conversation_sets_should_end(self):
        state = _make_state_with_tool_call("end_conversation", {})
        result = tool_node(state)
        assert result["should_end"] is True
        assert "ENCERRAMENTO" in result["messages"][0].content

    def test_unknown_tool_returns_error(self):
        state = _make_state_with_tool_call("ferramenta_inexistente", {"arg": "x"})
        result = tool_node(state)
        assert len(result["messages"]) == 1
        assert "ERRO" in result["messages"][0].content
        assert "não disponível" in result["messages"][0].content

    def test_tool_invoke_exception_returns_erro_sistema(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "end_conversation"
        fake_tool.invoke.side_effect = RuntimeError("falha")
        monkeypatch.setitem(TOOL_MAP, "end_conversation", fake_tool)

        state = _make_state_with_tool_call("end_conversation", {})
        result = tool_node(state)
        assert "ERRO_SISTEMA" in result["messages"][0].content


# --- tool_node: transfer tools ---


class TestToolNodeTransfer:
    def test_transfer_to_credit_updates_state(self):
        state = _make_state_with_tool_call("transfer_to_credit", {})
        result = tool_node(state)
        assert result["current_agent"] == "credit"
        assert "TRANSFERÊNCIA" in result["messages"][0].content

    def test_transfer_to_interview_updates_state(self):
        state = _make_state_with_tool_call("transfer_to_interview", {})
        result = tool_node(state)
        assert result["current_agent"] == "interview"

    def test_transfer_to_triage_updates_state(self):
        state = _make_state_with_tool_call("transfer_to_triage", {})
        result = tool_node(state)
        assert result["current_agent"] == "triage"

    def test_transfer_to_exchange_updates_state(self):
        state = _make_state_with_tool_call("transfer_to_exchange", {})
        result = tool_node(state)
        assert result["current_agent"] == "exchange"
        assert "TRANSFERÊNCIA" in result["messages"][0].content


# --- tool_node: authenticate_client ---


class TestToolNodeAuth:
    def test_auth_success_updates_state(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))

        state = _make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "12345678901", "birth_date": "15/03/1990"},
        )
        result = tool_node(state)
        assert result["authenticated"] is True
        assert result["client_data"]["nome"] == "João Silva"
        assert result["client_data"]["score"] == 650
        assert result["client_data"]["limite_credito"] == 5000.00

    def test_auth_failure_increments_attempts(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))

        state = _make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "00000000000", "birth_date": "01/01/2000"},
        )
        result = tool_node(state)
        assert result["auth_attempts"] == 1

    def test_auth_failure_increments_from_existing(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))

        state = _make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "00000000000", "birth_date": "01/01/2000"},
            auth_attempts=1,
        )
        result = tool_node(state)
        assert result["auth_attempts"] == 2

    def test_auth_success_parse_error_falls_back_to_raw(self, monkeypatch):
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

        state = _make_state_with_tool_call(
            "authenticate_client",
            {"cpf": "12345678901", "birth_date": "15/03/1990"},
        )
        result = tool_node(state)
        assert result["authenticated"] is True
        assert result["client_data"] == {"raw": fake_tool.invoke.return_value}


# --- tool_node: update_client_score ---


class TestToolNodeScoreUpdate:
    def test_score_update_updates_client_data(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "update_client_score"
        fake_tool.invoke.return_value = (
            "ATUALIZADO: Score do cliente atualizado de 650 para 800."
        )
        monkeypatch.setitem(TOOL_MAP, "update_client_score", fake_tool)

        state = _make_state_with_tool_call(
            "update_client_score",
            {"cpf": "12345678901", "new_score": 800},
            client_data={"nome": "João", "cpf": "12345678901", "score": 650},
        )
        result = tool_node(state)
        assert result["client_data"]["score"] == 800

    def test_score_update_no_client_data_is_noop(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "update_client_score"
        fake_tool.invoke.return_value = (
            "ATUALIZADO: Score do cliente atualizado de 650 para 800."
        )
        monkeypatch.setitem(TOOL_MAP, "update_client_score", fake_tool)

        state = _make_state_with_tool_call(
            "update_client_score",
            {"cpf": "12345678901", "new_score": 800},
        )
        result = tool_node(state)
        assert "client_data" not in result

    def test_score_update_parse_error_is_noop(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "update_client_score"
        fake_tool.invoke.return_value = "ATUALIZADO: formato inesperado"
        monkeypatch.setitem(TOOL_MAP, "update_client_score", fake_tool)

        state = _make_state_with_tool_call(
            "update_client_score",
            {"cpf": "12345678901", "new_score": 800},
            client_data={"nome": "João", "score": 650},
        )
        result = tool_node(state)
        # Parse fails → no client_data update
        assert "client_data" not in result


# --- tool_node: request_limit_increase ---


class TestToolNodeLimitIncrease:
    def test_approved_updates_client_data(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "request_limit_increase"
        fake_tool.invoke.return_value = (
            "APROVADO: Solicitação aprovada! "
            "Limite anterior: R$ 5000.00. "
            "Novo limite: R$ 7000.00."
        )
        monkeypatch.setitem(TOOL_MAP, "request_limit_increase", fake_tool)

        state = _make_state_with_tool_call(
            "request_limit_increase",
            {"cpf": "12345678901", "new_limit": 7000.0},
            client_data={"nome": "João", "limite_credito": 5000.0},
        )
        result = tool_node(state)
        assert result["client_data"]["limite_credito"] == 7000.0

    def test_rejected_does_not_update_client_data(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "request_limit_increase"
        fake_tool.invoke.return_value = "REJEITADO: Score insuficiente."
        monkeypatch.setitem(TOOL_MAP, "request_limit_increase", fake_tool)

        state = _make_state_with_tool_call(
            "request_limit_increase",
            {"cpf": "12345678901", "new_limit": 15000.0},
            client_data={"nome": "João", "limite_credito": 5000.0},
        )
        result = tool_node(state)
        assert "client_data" not in result

    def test_approved_no_client_data_is_noop(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "request_limit_increase"
        fake_tool.invoke.return_value = "APROVADO: OK"
        monkeypatch.setitem(TOOL_MAP, "request_limit_increase", fake_tool)

        state = _make_state_with_tool_call(
            "request_limit_increase",
            {"cpf": "12345678901", "new_limit": 7000.0},
        )
        result = tool_node(state)
        assert "client_data" not in result

    def test_approved_bad_args_is_noop(self, monkeypatch):
        fake_tool = MagicMock()
        fake_tool.name = "request_limit_increase"
        fake_tool.invoke.return_value = "APROVADO: OK"
        monkeypatch.setitem(TOOL_MAP, "request_limit_increase", fake_tool)

        state = _make_state_with_tool_call(
            "request_limit_increase",
            {"cpf": "12345678901"},  # Missing new_limit
            client_data={"nome": "João", "limite_credito": 5000.0},
        )
        result = tool_node(state)
        # KeyError on new_limit → no update
        assert "client_data" not in result


# --- _handle_* helpers (direct unit tests) ---


class TestHandleAuthResult:
    def test_success_sets_authenticated(self):
        state_updates = {}
        _handle_auth_result(
            "SUCESSO: Nome: Ana, CPF: 123, Limite de crédito: R$ 5000.00, Score: 700",
            {},
            state_updates,
        )
        assert state_updates["authenticated"] is True
        assert state_updates["client_data"]["nome"] == "Ana"

    def test_failure_increments_attempts(self):
        state_updates = {}
        _handle_auth_result("FALHA: dados incorretos", {"auth_attempts": 2}, state_updates)
        assert state_updates["auth_attempts"] == 3

    def test_other_result_no_update(self):
        state_updates = {}
        _handle_auth_result("ERRO_SISTEMA: algo deu errado", {}, state_updates)
        assert state_updates == {}


class TestHandleScoreUpdate:
    def test_no_update_when_not_atualizado(self):
        state_updates = {}
        _handle_score_update("ERRO: falha", {"client_data": {"score": 1}}, state_updates)
        assert state_updates == {}


class TestHandleLimitIncrease:
    def test_no_update_when_not_aprovado(self):
        state_updates = {}
        _handle_limit_increase(
            "REJEITADO: nope", {"new_limit": 9999}, {"client_data": {"x": 1}}, state_updates
        )
        assert state_updates == {}

    def test_no_update_when_bad_new_limit_type(self):
        state_updates = {}
        _handle_limit_increase(
            "APROVADO: ok",
            {"new_limit": "not_a_number"},
            {"client_data": {"limite_credito": 5000}},
            state_updates,
        )
        # "not_a_number" can be converted to float? No → ValueError → noop
        assert state_updates == {}
