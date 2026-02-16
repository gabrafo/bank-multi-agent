import logging

from langchain_core.messages import AIMessage, ToolMessage

from src.agents.credit import CREDIT_SYSTEM_PROMPT, CREDIT_TOOLS
from src.agents.exchange import EXCHANGE_SYSTEM_PROMPT, EXCHANGE_TOOLS
from src.agents.interview import INTERVIEW_SYSTEM_PROMPT, INTERVIEW_TOOLS
from src.agents.triage import TRIAGE_SYSTEM_PROMPT, TRIAGE_TOOLS
from src.config import get_llm
from src.state import AgentState

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# Configuração de cada agente: prompt de sistema e ferramentas
AGENT_CONFIG = {
    "triage": {"prompt": TRIAGE_SYSTEM_PROMPT, "tools": TRIAGE_TOOLS},
    "credit": {"prompt": CREDIT_SYSTEM_PROMPT, "tools": CREDIT_TOOLS},
    "interview": {"prompt": INTERVIEW_SYSTEM_PROMPT, "tools": INTERVIEW_TOOLS},
    "exchange": {"prompt": EXCHANGE_SYSTEM_PROMPT, "tools": EXCHANGE_TOOLS},
}

# Mapa global de ferramentas por nome (deduplicado)
ALL_TOOLS = list(
    {
        tool.name: tool
        for config in AGENT_CONFIG.values()
        for tool in config["tools"]
    }.values()
)
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}

# Mapa de ferramentas de transferência → agente de destino
TRANSFER_MAP = {
    "transfer_to_credit": "credit",
    "transfer_to_exchange": "exchange",
    "transfer_to_interview": "interview",
    "transfer_to_triage": "triage",
}


def _get_initial_state() -> dict:
    """Retorna o estado inicial padrão para uma nova conversa."""
    return {
        "authenticated": False,
        "client_data": None,
        "auth_attempts": 0,
        "current_agent": "triage",
        "should_end": False,
    }


# --- Nó genérico de agente ---


def _agent_node(state: AgentState, agent_name: str) -> dict:
    """Nó genérico: chama o LLM com o prompt e ferramentas do agente."""
    config = AGENT_CONFIG[agent_name]
    llm = get_llm()
    llm_with_tools = llm.bind_tools(config["tools"])

    messages = [config["prompt"]] + state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        logger.error("Erro ao chamar o LLM (%s): %s", agent_name, e)
        error_msg = AIMessage(
            content=(
                "Peço desculpas, mas estou com dificuldades técnicas no momento. "
                "Por favor, tente novamente em alguns instantes."
            )
        )
        return {"messages": [error_msg]}

    return {"messages": [response]}


# --- Nós específicos (wrappers) ---


def triage_node(state: AgentState) -> dict:
    """Nó do agente de triagem."""
    return _agent_node(state, "triage")


def credit_node(state: AgentState) -> dict:
    """Nó do agente de crédito."""
    return _agent_node(state, "credit")


def interview_node(state: AgentState) -> dict:
    """Nó do agente de entrevista de crédito."""
    return _agent_node(state, "interview")


def exchange_node(state: AgentState) -> dict:
    """Nó do agente de câmbio."""
    return _agent_node(state, "exchange")


# --- Nó de ferramentas ---


def tool_node(state: AgentState) -> dict:
    """Executa as tool calls da última mensagem e atualiza o estado."""
    last_message: AIMessage = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    state_updates: dict = {}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool = TOOL_MAP.get(tool_name)

        if tool is None:
            logger.warning("Ferramenta desconhecida chamada: %s", tool_name)
            tool_messages.append(
                ToolMessage(
                    content=f"ERRO: Ferramenta '{tool_name}' não disponível.",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
            continue

        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            logger.error("Erro ao executar ferramenta %s: %s", tool_name, e)
            result = (
                "ERRO_SISTEMA: Ocorreu um erro ao executar esta operação. "
                "Tente novamente."
            )

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

        result_str = str(result)

        # --- Atualizações de estado por ferramenta ---

        if tool_name == "end_conversation":
            state_updates["should_end"] = True

        elif tool_name == "authenticate_client":
            _handle_auth_result(result_str, state, state_updates)

        elif tool_name in TRANSFER_MAP:
            state_updates["current_agent"] = TRANSFER_MAP[tool_name]

        elif tool_name == "update_client_score":
            _handle_score_update(result_str, state, state_updates)

        elif tool_name == "request_limit_increase":
            _handle_limit_increase(result_str, tool_args, state, state_updates)

    return {"messages": tool_messages, **state_updates}


def _handle_auth_result(
    result_str: str, state: AgentState, state_updates: dict
) -> None:
    """Processa o resultado de authenticate_client e atualiza o estado."""
    if result_str.startswith("SUCESSO"):
        state_updates["authenticated"] = True
        try:
            parts = result_str.split(", ")
            client_data = {}
            for part in parts:
                if "Nome:" in part:
                    client_data["nome"] = part.split("Nome:")[1].strip()
                elif "CPF:" in part:
                    client_data["cpf"] = part.split("CPF:")[1].strip()
                elif "Limite de crédito:" in part:
                    val = part.split("R$")[1].strip()
                    client_data["limite_credito"] = float(val)
                elif "Score:" in part:
                    client_data["score"] = int(part.split("Score:")[1].strip())
            state_updates["client_data"] = client_data
        except (IndexError, ValueError) as e:
            logger.warning("Erro ao parsear dados do cliente: %s", e)
            state_updates["client_data"] = {"raw": result_str}
    elif result_str.startswith("FALHA"):
        state_updates["auth_attempts"] = state.get("auth_attempts", 0) + 1


def _handle_score_update(
    result_str: str, state: AgentState, state_updates: dict
) -> None:
    """Processa o resultado de update_client_score e atualiza client_data."""
    if result_str.startswith("ATUALIZADO") and state.get("client_data"):
        try:
            new_score = int(result_str.split("para ")[1].rstrip("."))
            client_data = dict(state["client_data"])
            client_data["score"] = new_score
            state_updates["client_data"] = client_data
        except (IndexError, ValueError) as e:
            logger.warning("Erro ao parsear novo score: %s", e)


def _handle_limit_increase(
    result_str: str,
    tool_args: dict,
    state: AgentState,
    state_updates: dict,
) -> None:
    """Processa o resultado de request_limit_increase e atualiza client_data."""
    if result_str.startswith("APROVADO") and state.get("client_data"):
        try:
            new_limit = float(tool_args["new_limit"])
            client_data = dict(state["client_data"])
            client_data["limite_credito"] = new_limit
            state_updates["client_data"] = client_data
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Erro ao atualizar limite no estado: %s", e)


# --- Roteamento condicional ---


def route_entry(state: AgentState) -> str:
    """Decide qual agente deve processar a mensagem na entrada do grafo."""
    return state.get("current_agent", "triage")


def should_continue(state: AgentState) -> str:
    """Decide se deve executar ferramentas ou encerrar o turno."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def route_after_tools(state: AgentState) -> str:
    """Após ferramentas, roteia para o agente atual (que pode ter mudado
    por uma ferramenta de transferência)."""
    return state.get("current_agent", "triage")


# --- Construção do grafo ---


def build_graph() -> StateGraph:
    """Constrói e retorna o grafo compilado do sistema multi-agente."""
    builder = StateGraph(AgentState)

    # Nós de agentes
    agent_names = ["triage", "credit", "interview", "exchange"]
    agent_map = {n: n for n in agent_names}

    # Nós de agentes
    builder.add_node("triage", triage_node)
    builder.add_node("credit", credit_node)
    builder.add_node("interview", interview_node)
    builder.add_node("exchange", exchange_node)
    builder.add_node("tools", tool_node)

    # Entrada condicional: roteia para o agente atual
    builder.add_conditional_edges(START, route_entry, agent_map)

    # Cada agente → should_continue → (tools | END)
    for name in agent_names:
        builder.add_conditional_edges(
            name, should_continue, {"tools": "tools", END: END}
        )

    # Após ferramentas → roteia para o agente correto
    builder.add_conditional_edges("tools", route_after_tools, agent_map)

    return builder.compile()


graph = build_graph()
