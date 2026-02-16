import json
import logging

from langchain_core.messages import AIMessage, ToolMessage

from src.agents.triage import TRIAGE_SYSTEM_PROMPT, TRIAGE_TOOLS
from src.config import get_llm
from src.state import AgentState

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# Mapa de ferramentas por nome para execução
TOOL_MAP = {tool.name: tool for tool in TRIAGE_TOOLS}


def _get_initial_state() -> dict:
    """Retorna o estado inicial padrão para uma nova conversa."""
    return {
        "authenticated": False,
        "client_data": None,
        "auth_attempts": 0,
        "current_agent": "triage",
        "should_end": False,
    }


# --- Nós do grafo ---


def triage_node(state: AgentState) -> dict:
    """Nó do agente de triagem: chama o LLM com o prompt de sistema e ferramentas."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TRIAGE_TOOLS)

    # Monta as mensagens: system prompt + histórico
    messages = [TRIAGE_SYSTEM_PROMPT] + state["messages"]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        logger.error("Erro ao chamar o LLM: %s", e)
        error_msg = AIMessage(
            content=(
                "Peço desculpas, mas estou com dificuldades técnicas no momento. "
                "Por favor, tente novamente em alguns instantes."
            )
        )
        return {"messages": [error_msg]}

    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Nó de execução de ferramentas: executa as tool calls e atualiza o estado."""
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

        # Atualiza o estado com base no resultado da ferramenta
        if tool_name == "end_conversation":
            state_updates["should_end"] = True

        elif tool_name == "authenticate_client":
            result_str = str(result)
            if result_str.startswith("SUCESSO"):
                # Extrai dados do cliente da resposta
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
                            client_data["score"] = int(
                                part.split("Score:")[1].strip()
                            )
                    state_updates["client_data"] = client_data
                except (IndexError, ValueError) as e:
                    logger.warning("Erro ao parsear dados do cliente: %s", e)
                    state_updates["client_data"] = {"raw": result_str}
            elif result_str.startswith("FALHA"):
                state_updates["auth_attempts"] = state.get("auth_attempts", 0) + 1

    return {"messages": tool_messages, **state_updates}


# --- Roteamento condicional ---


def should_continue(state: AgentState) -> str:
    """Decide o próximo passo após o nó do agente."""
    last_message = state["messages"][-1]

    # Se o LLM fez chamadas de ferramenta, vá para o nó de ferramentas
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Caso contrário, a resposta é final (texto puro)
    return END


def after_tools(state: AgentState) -> str:
    """Decide o próximo passo após a execução de ferramentas.

    Sempre volta ao agente para que ele possa processar o resultado das
    ferramentas e produzir uma resposta adequada (inclusive mensagem de
    despedida após end_conversation).
    """
    return "triage"

def build_graph() -> StateGraph:
    """Constrói e retorna o grafo compilado do sistema de atendimento."""
    builder = StateGraph(AgentState)

    # Adiciona nós
    builder.add_node("triage", triage_node)
    builder.add_node("tools", tool_node)

    # Ponto de entrada
    builder.set_entry_point("triage")

    # Arestas condicionais
    builder.add_conditional_edges("triage", should_continue, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", after_tools, {"triage": "triage"})

    return builder.compile()

graph = build_graph()
