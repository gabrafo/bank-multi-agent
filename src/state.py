from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado compartilhado entre todos os agentes do grafo."""

    # Histórico de mensagens da conversa (com merge automático do LangGraph)
    messages: Annotated[list, add_messages]

    # Flag de autenticação do cliente
    authenticated: bool

    # Dados do cliente autenticado (nome, cpf, limite, score, etc.)
    client_data: Optional[dict[str, Any]]

    # Número de tentativas de autenticação realizadas
    auth_attempts: int

    # Agente ativo no momento (triagem, credito, entrevista, cambio)
    current_agent: str

    # Flag para indicar que a conversa deve ser encerrada
    should_end: bool