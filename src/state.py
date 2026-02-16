"""Definição do estado compartilhado entre os agentes."""

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado compartilhado entre todos os nós do grafo.

    Attributes:
        messages: Histórico de mensagens (merge automático via ``add_messages``).
        authenticated: Indica se o cliente foi autenticado.
        client_data: Dados do cliente autenticado (nome, cpf, limite, score).
        auth_attempts: Quantidade de tentativas de autenticação realizadas.
        current_agent: Identificador do agente ativo.
        should_end: Sinaliza que a conversa deve ser encerrada.
    """

    messages: Annotated[list, add_messages]
    authenticated: bool
    client_data: Optional[dict[str, Any]]
    auth_attempts: int
    current_agent: str
    should_end: bool