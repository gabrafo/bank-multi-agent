"""Ferramentas comuns a todos os agentes."""

from langchain_core.tools import tool


@tool
def end_conversation() -> str:
    """Encerra o atendimento com o cliente.

    Returns:
        Confirmação de encerramento.
    """
    return "ENCERRAMENTO: Atendimento finalizado com sucesso."
