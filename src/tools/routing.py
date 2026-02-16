"""Ferramentas de transferência implícita entre agentes."""

from langchain_core.tools import tool


@tool
def transfer_to_credit() -> str:
    """Transfere o atendimento para o serviço de crédito.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para o serviço de crédito."


@tool
def transfer_to_interview() -> str:
    """Transfere o atendimento para a entrevista de crédito.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para a entrevista de crédito."


@tool
def transfer_to_exchange() -> str:
    """Transfere o atendimento para o serviço de câmbio.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para o serviço de câmbio."


@tool
def transfer_to_triage() -> str:
    """Transfere o atendimento de volta para a triagem.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para a triagem."
