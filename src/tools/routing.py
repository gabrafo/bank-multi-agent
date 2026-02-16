from langchain_core.tools import tool


@tool
def transfer_to_credit() -> str:
    """Transfere o atendimento para o serviço de crédito do Banco Ágil.
    Use quando o cliente autenticado deseja consultar ou alterar seu limite
    de crédito.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para o serviço de crédito."


@tool
def transfer_to_interview() -> str:
    """Transfere o atendimento para a entrevista de crédito do Banco Ágil.
    Use quando o cliente deseja tentar recalcular seu score de crédito
    por meio de uma entrevista financeira.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para a entrevista de crédito."


@tool
def transfer_to_triage() -> str:
    """Transfere o atendimento de volta para a triagem do Banco Ágil.
    Use quando o atendimento no serviço atual foi concluído e o cliente
    pode ter outras necessidades.

    Returns:
        Confirmação de transferência.
    """
    return "TRANSFERÊNCIA: Cliente encaminhado para a triagem."
