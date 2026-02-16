from langchain_core.tools import tool


@tool
def end_conversation() -> str:
    """Encerra o atendimento com o cliente. Deve ser chamada quando o cliente
    solicitar o fim da conversa ou quando o atendimento for concluído.

    Returns:
        Confirmação de encerramento.
    """
    return "ENCERRAMENTO: Atendimento finalizado com sucesso."
