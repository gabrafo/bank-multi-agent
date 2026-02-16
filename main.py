import logging

from langchain_core.messages import HumanMessage

from src.graph import build_graph, _get_initial_state

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Loop principal de interação via CLI para testes."""
    print("=" * 60)
    print("  Banco Ágil - Atendimento Virtual (CLI)")
    print("=" * 60)
    print("(Digite 'sair' para encerrar)\n")

    try:
        graph = build_graph()
    except Exception as e:
        logger.error("Falha ao inicializar o sistema: %s", e)
        print(f"Erro ao inicializar: {e}")
        return

    state = _get_initial_state()
    state["messages"] = []

    # Primeira interação: o agente inicia a conversa
    try:
        result = graph.invoke(state)
        state = result
        _print_last_ai_message(state)
    except Exception as e:
        logger.error("Erro na execução do grafo: %s", e)
        print("Ocorreu um erro técnico. Tente novamente mais tarde.")
        return

    # Loop de conversa
    while not state.get("should_end", False):
        try:
            user_input = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAtendimento encerrado. Até logo!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("sair", "exit", "quit"):
            # Envia a mensagem de encerramento para o agente processar
            user_input = "Quero encerrar o atendimento."

        state["messages"].append(HumanMessage(content=user_input))

        try:
            result = graph.invoke(state)
            state = result
            _print_last_ai_message(state)
        except Exception as e:
            logger.error("Erro na execução do grafo: %s", e)
            print(
                "\nAssistente: Peço desculpas, ocorreu um erro técnico. "
                "Tente novamente."
            )

    print("\n" + "=" * 60)
    print("  Atendimento encerrado. Obrigado por usar o Banco Ágil!")
    print("=" * 60)


def _print_last_ai_message(state: dict):
    """Imprime a última mensagem do assistente."""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
            print(f"\nAssistente: {msg.content}")
            return


if __name__ == "__main__":
    main()
