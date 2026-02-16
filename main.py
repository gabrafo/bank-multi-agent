"""Interface CLI do atendimento virtual do Banco Ágil."""

from langchain_core.messages import AIMessage, HumanMessage

from src.graph import build_graph, _get_initial_state


def main() -> None:
    """Loop principal de interação via terminal."""
    graph = build_graph()
    state = {**_get_initial_state(), "messages": []}

    # Invoca o grafo sem mensagem para obter a saudação inicial
    try:
        state = graph.invoke(state)
    except Exception as e:
        print(f"Erro ao inicializar: {e}")
        return

    # Exibe a saudação do agente de triagem
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"Atendente: {msg.content}")
            break

    while not state.get("should_end", False):
        user_input = input("\nVocê: ").strip()
        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"\nErro de sistema: {e}")
            continue

        # Exibe a última resposta do agente
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nAtendente: {msg.content}")
                break

    print("\nAtendimento encerrado. Até logo!")


if __name__ == "__main__":
    main()
