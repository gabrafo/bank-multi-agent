import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import build_graph, _get_initial_state

st.set_page_config(
    page_title="Banco Ágil",
    page_icon="🏦",
    layout="centered",
)

# --- Inicialização do estado da sessão ---

if "graph" not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except Exception as e:
        st.error(f"Erro ao inicializar o sistema: {e}")
        st.stop()

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {**_get_initial_state(), "messages": []}
    # Primeira interação: agente inicia a conversa
    try:
        result = st.session_state.graph.invoke(st.session_state.agent_state)
        st.session_state.agent_state = result
    except Exception:
        pass

if "chat_history" not in st.session_state:
    # Extrai a saudação inicial do agente
    st.session_state.chat_history = []
    for msg in st.session_state.agent_state["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": msg.content}
            )
            break

# --- Header ---

st.title("🏦 Banco Ágil")
st.caption("Atendimento Virtual Inteligente")

# --- Exibir histórico do chat ---

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input do usuário ---

if st.session_state.agent_state.get("should_end", False):
    st.info("Atendimento encerrado. Atualize a página para iniciar uma nova conversa.")
    if st.button("Nova conversa"):
        for key in ["agent_state", "chat_history"]:
            del st.session_state[key]
        st.rerun()
else:
    if user_input := st.chat_input("Digite sua mensagem..."):
        # Mostra a mensagem do usuário
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Adiciona ao estado do agente e invoca o grafo
        st.session_state.agent_state["messages"].append(
            HumanMessage(content=user_input)
        )

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    result = st.session_state.graph.invoke(
                        st.session_state.agent_state
                    )
                    st.session_state.agent_state = result

                    # Extrai a última resposta do assistente
                    response = ""
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage) and msg.content:
                            response = msg.content
                            break

                    if not response:
                        response = "Desculpe, não consegui processar sua solicitação."

                except Exception:
                    response = (
                        "Peço desculpas, estou com dificuldades técnicas. "
                        "Tente novamente em alguns instantes."
                    )

                st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )

        # Rerun para atualizar estado (ex: mostrar botão de nova conversa)
        if st.session_state.agent_state.get("should_end", False):
            st.rerun()
