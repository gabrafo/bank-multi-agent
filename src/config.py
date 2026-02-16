import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def get_llm() -> ChatGroq:
    """Retorna a instância do LLM configurada com a API do Groq."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variável de ambiente LLM_API_KEY não encontrada. "
            "Configure-a no arquivo .env na raiz do projeto."
        )
    return ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.3,
    )