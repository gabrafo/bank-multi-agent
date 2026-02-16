"""Configuração do LLM e caminhos de dados do projeto."""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def get_llm() -> ChatGroq:
    """Cria e retorna uma instância do ``ChatGroq``.

    Lê a chave de API da variável de ambiente ``LLM_API_KEY`` e o nome do modelo da variável ``LLM_MODEL_NAME``.

    Returns:
        Instância configurada do LLM.

    Raises:
        EnvironmentError: Se ``LLM_API_KEY`` ou ``LLM_MODEL_NAME`` não estiverem definidas.
    """
    model_name = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or not model_name:
        raise EnvironmentError(
            "Variável de ambiente LLM_API_KEY ou LLM_MODEL_NAME não encontrada. "
            "Configure-as no arquivo .env na raiz do projeto."
        )
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.3,
    )