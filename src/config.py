"""Configuração do LLM e caminhos de dados do projeto."""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def get_llm() -> ChatGoogleGenerativeAI:
    """Cria e retorna uma instância do ``ChatGoogleGenerativeAI``.

    Lê a chave de API da variável de ambiente ``LLM_API_KEY`` e o nome do
    modelo da variável ``LLM_MODEL_NAME``.

    Returns:
        Instância configurada do LLM.

    Raises:
        EnvironmentError: Se ``LLM_API_KEY`` ou ``LLM_MODEL_NAME`` não estiverem definidas.
    """
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or not model_name:
        raise EnvironmentError(
            "Variável de ambiente LLM_API_KEY ou LLM_MODEL_NAME não encontrada. "
            "Configure-as no arquivo .env na raiz do projeto."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )