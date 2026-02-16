"""Testes para configuração do LLM e variáveis de ambiente."""

import pytest

from src.config import get_llm


class TestGetLlm:
    """Testes para a função get_llm."""

    def test_raises_when_api_key_missing(self, monkeypatch):
        """Deve lançar EnvironmentError quando LLM_API_KEY não está definida."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
            get_llm()

    def test_returns_llm_when_api_key_present(self, monkeypatch):
        """Deve retornar uma instância de ChatGoogleGenerativeAI quando a chave existe."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key-for-testing")
        monkeypatch.setenv("LLM_MODEL_NAME", "gemini-2.5-flash")
        llm = get_llm()
        assert llm is not None
        assert llm.model == "gemini-2.5-flash"

    def test_default_model_name(self, monkeypatch):
        """Deve usar gemini-2.5-flash como modelo padrão."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key-for-testing")
        monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
        llm = get_llm()
        assert llm.model == "gemini-2.5-flash"

    def test_raises_when_model_name_empty(self, monkeypatch):
        """Deve lançar EnvironmentError quando LLM_MODEL_NAME está vazia."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key-for-testing")
        monkeypatch.setenv("LLM_MODEL_NAME", "")
        with pytest.raises(EnvironmentError, match="LLM_MODEL_NAME"):
            get_llm()
