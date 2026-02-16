"""Testes para configuração do LLM e variáveis de ambiente."""

import pytest

from src.config import get_llm, MODEL_NAME


class TestGetLlm:
    """Testes para a função get_llm."""

    def test_raises_when_api_key_missing(self, monkeypatch):
        """Deve lançar EnvironmentError quando LLM_API_KEY não está definida."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
            get_llm()

    def test_returns_llm_when_api_key_present(self, monkeypatch):
        """Deve retornar uma instância de ChatGroq quando a chave existe."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key-for-testing")
        llm = get_llm()
        assert llm is not None
        assert llm.model_name == MODEL_NAME

    def test_model_name_is_expected(self):
        """O modelo configurado deve ser o llama-3.3-70b-versatile."""
        assert MODEL_NAME == "llama-3.3-70b-versatile"
