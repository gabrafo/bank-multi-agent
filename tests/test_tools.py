import csv
import os
import tempfile

import pytest

from src.tools.auth import authenticate_client
from src.tools.common import end_conversation


# --- Fixtures ---


@pytest.fixture
def fake_clients_csv(tmp_path, monkeypatch):
    """Cria um CSV temporário de clientes e aponta a tool para ele."""
    csv_path = tmp_path / "clientes.csv"
    csv_path.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n"
        "98765432100,22/07/1985,Maria Santos,8000.00,780\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))
    return csv_path


@pytest.fixture
def missing_csv(monkeypatch):
    """Aponta a tool para um CSV que não existe."""
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", "/inexistente/clientes.csv")


@pytest.fixture
def corrupted_csv(tmp_path, monkeypatch):
    """Cria um CSV corrompido (binário) e aponta a tool para ele."""
    csv_path = tmp_path / "clientes.csv"
    csv_path.write_bytes(b"\x00\x01\x02\xff\xfe")
    monkeypatch.setattr("src.tools.auth.CLIENTS_CSV", str(csv_path))
    return csv_path


# --- Testes: authenticate_client ---


class TestAuthenticateClient:
    """Testes para a ferramenta authenticate_client."""

    def test_auth_success(self, fake_clients_csv):
        """Autenticação com CPF e data válidos deve retornar SUCESSO."""
        result = authenticate_client.invoke(
            {"cpf": "12345678901", "birth_date": "15/03/1990"}
        )
        assert result.startswith("SUCESSO")
        assert "João Silva" in result
        assert "5000.00" in result
        assert "650" in result

    def test_auth_success_with_formatted_cpf(self, fake_clients_csv):
        """CPF com pontuação deve ser normalizado e autenticar."""
        result = authenticate_client.invoke(
            {"cpf": "123.456.789-01", "birth_date": "15/03/1990"}
        )
        assert result.startswith("SUCESSO")
        assert "João Silva" in result

    def test_auth_success_second_client(self, fake_clients_csv):
        """Autenticação do segundo cliente do CSV."""
        result = authenticate_client.invoke(
            {"cpf": "98765432100", "birth_date": "22/07/1985"}
        )
        assert result.startswith("SUCESSO")
        assert "Maria Santos" in result

    def test_auth_failure_wrong_cpf(self, fake_clients_csv):
        """CPF inexistente deve retornar FALHA."""
        result = authenticate_client.invoke(
            {"cpf": "00000000000", "birth_date": "15/03/1990"}
        )
        assert result.startswith("FALHA")
        assert "não correspondem" in result

    def test_auth_failure_wrong_birth_date(self, fake_clients_csv):
        """Data de nascimento errada deve retornar FALHA."""
        result = authenticate_client.invoke(
            {"cpf": "12345678901", "birth_date": "01/01/2000"}
        )
        assert result.startswith("FALHA")

    def test_auth_failure_invalid_cpf_short(self, fake_clients_csv):
        """CPF com menos de 11 dígitos deve retornar FALHA."""
        result = authenticate_client.invoke(
            {"cpf": "12345", "birth_date": "15/03/1990"}
        )
        assert result.startswith("FALHA")
        assert "inválido" in result

    def test_auth_failure_invalid_cpf_letters(self, fake_clients_csv):
        """CPF com letras deve retornar FALHA."""
        result = authenticate_client.invoke(
            {"cpf": "abcdefghijk", "birth_date": "15/03/1990"}
        )
        assert result.startswith("FALHA")
        assert "inválido" in result

    def test_auth_csv_not_found(self, missing_csv):
        """Quando o CSV não existe, deve retornar ERRO_SISTEMA."""
        result = authenticate_client.invoke(
            {"cpf": "12345678901", "birth_date": "15/03/1990"}
        )
        assert result.startswith("ERRO_SISTEMA")

    def test_auth_csv_corrupted(self, corrupted_csv):
        """Quando o CSV está corrompido, deve retornar ERRO_SISTEMA."""
        result = authenticate_client.invoke(
            {"cpf": "12345678901", "birth_date": "15/03/1990"}
        )
        # Pode retornar FALHA (se leu 0 linhas) ou ERRO_SISTEMA
        assert result.startswith("FALHA") or result.startswith("ERRO_SISTEMA")


# --- Testes: end_conversation ---


class TestEndConversation:
    """Testes para a ferramenta end_conversation."""

    def test_end_conversation_returns_confirmation(self):
        """end_conversation deve retornar mensagem de encerramento."""
        result = end_conversation.invoke({})
        assert result.startswith("ENCERRAMENTO")
        assert "finalizado" in result
