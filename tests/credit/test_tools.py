"""Testes para as ferramentas de crédito: query_credit_limit e request_limit_increase."""

import csv
import os

import pytest

from src.tools.credit import query_credit_limit, request_limit_increase


# --- Fixtures ---


@pytest.fixture
def credit_csvs(tmp_path, monkeypatch):
    """Cria CSVs temporários de clientes, score_limite e solicitações."""
    clients = tmp_path / "clientes.csv"
    clients.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n"
        "98765432100,22/07/1985,Maria Santos,8000.00,780\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(clients))

    score_limit = tmp_path / "score_limite.csv"
    score_limit.write_text(
        "score_minimo,score_maximo,limite_maximo\n"
        "0,399,2000.00\n"
        "400,549,3500.00\n"
        "550,649,5000.00\n"
        "650,749,8000.00\n"
        "750,849,12000.00\n"
        "850,1000,20000.00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.credit.SCORE_LIMIT_CSV", str(score_limit))

    requests = tmp_path / "solicitacoes.csv"
    requests.write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.credit.REQUESTS_CSV", str(requests))

    return {
        "clients": clients,
        "score_limit": score_limit,
        "requests": requests,
    }


@pytest.fixture
def missing_clients_csv(monkeypatch):
    monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", "/inexistente/clientes.csv")


@pytest.fixture
def missing_score_csv(tmp_path, monkeypatch):
    clients = tmp_path / "clientes.csv"
    clients.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(clients))
    monkeypatch.setattr("src.tools.credit.SCORE_LIMIT_CSV", "/inexistente/score.csv")


# --- query_credit_limit ---


class TestQueryCreditLimit:
    def test_success(self, credit_csvs):
        result = query_credit_limit.invoke({"cpf": "12345678901"})
        assert result.startswith("LIMITE")
        assert "João Silva" in result
        assert "5000.00" in result
        assert "650" in result

    def test_success_formatted_cpf(self, credit_csvs):
        result = query_credit_limit.invoke({"cpf": "123.456.789-01"})
        assert result.startswith("LIMITE")
        assert "João Silva" in result

    def test_client_not_found(self, credit_csvs):
        result = query_credit_limit.invoke({"cpf": "00000000000"})
        assert result.startswith("ERRO")
        assert "não encontrado" in result

    def test_csv_not_found(self, missing_clients_csv):
        result = query_credit_limit.invoke({"cpf": "12345678901"})
        assert result.startswith("ERRO_SISTEMA")

    def test_csv_read_error(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_bytes(b"\x00\x01\x02\xff\xfe")
        monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(csv_path))
        result = query_credit_limit.invoke({"cpf": "12345678901"})
        # May return ERRO or ERRO_SISTEMA depending on how parsing fails
        assert "ERRO" in result


# --- request_limit_increase ---


class TestRequestLimitIncrease:
    def test_approved(self, credit_csvs):
        """Score 650 → max 8000. Solicita 7000 → aprovado."""
        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert result.startswith("APROVADO")
        assert "7000.00" in result

        # Verifica que o CSV de solicitações foi atualizado
        with open(credit_csvs["requests"], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["status_pedido"] == "aprovado"
        assert rows[0]["novo_limite_solicitado"] == "7000.0"

        # Verifica que o limite foi atualizado no CSV de clientes
        with open(credit_csvs["clients"], "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["cpf"] == "12345678901":
                    assert row["limite_credito"] == "7000.00"

    def test_rejected_score_insufficient(self, credit_csvs):
        """Score 650 → max 8000. Solicita 10000 → rejeitado."""
        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 10000.0}
        )
        assert result.startswith("REJEITADO")
        assert "8000.00" in result
        assert "entrevista" in result

    def test_limit_lower_than_current(self, credit_csvs):
        """Solicita limite menor que o atual → INFORMAÇÃO."""
        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 3000.0}
        )
        assert result.startswith("INFORMAÇÃO")

    def test_limit_equal_to_current(self, credit_csvs):
        """Solicita limite igual ao atual → INFORMAÇÃO."""
        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 5000.0}
        )
        assert result.startswith("INFORMAÇÃO")

    def test_client_not_found(self, credit_csvs):
        result = request_limit_increase.invoke(
            {"cpf": "00000000000", "new_limit": 7000.0}
        )
        assert result.startswith("ERRO")
        assert "não encontrado" in result

    def test_clients_csv_not_found(self, missing_clients_csv):
        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert result.startswith("ERRO_SISTEMA")

    def test_score_csv_not_found(self, missing_score_csv, tmp_path, monkeypatch):
        requests = tmp_path / "solicitacoes.csv"
        requests.write_text(
            "cpf_cliente,data_hora_solicitacao,limite_atual,"
            "novo_limite_solicitado,status_pedido\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.credit.REQUESTS_CSV", str(requests))

        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert result.startswith("ERRO_SISTEMA")

    def test_requests_csv_write_error(self, credit_csvs, monkeypatch):
        """Quando não é possível escrever no CSV de solicitações."""
        monkeypatch.setattr(
            "src.tools.credit.REQUESTS_CSV", "/readonly/solicitacoes.csv"
        )
        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert result.startswith("ERRO_SISTEMA")

    def test_approved_but_client_update_fails(self, credit_csvs, monkeypatch):
        """Aprovado mas falha ao atualizar limite no CSV de clientes."""
        # Torna o CSV de clientes read-only após a leitura inicial
        original_open = open

        call_count = 0

        def mock_open(path, *args, **kwargs):
            nonlocal call_count
            if str(path) == str(credit_csvs["clients"]) and "w" in str(args):
                raise PermissionError("read-only")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert result.startswith("APROVADO")
        assert "erro" in result.lower()

    def test_formatted_cpf(self, credit_csvs):
        result = request_limit_increase.invoke(
            {"cpf": "123.456.789-01", "new_limit": 6000.0}
        )
        assert result.startswith("APROVADO")

    def test_clients_csv_generic_read_error(self, tmp_path, monkeypatch):
        """Erro genérico (não FileNotFoundError) ao ler clientes."""
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_bytes(b"\x00\x01\x02\xff\xfe")
        monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(csv_path))

        score_limit = tmp_path / "score_limite.csv"
        score_limit.write_text(
            "score_minimo,score_maximo,limite_maximo\n"
            "0,1000,20000.00\n",
        )
        monkeypatch.setattr("src.tools.credit.SCORE_LIMIT_CSV", str(score_limit))

        requests = tmp_path / "sol.csv"
        requests.write_text(
            "cpf_cliente,data_hora_solicitacao,limite_atual,"
            "novo_limite_solicitado,status_pedido\n",
        )
        monkeypatch.setattr("src.tools.credit.REQUESTS_CSV", str(requests))

        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert "ERRO" in result

    def test_score_csv_generic_read_error(self, tmp_path, monkeypatch):
        """Erro genérico (não FileNotFoundError) ao ler score_limite."""
        clients = tmp_path / "clientes.csv"
        clients.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,650\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(clients))

        score_limit = tmp_path / "score_limite.csv"
        score_limit.write_bytes(b"\x00\x01\x02\xff\xfe")
        monkeypatch.setattr("src.tools.credit.SCORE_LIMIT_CSV", str(score_limit))

        requests = tmp_path / "sol.csv"
        requests.write_text(
            "cpf_cliente,data_hora_solicitacao,limite_atual,"
            "novo_limite_solicitado,status_pedido\n",
        )
        monkeypatch.setattr("src.tools.credit.REQUESTS_CSV", str(requests))

        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 7000.0}
        )
        assert "ERRO_SISTEMA" in result

    def test_score_out_of_range_returns_error(self, tmp_path, monkeypatch):
        """Score fora de qualquer faixa na tabela → ERRO_SISTEMA."""
        clients = tmp_path / "clientes.csv"
        clients.write_text(
            "cpf,data_nascimento,nome,limite_credito,score\n"
            "12345678901,15/03/1990,João Silva,5000.00,1500\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.credit.CLIENTS_CSV", str(clients))

        score_limit = tmp_path / "score_limite.csv"
        score_limit.write_text(
            "score_minimo,score_maximo,limite_maximo\n"
            "0,1000,20000.00\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("src.tools.credit.SCORE_LIMIT_CSV", str(score_limit))

        requests = tmp_path / "sol.csv"
        requests.write_text(
            "cpf_cliente,data_hora_solicitacao,limite_atual,"
            "novo_limite_solicitado,status_pedido\n",
        )
        monkeypatch.setattr("src.tools.credit.REQUESTS_CSV", str(requests))

        result = request_limit_increase.invoke(
            {"cpf": "12345678901", "new_limit": 25000.0}
        )
        assert "ERRO_SISTEMA" in result
