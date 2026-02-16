"""Testes para as ferramentas de entrevista de crédito."""

import csv

import pytest

from src.tools.interview import (
    PESO_DEPENDENTES,
    PESO_DEPENDENTES_3_MAIS,
    PESO_DIVIDAS,
    PESO_EMPREGO,
    PESO_RENDA,
    calculate_credit_score,
    update_client_score,
)


# --- Fixtures ---


@pytest.fixture
def fake_clients_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "clientes.csv"
    csv_path.write_text(
        "cpf,data_nascimento,nome,limite_credito,score\n"
        "12345678901,15/03/1990,João Silva,5000.00,650\n"
        "98765432100,22/07/1985,Maria Santos,8000.00,780\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.tools.interview.CLIENTS_CSV", str(csv_path))
    return csv_path


@pytest.fixture
def missing_csv(monkeypatch):
    monkeypatch.setattr("src.tools.interview.CLIENTS_CSV", "/inexistente/c.csv")


# --- calculate_credit_score ---


class TestCalculateCreditScore:
    def test_formal_no_debts(self):
        """Emprego formal, sem dívidas, 0 dependentes."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 2000.0,
            "num_dependentes": 0,
            "tem_dividas": "não",
        })
        assert result.startswith("SCORE_CALCULADO")
        # (5000/2001)*30 + 300 + 100 + 100 = 74.96 + 300 + 100 + 100 ≈ 575
        assert "575" in result

    def test_autonomo_with_debts(self):
        """Emprego autônomo, com dívidas, 1 dependente."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 3000.0,
            "tipo_emprego": "autônomo",
            "despesas_fixas": 1500.0,
            "num_dependentes": 1,
            "tem_dividas": "sim",
        })
        assert result.startswith("SCORE_CALCULADO")
        # (3000/1501)*30 + 200 + 80 + (-100) = 59.96 + 200 + 80 - 100 = 240
        assert "240" in result

    def test_desempregado(self):
        """Emprego desempregado, alto custo."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 0.0,
            "tipo_emprego": "desempregado",
            "despesas_fixas": 500.0,
            "num_dependentes": 2,
            "tem_dividas": "sim",
        })
        assert result.startswith("SCORE_CALCULADO")
        # (0/501)*30 + 0 + 60 + (-100) = 0 + 0 + 60 - 100 = -40 → clamped to 0
        assert "SCORE_CALCULADO: 0." in result

    def test_3_plus_dependentes(self):
        """3+ dependentes usa peso especial."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 10000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 0.0,
            "num_dependentes": 5,
            "tem_dividas": "não",
        })
        assert result.startswith("SCORE_CALCULADO")
        # (10000/1)*30 + 300 + 30 + 100 = 300000 + 430 → clamped 1000
        assert "SCORE_CALCULADO: 1000." in result

    def test_score_clamped_to_1000(self):
        """Score muito alto é limitado a 1000."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 50000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 0.0,
            "num_dependentes": 0,
            "tem_dividas": "não",
        })
        assert "SCORE_CALCULADO: 1000." in result

    def test_score_clamped_to_0(self):
        """Score negativo é limitado a 0."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 0.0,
            "tipo_emprego": "desempregado",
            "despesas_fixas": 1000.0,
            "num_dependentes": 0,
            "tem_dividas": "sim",
        })
        # 0 + 0 + 100 + (-100) = 0
        assert "SCORE_CALCULADO: 0." in result

    def test_invalid_tipo_emprego(self):
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "estagiário",
            "despesas_fixas": 1000.0,
            "num_dependentes": 0,
            "tem_dividas": "não",
        })
        assert result.startswith("ERRO")
        assert "inválido" in result

    def test_invalid_tem_dividas(self):
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 1000.0,
            "num_dependentes": 0,
            "tem_dividas": "talvez",
        })
        assert result.startswith("ERRO")
        assert "inválida" in result

    def test_negative_renda(self):
        result = calculate_credit_score.invoke({
            "renda_mensal": -1000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 500.0,
            "num_dependentes": 0,
            "tem_dividas": "não",
        })
        assert result.startswith("ERRO")
        assert "negativa" in result

    def test_negative_despesas(self):
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": -500.0,
            "num_dependentes": 0,
            "tem_dividas": "não",
        })
        assert result.startswith("ERRO")
        assert "negativas" in result

    def test_negative_dependentes(self):
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 500.0,
            "num_dependentes": -1,
            "tem_dividas": "não",
        })
        assert result.startswith("ERRO")
        assert "negativo" in result

    def test_case_insensitive_inputs(self):
        """Inputs com capitalização diferente devem funcionar."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "FORMAL",
            "despesas_fixas": 2000.0,
            "num_dependentes": 0,
            "tem_dividas": "NÃO",
        })
        assert result.startswith("SCORE_CALCULADO")

    def test_exactly_2_dependentes(self):
        """Testa peso para exatamente 2 dependentes."""
        result = calculate_credit_score.invoke({
            "renda_mensal": 5000.0,
            "tipo_emprego": "formal",
            "despesas_fixas": 2000.0,
            "num_dependentes": 2,
            "tem_dividas": "não",
        })
        assert result.startswith("SCORE_CALCULADO")
        # (5000/2001)*30 + 300 + 60 + 100 = 74.96 + 460 ≈ 535
        assert "535" in result


# --- update_client_score ---


class TestUpdateClientScore:
    def test_success(self, fake_clients_csv):
        result = update_client_score.invoke(
            {"cpf": "12345678901", "new_score": 800}
        )
        assert result.startswith("ATUALIZADO")
        assert "650" in result  # old score
        assert "800" in result  # new score

        # Verifica que o CSV foi atualizado
        with open(fake_clients_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["cpf"] == "12345678901":
                    assert row["score"] == "800"

    def test_formatted_cpf(self, fake_clients_csv):
        result = update_client_score.invoke(
            {"cpf": "123.456.789-01", "new_score": 900}
        )
        assert result.startswith("ATUALIZADO")

    def test_client_not_found(self, fake_clients_csv):
        result = update_client_score.invoke(
            {"cpf": "00000000000", "new_score": 800}
        )
        assert result.startswith("ERRO")
        assert "não encontrado" in result

    def test_csv_not_found(self, missing_csv):
        result = update_client_score.invoke(
            {"cpf": "12345678901", "new_score": 800}
        )
        assert result.startswith("ERRO_SISTEMA")

    def test_csv_write_error(self, fake_clients_csv, monkeypatch):
        """Falha ao salvar as alterações."""
        original_open = open

        def mock_open(path, *args, **kwargs):
            if str(path) == str(fake_clients_csv) and args and "w" in str(args[0]):
                raise PermissionError("read-only")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        result = update_client_score.invoke(
            {"cpf": "12345678901", "new_score": 800}
        )
        assert result.startswith("ERRO_SISTEMA")

    def test_csv_generic_read_error(self, tmp_path, monkeypatch):
        """Erro genérico (não FileNotFoundError) ao ler clientes."""
        csv_path = tmp_path / "clientes.csv"
        csv_path.write_bytes(b"\x00\x01\x02\xff\xfe")
        monkeypatch.setattr("src.tools.interview.CLIENTS_CSV", str(csv_path))

        result = update_client_score.invoke(
            {"cpf": "12345678901", "new_score": 800}
        )
        assert "ERRO_SISTEMA" in result

    def test_preserves_other_clients(self, fake_clients_csv):
        """Atualizar um cliente não afeta os outros."""
        update_client_score.invoke({"cpf": "12345678901", "new_score": 900})

        with open(fake_clients_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        maria = [r for r in rows if "Maria" in r["nome"]][0]
        assert maria["score"] == "780"
