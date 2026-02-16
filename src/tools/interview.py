"""Ferramentas de entrevista financeira para recálculo de score de crédito."""

import csv
import logging
import os

from langchain_core.tools import tool

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

CLIENTS_CSV = os.path.join(DATA_DIR, "clientes.csv")

PESO_RENDA = 30

PESO_EMPREGO = {
    "formal": 300,
    "autônomo": 200,
    "desempregado": 0,
}

PESO_DEPENDENTES = {
    0: 100,
    1: 80,
    2: 60,
}
PESO_DEPENDENTES_3_MAIS = 30

PESO_DIVIDAS = {
    "sim": -100,
    "não": 100,
}


@tool
def calculate_credit_score(
    renda_mensal: float,
    tipo_emprego: str,
    despesas_fixas: float,
    num_dependentes: int,
    tem_dividas: str,
) -> str:
    """Calcula um novo score de crédito com base em dados da entrevista financeira.

    Fórmula: ``(renda / (despesas + 1)) * PESO_RENDA + emprego + dependentes + dívidas``.
    O resultado é limitado entre 0 e 1000.

    Args:
        renda_mensal: Renda mensal em reais.
        tipo_emprego: Tipo de emprego (formal, autônomo ou desempregado).
        despesas_fixas: Despesas fixas mensais em reais.
        num_dependentes: Quantidade de dependentes.
        tem_dividas: Possui dívidas ativas (sim ou não).

    Returns:
        Score calculado com detalhes ou mensagem de erro.
    """
    # Validar entradas
    if renda_mensal < 0:
        return "ERRO: Renda mensal não pode ser negativa."

    if despesas_fixas < 0:
        return "ERRO: Despesas fixas não podem ser negativas."

    if num_dependentes < 0:
        return "ERRO: Número de dependentes não pode ser negativo."

    tipo_emprego_lower = tipo_emprego.lower().strip()
    if tipo_emprego_lower not in PESO_EMPREGO:
        return (
            "ERRO: Tipo de emprego inválido. "
            "Use: formal, autônomo ou desempregado."
        )

    tem_dividas_lower = tem_dividas.lower().strip()
    if tem_dividas_lower not in PESO_DIVIDAS:
        return "ERRO: Resposta sobre dívidas inválida. Use: sim ou não."

    # Calcular score
    peso_dep = PESO_DEPENDENTES.get(num_dependentes, PESO_DEPENDENTES_3_MAIS)

    score = (
        (renda_mensal / (despesas_fixas + 1)) * PESO_RENDA
        + PESO_EMPREGO[tipo_emprego_lower]
        + peso_dep
        + PESO_DIVIDAS[tem_dividas_lower]
    )

    score = max(0, min(1000, int(round(score))))

    return (
        f"SCORE_CALCULADO: {score}. "
        f"Detalhes: Renda mensal=R$ {renda_mensal:.2f}, "
        f"Despesas fixas=R$ {despesas_fixas:.2f}, "
        f"Emprego={tipo_emprego_lower}, "
        f"Dependentes={num_dependentes}, "
        f"Dívidas={tem_dividas_lower}."
    )


@tool
def update_client_score(cpf: str, new_score: int) -> str:
    """Atualiza o score de crédito de um cliente em ``clientes.csv``.

    Args:
        cpf: CPF do cliente (apenas números, 11 dígitos).
        new_score: Novo score de crédito (0 a 1000).

    Returns:
        Confirmação com scores anterior e novo, ou mensagem de erro.
    """
    cpf_clean = cpf.replace(".", "").replace("-", "").replace(" ", "")

    try:
        with open(CLIENTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except FileNotFoundError:
        logger.error("Arquivo %s não encontrado.", CLIENTS_CSV)
        return (
            "ERRO_SISTEMA: Não foi possível acessar a base de dados. "
            "Por favor, tente novamente mais tarde."
        )
    except Exception as e:
        logger.error("Erro ao ler %s: %s", CLIENTS_CSV, e)
        return (
            "ERRO_SISTEMA: Ocorreu um erro inesperado ao acessar os dados. "
            "Por favor, tente novamente mais tarde."
        )

    found = False
    old_score = None
    for row in rows:
        row_cpf = row["cpf"].strip().replace(".", "").replace("-", "")
        if row_cpf == cpf_clean:
            old_score = row["score"]
            row["score"] = str(new_score)
            found = True
            break

    if not found:
        return "ERRO: Cliente não encontrado na base de dados."

    try:
        with open(CLIENTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logger.error("Erro ao salvar %s: %s", CLIENTS_CSV, e)
        return (
            "ERRO_SISTEMA: Não foi possível salvar a atualização do score. "
            "Por favor, tente novamente mais tarde."
        )

    return (
        f"ATUALIZADO: Score do cliente atualizado de {old_score} para "
        f"{new_score}."
    )
