import csv
import logging
import os
from datetime import datetime

from langchain_core.tools import tool

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

CLIENTS_CSV = os.path.join(DATA_DIR, "clientes.csv")
SCORE_LIMIT_CSV = os.path.join(DATA_DIR, "score_limite.csv")
REQUESTS_CSV = os.path.join(DATA_DIR, "solicitacoes_aumento_limite.csv")


def _read_clients() -> tuple[list[dict], list[str] | None]:
    """Lê todos os clientes do CSV. Retorna (rows, fieldnames) ou levanta exceção."""
    with open(CLIENTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return rows, fieldnames


def _find_client(rows: list[dict], cpf_clean: str) -> dict | None:
    """Encontra um cliente pelo CPF normalizado."""
    for row in rows:
        row_cpf = row["cpf"].strip().replace(".", "").replace("-", "")
        if row_cpf == cpf_clean:
            return row
    return None


def _get_max_limit_for_score(score: int) -> float | None:
    """Consulta o limite máximo permitido para um dado score."""
    with open(SCORE_LIMIT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["score_minimo"]) <= score <= int(row["score_maximo"]):
                return float(row["limite_maximo"])
    return None


def _normalize_cpf(cpf: str) -> str:
    """Remove formatação do CPF."""
    return cpf.replace(".", "").replace("-", "").replace(" ", "")


@tool
def query_credit_limit(cpf: str) -> str:
    """Consulta o limite de crédito atual do cliente pelo CPF.

    Args:
        cpf: CPF do cliente (apenas números, 11 dígitos).

    Returns:
        Informações sobre o limite de crédito do cliente.
    """
    cpf_clean = _normalize_cpf(cpf)

    try:
        rows, _ = _read_clients()
    except FileNotFoundError:
        logger.error("Arquivo %s não encontrado.", CLIENTS_CSV)
        return (
            "ERRO_SISTEMA: Não foi possível acessar a base de dados de clientes. "
            "Por favor, tente novamente mais tarde."
        )
    except Exception as e:
        logger.error("Erro ao ler %s: %s", CLIENTS_CSV, e)
        return (
            "ERRO_SISTEMA: Ocorreu um erro inesperado ao consultar os dados. "
            "Por favor, tente novamente mais tarde."
        )

    client = _find_client(rows, cpf_clean)
    if client is None:
        return "ERRO: Cliente não encontrado na base de dados."

    return (
        f"LIMITE: Cliente {client['nome']} (CPF: {client['cpf']}). "
        f"Limite de crédito atual: R$ {client['limite_credito']}. "
        f"Score: {client['score']}."
    )


@tool
def request_limit_increase(cpf: str, new_limit: float) -> str:
    """Solicita aumento do limite de crédito do cliente.

    O sistema verifica automaticamente se o score do cliente permite o novo
    limite solicitado, consulta a tabela de score/limite e registra a
    solicitação no arquivo de controle.

    Args:
        cpf: CPF do cliente (apenas números, 11 dígitos).
        new_limit: Novo limite de crédito desejado em reais.

    Returns:
        Resultado da solicitação (aprovado ou rejeitado) com detalhes.
    """
    cpf_clean = _normalize_cpf(cpf)

    # 1. Ler dados do cliente
    try:
        rows, fieldnames = _read_clients()
    except FileNotFoundError:
        logger.error("Arquivo %s não encontrado.", CLIENTS_CSV)
        return (
            "ERRO_SISTEMA: Não foi possível acessar a base de dados de clientes. "
            "Por favor, tente novamente mais tarde."
        )
    except Exception as e:
        logger.error("Erro ao ler %s: %s", CLIENTS_CSV, e)
        return (
            "ERRO_SISTEMA: Ocorreu um erro inesperado ao consultar os dados. "
            "Por favor, tente novamente mais tarde."
        )

    client = _find_client(rows, cpf_clean)
    if client is None:
        return "ERRO: Cliente não encontrado na base de dados."

    current_limit = float(client["limite_credito"])
    score = int(client["score"])

    if new_limit <= current_limit:
        return (
            f"INFORMAÇÃO: O limite solicitado (R$ {new_limit:.2f}) é menor ou "
            f"igual ao limite atual (R$ {current_limit:.2f}). "
            f"Não é necessário solicitar aumento."
        )

    # 2. Verificar limite máximo permitido pelo score
    try:
        max_allowed = _get_max_limit_for_score(score)
    except FileNotFoundError:
        logger.error("Arquivo %s não encontrado.", SCORE_LIMIT_CSV)
        return (
            "ERRO_SISTEMA: Não foi possível acessar a tabela de score. "
            "Por favor, tente novamente mais tarde."
        )
    except Exception as e:
        logger.error("Erro ao ler %s: %s", SCORE_LIMIT_CSV, e)
        return (
            "ERRO_SISTEMA: Ocorreu um erro inesperado ao verificar o score. "
            "Por favor, tente novamente mais tarde."
        )

    if max_allowed is None:
        return (
            "ERRO_SISTEMA: Não foi possível determinar o limite máximo para "
            f"o score {score}. Por favor, tente novamente mais tarde."
        )

    # 3. Determinar status
    status = "aprovado" if new_limit <= max_allowed else "rejeitado"

    # 4. Registrar solicitação
    timestamp = datetime.now().isoformat()
    try:
        with open(REQUESTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [cpf_clean, timestamp, current_limit, new_limit, status]
            )
    except Exception as e:
        logger.error("Erro ao gravar em %s: %s", REQUESTS_CSV, e)
        return (
            "ERRO_SISTEMA: Não foi possível registrar a solicitação. "
            "Por favor, tente novamente mais tarde."
        )

    # 5. Se aprovado, atualizar limite do cliente
    if status == "aprovado":
        try:
            client["limite_credito"] = f"{new_limit:.2f}"
            with open(CLIENTS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            logger.error("Erro ao atualizar limite em %s: %s", CLIENTS_CSV, e)
            # Solicitação já registrada, informa parcialmente
            return (
                f"APROVADO: Solicitação aprovada, porém houve um erro ao "
                f"atualizar o limite na base de dados. "
                f"Limite anterior: R$ {current_limit:.2f}. "
                f"Novo limite solicitado: R$ {new_limit:.2f}."
            )

        return (
            f"APROVADO: Solicitação de aumento de limite aprovada! "
            f"Limite anterior: R$ {current_limit:.2f}. "
            f"Novo limite: R$ {new_limit:.2f}. "
            f"Score atual: {score}."
        )

    return (
        f"REJEITADO: Solicitação de aumento de limite rejeitada. "
        f"Limite atual: R$ {current_limit:.2f}. "
        f"Limite solicitado: R$ {new_limit:.2f}. "
        f"Limite máximo permitido para score {score}: R$ {max_allowed:.2f}. "
        f"O cliente pode realizar uma entrevista de crédito para tentar "
        f"melhorar seu score."
    )
