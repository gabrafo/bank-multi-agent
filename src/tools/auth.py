import csv
import logging
import os

from langchain_core.tools import tool

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

CLIENTS_CSV = os.path.join(DATA_DIR, "clientes.csv")


@tool
def authenticate_client(cpf: str, birth_date: str) -> str:
    """Autentica um cliente do Banco Ágil verificando CPF e data de nascimento
    contra a base de dados. O CPF deve conter apenas números (11 dígitos) e a
    data de nascimento deve estar no formato DD/MM/AAAA.

    Args:
        cpf: CPF do cliente (apenas números, 11 dígitos).
        birth_date: Data de nascimento no formato DD/MM/AAAA.

    Returns:
        Resultado da autenticação com dados do cliente ou mensagem de erro.
    """
    # Normaliza o CPF removendo pontuação
    cpf_clean = cpf.replace(".", "").replace("-", "").replace(" ", "")

    if len(cpf_clean) != 11 or not cpf_clean.isdigit():
        return (
            "FALHA: CPF inválido. O CPF deve conter exatamente 11 dígitos numéricos."
        )

    # Normaliza a data (aceita formatos comuns)
    birth_clean = birth_date.strip()

    try:
        with open(CLIENTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_cpf = row["cpf"].strip().replace(".", "").replace("-", "")
                row_birth = row["data_nascimento"].strip()
                if row_cpf == cpf_clean and row_birth == birth_clean:
                    return (
                        f"SUCESSO: Cliente autenticado. "
                        f"Nome: {row['nome']}, "
                        f"CPF: {row['cpf']}, "
                        f"Limite de crédito: R$ {row['limite_credito']}, "
                        f"Score: {row['score']}"
                    )
    except FileNotFoundError:
        logger.error("Arquivo %s não encontrado.", CLIENTS_CSV)
        return (
            "ERRO_SISTEMA: Não foi possível acessar a base de dados de clientes. "
            "Por favor, tente novamente mais tarde."
        )
    except Exception as e:
        logger.error("Erro ao ler %s: %s", CLIENTS_CSV, e)
        return (
            "ERRO_SISTEMA: Ocorreu um erro inesperado ao verificar os dados. "
            "Por favor, tente novamente mais tarde."
        )

    return "FALHA: CPF ou data de nascimento não correspondem a nenhum cliente cadastrado."
