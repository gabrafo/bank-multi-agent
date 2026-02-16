import logging

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

AWESOME_API_URL = "https://economia.awesomeapi.com.br/json/last"

# Moedas comuns e seus nomes em português
CURRENCY_NAMES = {
    "USD": "Dólar Americano",
    "EUR": "Euro",
    "GBP": "Libra Esterlina",
    "ARS": "Peso Argentino",
    "CAD": "Dólar Canadense",
    "AUD": "Dólar Australiano",
    "JPY": "Iene Japonês",
    "CNY": "Yuan Chinês",
    "BTC": "Bitcoin",
}


@tool
def get_exchange_rate(currency_code: str) -> str:
    """Consulta a cotação atual de uma moeda estrangeira em relação ao Real
    brasileiro (BRL) usando a API AwesomeAPI.

    Args:
        currency_code: Código da moeda (ex: USD, EUR, GBP, ARS, BTC).

    Returns:
        Cotação atual com detalhes ou mensagem de erro.
    """
    code = currency_code.upper().strip()

    if not code.isalpha() or len(code) < 2 or len(code) > 5:
        return (
            "ERRO: Código de moeda inválido. "
            "Use códigos como USD, EUR, GBP, ARS, BTC."
        )

    pair = f"{code}-BRL"
    url = f"{AWESOME_API_URL}/{pair}"

    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.Timeout:
        logger.error("Timeout ao consultar cotação de %s.", code)
        return (
            "ERRO_SISTEMA: A consulta de cotação demorou muito. "
            "Por favor, tente novamente em alguns instantes."
        )
    except requests.exceptions.ConnectionError:
        logger.error("Erro de conexão ao consultar cotação de %s.", code)
        return (
            "ERRO_SISTEMA: Não foi possível conectar ao serviço de cotação. "
            "Verifique sua conexão e tente novamente."
        )
    except requests.exceptions.RequestException as e:
        logger.error("Erro ao consultar cotação de %s: %s", code, e)
        return (
            "ERRO_SISTEMA: Ocorreu um erro ao consultar a cotação. "
            "Por favor, tente novamente mais tarde."
        )

    if response.status_code != 200:
        logger.error(
            "API retornou status %d para %s.", response.status_code, pair
        )
        return (
            f"ERRO_SISTEMA: O serviço de cotação retornou um erro "
            f"(status {response.status_code}). "
            f"Por favor, tente novamente mais tarde."
        )

    try:
        data = response.json()
    except (ValueError, KeyError):
        logger.error("Resposta inválida da API para %s.", pair)
        return (
            "ERRO_SISTEMA: A resposta do serviço de cotação foi inválida. "
            "Por favor, tente novamente mais tarde."
        )

    # A chave no JSON é "USDBRL", "EURBRL", etc.
    key = f"{code}BRL"
    if key not in data:
        return (
            f"ERRO: Moeda '{code}' não encontrada. "
            f"Verifique se o código está correto. "
            f"Exemplos válidos: USD, EUR, GBP, ARS, BTC."
        )

    quote = data[key]
    currency_name = CURRENCY_NAMES.get(code, code)

    try:
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        high = float(quote["high"])
        low = float(quote["low"])
        variation = float(quote["pctChange"])
    except (KeyError, ValueError, TypeError) as e:
        logger.error("Erro ao parsear dados de cotação de %s: %s", code, e)
        return (
            "ERRO_SISTEMA: Os dados de cotação estão em formato inesperado. "
            "Por favor, tente novamente mais tarde."
        )

    return (
        f"COTAÇÃO: {currency_name} ({code}/BRL). "
        f"Compra: R$ {bid:.4f}. "
        f"Venda: R$ {ask:.4f}. "
        f"Máxima do dia: R$ {high:.4f}. "
        f"Mínima do dia: R$ {low:.4f}. "
        f"Variação: {variation:+.2f}%."
    )
