"""Testes para a ferramenta get_exchange_rate (AwesomeAPI mockada)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tools.exchange import get_exchange_rate, AWESOME_API_URL


# --- Helpers ---


def _mock_response(status_code=200, json_data=None, raise_json=False):
    """Cria um objeto Response mockado."""
    resp = MagicMock()
    resp.status_code = status_code
    if raise_json:
        resp.json.side_effect = ValueError("No JSON")
    else:
        resp.json.return_value = json_data or {}
    return resp


SAMPLE_USD_RESPONSE = {
    "USDBRL": {
        "code": "USD",
        "codein": "BRL",
        "name": "Dólar Americano/Real Brasileiro",
        "high": "5.1234",
        "low": "5.0012",
        "varBid": "0.0234",
        "pctChange": "0.45",
        "bid": "5.0500",
        "ask": "5.0600",
        "timestamp": "1700000000",
        "create_date": "2025-01-01 12:00:00",
    }
}

SAMPLE_EUR_RESPONSE = {
    "EURBRL": {
        "code": "EUR",
        "codein": "BRL",
        "name": "Euro/Real Brasileiro",
        "high": "5.5000",
        "low": "5.3000",
        "varBid": "-0.0100",
        "pctChange": "-0.18",
        "bid": "5.4000",
        "ask": "5.4100",
        "timestamp": "1700000000",
        "create_date": "2025-01-01 12:00:00",
    }
}


# --- Testes ---


class TestGetExchangeRate:
    @patch("src.tools.exchange.requests.get")
    def test_usd_success(self, mock_get):
        """Consulta USD/BRL com sucesso."""
        mock_get.return_value = _mock_response(json_data=SAMPLE_USD_RESPONSE)

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("COTAÇÃO")
        assert "Dólar Americano" in result
        assert "5.0500" in result
        assert "5.0600" in result
        assert "5.1234" in result
        assert "5.0012" in result
        assert "+0.45%" in result
        mock_get.assert_called_once_with(
            f"{AWESOME_API_URL}/USD-BRL", timeout=10
        )

    @patch("src.tools.exchange.requests.get")
    def test_eur_success(self, mock_get):
        """Consulta EUR/BRL com sucesso."""
        mock_get.return_value = _mock_response(json_data=SAMPLE_EUR_RESPONSE)

        result = get_exchange_rate.invoke({"currency_code": "EUR"})

        assert result.startswith("COTAÇÃO")
        assert "Euro" in result
        assert "-0.18%" in result

    @patch("src.tools.exchange.requests.get")
    def test_lowercase_input(self, mock_get):
        """Aceita código em minúsculas."""
        mock_get.return_value = _mock_response(json_data=SAMPLE_USD_RESPONSE)

        result = get_exchange_rate.invoke({"currency_code": "usd"})

        assert result.startswith("COTAÇÃO")
        mock_get.assert_called_once_with(
            f"{AWESOME_API_URL}/USD-BRL", timeout=10
        )

    @patch("src.tools.exchange.requests.get")
    def test_unknown_currency_name(self, mock_get):
        """Moeda sem nome mapeado usa o código como nome."""
        mock_get.return_value = _mock_response(json_data={
            "CHFBRL": {
                "code": "CHF", "codein": "BRL",
                "high": "6.0", "low": "5.8",
                "pctChange": "0.10",
                "bid": "5.9000", "ask": "5.9100",
            }
        })

        result = get_exchange_rate.invoke({"currency_code": "CHF"})

        assert result.startswith("COTAÇÃO")
        assert "CHF" in result  # Usa código como nome

    def test_invalid_code_numeric(self):
        """Código numérico deve retornar ERRO."""
        result = get_exchange_rate.invoke({"currency_code": "123"})
        assert result.startswith("ERRO")
        assert "inválido" in result

    def test_invalid_code_too_long(self):
        result = get_exchange_rate.invoke({"currency_code": "ABCDEF"})
        assert result.startswith("ERRO")

    def test_invalid_code_single_char(self):
        result = get_exchange_rate.invoke({"currency_code": "A"})
        assert result.startswith("ERRO")

    def test_invalid_code_special_chars(self):
        result = get_exchange_rate.invoke({"currency_code": "US$"})
        assert result.startswith("ERRO")

    @patch("src.tools.exchange.requests.get")
    def test_timeout_error(self, mock_get):
        """Timeout deve retornar ERRO_SISTEMA."""
        mock_get.side_effect = requests.exceptions.Timeout("timeout")

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("ERRO_SISTEMA")
        assert "demorou" in result

    @patch("src.tools.exchange.requests.get")
    def test_connection_error(self, mock_get):
        """Erro de conexão deve retornar ERRO_SISTEMA."""
        mock_get.side_effect = requests.exceptions.ConnectionError("no net")

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("ERRO_SISTEMA")
        assert "conectar" in result

    @patch("src.tools.exchange.requests.get")
    def test_generic_request_error(self, mock_get):
        """Erro genérico de requests deve retornar ERRO_SISTEMA."""
        mock_get.side_effect = requests.exceptions.RequestException("fail")

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("ERRO_SISTEMA")

    @patch("src.tools.exchange.requests.get")
    def test_non_200_status(self, mock_get):
        """Status HTTP não-200 deve retornar ERRO_SISTEMA."""
        mock_get.return_value = _mock_response(status_code=500)

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("ERRO_SISTEMA")
        assert "500" in result

    @patch("src.tools.exchange.requests.get")
    def test_invalid_json_response(self, mock_get):
        """Resposta que não é JSON válido deve retornar ERRO_SISTEMA."""
        mock_get.return_value = _mock_response(raise_json=True)

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("ERRO_SISTEMA")
        assert "inválida" in result

    @patch("src.tools.exchange.requests.get")
    def test_currency_not_in_response(self, mock_get):
        """Moeda não encontrada na resposta JSON deve retornar ERRO."""
        mock_get.return_value = _mock_response(json_data={})

        result = get_exchange_rate.invoke({"currency_code": "XYZ"})

        assert result.startswith("ERRO")
        assert "não encontrada" in result

    @patch("src.tools.exchange.requests.get")
    def test_malformed_quote_data(self, mock_get):
        """Dados de cotação com campos ausentes/inválidos → ERRO_SISTEMA."""
        mock_get.return_value = _mock_response(json_data={
            "USDBRL": {"code": "USD"}  # Missing bid, ask, etc.
        })

        result = get_exchange_rate.invoke({"currency_code": "USD"})

        assert result.startswith("ERRO_SISTEMA")
        assert "formato inesperado" in result

    @patch("src.tools.exchange.requests.get")
    def test_whitespace_in_code(self, mock_get):
        """Código com espaços deve ser normalizado."""
        mock_get.return_value = _mock_response(json_data=SAMPLE_USD_RESPONSE)

        result = get_exchange_rate.invoke({"currency_code": " usd "})

        assert result.startswith("COTAÇÃO")
