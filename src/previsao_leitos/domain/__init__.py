"""Camada de domínio: tipos nominais validados na fronteira e hierarquia única de exceções."""

from previsao_leitos.domain.exceptions import (
    ConfigurationError,
    DataContractError,
    LeakageViolationError,
    PrevisaoLeitosError,
    SourceUnavailableError,
)
from previsao_leitos.domain.types import (
    CodigoCNES,
    CodigoMunicipioIBGE,
    CodigoUF,
    CompetenciaMensal,
    parse_codigo_cnes,
    parse_codigo_municipio_ibge,
    parse_codigo_uf,
    parse_competencia_mensal,
)

__all__ = [
    "CodigoCNES",
    "CodigoMunicipioIBGE",
    "CodigoUF",
    "CompetenciaMensal",
    "ConfigurationError",
    "DataContractError",
    "LeakageViolationError",
    "PrevisaoLeitosError",
    "SourceUnavailableError",
    "parse_codigo_cnes",
    "parse_codigo_municipio_ibge",
    "parse_codigo_uf",
    "parse_competencia_mensal",
]
