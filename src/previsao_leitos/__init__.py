"""Pacote modular para previsão de leitos necessários em hospitais-escola sob risco de enchentes.

Fronteira clínica e operacional: projeto exclusivamente de pesquisa aberta.
Not for patient care or live clinical use.
"""

from previsao_leitos.domain import (
    CodigoCNES,
    CodigoMunicipioIBGE,
    CodigoUF,
    CompetenciaMensal,
    ConfigurationError,
    DataContractError,
    LeakageViolationError,
    PrevisaoLeitosError,
    SourceUnavailableError,
    parse_codigo_cnes,
    parse_codigo_municipio_ibge,
    parse_codigo_uf,
    parse_competencia_mensal,
)

__version__ = "1.4.0"

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
    "__version__",
    "parse_codigo_cnes",
    "parse_codigo_municipio_ibge",
    "parse_codigo_uf",
    "parse_competencia_mensal",
]
