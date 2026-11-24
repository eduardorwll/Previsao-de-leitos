"""Tipos nominais (NewType) e validadores de fronteira para chaves oficiais brasileiras.

Preserva zeros à esquerda em códigos CNES e IBGE e impede troca acidental entre
identificadores de estabelecimento, município e UF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType

from previsao_leitos.domain.exceptions import DataContractError

CodigoCNES = NewType("CodigoCNES", str)
CodigoMunicipioIBGE = NewType("CodigoMunicipioIBGE", str)
CodigoUF = NewType("CodigoUF", str)

_CNES_RE = re.compile(r"^\d{7}$")
_IBGE_7_RE = re.compile(r"^\d{7}$")
_VALID_UF_CODES: frozenset[str] = frozenset(
    {
        "11", "12", "13", "14", "15", "16", "17",
        "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "31", "32", "33", "35",
        "41", "42", "43",
        "50", "51", "52", "53",
    }
)


@dataclass(frozen=True, order=True)
class CompetenciaMensal:
    """Competência mensal CNES/SIH (ano e mês calendário)."""

    ano: int
    mes: int

    def __post_init__(self) -> None:
        if not (1990 <= self.ano <= 2100):
            raise DataContractError(f"Ano fora do intervalo esperado (1990..2100): {self.ano}")
        if not (1 <= self.mes <= 12):
            raise DataContractError(f"Mês calendário inválido (1..12): {self.mes}")

    def as_yyyymm(self) -> str:
        """Retorna representação padronizada AAAAMM."""
        return f"{self.ano:04d}{self.mes:02d}"


def parse_codigo_cnes(raw: str) -> CodigoCNES:
    """Valida e normaliza código CNES de 7 dígitos preservando zeros à esquerda."""
    cleaned = raw.strip()
    if not _CNES_RE.match(cleaned):
        raise DataContractError(f"Código CNES inválido (esperado 7 dígitos numéricos): {raw!r}")
    return CodigoCNES(cleaned)


def parse_codigo_municipio_ibge(raw: str) -> CodigoMunicipioIBGE:
    """Valida código IBGE municipal de 7 dígitos e verifica prefixo de UF oficial."""
    cleaned = raw.strip()
    if not _IBGE_7_RE.match(cleaned) or cleaned[:2] not in _VALID_UF_CODES:
        raise DataContractError(
            f"Código IBGE municipal inválido (esperado 7 dígitos com UF oficial): {raw!r}"
        )
    return CodigoMunicipioIBGE(cleaned)


def parse_codigo_uf(raw: str) -> CodigoUF:
    """Valida código IBGE oficial da Unidade Federativa (2 dígitos)."""
    cleaned = raw.strip()
    if cleaned not in _VALID_UF_CODES:
        raise DataContractError(f"Código de UF IBGE inválido: {raw!r}")
    return CodigoUF(cleaned)


def parse_competencia_mensal(ano: int, mes: int) -> CompetenciaMensal:
    """Constrói e valida uma competência mensal calendário."""
    return CompetenciaMensal(ano=ano, mes=mes)
