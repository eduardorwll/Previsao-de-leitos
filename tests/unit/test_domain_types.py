"""Testes unitários de fronteira para tipos de domínio e importação sem efeitos colaterais."""

from __future__ import annotations

import pytest

import previsao_leitos
from previsao_leitos.domain import (
    CompetenciaMensal,
    DataContractError,
    parse_codigo_cnes,
    parse_codigo_municipio_ibge,
    parse_codigo_uf,
    parse_competencia_mensal,
)


def test_package_imports_without_env_vars_or_side_effects() -> None:
    assert previsao_leitos.__version__ == "0.1.0"


def test_parse_codigo_cnes_preserves_leading_zeros() -> None:
    cnes = parse_codigo_cnes(" 0012345 ")
    assert cnes == "0012345"


@pytest.mark.parametrize("invalid_cnes", ["", "123456", "12345678", "223760A"])
def test_parse_codigo_cnes_rejects_invalid_values(invalid_cnes: str) -> None:
    with pytest.raises(DataContractError):
        parse_codigo_cnes(invalid_cnes)


def test_parse_codigo_municipio_ibge_accepts_porto_alegre_rs() -> None:
    poa = parse_codigo_municipio_ibge("4314902")
    assert poa == "4314902"


@pytest.mark.parametrize("invalid_ibge", ["431490", "9914902", "43149020", "abcdefg"])
def test_parse_codigo_municipio_ibge_rejects_invalid_length_or_uf(invalid_ibge: str) -> None:
    with pytest.raises(DataContractError):
        parse_codigo_municipio_ibge(invalid_ibge)


def test_parse_codigo_uf_accepts_rs_and_rejects_unknown() -> None:
    assert parse_codigo_uf("43") == "43"
    with pytest.raises(DataContractError):
        parse_codigo_uf("99")


def test_competencia_mensal_validates_bounds_and_orders_chronologically() -> None:
    abr = parse_competencia_mensal(2024, 4)
    mai = parse_competencia_mensal(2024, 5)
    assert abr < mai
    assert mai.as_yyyymm() == "202405"
    assert isinstance(mai, CompetenciaMensal)


@pytest.mark.parametrize(("ano", "mes"), [(2024, 0), (2024, 13), (1980, 5), (2150, 1)])
def test_competencia_mensal_rejects_out_of_range_values(ano: int, mes: int) -> None:
    with pytest.raises(DataContractError):
        parse_competencia_mensal(ano, mes)
