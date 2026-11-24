"""Testes unitários de conformidade LGPD: bloqueio de PII e supressão de pequenas células."""

from __future__ import annotations

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.ingestion.lgpd import (
    suppress_small_cells,
    validate_no_pii_columns,
)


@pytest.mark.parametrize(
    "pii_col",
    ["cpf", "cns_paciente", "nome_paciente", "dt_nasc", "prontuario", "cep_residencia"],
)
def test_validate_no_pii_columns_rejects_sensitive_fields(pii_col: str) -> None:
    df = pl.DataFrame({"id_estabelecimento_cnes": ["2237601"], pii_col: ["X"]})

    with pytest.raises(DataContractError, match="PII"):
        validate_no_pii_columns(df, allowed_columns=("id_estabelecimento_cnes", pii_col))


def test_validate_no_pii_columns_rejects_unexpected_extra_column() -> None:
    df = pl.DataFrame({"id_estabelecimento_cnes": ["2237601"], "coluna_nao_declarada": [1]})

    with pytest.raises(DataContractError, match="allowlist"):
        validate_no_pii_columns(df, allowed_columns=("id_estabelecimento_cnes",))


def test_suppress_small_cells_preserves_true_zero_and_masks_one_to_four() -> None:
    df = pl.DataFrame(
        {
            "estrato": ["A", "A", "A", "A"],
            "categoria": ["c1", "c2", "c3", "c4"],
            "internacoes": [0, 2, 4, 12],
        }
    )

    result = suppress_small_cells(
        df,
        count_columns=("internacoes",),
        stratum_columns=("estrato",),
        min_cell_count=5,
    )

    # 2 células primariamente suprimidas (2 e 4) -> sem necessidade de supressão complementar
    assert result["internacoes"].to_list() == [0, None, None, 12]


def test_suppress_small_cells_applies_complementary_suppression_on_single_masked_cell() -> None:
    df = pl.DataFrame(
        {
            "estrato": ["HCPA", "HCPA", "HCPA", "HUSM", "HUSM"],
            "grupo_cid": ["leptospirose", "trauma", "respiratorio", "trauma", "respiratorio"],
            "internacoes": [3, 7, 25, 10, 18],
        }
    )

    result = suppress_small_cells(
        df,
        count_columns=("internacoes",),
        stratum_columns=("estrato",),
        min_cell_count=5,
    )

    # Em HCPA há exatamente 1 célula < 5 (3) -> suprime também a segunda menor positiva (7)
    # Em HUSM todas >= 5 -> nenhuma suprimida
    assert result["internacoes"].to_list() == [None, None, 25, 10, 18]


def test_suppress_small_cells_rejects_invalid_threshold() -> None:
    df = pl.DataFrame({"estrato": ["A"], "internacoes": [5]})

    with pytest.raises(DataContractError):
        suppress_small_cells(
            df,
            count_columns=("internacoes",),
            stratum_columns=("estrato",),
            min_cell_count=1,
        )
