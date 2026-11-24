"""Controles LGPD na fronteira de ingestão: bloqueio de PII e supressão de pequenas células."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from previsao_leitos.domain import DataContractError

_FORBIDDEN_PII_TOKENS: tuple[str, ...] = (
    "cpf",
    "cns",
    "nome_paciente",
    "nome_mae",
    "dt_nasc",
    "data_nasc",
    "prontuario",
    "cep",
    "endereco",
    "telefone",
    "email",
    "rg_paciente",
)


def validate_no_pii_columns(
    df: pl.DataFrame,
    *,
    allowed_columns: Sequence[str],
) -> None:
    """Valida que nenhuma coluna sugere PII e que todas pertencem à allowlist do contrato."""
    allowed_set = set(allowed_columns)
    for col in df.columns:
        lowered = col.lower()
        for token in _FORBIDDEN_PII_TOKENS:
            if token in lowered:
                raise DataContractError(
                    f"Coluna potencialmente identificável (PII) bloqueada pela política LGPD: "
                    f"{col!r} (token={token!r})."
                )
        if col not in allowed_set:
            raise DataContractError(
                f"Coluna fora da allowlist do contrato de ingestão: {col!r}."
            )


def suppress_small_cells(
    df: pl.DataFrame,
    *,
    count_columns: Sequence[str],
    stratum_columns: Sequence[str],
    min_cell_count: int = 5,
) -> pl.DataFrame:
    """Aplica supressão primária (0 < n < min_cell_count) e complementar por estrato.

    Zeros verdadeiros (`0`) permanecem intactos para distinguir ausência real de dado suprimido
    (`null`). Se exatamente uma célula de um estrato for suprimida primariamente, a menor célula
    positiva restante no mesmo estrato também é suprimida para impedir reconstrução por subtração.
    """
    if min_cell_count < 2:
        raise DataContractError(
            f"Limiar de supressão LGPD deve ser >= 2 (recebido: {min_cell_count})."
        )
    if df.is_empty():
        return df.clone()

    for col in (*count_columns, *stratum_columns):
        if col not in df.columns:
            raise DataContractError(f"Coluna ausente para supressão LGPD: {col!r}.")

    result = df.clone()
    row_id_col = "__lgpd_row_id__"
    result = result.with_row_index(row_id_col)

    for col in count_columns:
        is_primary = (pl.col(col) > 0) & (pl.col(col) < min_cell_count)
        is_remaining_positive = pl.col(col) >= min_cell_count

        candidate_key = (
            pl.when(is_remaining_positive)
            .then(pl.struct(pl.col(col), pl.col(row_id_col)))
            .otherwise(None)
        )

        if stratum_columns:
            primary_count_in_stratum = is_primary.sum().over(list(stratum_columns))
            # Ranqueia apenas as células >= min_cell_count no estrato (nulls ficam fora do rank)
            candidate_rank = candidate_key.rank(method="ordinal").over(list(stratum_columns))
            is_complementary = (primary_count_in_stratum == 1) & (candidate_rank == 1)
        else:
            primary_total = is_primary.sum()
            candidate_rank = candidate_key.rank(method="ordinal")
            is_complementary = (primary_total == 1) & (candidate_rank == 1)

        result = result.with_columns(
            pl.when(is_primary | is_complementary)
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col)
        )

    return result.drop(row_id_col)


def count_suppressed_cells(
    before: pl.DataFrame,
    after: pl.DataFrame,
    *,
    count_columns: Sequence[str],
) -> int:
    """Conta quantas células não nulas foram mascaradas para `null` pela supressão LGPD."""
    total = 0
    for col in count_columns:
        before_nulls = int(before[col].null_count())
        after_nulls = int(after[col].null_count())
        total += max(0, after_nulls - before_nulls)
    return total
