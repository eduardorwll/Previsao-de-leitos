"""Construção de splits causais globais multi-região para modelagem de demanda (F04)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from previsao_leitos.domain import DataContractError, LeakageViolationError
from previsao_leitos.geo import SplitPolicy, prepare_causal_weekly_panel

_FORBIDDEN_PII_COLUMNS: frozenset[str] = frozenset(
    {
        "cpf",
        "cns",
        "nome_paciente",
        "nome_mae",
        "data_nascimento",
        "prontuario",
        "num_aih",
        "n_aih",
        "endereco_paciente",
    }
)


@dataclass(frozen=True)
class CanonicalPanelSchema:
    """Contrato canônico de colunas do painel causal de modelagem de demanda (F04 / V1.2).

    Impede desalinhamentos silenciosos entre produtores de features (`datasets.py`) e
    consumidores de modelagem (`two_stage.py`, `enso_what_if.py`), além de reforçar
    conformidade LGPD (zero colunas PII) e invariantes numéricos na fronteira.
    """

    entity_col: str = "nome_regiao_saude"
    target_year_col: str = "target_ano_epi"
    target_week_col: str = "target_semana_epi"
    target_col: str = "target_y"
    harmonic_sin_col: str = "target_week_sin"
    harmonic_cos_col: str = "target_week_cos"
    capacity_col: str | None = "capacidade_sem"
    required_extra_cols: tuple[str, ...] = ()

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Valida o DataFrame contra o contrato canônico, lançando DataContractError se violado."""
        if df.is_empty():
            raise DataContractError("CanonicalPanelSchema: DataFrame vazio.")

        lower_cols = {c.lower() for c in df.columns}
        pii_found = sorted(lower_cols & _FORBIDDEN_PII_COLUMNS)
        if pii_found:
            raise DataContractError(
                f"Violação LGPD em CanonicalPanelSchema: colunas PII proibidas {pii_found}."
            )

        required: list[str] = [
            self.entity_col,
            self.target_year_col,
            self.target_week_col,
            self.target_col,
            self.harmonic_sin_col,
            self.harmonic_cos_col,
            *self.required_extra_cols,
        ]
        if self.capacity_col is not None:
            required.append(self.capacity_col)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataContractError(
                f"CanonicalPanelSchema: colunas obrigatórias ausentes: {missing}"
            )

        null_keys = df.filter(
            pl.col(self.entity_col).is_null()
            | pl.col(self.target_year_col).is_null()
            | pl.col(self.target_week_col).is_null()
        )
        if null_keys.height > 0:
            raise DataContractError(
                "CanonicalPanelSchema: valores nulos nas chaves de entidade/tempo."
            )

        week_i64 = pl.col(self.target_week_col).cast(pl.Int64, strict=False)
        bad_week = df.filter(week_i64.is_null() | (week_i64 < 1) | (week_i64 > 53))
        if bad_week.height > 0:
            raise DataContractError(
                f"CanonicalPanelSchema: '{self.target_week_col}' fora do intervalo 1..53."
            )

        target_f64 = pl.col(self.target_col).cast(pl.Float64, strict=False)
        bad_target = df.filter(
            target_f64.is_null()
            | target_f64.is_nan()
            | target_f64.is_infinite()
            | (target_f64 < 0.0)
        )
        if bad_target.height > 0:
            raise DataContractError(
                f"CanonicalPanelSchema: valores inválidos (null/NaN/inf/<0) em '{self.target_col}'."
            )

        for harm_col in (self.harmonic_sin_col, self.harmonic_cos_col):
            harm_f64 = pl.col(harm_col).cast(pl.Float64, strict=False)
            bad_harm = df.filter(
                harm_f64.is_null()
                | harm_f64.is_nan()
                | harm_f64.is_infinite()
                | (harm_f64.abs() > 1.0 + 1e-6)
            )
            if bad_harm.height > 0:
                raise DataContractError(
                    f"CanonicalPanelSchema: harmônico '{harm_col}' fora de [-1, 1] ou não finito."
                )

        if self.capacity_col is not None:
            cap_f64 = pl.col(self.capacity_col).cast(pl.Float64, strict=False)
            bad_cap = df.filter(
                cap_f64.is_null()
                | cap_f64.is_nan()
                | cap_f64.is_infinite()
                | (cap_f64 <= 0.0)
            )
            if bad_cap.height > 0:
                raise DataContractError(
                    f"CanonicalPanelSchema: '{self.capacity_col}' deve ser finito e "
                    "estritamente positivo."
                )

        return df


def validate_canonical_panel_frame(
    df: pl.DataFrame,
    *,
    schema: CanonicalPanelSchema | None = None,
) -> pl.DataFrame:
    """Valida um painel de demanda contra `CanonicalPanelSchema`."""
    active_schema = schema or CanonicalPanelSchema()
    return active_schema.validate(df)


@dataclass(frozen=True)
class GlobalDemandSplits:
    """Partições temporais causais multi-região prontas para treino e avaliação."""

    train_df: pl.DataFrame
    eval_df: pl.DataFrame
    feature_cols: tuple[str, ...]
    entity_col: str
    target_col: str
    horizon_weeks: int
    eval_year: int


def prepare_global_demand_splits(
    weekly_df: pl.DataFrame,
    *,
    entity_col: str,
    target_col: str,
    dynamic_feature_cols: Sequence[str],
    static_feature_cols: Sequence[str] = (),
    extra_feature_cols: Sequence[str] = (),
    horizon_weeks: int = 2,
    lag_weeks: Sequence[int] = (0, 1, 2, 3, 4),
    eval_year: int = 2024,
    policy: SplitPolicy | None = None,
    allow_holdout: bool = False,
) -> GlobalDemandSplits:
    """Constrói painel causal global (todas as regiões) bloqueando o holdout `2025`.

    Adiciona estatísticas autorregressivas estritamente causais (`y_roll4`, `y_roll12`)
    calculadas apenas sobre `y <= t_0` antes de qualquer divisão temporal.
    """
    active_policy = policy or SplitPolicy()
    if eval_year >= active_policy.holdout_year and not allow_holdout:
        raise LeakageViolationError(
            f"Acesso bloqueado ao holdout intocado ({active_policy.holdout_year}): "
            f"eval_year={eval_year} exige `allow_holdout=True` apenas na auditoria final."
        )

    panel = prepare_causal_weekly_panel(
        weekly_df,
        entity_col=entity_col,
        target_col=target_col,
        feature_cols=dynamic_feature_cols,
        horizon_weeks=horizon_weeks,
        lag_weeks=lag_weeks,
    )

    # Estatísticas autorregressivas estritamente causais (<= t_0) para capturar porte regional
    enriched = panel.with_columns(
        pl.col("y_origin")
        .rolling_mean(window_size=4, min_samples=1)
        .over(entity_col)
        .alias("y_roll4"),
        pl.col("y_origin")
        .rolling_mean(window_size=12, min_samples=1)
        .over(entity_col)
        .alias("y_roll12"),
        (
            pl.col("y_origin")
            - pl.col("y_origin").rolling_mean(window_size=12, min_samples=1).over(entity_col)
        ).alias("y_dev_roll12"),
    )

    lag_cols = [f"{col}_lag{k}" for col in dynamic_feature_cols for k in lag_weeks]
    valid_static = [c for c in static_feature_cols if c in enriched.columns]
    # Colunas extras entram sem defasagem: já carregam memória (ex.: soma móvel do choque).
    valid_extra = [c for c in extra_feature_cols if c in enriched.columns]
    feature_cols = tuple(
        [
            *lag_cols,
            *valid_static,
            *valid_extra,
            "y_origin",
            "y_roll4",
            "y_roll12",
            "y_dev_roll12",
            "target_week_sin",
            "target_week_cos",
        ]
    )

    required = [*feature_cols, "target_y", "target_delta", "target_ano_epi", "target_semana_epi"]
    clean = enriched.drop_nulls(subset=required)

    train_df = clean.filter(pl.col("target_ano_epi") <= active_policy.train_end_year)
    eval_df = clean.filter(pl.col("target_ano_epi") == eval_year)

    if train_df.height < 20 or eval_df.height == 0:
        raise DataContractError(
            f"Amostras insuficientes no painel global "
            f"(train={train_df.height}, eval={eval_df.height})."
        )

    return GlobalDemandSplits(
        train_df=train_df,
        eval_df=eval_df,
        feature_cols=feature_cols,
        entity_col=entity_col,
        target_col=target_col,
        horizon_weeks=horizon_weeks,
        eval_year=eval_year,
    )
