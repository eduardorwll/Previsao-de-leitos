"""Dataset 2D indexado sob demanda respeitando fronteiras de entidade e lacunas ISO."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from previsao_leitos.domain import DataContractError


class IndexedHydroSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Emite janelas `[seq_len, dyn_dim]` sob demanda sem jamais cruzar lacunas ou entidades.

    Garante separação física (ADR-0001): `dyn_cols` contém apenas dinâmica meteorológica/
    hidrológica pura; atributos de impermeabilização/aterro/vegetação ficam estritamente em
    `mono_pos_cols` e `mono_neg_cols`.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        entity_col: str,
        dyn_cols: Sequence[str],
        ctx_cols: Sequence[str],
        mono_pos_cols: Sequence[str],
        mono_neg_cols: Sequence[str],
        event_col: str,
        severity_col: str,
        seq_len: int = 8,
    ) -> None:
        if seq_len < 1:
            raise DataContractError(f"seq_len deve ser >= 1 (recebido {seq_len}).")

        overlap = (set(dyn_cols) | set(ctx_cols)) & (set(mono_pos_cols) | set(mono_neg_cols))
        if overlap:
            raise DataContractError(
                f"Violação ADR-0001: features monotônicas no ramo irrestrito: {sorted(overlap)}"
            )

        ordered = df.sort([entity_col, "ano_epi", "semana_epi"])
        self._seq_len = seq_len
        self._x_dyn = ordered.select(list(dyn_cols)).to_numpy().astype(np.float32)
        self._x_ctx = ordered.select(list(ctx_cols)).to_numpy().astype(np.float32)
        self._x_pos = ordered.select(list(mono_pos_cols)).to_numpy().astype(np.float32)
        self._x_neg = ordered.select(list(mono_neg_cols)).to_numpy().astype(np.float32)
        self._y_event = ordered[event_col].to_numpy().astype(np.float32)
        self._y_sev = ordered[severity_col].to_numpy().astype(np.float32)

        entities = ordered[entity_col].to_list()
        years = ordered["ano_epi"].to_list()
        weeks = ordered["semana_epi"].to_list()
        mondays = [
            date.fromisocalendar(int(y), int(w), 1)
            for y, w in zip(years, weeks, strict=True)
        ]

        # Constrói índices válidos finais `end_idx` garantindo exatamente 7 dias entre passos
        valid_ends: list[int] = []
        run_len = 0
        for i in range(len(entities)):
            if i > 0 and entities[i] == entities[i - 1] and (mondays[i] - mondays[i - 1]).days == 7:
                run_len += 1
            else:
                run_len = 1

            has_finite_target = bool(
                np.isfinite(self._y_event[i]) and np.isfinite(self._y_sev[i])
            )
            if run_len >= seq_len and has_finite_target:
                window_dyn = self._x_dyn[i - seq_len + 1 : i + 1]
                if bool(np.isfinite(window_dyn).all()):
                    valid_ends.append(i)

        self._valid_ends = np.asarray(valid_ends, dtype=np.int64)
        self._row_meta = ordered.select([entity_col, "ano_epi", "semana_epi"])

    @property
    def valid_end_indices(self) -> np.ndarray:
        """Índices das linhas finais correspondentes a cada amostra válida."""
        return self._valid_ends

    @property
    def sample_metadata(self) -> pl.DataFrame:
        """Metadados `(entity, ano_epi, semana_epi)` alinhados 1-a-1 com o dataset."""
        return (
            self._row_meta.with_row_index("_idx")
            .filter(pl.col("_idx").is_in(self._valid_ends.tolist()))
            .drop("_idx")
        )

    def __len__(self) -> int:
        return len(self._valid_ends)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = int(self._valid_ends[idx])
        start = end - self._seq_len + 1
        return {
            "x_dyn": torch.from_numpy(self._x_dyn[start : end + 1].copy()),
            "x_ctx": torch.from_numpy(self._x_ctx[end].copy()),
            "x_mono_pos": torch.from_numpy(self._x_pos[end].copy()),
            "x_mono_neg": torch.from_numpy(self._x_neg[end].copy()),
            "y_event": torch.tensor(self._y_event[end], dtype=torch.float32),
            "y_sev": torch.tensor(self._y_sev[end], dtype=torch.float32),
        }
