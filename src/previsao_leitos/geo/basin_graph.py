"""Grafo dirigido de bacias hidrográficas (DAG ponderado + tempo de trânsito + remanso)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import polars as pl

from previsao_leitos.domain import DataContractError
from previsao_leitos.geo.calendar_grid import build_complete_iso_week_grid


@dataclass(frozen=True)
class WeightedBasinEdge:
    """Aresta dirigida montante -> jusante com peso físico de contribuição e tempo de trânsito."""

    src: str
    dst: str
    weight: float = 1.0
    travel_weeks: int = 0

    def __post_init__(self) -> None:
        if self.weight <= 0.0:
            raise DataContractError(f"Peso hidrológico deve ser > 0 (recebido {self.weight}).")
        if self.travel_weeks < 0:
            raise DataContractError(
                f"Tempo de trânsito negativo proibido (vazamento futuro): {self.travel_weeks}"
            )


# Topologia hidrográfica simplificada montante -> jusante do Rio Grande do Sul (Ottobacias ANA)
RS_BASIN_EDGES: tuple[tuple[str, str], ...] = (
    ("serra_taquari_antas", "taquari_antas"),
    ("planalto_alto_jacui", "jacui_medio"),
    ("taquari_antas", "jacui_baixo"),
    ("jacui_medio", "jacui_baixo"),
    ("jacui_baixo", "guaiba_capital"),
    ("cai", "guaiba_capital"),
    ("sinos", "guaiba_capital"),
    ("gravatai", "guaiba_capital"),
    ("guaiba_capital", "lagoa_patos_sul"),
    ("camaqua", "lagoa_patos_sul"),
    ("alto_uruguai", "medio_uruguai"),
    ("medio_uruguai", "fronteira_oeste"),
    ("ibicui", "fronteira_oeste"),
)

# Arestas ponderadas com contribuição relativa de vazão e defasagem física de trânsito (semanas)
RS_BASIN_WEIGHTED_EDGES: tuple[WeightedBasinEdge, ...] = (
    WeightedBasinEdge("serra_taquari_antas", "taquari_antas", weight=1.0, travel_weeks=0),
    WeightedBasinEdge("planalto_alto_jacui", "jacui_medio", weight=1.0, travel_weeks=0),
    WeightedBasinEdge("taquari_antas", "jacui_baixo", weight=0.55, travel_weeks=0),
    WeightedBasinEdge("jacui_medio", "jacui_baixo", weight=0.45, travel_weeks=0),
    WeightedBasinEdge("jacui_baixo", "guaiba_capital", weight=0.45, travel_weeks=0),
    WeightedBasinEdge("taquari_antas", "guaiba_capital", weight=0.25, travel_weeks=0),
    WeightedBasinEdge("cai", "guaiba_capital", weight=0.12, travel_weeks=0),
    WeightedBasinEdge("sinos", "guaiba_capital", weight=0.12, travel_weeks=0),
    WeightedBasinEdge("gravatai", "guaiba_capital", weight=0.06, travel_weeks=0),
    # Trânsito lento pela Lagoa dos Patos (7-14 dias) até Pelotas/Rio Grande
    WeightedBasinEdge("guaiba_capital", "lagoa_patos_sul", weight=0.75, travel_weeks=1),
    WeightedBasinEdge("camaqua", "lagoa_patos_sul", weight=0.25, travel_weeks=0),
    WeightedBasinEdge("alto_uruguai", "medio_uruguai", weight=1.0, travel_weeks=0),
    WeightedBasinEdge("medio_uruguai", "fronteira_oeste", weight=0.65, travel_weeks=1),
    WeightedBasinEdge("ibicui", "fronteira_oeste", weight=0.35, travel_weeks=0),
)

ESTUARINE_BACKWATER_BASINS: frozenset[str] = frozenset({"guaiba_capital", "lagoa_patos_sul"})

# Mapeamento das 30 Regiões de Saúde oficiais do RS (prefixo 01..30) para sua bacia receptora
RS_REGION_PREFIX_TO_BASIN: dict[str, str] = {
    "01": "jacui_medio",
    "02": "jacui_medio",
    "03": "fronteira_oeste",
    "04": "litoral_norte",
    "05": "litoral_norte",
    "06": "sinos",
    "07": "sinos",
    "08": "cai",
    "09": "jacui_baixo",
    "10": "guaiba_capital",
    "11": "alto_uruguai",
    "12": "medio_uruguai",
    "13": "alto_uruguai",
    "14": "medio_uruguai",
    "15": "alto_uruguai",
    "16": "alto_uruguai",
    "17": "planalto_alto_jacui",
    "18": "planalto_alto_jacui",
    "19": "planalto_alto_jacui",
    "20": "planalto_alto_jacui",
    "21": "lagoa_patos_sul",
    "22": "camaqua",
    "23": "serra_taquari_antas",
    "24": "serra_taquari_antas",
    "25": "serra_taquari_antas",
    "26": "serra_taquari_antas",
    "27": "jacui_baixo",
    "28": "jacui_baixo",
    "29": "taquari_antas",
    "30": "taquari_antas",
}


def map_rs_health_region_to_basin(nome_regiao_saude: str) -> str:
    """Resolve o nome oficial da Região de Saúde do RS para sua bacia hidrográfica."""
    digits = "".join(ch for ch in nome_regiao_saude[:12] if ch.isdigit())
    prefix = digits[:2].zfill(2) if digits else ""
    return RS_REGION_PREFIX_TO_BASIN.get(prefix, "outras_bacias")


def compute_estuarine_backwater_index(
    df: pl.DataFrame,
    *,
    basin_col: str = "bacia_id",
    south_wind_col: str = "vento_sul_ms",
    storage_col: str = "api_montante_max",
    output_col: str = "remanso_lagunar_idx",
    estuarine_basins: frozenset[str] = ESTUARINE_BACKWATER_BASINS,
) -> pl.DataFrame:
    """Calcula índice físico de remanso (vento Sul positivo x estoque hídrico) em bacias lagunares.

    Em bacias de cabeceira/planalto ou sob vento Norte (`vento_sul_ms <= 0`), o remanso é 0.0.
    """
    estuarine_list = sorted(estuarine_basins)
    return df.with_columns(
        pl.when(pl.col(basin_col).is_in(estuarine_list))
        .then(pl.col(south_wind_col).fill_null(0.0).clip(lower_bound=0.0) * pl.col(storage_col))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias(output_col)
    )


class BasinTopology:
    """Grafo dirigido acíclico (DAG) de bacias hidrográficas montante -> jusante."""

    def __init__(
        self,
        edges: tuple[tuple[str, str], ...],
        weighted_edges: tuple[WeightedBasinEdge, ...] | None = None,
    ) -> None:
        self._edges = edges
        self._weighted_edges: tuple[WeightedBasinEdge, ...] = (
            weighted_edges
            if weighted_edges is not None
            else tuple(WeightedBasinEdge(src, dst, 1.0, 0) for src, dst in edges)
        )
        self._downstream: dict[str, set[str]] = defaultdict(set)
        self._upstream_direct: dict[str, set[str]] = defaultdict(set)
        self._nodes: set[str] = set()

        for edge in self._weighted_edges:
            self._nodes.add(edge.src)
            self._nodes.add(edge.dst)
            self._downstream[edge.src].add(edge.dst)
            self._upstream_direct[edge.dst].add(edge.src)

        self._validate_acyclic()
        self._ancestors: dict[str, frozenset[str]] = {
            node: self._compute_ancestors(node) for node in self._nodes
        }

    @classmethod
    def from_weighted_edges(
        cls, weighted_edges: tuple[WeightedBasinEdge, ...]
    ) -> BasinTopology:
        """Constrói DAG diretamente a partir de arestas ponderadas com tempo de trânsito."""
        raw_edges = tuple((e.src, e.dst) for e in weighted_edges)
        return cls(edges=raw_edges, weighted_edges=weighted_edges)

    def _validate_acyclic(self) -> None:
        in_degree = {node: len(self._upstream_direct[node]) for node in self._nodes}
        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for nxt in self._downstream[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        if visited != len(self._nodes):
            raise DataContractError(
                "Topologia de bacias inválida: ciclo detectado no grafo hidrográfico."
            )

    def _compute_ancestors(self, node: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(self._upstream_direct[node])
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self._upstream_direct[cur])
        return frozenset(seen)

    def upstream_of(self, basin_id: str) -> frozenset[str]:
        """Retorna todas as sub-bacias estritamente a montante de `basin_id`."""
        return self._ancestors.get(basin_id, frozenset())

    def transitive_upstream_paths(self) -> tuple[WeightedBasinEdge, ...]:
        """Expande todos os caminhos dirigidos montante->jusante compondo pesos e lags.

        Em um DAG hidrográfico, cada caminho simples `src -> ... -> dst` contribui com
        peso composto igual ao produto dos pesos das arestas e tempo de trânsito igual
        à soma das defasagens (`travel_weeks`), consolidados por `(src, dst, lag)`.
        """
        out_adj: dict[str, list[WeightedBasinEdge]] = defaultdict(list)
        for edge in self._weighted_edges:
            out_adj[edge.src].append(edge)

        accum: dict[tuple[str, str, int], float] = defaultdict(float)
        for origin in sorted(self._nodes):
            stack: list[tuple[str, float, int]] = [(origin, 1.0, 0)]
            while stack:
                cur, cur_w, cur_lag = stack.pop()
                for edge in out_adj.get(cur, ()):
                    nxt_w = cur_w * edge.weight
                    nxt_lag = cur_lag + edge.travel_weeks
                    accum[(origin, edge.dst, nxt_lag)] += nxt_w
                    stack.append((edge.dst, nxt_w, nxt_lag))

        return tuple(
            WeightedBasinEdge(src=src, dst=dst, weight=weight, travel_weeks=lag)
            for (src, dst, lag), weight in sorted(accum.items())
        )

    def propagate_upstream(
        self,
        weekly_df: pl.DataFrame,
        *,
        basin_col: str = "bacia_id",
        value_col: str = "chuva_sem_mm",
        output_col: str = "chuva_montante_mm",
    ) -> pl.DataFrame:
        """Calcula média dos valores observados em todas as bacias estritamente a montante."""
        if weekly_df.is_empty():
            return weekly_df.with_columns(pl.lit(0.0).alias(output_col))

        pair_rows = [
            {"_dst_basin": dst, "_src_basin": src}
            for dst in sorted(self._nodes)
            for src in sorted(self._ancestors[dst])
        ]
        if not pair_rows:
            return weekly_df.with_columns(pl.lit(0.0).alias(output_col))

        pairs = pl.DataFrame(pair_rows)
        src_vals = (
            weekly_df.group_by([basin_col, "ano_epi", "semana_epi"])
            .agg(pl.col(value_col).mean().alias("_src_val"))
            .rename({basin_col: "_src_basin"})
        )

        upstream_agg = (
            pairs.join(src_vals, on="_src_basin", how="inner")
            .group_by(["_dst_basin", "ano_epi", "semana_epi"])
            .agg(pl.col("_src_val").mean().alias(output_col))
            .rename({"_dst_basin": basin_col})
        )

        return weekly_df.join(
            upstream_agg, on=[basin_col, "ano_epi", "semana_epi"], how="left"
        ).with_columns(pl.col(output_col).fill_null(0.0))

    def propagate_weighted_upstream(
        self,
        weekly_df: pl.DataFrame,
        *,
        basin_col: str = "bacia_id",
        value_col: str = "chuva_sem_mm",
        output_col: str = "chuva_montante_pond",
        multi_hop: bool = True,
    ) -> pl.DataFrame:
        """Propaga sinais a montante usando pesos físicos (`weight`) e trânsito (`travel_weeks`).

        Quando `multi_hop=True` (padrão V1.1), realiza convolução transitiva completa sobre
        todos os ancestrais do DAG (levando Serra/Taquari/Jacuí até POA e Lagoa dos Patos).
        """
        if weekly_df.is_empty() or not self._weighted_edges:
            return weekly_df.with_columns(pl.lit(0.0).alias(output_col))

        active_edges = self.transitive_upstream_paths() if multi_hop else self._weighted_edges

        # Agrega primeiro por bacia e garante grade ISO semanal contínua antes de aplicar lag
        basin_series = weekly_df.group_by([basin_col, "ano_epi", "semana_epi"]).agg(
            pl.col(value_col).mean().alias("_b_val")
        )
        basin_grid = build_complete_iso_week_grid(basin_series, entity_col=basin_col)

        distinct_lags = sorted({e.travel_weeks for e in active_edges})
        lag_exprs = [
            pl.col("_b_val").shift(lag).over(basin_col).alias(f"_lag_{lag}")
            for lag in distinct_lags
        ]
        lagged_basins = basin_grid.with_columns(lag_exprs)

        parts: list[pl.DataFrame] = []
        for lag in distinct_lags:
            edges_for_lag = [e for e in active_edges if e.travel_weeks == lag]
            edge_df = pl.DataFrame(
                {
                    "_src_basin": [e.src for e in edges_for_lag],
                    "_dst_basin": [e.dst for e in edges_for_lag],
                    "_w": [e.weight for e in edges_for_lag],
                }
            )
            src_df = lagged_basins.select(
                [basin_col, "ano_epi", "semana_epi", f"_lag_{lag}"]
            ).rename({basin_col: "_src_basin", f"_lag_{lag}": "_src_val"})
            joined = edge_df.join(src_df, on="_src_basin", how="inner").filter(
                pl.col("_src_val").is_not_null()
            )
            parts.append(joined)

        if not parts:
            return weekly_df.with_columns(pl.lit(0.0).alias(output_col))

        combined = pl.concat(parts)
        weighted_agg = (
            combined.group_by(["_dst_basin", "ano_epi", "semana_epi"])
            .agg(
                ((pl.col("_src_val") * pl.col("_w")).sum() / pl.col("_w").sum()).alias(output_col)
            )
            .rename({"_dst_basin": basin_col})
        )

        return weekly_df.join(
            weighted_agg, on=[basin_col, "ano_epi", "semana_epi"], how="left"
        ).with_columns(pl.col(output_col).fill_null(0.0))
