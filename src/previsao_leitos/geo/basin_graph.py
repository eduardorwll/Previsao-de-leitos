"""Grafo dirigido de bacias hidrográficas (DAG montante -> jusante) para o RS."""

from __future__ import annotations

from collections import defaultdict, deque

import polars as pl

from previsao_leitos.domain import DataContractError

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

# Mapeamento das 30 Regiões de Saúde oficiais do RS (prefixo 01..30) para sua bacia receptora
RS_REGION_PREFIX_TO_BASIN: dict[str, str] = {
    "01": "jacui_medio",          # Verdes Campos (Santa Maria)
    "02": "jacui_medio",          # Entre Rios
    "03": "fronteira_oeste",      # Fronteira Oeste
    "04": "litoral_norte",        # Belas Praias
    "05": "litoral_norte",        # Bons Ventos
    "06": "sinos",                # Vale do Paranhana e Costa da Serra
    "07": "sinos",                # Vale dos Sinos
    "08": "cai",                  # Vale do Caí e Metropolitana
    "09": "jacui_baixo",          # Carbonífera / Costa Doce
    "10": "guaiba_capital",       # Capital e Vale do Gravataí (Porto Alegre)
    "11": "alto_uruguai",         # Sete Povos das Missões
    "12": "medio_uruguai",        # Portal das Missões
    "13": "alto_uruguai",         # Diversidade
    "14": "medio_uruguai",        # Fronteira Noroeste
    "15": "alto_uruguai",         # Caminho das Águas
    "16": "alto_uruguai",         # Alto Uruguai Gaúcho
    "17": "planalto_alto_jacui",  # Planalto (Passo Fundo)
    "18": "planalto_alto_jacui",  # Araucárias
    "19": "planalto_alto_jacui",  # Botucaraí
    "20": "planalto_alto_jacui",  # Rota da Produção
    "21": "lagoa_patos_sul",      # Sul (Pelotas / Rio Grande)
    "22": "camaqua",              # Pampa (Bagé)
    "23": "serra_taquari_antas",  # Caxias e Hortênsias
    "24": "serra_taquari_antas",  # Campos de Cima da Serra
    "25": "serra_taquari_antas",  # Vinhedos e Basalto
    "26": "serra_taquari_antas",  # Uva e Vale
    "27": "jacui_baixo",          # Jacuí Centro
    "28": "jacui_baixo",          # Vales e Montanhas (Santa Cruz do Sul)
    "29": "taquari_antas",        # Vales e Montanhas / Vale do Taquari (Lajeado)
    "30": "taquari_antas",        # Vale do Taquari
}


def map_rs_health_region_to_basin(nome_regiao_saude: str) -> str:
    """Resolve o nome oficial da Região de Saúde do RS para sua bacia hidrográfica."""
    digits = "".join(ch for ch in nome_regiao_saude[:12] if ch.isdigit())
    prefix = digits[:2].zfill(2) if digits else ""
    return RS_REGION_PREFIX_TO_BASIN.get(prefix, "outras_bacias")


class BasinTopology:
    """Grafo dirigido acíclico (DAG) de bacias hidrográficas montante -> jusante."""

    def __init__(self, edges: tuple[tuple[str, str], ...]) -> None:
        self._edges = edges
        self._downstream: dict[str, set[str]] = defaultdict(set)
        self._upstream_direct: dict[str, set[str]] = defaultdict(set)
        self._nodes: set[str] = set()

        for src, dst in edges:
            self._nodes.add(src)
            self._nodes.add(dst)
            self._downstream[src].add(dst)
            self._upstream_direct[dst].add(src)

        self._validate_acyclic()
        self._ancestors: dict[str, frozenset[str]] = {
            node: self._compute_ancestors(node) for node in self._nodes
        }

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
