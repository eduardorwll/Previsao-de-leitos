"""Teste de conectividade e validação SQL gratuita (0 bytes cobrados) no BigQuery real."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from previsao_leitos.domain import parse_codigo_uf
from previsao_leitos.ingestion import (
    BigQueryAdapter,
    build_cnes_beds_by_specialty_query,
    build_cnes_teaching_hospitals_query,
    build_ibge_municipios_query,
    build_sih_referral_flows_query,
    build_sih_weekly_demand_query,
)


def _read_gcp_project_id() -> str | None:
    if env_val := os.getenv("GCP_PROJECT_ID"):
        return env_val
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GCP_PROJECT_ID="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


@pytest.mark.bigquery
def test_schema_introspection_and_dry_run_rs_queries_cost_zero_bytes() -> None:
    project_id = _read_gcp_project_id()
    if not project_id:
        pytest.skip("GCP_PROJECT_ID não configurado no ambiente nem em .env")

    adapter = BigQueryAdapter(project_id=project_id)
    cols = adapter.inspect_columns("basedosdados.br_ms_cnes.leito")
    assert "quantidade_total" in cols
    assert "quantidade_sus" in cols

    uf_rs = parse_codigo_uf("43")
    for builder in (
        build_cnes_teaching_hospitals_query,
        build_cnes_beds_by_specialty_query,
        build_sih_weekly_demand_query,
        build_sih_referral_flows_query,
        build_ibge_municipios_query,
    ):
        spec = builder(uf=uf_rs, ano_inicio=2010, ano_fim=2025)
        estimated = adapter.estimate_bytes(spec)
        # Em tabelas com RLS (CNES/SIH), dry_run valida SQL/parâmetros e retorna None;
        # em tabelas sem RLS (IBGE), retorna bytes >= 0.
        assert estimated is None or estimated >= 0
