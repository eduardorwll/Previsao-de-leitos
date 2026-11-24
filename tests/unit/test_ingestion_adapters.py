"""Testes unitários de adaptadores (BigQuery dry_run + Fixture offline), schemas e queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from previsao_leitos.domain import (
    ConfigurationError,
    DataContractError,
    SourceUnavailableError,
    parse_codigo_uf,
)
from previsao_leitos.ingestion.adapters import BigQueryAdapter, ParquetFixtureAdapter
from previsao_leitos.ingestion.ports import QueryParam, QuerySpec
from previsao_leitos.ingestion.queries import (
    build_cnes_beds_by_specialty_query,
    build_cnes_teaching_hospitals_query,
    build_ibge_municipios_query,
    build_sih_referral_flows_query,
    build_sih_weekly_demand_query,
)
from previsao_leitos.ingestion.schemas import (
    validate_cnes_beds_frame,
    validate_cnes_teaching_hospitals_frame,
    validate_sih_weekly_demand_frame,
)


@dataclass
class _FakeBQJob:
    total_bytes_processed: int | None
    df: pl.DataFrame

    def to_arrow(self) -> Any:
        return self.df.to_arrow()


class _FakeBQClient:
    """Fake in-memory do google.cloud.bigquery.Client para testar dry_run sem rede."""

    def __init__(
        self,
        dry_run_bytes: int | None,
        result_df: pl.DataFrame,
        exec_bytes: int | None = None,
    ) -> None:
        self.dry_run_bytes = dry_run_bytes
        self.exec_bytes = dry_run_bytes if exec_bytes is None else exec_bytes
        self.result_df = result_df
        self.calls: list[dict[str, Any]] = []

    def get_table(self, table_ref: str) -> Any:
        class _Field:
            def __init__(self, name: str) -> None:
                self.name = name

        class _Table:
            schema = (_Field("ano"), _Field("sigla_uf"), _Field("id_estabelecimento_cnes"))

        return _Table()

    def query(self, sql: str, job_config: Any = None) -> _FakeBQJob:
        is_dry = bool(getattr(job_config, "dry_run", False))
        self.calls.append({"sql": sql, "dry_run": is_dry, "job_config": job_config})
        total = self.dry_run_bytes if is_dry else self.exec_bytes
        return _FakeBQJob(total_bytes_processed=total, df=self.result_df)


def test_bigquery_adapter_rejects_query_exceeding_dry_run_budget() -> None:
    fake_client = _FakeBQClient(
        dry_run_bytes=500 * 1024 * 1024,  # 500 MB
        result_df=pl.DataFrame({"x": [1]}),
    )
    adapter = BigQueryAdapter(client=fake_client)
    spec = QuerySpec(
        name="expensive_query",
        sql="SELECT 1 AS x WHERE @ano > 0",
        params=(QueryParam("ano", "INT64", 2024),),
        schema_version="v1",
        expected_columns=("x",),
    )

    with pytest.raises(ConfigurationError, match="dry_run"):
        adapter.fetch(spec, max_bytes_billed=100 * 1024 * 1024)  # teto 100 MB

    # Deve ter chamado APENAS o dry_run e nunca a consulta faturável
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["dry_run"] is True


def test_bigquery_adapter_executes_parameterized_query_within_budget() -> None:
    expected = pl.DataFrame({"id_estabelecimento_cnes": ["0012345"]})
    fake_client = _FakeBQClient(dry_run_bytes=10 * 1024 * 1024, result_df=expected)
    adapter = BigQueryAdapter(client=fake_client)
    spec = QuerySpec(
        name="cheap_query",
        sql="SELECT id_estabelecimento_cnes FROM t WHERE sigla_uf = @sigla_uf",
        params=(QueryParam("sigla_uf", "STRING", "RS"),),
        schema_version="v1",
        expected_columns=("id_estabelecimento_cnes",),
    )

    df = adapter.fetch(spec, max_bytes_billed=50 * 1024 * 1024)
    assert df["id_estabelecimento_cnes"].to_list() == ["0012345"]
    assert [c["dry_run"] for c in fake_client.calls] == [True, False]


def test_bigquery_adapter_handles_rls_none_dry_run_and_keeps_server_side_limit() -> None:
    expected = pl.DataFrame({"id_estabelecimento_cnes": ["2237601"]})
    fake_client = _FakeBQClient(
        dry_run_bytes=None,
        exec_bytes=12 * 1024 * 1024,
        result_df=expected,
    )
    adapter = BigQueryAdapter(client=fake_client)
    spec = QuerySpec(
        name="rls_query",
        sql="SELECT id_estabelecimento_cnes FROM t WHERE sigla_uf = @sigla_uf",
        params=(QueryParam("sigla_uf", "STRING", "RS"),),
        schema_version="v1",
        expected_columns=("id_estabelecimento_cnes",),
    )

    assert adapter.estimate_bytes(spec) is None
    df = adapter.fetch(spec, max_bytes_billed=50 * 1024 * 1024)
    assert df["id_estabelecimento_cnes"].to_list() == ["2237601"]
    assert adapter.last_bytes_processed == 12 * 1024 * 1024
    assert fake_client.calls[-1]["job_config"].maximum_bytes_billed == 50 * 1024 * 1024


def test_bigquery_adapter_rejects_invalid_table_identifier() -> None:
    fake_client = _FakeBQClient(dry_run_bytes=0, result_df=pl.DataFrame())
    adapter = BigQueryAdapter(client=fake_client)

    with pytest.raises(DataContractError):
        adapter.inspect_columns("basedosdados.br_ms_cnes.leito; DROP TABLE x")


def test_parquet_fixture_adapter_loads_offline_and_errors_when_missing(tmp_path: Path) -> None:
    df_in = pl.DataFrame({"id_estabelecimento_cnes": ["0012345"]})
    df_in.write_parquet(tmp_path / "cnes_hospitais_escola.parquet")
    adapter = ParquetFixtureAdapter(fixtures_dir=tmp_path)

    spec_ok = QuerySpec(
        name="cnes_hospitais_escola",
        sql="SELECT 1",
        params=(),
        schema_version="v1",
        expected_columns=("id_estabelecimento_cnes",),
    )
    out = adapter.fetch(spec_ok, max_bytes_billed=1024)
    assert out["id_estabelecimento_cnes"].to_list() == ["0012345"]
    assert adapter.estimate_bytes(spec_ok) == 0

    spec_missing = QuerySpec(
        name="nao_existe",
        sql="SELECT 1",
        params=(),
        schema_version="v1",
        expected_columns=("x",),
    )
    with pytest.raises(SourceUnavailableError):
        adapter.fetch(spec_missing, max_bytes_billed=1024)


def test_all_rs_query_specs_use_named_parameters_only() -> None:
    uf_rs = parse_codigo_uf("43")
    specs = [
        build_cnes_teaching_hospitals_query(uf=uf_rs, ano_inicio=2010, ano_fim=2025),
        build_cnes_beds_by_specialty_query(uf=uf_rs, ano_inicio=2010, ano_fim=2025),
        build_sih_weekly_demand_query(uf=uf_rs, ano_inicio=2010, ano_fim=2025),
        build_sih_referral_flows_query(uf=uf_rs, ano_inicio=2010, ano_fim=2025),
        build_ibge_municipios_query(uf=uf_rs, ano_inicio=2010, ano_fim=2025),
    ]

    for spec in specs:
        assert spec.params, f"{spec.name} deve declarar parâmetros explícitos"
        for param in spec.params:
            assert f"@{param.name}" in spec.sql
        assert "SELECT *" not in spec.sql.upper()

    # Prove-It: dicionário oficial CNES e SIH
    hosp_sql = specs[0].sql
    beds_sql = specs[1].sql
    sih_sql = specs[2].sql
    flows_sql = specs[3].sql
    assert "'5'" in hosp_sql
    assert "('74', '75', '76', '83', '85', '86') THEN 'uti_adulto'" in beds_sql
    assert "('77', '78', '79') THEN 'uti_pediatrica'" in beds_sql
    assert "STARTS_WITH(id_municipio_estabelecimento, @uf_code)" in sih_sql
    assert "STARTS_WITH(id_municipio_estabelecimento, @uf_code)" in flows_sql
    assert "tipo_uti IN ('51', '74', '75', '76', '83', '85', '86') THEN 'uti_adulto'" in sih_sql
    assert "COALESCE(cid_principal_subcategoria, cid_principal_categoria)" in sih_sql


def test_schema_validators_preserve_leading_zeros_and_enforce_bounds() -> None:
    df_hosp = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["0012345"],
            "id_municipio": ["4314902"],
            "sigla_uf": ["RS"],
            "ano": [2024],
            "mes": [5],
            "tipo_unidade": ["05"],
            "is_hospital_ensino": [True],
        }
    )
    validated_hosp = validate_cnes_teaching_hospitals_frame(df_hosp)
    assert validated_hosp.schema["id_estabelecimento_cnes"] == pl.String
    assert validated_hosp["id_estabelecimento_cnes"][0] == "0012345"

    df_beds_bad = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["0012345"],
            "id_municipio": ["4314902"],
            "ano": [2024],
            "mes": [5],
            "categoria_leito": ["uti_adulto"],
            "quantidade_existente": [10],
            "quantidade_sus": [12],  # inválido: SUS > existente
        }
    )
    with pytest.raises(DataContractError, match="quantidade_sus"):
        validate_cnes_beds_frame(df_beds_bad)

    df_sih_bad = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601"],
            "ano_epi": [2024],
            "semana_epi": [60],  # inválido: semana epidemiológica > 53
            "especialidade": ["clinico"],
            "grupo_cid": ["leptospirose"],
            "carater_atendimento": ["urgencia"],
            "internacoes": [10],
            "dias_permanencia_total": [50],
            "diarias_uti_total": [15],
            "obitos": [1],
            "altas_vivas": [9],
        }
    )
    with pytest.raises(DataContractError, match="semana_epi"):
        validate_sih_weekly_demand_frame(df_sih_bad)
