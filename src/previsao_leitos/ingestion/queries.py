"""Construtores de QuerySpec 100% parametrizados para BigQuery (basedosdados)."""

from __future__ import annotations

from previsao_leitos.domain import CodigoUF
from previsao_leitos.ingestion.ports import UF_CODE_TO_SIGLA, QueryParam, QuerySpec

_SQL_CNES_TEACHING_HOSPITALS = """
WITH ensino AS (
    SELECT DISTINCT
        ano,
        mes,
        id_estabelecimento_cnes
    FROM `basedosdados.br_ms_cnes.estabelecimento_ensino`
    WHERE ano BETWEEN @ano_inicio AND @ano_fim
      AND sigla_uf = @sigla_uf
)
SELECT
    est.id_estabelecimento_cnes,
    est.id_municipio,
    est.sigla_uf,
    est.ano,
    est.mes,
    est.tipo_unidade,
    (
        ens.id_estabelecimento_cnes IS NOT NULL
        OR est.tipo_atividade_ensino_pesquisa IN ('1', '2', '3', '5')
    ) AS is_hospital_ensino
FROM `basedosdados.br_ms_cnes.estabelecimento` AS est
LEFT JOIN ensino AS ens
  ON est.ano = ens.ano
 AND est.mes = ens.mes
 AND est.id_estabelecimento_cnes = ens.id_estabelecimento_cnes
WHERE est.ano BETWEEN @ano_inicio AND @ano_fim
  AND est.sigla_uf = @sigla_uf
  AND est.id_municipio IS NOT NULL
  AND (est.indicador_leito_hospitalar = 1 OR est.tipo_unidade IN ('05', '07', '20', '21'))
"""

_SQL_CNES_BEDS_BY_SPECIALTY = """
SELECT
    id_estabelecimento_cnes,
    id_municipio,
    ano,
    mes,
    CASE
        WHEN tipo_especialidade_leito IN ('74', '75', '76', '83', '85', '86') THEN 'uti_adulto'
        WHEN tipo_especialidade_leito IN ('77', '78', '79') THEN 'uti_pediatrica'
        WHEN tipo_especialidade_leito IN ('80', '81', '82') THEN 'uti_neonatal'
        WHEN tipo_especialidade_leito IN ('64', '65', '92', '93', '94', '95') THEN 'uci'
        WHEN tipo_leito = '1' THEN 'cirurgico'
        WHEN tipo_leito = '2' THEN 'clinico'
        WHEN tipo_leito = '4' THEN 'obstetrico'
        WHEN tipo_leito = '5' THEN 'pediatrico'
        ELSE 'outros'
    END AS categoria_leito,
    SUM(IFNULL(quantidade_total, 0)) AS quantidade_existente,
    SUM(LEAST(IFNULL(quantidade_sus, 0), IFNULL(quantidade_total, 0))) AS quantidade_sus
FROM `basedosdados.br_ms_cnes.leito`
WHERE ano BETWEEN @ano_inicio AND @ano_fim
  AND sigla_uf = @sigla_uf
  AND id_municipio IS NOT NULL
GROUP BY
    id_estabelecimento_cnes,
    id_municipio,
    ano,
    mes,
    categoria_leito
"""

_SQL_SIH_WEEKLY_DEMAND = """
WITH base AS (
    SELECT
        id_estabelecimento_cnes,
        EXTRACT(ISOYEAR FROM data_internacao) AS ano_epi,
        EXTRACT(ISOWEEK FROM data_internacao) AS semana_epi,
        tipo_uti,
        especialidade_leito,
        LEFT(COALESCE(cid_principal_subcategoria, cid_principal_categoria), 3) AS cid3,
        carater_internacao,
        quantidade_dias_permanencia,
        quantidade_dias_uti_mes,
        indicador_obito
    FROM `basedosdados.br_ms_sih.aihs_reduzidas`
    WHERE ano BETWEEN @ano_inicio AND @ano_fim
      AND STARTS_WITH(id_municipio_estabelecimento, @uf_code)
      AND data_internacao IS NOT NULL
      AND id_estabelecimento_cnes IS NOT NULL
)
SELECT
    id_estabelecimento_cnes,
    ano_epi,
    semana_epi,
    CASE
        WHEN tipo_uti IN ('51', '74', '75', '76', '83', '85', '86') THEN 'uti_adulto'
        WHEN tipo_uti IN ('52', '77', '78', '79') THEN 'uti_pediatrica'
        WHEN tipo_uti IN ('80', '81', '82') THEN 'uti_neonatal'
        WHEN especialidade_leito IN ('1', '01') THEN 'cirurgico'
        WHEN especialidade_leito IN ('2', '02') THEN 'obstetrico'
        WHEN especialidade_leito IN ('3', '03') THEN 'clinico'
        WHEN especialidade_leito IN ('7', '07') THEN 'pediatrico'
        ELSE 'outros'
    END AS especialidade,
    CASE
        WHEN cid3 = 'A27' THEN 'leptospirose'
        WHEN cid3 BETWEEN 'A00' AND 'A09' OR cid3 = 'B15' THEN 'infecciosas_hidricas'
        WHEN cid3 BETWEEN 'S00' AND 'T98' THEN 'trauma'
        WHEN cid3 BETWEEN 'J00' AND 'J99' THEN 'respiratorio'
        WHEN cid3 BETWEEN 'I00' AND 'I99' OR cid3 BETWEEN 'N17' AND 'N19'
            THEN 'cardiovascular_renal'
        ELSE 'outros'
    END AS grupo_cid,
    CASE
        WHEN carater_internacao IN ('1', '01') THEN 'eletivo'
        ELSE 'urgencia'
    END AS carater_atendimento,
    COUNT(*) AS internacoes,
    SUM(IFNULL(quantidade_dias_permanencia, 0)) AS dias_permanencia_total,
    SUM(IFNULL(quantidade_dias_uti_mes, 0)) AS diarias_uti_total,
    SUM(IF(indicador_obito = 1, 1, 0)) AS obitos,
    SUM(IF(indicador_obito = 1, 0, 1)) AS altas_vivas
FROM base
GROUP BY
    id_estabelecimento_cnes,
    ano_epi,
    semana_epi,
    especialidade,
    grupo_cid,
    carater_atendimento
"""

_SQL_SIH_REFERRAL_FLOWS = """
SELECT
    SUBSTR(id_municipio_paciente, 1, 6) AS id_municipio_residencia_6d,
    id_estabelecimento_cnes,
    EXTRACT(ISOYEAR FROM data_internacao) AS ano_epi,
    EXTRACT(ISOWEEK FROM data_internacao) AS semana_epi,
    COUNT(*) AS internacoes
FROM `basedosdados.br_ms_sih.aihs_reduzidas`
WHERE ano BETWEEN @ano_inicio AND @ano_fim
  AND STARTS_WITH(id_municipio_estabelecimento, @uf_code)
  AND data_internacao IS NOT NULL
  AND id_estabelecimento_cnes IS NOT NULL
  AND id_municipio_paciente IS NOT NULL
GROUP BY
    id_municipio_residencia_6d,
    id_estabelecimento_cnes,
    ano_epi,
    semana_epi
"""

# Nota empirica BigQuery (inspect_columns + sonda 2024-05 em basedosdados.br_ms_sih.aihs_reduzidas):
# A coluna `sigla_uf` nessa tabela e INT64 e 100% NULL; portanto o filtro seguro por UF
# permanece `STARTS_WITH(id_municipio_estabelecimento, @uf_code)`.
_SQL_SIH_ACTIVE_CENSUS = """
WITH base AS (
    SELECT
        id_estabelecimento_cnes,
        ano AS ano_cmpt,
        data_internacao,
        LEAST(
            GREATEST(
                IF(data_saida > data_internacao, DATE_DIFF(data_saida, data_internacao, DAY), 0),
                IFNULL(quantidade_dias_permanencia, 1),
                IFNULL(quantidade_dias_uti_mes, 0),
                1
            ),
            180
        ) AS n_dias_ativos,
        CASE
            WHEN tipo_uti IN ('51', '74', '75', '76', '83', '85', '86') THEN 'uti_adulto'
            WHEN tipo_uti IN ('52', '77', '78', '79') THEN 'uti_pediatrica'
            WHEN tipo_uti IN ('80', '81', '82') THEN 'uti_neonatal'
            WHEN especialidade_leito IN ('1', '01') THEN 'cirurgico'
            WHEN especialidade_leito IN ('2', '02') THEN 'obstetrico'
            WHEN especialidade_leito IN ('3', '03') THEN 'clinico'
            WHEN especialidade_leito IN ('7', '07') THEN 'pediatrico'
            ELSE 'outros'
        END AS especialidade,
        CASE
            WHEN LEFT(cid_principal_categoria, 3) = 'A27' THEN 'leptospirose'
            WHEN LEFT(cid_principal_categoria, 3) BETWEEN 'A00' AND 'A09'
                OR LEFT(cid_principal_categoria, 3) = 'B15'
                THEN 'infecciosas_hidricas'
            WHEN LEFT(cid_principal_categoria, 3) BETWEEN 'S00' AND 'T98' THEN 'trauma'
            WHEN LEFT(cid_principal_categoria, 3) BETWEEN 'J00' AND 'J99' THEN 'respiratorio'
            WHEN LEFT(cid_principal_categoria, 3) BETWEEN 'I00' AND 'I99'
                OR LEFT(cid_principal_categoria, 3) BETWEEN 'N17' AND 'N19'
                THEN 'cardiovascular_renal'
            ELSE 'outros'
        END AS grupo_cid,
        CASE
            WHEN carater_internacao IN ('1', '01') THEN 'eletivo'
            ELSE 'urgencia'
        END AS carater_atendimento,
        GREATEST(IFNULL(quantidade_dias_uti_mes, 0), 0) AS diarias_uti_aih
    FROM `basedosdados.br_ms_sih.aihs_reduzidas`
    WHERE ano BETWEEN @ano_inicio - 1 AND @ano_fim + 1
      AND STARTS_WITH(id_municipio_estabelecimento, @uf_code)
      AND data_internacao IS NOT NULL
      AND id_estabelecimento_cnes IS NOT NULL
),
expanded AS (
    SELECT
        b.id_estabelecimento_cnes,
        EXTRACT(ISOYEAR FROM dia_ativo) AS ano_epi,
        EXTRACT(ISOWEEK FROM dia_ativo) AS semana_epi,
        b.especialidade,
        b.grupo_cid,
        b.carater_atendimento,
        IF(dia_ativo = b.data_internacao, 1, 0) AS is_admissao,
        1.0 AS paciente_dia,
        SAFE_DIVIDE(
            CAST(b.diarias_uti_aih AS FLOAT64),
            CAST(b.n_dias_ativos AS FLOAT64)
        ) AS diaria_uti_dia,
        b.ano_cmpt
    FROM base AS b,
    UNNEST(
        GENERATE_DATE_ARRAY(
            b.data_internacao,
            DATE_ADD(b.data_internacao, INTERVAL (b.n_dias_ativos - 1) DAY)
        )
    ) AS dia_ativo
)
SELECT
    id_estabelecimento_cnes,
    ano_epi,
    semana_epi,
    especialidade,
    grupo_cid,
    carater_atendimento,
    SUM(is_admissao) AS internacoes_iniciadas,
    SUM(paciente_dia) AS pacientes_dia_ativos,
    SUM(IFNULL(diaria_uti_dia, 0.0)) AS diarias_uti_ativas,
    MAX(ano_cmpt) AS ano_cmpt_max
FROM expanded
WHERE ano_epi BETWEEN @ano_inicio AND @ano_fim
GROUP BY
    id_estabelecimento_cnes,
    ano_epi,
    semana_epi,
    especialidade,
    grupo_cid,
    carater_atendimento
"""

_SQL_SIH_EXTERNAL_REFERRAL_MATRIX = """
SELECT
    SUBSTR(id_municipio_paciente, 1, 6) AS id_municipio_residencia_6d,
    SUBSTR(id_municipio_estabelecimento, 1, 6) AS id_municipio_estabelecimento_6d,
    id_estabelecimento_cnes,
    COUNT(*) AS internacoes_urgencia_total,
    SUM(GREATEST(IFNULL(quantidade_dias_uti_mes, 0), 0)) AS diarias_uti_total
FROM `basedosdados.br_ms_sih.aihs_reduzidas`
WHERE ano BETWEEN @ano_inicio AND @train_end_year + 1
  AND EXTRACT(ISOYEAR FROM data_internacao) BETWEEN @ano_inicio AND @train_end_year
  AND STARTS_WITH(id_municipio_estabelecimento, @uf_code)
  AND data_internacao IS NOT NULL
  AND id_estabelecimento_cnes IS NOT NULL
  AND id_municipio_paciente IS NOT NULL
  AND SUBSTR(id_municipio_paciente, 1, 6) != SUBSTR(id_municipio_estabelecimento, 1, 6)
  AND (
      carater_internacao NOT IN ('1', '01')
      OR IFNULL(quantidade_dias_uti_mes, 0) > 0
  )
GROUP BY
    id_municipio_residencia_6d,
    id_municipio_estabelecimento_6d,
    id_estabelecimento_cnes
HAVING COUNT(*) >= @min_cell_count
"""

_SQL_SIH_DELAY_TRIANGLE = """
WITH incremental AS (
    SELECT
        EXTRACT(ISOYEAR FROM data_internacao) AS ano_epi,
        EXTRACT(ISOWEEK FROM data_internacao) AS semana_epi,
        GREATEST(
            0,
            LEAST(
                @max_delay_weeks,
                CAST(FLOOR(DATE_DIFF(DATE(ano, mes, 28), data_internacao, DAY) / 7.0) AS INT64)
            )
        ) AS delay_weeks,
        COUNT(*) AS incremental_count
    FROM `basedosdados.br_ms_sih.aihs_reduzidas`
    WHERE ano BETWEEN @ano_inicio AND @ano_fim + 1
      AND STARTS_WITH(id_municipio_estabelecimento, @uf_code)
      AND data_internacao IS NOT NULL
      AND mes BETWEEN 1 AND 12
    GROUP BY
        ano_epi,
        semana_epi,
        delay_weeks
)
SELECT
    ano_epi,
    semana_epi,
    delay_weeks,
    incremental_count,
    SUM(incremental_count) OVER (
        PARTITION BY ano_epi, semana_epi
        ORDER BY delay_weeks
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_count
FROM incremental
WHERE ano_epi BETWEEN @ano_inicio AND @ano_fim
ORDER BY
    ano_epi,
    semana_epi,
    delay_weeks
"""

_SQL_IBGE_MUNICIPIOS = """
SELECT
    m.id_municipio,
    m.nome,
    m.sigla_uf,
    p.ano,
    CAST(p.populacao AS FLOAT64) AS populacao,
    m.id_regiao_saude,
    m.nome_regiao_saude
FROM `basedosdados.br_bd_diretorios_brasil.municipio` AS m
LEFT JOIN `basedosdados.br_ibge_populacao.municipio` AS p
  ON m.id_municipio = p.id_municipio
 AND p.ano BETWEEN @ano_inicio AND @ano_fim
WHERE m.sigla_uf = @sigla_uf
"""


def build_cnes_teaching_hospitals_query(
    *, uf: CodigoUF, ano_inicio: int, ano_fim: int
) -> QuerySpec:
    """Retorna QuerySpec parametrizada do catálogo CNES de hospitais-escola."""
    return QuerySpec(
        name="cnes_hospitais_escola",
        sql=_SQL_CNES_TEACHING_HOSPITALS,
        params=(
            QueryParam("sigla_uf", "STRING", UF_CODE_TO_SIGLA[str(uf)]),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version="v2",
        expected_columns=(
            "id_estabelecimento_cnes",
            "id_municipio",
            "sigla_uf",
            "ano",
            "mes",
            "tipo_unidade",
            "is_hospital_ensino",
        ),
    )


def build_cnes_beds_by_specialty_query(
    *, uf: CodigoUF, ano_inicio: int, ano_fim: int
) -> QuerySpec:
    """Retorna QuerySpec parametrizada de leitos CNES por tipo específico de UTI/especialidade."""
    return QuerySpec(
        name="cnes_leitos_especialidade",
        sql=_SQL_CNES_BEDS_BY_SPECIALTY,
        params=(
            QueryParam("sigla_uf", "STRING", UF_CODE_TO_SIGLA[str(uf)]),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version="v2",
        expected_columns=(
            "id_estabelecimento_cnes",
            "id_municipio",
            "ano",
            "mes",
            "categoria_leito",
            "quantidade_existente",
            "quantidade_sus",
        ),
    )


def build_sih_weekly_demand_query(*, uf: CodigoUF, ano_inicio: int, ano_fim: int) -> QuerySpec:
    """Retorna QuerySpec parametrizada da demanda semanal agregada SIH-SUS (zero PII)."""
    return QuerySpec(
        name="sih_demanda_semanal_agg",
        sql=_SQL_SIH_WEEKLY_DEMAND,
        params=(
            QueryParam("uf_code", "STRING", str(uf)),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version="v3",
        expected_columns=(
            "id_estabelecimento_cnes",
            "ano_epi",
            "semana_epi",
            "especialidade",
            "grupo_cid",
            "carater_atendimento",
            "internacoes",
            "dias_permanencia_total",
            "diarias_uti_total",
            "obitos",
            "altas_vivas",
        ),
    )


def build_sih_referral_flows_query(*, uf: CodigoUF, ano_inicio: int, ano_fim: int) -> QuerySpec:
    """Retorna QuerySpec parametrizada dos fluxos semanais MUNIC_RES -> CNES (zero PII)."""
    return QuerySpec(
        name="sih_fluxos_referencia_semanal",
        sql=_SQL_SIH_REFERRAL_FLOWS,
        params=(
            QueryParam("uf_code", "STRING", str(uf)),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version="v2",
        expected_columns=(
            "id_municipio_residencia_6d",
            "id_estabelecimento_cnes",
            "ano_epi",
            "semana_epi",
            "internacoes",
        ),
    )


def build_sih_active_census_query(
    *, uf: CodigoUF, ano_inicio: int, ano_fim: int
) -> QuerySpec:
    """Retorna QuerySpec parametrizada de censo ativo semanal SIH-SUS (sem double-count +1)."""
    return QuerySpec(
        name="sih_censo_ativo_semanal",
        sql=_SQL_SIH_ACTIVE_CENSUS,
        params=(
            QueryParam("uf_code", "STRING", str(uf)),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version="v2",
        expected_columns=(
            "id_estabelecimento_cnes",
            "ano_epi",
            "semana_epi",
            "especialidade",
            "grupo_cid",
            "carater_atendimento",
            "internacoes_iniciadas",
            "pacientes_dia_ativos",
            "diarias_uti_ativas",
            "ano_cmpt_max",
        ),
    )


def build_sih_external_referral_matrix_query(
    *,
    uf: CodigoUF,
    ano_inicio: int = 2010,
    train_end_year: int = 2022,
    min_cell_count: int = 5,
) -> QuerySpec:
    """Retorna QuerySpec da matriz W_ref externa (origem != destino, <= train_end_year)."""
    return QuerySpec(
        name="sih_w_ref_externa_treino",
        sql=_SQL_SIH_EXTERNAL_REFERRAL_MATRIX,
        params=(
            QueryParam("uf_code", "STRING", str(uf)),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("train_end_year", "INT64", train_end_year),
            QueryParam("min_cell_count", "INT64", min_cell_count),
        ),
        schema_version="v2",
        expected_columns=(
            "id_municipio_residencia_6d",
            "id_municipio_estabelecimento_6d",
            "id_estabelecimento_cnes",
            "internacoes_urgencia_total",
            "diarias_uti_total",
        ),
    )


def build_sih_delay_triangle_query(
    *,
    uf: CodigoUF,
    ano_inicio: int = 2020,
    ano_fim: int = 2026,
    max_delay_weeks: int = 12,
) -> QuerySpec:
    """Retorna QuerySpec do triângulo de atraso de processamento SIH para ChainLadderNowcaster."""
    return QuerySpec(
        name="sih_delay_triangle",
        sql=_SQL_SIH_DELAY_TRIANGLE,
        params=(
            QueryParam("uf_code", "STRING", str(uf)),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
            QueryParam("max_delay_weeks", "INT64", max_delay_weeks),
        ),
        schema_version="v1",
        expected_columns=(
            "ano_epi",
            "semana_epi",
            "delay_weeks",
            "incremental_count",
            "cumulative_count",
        ),
    )


def build_ibge_municipios_query(*, uf: CodigoUF, ano_inicio: int, ano_fim: int) -> QuerySpec:
    """Retorna QuerySpec parametrizada de diretório municipal, região de saúde e população IBGE."""
    return QuerySpec(
        name="ibge_municipios_rs",
        sql=_SQL_IBGE_MUNICIPIOS,
        params=(
            QueryParam("sigla_uf", "STRING", UF_CODE_TO_SIGLA[str(uf)]),
            QueryParam("ano_inicio", "INT64", ano_inicio),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version="v1",
        expected_columns=(
            "id_municipio",
            "nome",
            "sigla_uf",
            "ano",
            "populacao",
            "id_regiao_saude",
            "nome_regiao_saude",
        ),
    )
