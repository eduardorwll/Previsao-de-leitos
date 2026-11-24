# Especificação V1 — Módulo `ingestion` (`spec/SPEC-ingestion.md`)

- **Status:** `proposed` *(aguardando aprovação antes de iniciar a implementação em código)*
- **Etapa / Feature:** Etapa 1 — Trilha 1A (Feature `F01 — Ingestão Segura e Contratos LGPD`)
- **Escopo V1:** Rio Grande do Sul (`sigla_uf = "RS"`, `uf_code = "43"`, 497 municípios, 7 Macrorregiões de Saúde, período `2010–2025`), com arquitetura pronta para expansão nacional (`V2`).

> **Fronteira Clínica e LGPD:** Este módulo extrai exclusivamente **dados públicos agregados** para pesquisa. É proibido baixar, armazenar em cache ou registrar em log qualquer campo identificável de paciente (CPF, CNS, nome, data de nascimento, prontuário ou endereço residencial).

---

## 1. Objetivo e Critérios Testáveis de Aceite

Substituir a extração acoplada e insegura do protótipo legado (`legacy/previsao_leitos_colab.py`) por um módulo profundo `src/previsao_leitos/ingestion/` que:

1. **Parametrize 100% das consultas BigQuery** via `bigquery.ScalarQueryParameter` (zero interpolação f-string de valores; identificadores validados por allowlist regex `^[a-z0-9_]+$`).
2. **Audite custo antes de executar** (`dry_run=True`), rejeitando automaticamente qualquer consulta cujo `total_bytes_processed` exceda `max_mb_scanned` (`PREVISAO_MAX_BQ_MB`, padrão `2000 MB` na V1 RS) sem cobrar um único byte.
3. **Preserve a granularidade de hospitais-escola e especialidades de UTI**, consultando `basedosdados.br_ms_cnes.estabelecimento`, `basedosdados.br_ms_cnes.leito` (separando explicitamente **UTI Adulto**, **UTI Pediátrica**, **UTI Neonatal**, **UCI**, **Clínico** e **Cirúrgico**, além de SUS vs. privado) e `basedosdados.br_ms_cnes.estabelecimento_ensino`.
4. **Extraia demanda real semanal agregada na origem** via `basedosdados.br_ms_sih.aihs_reduzidas` (por `CNES`, semana epidemiológica, especialidade, grupo CID-10, caráter `CAR_INT` eletivo vs. urgência, `DIAS_PERM`, `UTI_MES_TO`, `MORTE` e fluxo `MUNIC_RES → CNES`) e notificações semanais de agravos pós-enchente (`SINAN` Leptospirose/hídricas), **aplicando supressão primária e complementar de pequenas células ($n < 5$)** antes de persistir em disco.
5. **Mantenha 100% de testabilidade offline** atrás da costura `TabularSourcePort` com dois adaptadores reais: `BigQueryAdapter` (produção/extração real) e `ParquetFixtureAdapter` (testes unitários/integração e CI sem credenciais GCP).
6. **Garanta cache atômico e rastreável**, indexado por `SHA-256` de `(query_sql + parametros_ordenados + schema_version)`, escrevendo via arquivo `.tmp` + `os.replace()` sob diretório raiz permitido (`data/raw/`) e emitindo um manifesto JSON (`IngestionManifest`) com contagens, linhas descartadas, bytes escaneados e timestamp UTC.

---

## 2. Comandos de Verificação (Alvo após Implementação)

```bash
# Instalar extras de dados (polars, pyarrow, pandera, google-cloud-bigquery, pydantic)
uv sync --python 3.12 --extra dev --extra data

# Gates rápidos (CPU-only, offline, zero BigQuery)
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/check_constraints.py src tests
uv run pytest -m "not slow and not bigquery" --cov=src/previsao_leitos --cov-fail-under=80

# Teste opcional de conectividade gratuita (0 bytes cobrados) quando .env estiver configurado
uv run pytest -m "bigquery" -k "schema_introspection"
```

---

## 3. Estrutura do Módulo e Contratos de Interface

```text
src/previsao_leitos/ingestion/
├── __init__.py          # Superfície pública enxuta
├── ports.py             # TabularSourcePort (Protocol), QuerySpec, QueryParam, IngestionManifest, IngestionResult
├── adapters.py          # BigQueryAdapter (dry_run + ScalarQueryParameter) e ParquetFixtureAdapter (offline)
├── cache.py             # AtomicParquetCache (chave SHA-256, escrita .tmp -> os.replace, confinamento em data_dir)
├── lgpd.py              # Allowlist de colunas sem PII + supressão primária e complementar de pequenas células (n < 5)
├── schemas.py           # Contratos polars/pandera (preservando zeros à esquerda em CNES e IBGE)
└── queries.py           # Construtores de QuerySpec para CNES (estab+leito+ensino), SIH-SUS semanal, SINAN e IBGE
```

### Contrato Principal (Recorte de Estilo e Tipos)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import polars as pl

from previsao_leitos.domain import CodigoUF

@dataclass(frozen=True)
class QuerySpec:
    name: str
    sql: str
    params: tuple[tuple[str, str, str | int | float], ...]  # (nome, tipo_bq, valor)
    schema_version: str
    expected_columns: tuple[str, ...]

class TabularSourcePort(Protocol):
    """Costura real com 2 adaptadores: BigQueryAdapter e ParquetFixtureAdapter."""
    def inspect_columns(self, table_ref: str) -> tuple[str, ...]: ...
    def estimate_bytes(self, spec: QuerySpec) -> int: ...
    def fetch(self, spec: QuerySpec, *, max_bytes_billed: int) -> pl.DataFrame: ...

@dataclass(frozen=True)
class IngestionConfig:
    uf: CodigoUF
    ano_inicio: int
    ano_fim: int
    data_dir: Path
    max_mb_scanned: int = 2000
    min_cell_count: int = 5
```

---

## 4. Datasets e Esquemas da Trilha 1A (RS)

> **Regra obrigatória pré-query:** Antes de executar qualquer extração no BigQuery, `BigQueryAdapter.inspect_columns()` consulta metadados gratuitos (`0 bytes` via `client.get_table()` ou `INFORMATION_SCHEMA.COLUMNS`) para confirmar nomes de colunas e tipos nativos, evitando `CAST` redundante em chaves `STRING` clusterizadas.

| Artefato `data/raw/` | Tabela Origem (`basedosdados`) | Granularidade | Colunas-Chave & Garantias |
| :--- | :--- | :--- | :--- |
| `cnes_hospitais_escola` | `br_ms_cnes.estabelecimento` + `br_ms_cnes.estabelecimento_ensino` | `(id_estabelecimento_cnes, ano, mes)` | `id_estabelecimento_cnes` (`CodigoCNES` 7d), `id_municipio` (`CodigoMunicipioIBGE` 7d), `sigla_uf`, `tipo_unidade`, `esfera_administrativa`, `is_hospital_ensino` (`bool`), `vinculo_ensino` |
| `cnes_leitos_especialidade` | `br_ms_cnes.leito` | `(id_estabelecimento_cnes, ano, mes, categoria_leito)` | `categoria_leito` padronizada (`uti_adulto`, `uti_pediatrica`, `uti_neonatal`, `uci`, `clinico`, `cirurgico`, `obstetrico`, `pediatrico`), `quantidade_existente` ($\ge 0$), `quantidade_sus` ($\le$ existente) |
| `sih_demanda_semanal_agg` | `br_ms_sih.aihs_reduzidas` | `(id_estabelecimento_cnes, ano_epi, semana_epi, especialidade, grupo_cid, carater_atendimento)` | **Agregado no SQL:** `internacoes`, `dias_permanencia_total`, `diarias_uti_total`, `obitos`, `altas_vivas`; tabela auxiliar de fluxo `(id_municipio_residencia_6d, id_estabelecimento_cnes, ano_epi, semana_epi, internacoes)` — **ambas sujeitas à supressão LGPD $n < 5$** |
| `sinan_agravos_semanal_agg` | `br_ms_sinan` *(ou dump oficial DATASUS via adaptador local)* | `(id_municipio, ano_epi, semana_epi, agravo)` | `agravo` (`leptospirose`, `dengue`, `hepatite_a`, `tetano`), `notificacoes`, `hospitalizacoes`, `exposicao_enchente_30d` — **supressão LGPD $n < 5$** |
| `ibge_municipios_rs` | `br_bd_diretorios_brasil.municipio` + `br_ibge_populacao.municipio` | `(id_municipio, ano)` | `id_municipio` (7d), `nome`, `sigla_uf`, `id_macrorregiao_saude`, `area_km2_oficial`, `populacao` (sem imputação por mediana futura!) |

### Regra de Supressão de Pequenas Células (LGPD — `lgpd.py`)
1. **Allowlist estrita de saída:** qualquer coluna fora do contrato (ou cujo nome sugira PII: `cpf`, `cns`, `nome_paciente`, `dt_nasc`, `prontuario`, `cep`) levanta `DataContractError` imediatamente.
2. **Supressão primária:** toda célula de contagem com `0 < count < min_cell_count` (padrão `5`) é mascarada para `null` (mantendo zeros verdadeiros `0` intactos e distintos de `null`).
3. **Supressão complementar:** quando exatamente **uma** célula é suprimida dentro de um estrato cuja soma marginal está exposta, a segunda menor célula positiva do mesmo estrato também é suprimida para impedir reconstrução por subtração.

---

## 5. Estratégia de Testes (TDD — Offline & Determinístico)

- **Testes Unitários (*Small*, ms, sem rede):**
  - `test_sql_parameterization`: garante que todas as queries em `queries.py` usam `@param` e rejeitam identificadores fora da allowlist.
  - `test_dry_run_budget_guard`: verifica que `BigQueryAdapter` recusa consulta acima de `max_mb_scanned` com `ConfigurationError` antes de chamar execução faturável.
  - `test_atomic_cache_key_and_invalidation`: verifica que mudar `ano_fim` ou `schema_version` gera chave SHA-256 diferente, que escrita ocorre via `.tmp` e que caminhos fora de `data_dir` (*path traversal*) são bloqueados.
  - `test_lgpd_small_cell_suppression`: verifica rejeição de colunas PII, preservação de `0` verdadeiro, supressão primária (`1..4 → null`) e supressão complementar em estratos com 1 célula mascarada.
  - `test_schema_contracts_preserve_leading_zeros`: valida que códigos CNES (`"0012345"`) e IBGE (`"4314902"`) permanecem `pl.Utf8` com 7 dígitos e que `quantidade_sus <= quantidade_existente`.
- **Testes de Integração (*Medium*, s, Parquet local):**
  - `test_rs_fixture_ingestion_end_to_end`: exercita `ParquetFixtureAdapter` sobre uma mini-fixture realista do RS (HCPA, HUSM, HE-UFPel, HU-FURG) validando contratos, manifesto JSON e reconciliação de contagens.

---

## 6. Limites e Escopo Adiado (*Out of Scope — Deferred*)

- **Sempre:** executar `dry_run=True` antes de queries faturáveis; gravar `IngestionManifest` junto a cada snapshot Parquet; manter `ParquetFixtureAdapter` sincronizado com os schemas reais.
- **Pedir confirmação antes:** aumentar `PREVISAO_MAX_BQ_MB` acima de `2000 MB` ou extrair todas as 27 UFs (`V2`).
- **Nunca:** baixar microdados individuais de pacientes para disco local; preencher ausências estruturais com `0.0`; fazer imputação usando anos futuros (`2024–2025`).
- **Adiado para as próximas fatias (fora desta Spec V1):**
  - Conectores de grade climática e hidrologia (`MERGE-CPTEC`, `ANA HidroWeb`, `Ottobacias`, `S2iD`) → entrarão em `spec/SPEC-geo-hydro-etl.md` (Fatia 1A.3).
  - Enriquecimento da Trilha 1B (`GERINT-POA`, `ERA5-Land`, `Sentinel-1 SAR / HAND`, `OSMnx`) → Fatia 1B.
  - Orquestração distribuída `dask` para 27 UFs → V2 Brasil (Etapa 4).
