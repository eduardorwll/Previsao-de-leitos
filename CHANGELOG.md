# CHANGELOG.md

Histórico de mudanças orientado ao impacto para pesquisadores e mantenedores ([Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)).

## [Unreleased]

### Added
- **Etapa 1 / Feature F02 (`geo` — Especificação V1):** Especificação completa em `spec/SPEC-geo-hydro-etl.md` e documentação em `docs/features/02-geo-hidrologia-rs.md` (grade calendário 100% RS com `MERGE-CPTEC`, `ANA/Ottobacias` montante→jusante, `S2iD` e *Tracer Bullet* semanal $h=1..4$ semanas vs. Persistência).
- **Agregação Regional LGPD (`ingestion`):** Adicionada `aggregate_regional_weekly_demand` preservando sinal de agravos raros pós-enchente (ex.: leptospirose) na escala de Região de Saúde com supressão LGPD ($n < 5$) aplicada na camada de divulgação.
- **BigQuery Storage gRPC:** Adicionado `google-cloud-bigquery-storage` ao extra `[data]` em `pyproject.toml` para streaming Arrow de alta velocidade.

### Fixed
- **Códigos Oficiais CNES e SIH-SUS (`queries.py`):** Validação direta nos dicionários oficiais `br_ms_cnes.dicionario` e `br_ms_sih.dicionario`, corrigindo `tipo_atividade_ensino_pesquisa IN ('1','2','3','5')` (incluindo HCPA, Santa Casa, GHC e HUSM — `44` hospitais-escola ativos no RS em mai/2024), `tipo_especialidade_leito = '86'` (UCO III em `uti_adulto`), `'64'/'65'` em `uci`, filtro `STARTS_WITH(id_municipio_estabelecimento, @uf_code)` no SIH (`sigla_uf` é nulo na base), separação de `uti_adulto`/`uti_pediatrica`/`uti_neonatal` via `tipo_uti` e `LEFT(COALESCE(cid_principal_subcategoria, cid_principal_categoria), 3)` para captura integral de CID-10 (leptospirose, respiratório e trauma).
- **Observabilidade e Limite de Custo (`adapters.py`):** Leitura de `last_bytes_processed` após conclusão do job BigQuery e encapsulamento de `bytesBilledLimitExceeded` em `ConfigurationError`.

### Added (Fatia 1A.1)
- **Etapa 1 / Feature F01 (`ingestion` — Implementação TDD Fatia 1A.1):** Implementado módulo `src/previsao_leitos/ingestion/` (`ports.py`, `adapters.py`, `cache.py`, `lgpd.py`, `schemas.py`, `queries.py`, `pipeline.py`) com `BigQueryAdapter` parametrizado + `dry_run` RLS-aware, `ParquetFixtureAdapter` offline, cache Parquet atômico SHA-256, contratos `polars` preservando zeros à esquerda e supressão LGPD primária e complementar ($n < 5$).
- **Testes & CI F01:** Adicionados 21 novos testes (`tests/unit/test_ingestion_*.py` e `tests/integration/test_ingestion_rs_pipeline.py` totalizando 38 testes offline com `89,4%` de cobertura) + teste ao vivo `0 bytes` (`tests/integration/test_bigquery_schema_live.py`) e atualizado `.github/workflows/ci.yml` com `--extra data`.
- **Etapa 1 / Feature F01 (`ingestion` — Especificação V1):** Especificação completa em `spec/SPEC-ingestion.md` (costura `TabularSourcePort` com `BigQueryAdapter` parametrizado + `dry_run` + `ParquetFixtureAdapter`, cache atômico SHA-256, extração CNES por tipo específico de UTI e hospitais-escola, demanda semanal `SIH-SUS`/`SINAN` agregada com supressão LGPD $n < 5$), documentação da feature (`docs/features/01-ingestao-segura.md`) e guia seguro de acesso ao BigQuery sem chaves `.json` (`docs/runbooks/setup-bigquery.md`).

### Changed
- **Planejamento (Abordagem Mista — RS Semanal Multi-Horizonte):** Atualizados `spec/capability-map.md`, `docs/PROGRESSO.md`, `docs/data-catalog.md`, `docs/adr/0002` e `docs/adr/0004` para incorporar previsão semanal *ex-ante* ($h=1..4$ semanas, foco em $h=2$ semanas / Onda 1 + Onda 2 leptospirose), estratificação explícita por tipo de UTI e Macrorregião/Hospital-Escola, e execução mista em duas trilhas incrementais (Trilha 1A núcleo semanal RS → Trilha 1B enriquecimento físico e fila reprimida).

## [0.1.0] - 2026-11-24 (Etapa 0 — Fundação Modular)

### Added
- **Etapa 0 / Feature F00 (`domain`):** Empacotamento Python 3.12 (`pyproject.toml`, `.python-version`, `uv.lock`), tipos nominais validados (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`), hierarquia única de exceções (`src/previsao_leitos/domain/`) e 17 testes unitários com `100%` de cobertura de linhas e branches (`tests/unit/test_domain_types.py`).
- **Qualidade e CI:** Verificador mecânico de regras (`scripts/check_constraints.py`), pipeline CI rápido CPU-only (`.github/workflows/ci.yml`: `ruff`, `mypy`, `pytest --cov`, `pip-audit`, `gitleaks`), Dependabot (`.github/dependabot.yml`) e `.env.example`.
- **Documentação Técnica e Científica:** Painel de progresso por etapa e feature (`docs/PROGRESSO.md`), documentação da feature F00 (`docs/features/00-dominio-e-fundacao.md`), catálogo de 14 bases de dados (`docs/data-catalog.md`), mapa de capacidades (`spec/capability-map.md`), glossário em 3 contextos (`GLOSSARY-MAP.md` + `docs/glossaries/`), 5 ADRs (`docs/adr/0001–0005`), padrões de qualidade (`CONSTRAINTS.md`), diário de desempenho (`PERF.md`) e runbook de expurgo de segredos Git (`docs/runbooks/git-history-secret-purge.md`).

### Changed
- Arquivado o protótipo monolítico Colab em `legacy/previsao_leitos_colab.py` (via `git mv`, preservando histórico) para evitar conflito de importação com `src/previsao_leitos/`.
- Atualizado `README.md` com navegação para a documentação estruturada, instruções de início rápido e síntese metodológica das lacunas identificadas no protótipo inicial.

## [0.0.1] - 2026-11-23 (Protótipo Colab Legado)
- Versão inicial monolítica `previsao_leitos.py` (`HydroTFTNet` + delta mensal CNES via `HistGradientBoostingRegressor`) e relatório exploratório `README.md`.
