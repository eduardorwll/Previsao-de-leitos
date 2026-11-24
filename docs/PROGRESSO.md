# Mapa de Progresso do Projeto — Etapas e Features

> **Objetivo:** Prever, com antecedência (*ex-ante*) e intervalos de incerteza calibrados, a **necessidade de leitos hospitalares (clínicos, cirúrgicos, UTI e *surge capacity*) em hospitais-escola** frente a enchentes — começando pelo **Rio Grande do Sul (V1)** e escalando posteriormente para **todo o Brasil (V2)**.
>
> **Regime:** Pesquisa aberta | **Aviso:** Sistema voltado exclusivamente à pesquisa e ao planejamento agregado (*Not for patient care or live clinical use*).

---

## Painel Geral por Etapa

| Etapa | Escopo | Status | Progresso | Marco de Saída |
| :---: | :--- | :---: | :---: | :--- |
| **Etapa 0** | Fundação do Pacote, Segurança, Qualidade e Documentação Base | ✅ Concluída | `100%` | CI verde (`ruff`, `mypy`, `pytest 100%`, `pip-audit`) + pacote `domain` + catálogo de dados |
| **Etapa 1** | Arquitetura Modular, Ingestão Parametrizada e Base RS (Hospitais-Escola) | 🔄 Em andamento | `85%` | 100% dos 497 mun. RS + catálogo HUs + SIH-SUS agregado sem PII + *Tracer Bullet* semanal + testes de vazamento zero |
| **Etapa 2** | Pivô Científico RS: Demanda Real, LOS e Hidrologia Calibrada | ⏳ Planejada | `0%` | Modelo *ex-ante* supera Persistência/Sazonal-Ingênuo + calibração (`BSS > 0`) |
| **Etapa 3** | Incerteza Bayesiana, Simulação de Filas (`SimPy`) e Auditoria RS (2024/2025) | ⏳ Planejada | `0%` | **V1 RS validada** no holdout intocado `2025` + plano de gatilhos para hospitais-escola |
| **Etapa 4** | Escala Nacional Brasil (27 UFs, ST-GNN e Otimização Pareto) | ⏳ Planejada | `0%` | **V2 Brasil validada** via *Leave-UF-Out* e *Leave-Basin-Out* em todos os biomas |
| **Etapa 5** | Governança TRIPOD+AI / RECORD, Publicação e Release `v1.0.0` | ⏳ Planejada | `0%` | Checklists completos, reprodutibilidade fim a fim e relatório técnico aberto |

---

## Progresso Feature por Feature

| Módulo / Feature | Documentação Detalhada | Status | Resumo do Valor Entregue |
| :--- | :--- | :---: | :--- |
| **F00 — Domínio, Tipos Validados e Qualidade Base (`domain`)** | [00-dominio-e-fundacao.md](features/00-dominio-e-fundacao.md) | ✅ Concluída | Tipos nominais (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`), hierarquia única de exceções, pacote Python 3.12 (`uv`), verificador mecânico de qualidade e CI |
| **F01 — Ingestão Segura e Contratos LGPD (`ingestion`)** | [01-ingestao-segura.md](features/01-ingestao-segura.md) | ✅ Concluída (RS 2010–2025) | BigQuery parametrizado + gRPC Storage, CNES `v2` (44 HUs RS + UTI específica), SIH semanal `v3` (`COALESCE` CID-10 + fluxos), IBGE 497 mun. + 30 Regiões e agregação regional LGPD ($n < 5$) |
| **F02 — Geoespacial, Hidrologia 100% RS e *Tracer Bullet* (`geo`)** | [02-geo-hidrologia-rs.md](features/02-geo-hidrologia-rs.md) | ✅ Concluída (Trilha 1A) | Grade calendário diária/semanal sem compressão de lacunas, DAG de bacias RS montante→jusante (`30/30` Regiões), `NasaPowerHydroAdapter` + `OpenMeteoEra5Adapter`, e *Tracer Bullet* RS 2023–2024 superando Persistência e Sazonal em Diárias UTI Adulto (`MASE 0,89–0,93`) e Respiratório (`MASE 0,80–0,88`) |
| **F03 — Hidrologia Calibrada (`hydrology`)** | *(a criar na Etapa 2)* | ⏳ Planejada | Grade diária sem lacunas, `HydroFiLMMonotonicNet` (RS) e `HydroTFTNet` (BR) com monotonicidade real e calibração pós-hoc (ECE/BSS) |
| **F04 — Demanda de Internações e Tempo de Permanência (`demand`)** | *(a criar na Etapa 2)* | ⏳ Planejada | `SIH-SUS` agregado + `SINAN`, baselines `statsmodels`/`scikit-learn` quantil, LOS sob riscos competitivos (`scikit-survival`), modelo hierárquico `PyMC` e Hetero-GNN (`PyG` na V2) |
| **F05 — Prontidão, Filas Hospitalares e Otimização (`readiness`)** | *(a criar na Etapa 3)* | ⏳ Planejada | Simulador de filas de emergência/enfermaria/UTI (`SimPy`), explicabilidade `SHAP`, cenários de estresse composto e alocação multiobjetivo (`pymoo` na V2) |
| **F06 — Auditoria Estatística e Relatórios (`audit`)** | *(a criar na Etapa 3)* | ⏳ Planejada | Intervalos de confiança por *block-bootstrap*, Análise de Curva de Decisão (Net Benefit), figuras acessíveis WCAG 2.2 e conformidade TRIPOD+AI / PROBAST+AI / RECORD |
| **F07 — CLI e Orquestração Reprodutível (`cli`)** | *(a criar na Etapa 1–3)* | ⏳ Planejada | Ponto de entrada único, logging JSON estruturado, `run_manifest.json` e linhagem de dados reproduzível |

---

## Detalhamento Etapa por Etapa

### ✅ Etapa 0 — Fundação do Pacote, Segurança, Qualidade e Documentação Base
- [x] Empacotamento Python 3.12 via `pyproject.toml`, `.python-version` e `uv.lock`
- [x] Arquivamento do protótipo monolítico em `legacy/previsao_leitos_colab.py` (eliminando conflito de importação com `src/previsao_leitos/`)
- [x] Feature **F00 (`domain`)**: tipos validados na fronteira (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`) + hierarquia única de exceções + 17 testes unitários (`100%` de cobertura de linhas e branches)
- [x] Verificador mecânico de qualidade (`scripts/check_constraints.py`) calibrado + pipeline CI CPU-only (`.github/workflows/ci.yml` + `.github/dependabot.yml`)
- [x] Glossário de domínio em 3 contextos (`GLOSSARY-MAP.md` + `docs/glossaries/`), 5 registros de decisão arquitetural (`docs/adr/0001–0005`), mapa de capacidades (`spec/capability-map.md`) e catálogo de 14 bases de dados (`docs/data-catalog.md`)
- [x] Template seguro `.env.example` e runbook de expurgo do `.env` legado no histórico Git (`docs/runbooks/git-history-secret-purge.md`)
- [ ] **Pendente (ação operacional):** executar o runbook de expurgo de `b3ffad6:.env` após validação de IAM/billing no console GCP

### 🔄 Etapa 1 — Arquitetura Modular, Ingestão Parametrizada e Base RS (Abordagem Mista)
- **Trilha 1A — Núcleo Semanal RS (*Tracer Bullet* Prioritário):**
  - [x] Especificar contrato do módulo `ingestion` (`spec/SPEC-ingestion.md`) e guia de setup seguro do BigQuery (`docs/runbooks/setup-bigquery.md`)
  - [x] Implementar `TabularSourcePort` (`BigQueryAdapter` parametrizado com `dry_run` + `ParquetFixtureAdapter`), cache atômico por hash, contratos `polars` e supressão LGPD primária/complementar ($n < 5$)
  - [x] Executar extração real RS `2010–2025` (`CNES v2` + `SIH semanal v3` + `IBGE 497 mun. / 30 Regiões`) e validar cobertura completa de 2024 e 2025 no BigQuery
  - [x] Especificar contrato do módulo `geo` + ETL hidrológico 100% RS (`spec/SPEC-geo-hydro-etl.md`)
  - [x] Implementar módulo `geo` 100% RS (grade calendário completa + API causal + DAG de bacias RS montante→jusante + `NasaPowerHydroAdapter` / `OpenMeteoEra5Adapter`) e rodar *Tracer Bullet* semanal real (`2023` e `2024`, $h \in \{1,2,3,4\}$ semanas, `2025` bloqueado) com testes *Prove-It* (`54` testes, `88,2%` cobertura)
- **Trilha 1B — Enriquecimento Físico e Fila Reprimida (Incremental):**
  - [ ] Integrar `GERINT-POA` (demanda solicitada/fila reprimida na Região Metropolitana de Porto Alegre)
  - [ ] Integrar `ERA5-Land` (umidade do solo `swvl1–4` + temperatura) e relevo/inundação (`HAND` + `Sentinel-1 SAR`) + malha viária (`OSMnx`)
  - [ ] Aposentar `legacy/previsao_leitos_colab.py` em commit dedicado após paridade e cobertura $\ge 80\%$

### ⏳ Etapa 2 — Pivô Científico RS: Demanda Semanal Multi-Horizonte, LOS e Hidrologia Calibrada
- [ ] Congelar splits temporais (`<=2022` treino, `2023` calibração, `2024` auditoria RS, **`2025` holdout intocado**)
- [ ] Implementar `HydroFiLMMonotonicNet` em PyTorch Lightning com monotonicidade real (sem impermeabilidade no ramo irrestrito) e calibração pós-hoc (`ECE`, `BSS`)
- [ ] Acoplar previsões hidrológicas estritamente *ex-ante* com defasagens distribuídas (Onda 1: D+0–5; Onda 2: D+7–28) para horizontes de 1 a 4 semanas
- [ ] Comparar baselines por especialidade (**UTI Adulto**, **UTI Ped/Neo**, **Clínico**, **Cirúrgico**): `Persistência`, `Sazonal-Ingênuo`, `statsmodels` NegBin/Hurdle com *Distributed Lags*, `scikit-learn` quantil $P_{50}/P_{90}/P_{95}$, `aeon` MiniRocket e `TimesFM 2.5`
- [ ] Modelar Tempo de Permanência (LOS) sob riscos competitivos (`scikit-survival`: alta, UTI, transferência, óbito) convertendo chegadas em curva diária/semanal de leitos ocupados

### ⏳ Etapa 3 — Incerteza Bayesiana, Prontidão (`SimPy`) e Auditoria RS (2024/2025)
- [ ] Ajustar modelo hierárquico Bayesiano (`PyMC`: `Hospital-Escola ⊂ Macrorregião RS`) com intervalos preditivos $P_{50}/P_{90}/P_{95}$
- [ ] Construir simulador de filas hospitalares (`SimPy`: emergência, enfermaria, UTI, adiamento de eletivas, *lead time* 24–72h e cortes viários `OSMnx`)
- [ ] Gerar explicabilidade auditável (`SHAP` com verificação de aditividade) e cenários de estresse composto com tabela de gatilhos observáveis
- [ ] Executar avaliação final única no holdout `2025` com *block-bootstrap* 95% CI e Análise de Curva de Decisão (Net Benefit)

### ⏳ Etapa 4 — Escala Nacional Brasil (27 UFs e Rede Nacional de Hospitais-Escola)
- [ ] Escalar ingestão e cubos espaciais para 5.570 municípios e 27 UFs (`polars` + `dask`)
- [ ] Treinar `HydroTFTNet` multi-escala (janelas 15d–180d) + Hetero-GNN (`torch-geometric`) acoplando bacias e fluxos inter-hospitalares
- [ ] Executar validação espacial *Leave-UF-Out* e *Leave-Basin-Out* por bioma/porte/SUS e otimização multiobjetivo (`pymoo` NSGA-II/III)

### ⏳ Etapa 5 — Governança TRIPOD+AI / RECORD e Publicação
- [ ] Completar checklists **TRIPOD+AI**, **PROBAST+AI** e **RECORD** com diagrama de atrito amostral e tabelas por subgrupo
- [ ] Documentar plano de monitoramento de *drift*, recalibração, *rollback* e aposentadoria; publicar release `v1.0.0`
