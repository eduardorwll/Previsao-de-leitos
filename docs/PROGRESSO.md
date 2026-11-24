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
| **Etapa 2** | Pivô Científico RS: Demanda Real, LOS e Hidrologia Calibrada | ✅ Concluída (V1 RS) | `100%` | `F03 HydroFiLMMonotonicNet` (`autograd 0,00%`, `BSS=+0,076`) + `F04` (`NegBinHurdle 0,710`, `QuantileGBDT 0,820`) + LOS causal |
| **Etapa 3** | Incerteza Bayesiana, Simulação de Filas (`SimPy`) e Auditoria RS (2024/2025) | ✅ V1.1 RS Concluída | `100%` (V1.1) | **Plano Diretor V1.1 RS concluído** (Fases 1–4: Higiene `G1–G8`, Catraca, Dossiê `.md`/`.pdf`, `F01.1–F06.2`, `HydroFiLM-ENSO`, `BSTS`, `Nowcasting`; `2025` pronto para auditoria final) |
| **Etapa 4** | Escala Nacional Brasil (27 UFs, ST-GNN e Otimização Pareto) | ⏳ Planejada | `0%` | **V2 Brasil validada** via *Leave-UF-Out* e *Leave-Basin-Out* em todos os biomas |
| **Etapa 5** | Governança TRIPOD+AI / RECORD, Publicação e Release `v1.0.0` | ⏳ Planejada | `0%` | Checklists completos, reprodutibilidade fim a fim e relatório técnico aberto |

---

## Progresso Feature por Feature

| Módulo / Feature | Documentação Detalhada | Status | Resumo do Valor Entregue |
| :--- | :--- | :---: | :--- |
| **F00 — Domínio, Tipos Validados e Qualidade Base (`domain`)** | [00-dominio-e-fundacao.md](features/00-dominio-e-fundacao.md) | ✅ Concluída | Tipos nominais (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`), hierarquia única de exceções, pacote Python 3.12 (`uv`), verificador mecânico de qualidade e CI |
| **F01 — Ingestão Segura e Contratos LGPD (`ingestion`)** | [01-ingestao-segura.md](features/01-ingestao-segura.md) | 🔄 Evolução V1.1 (RS) | BigQuery parametrizado + gRPC, CNES `v2` temporal (`join_asof` causal), **censo ativo SIH `[DT_INTER..DT_SAIDA]`**, **`W_ref` externa pré-LGPD** e agregação regional LGPD ($n < 5$) |
| **F02 — Geoespacial, Hidrologia 100% RS, Aterros e Intermunicipal (`geo`)** | [02-geo-hidrologia-rs.md](features/02-geo-hidrologia-rs.md) | ✅ Fase 2 V1.1 Concluída | Grade sem lacunas, DAG multi-salto transitivo, `MapBiomas` anti-MAUP, `ANA HidroWeb`/`MERGE 0,1°`/`SAR`, grafo `W_ref` e **teleconexões `ENSO / El Niño`** (`enso_teleconnections.py`) |
| **F03 — Hidrologia Calibrada (`hydrology`)** | [03-hidrologia-calibrada-film.md](features/03-hidrologia-calibrada-film.md) | ✅ Concluída (`F03.1 HydroFiLM-ENSO`) | `IndexedHydroSequenceDataset`, `HydroFiLMMonotonicNet` com **modulação FiLM-ENSO** (`autograd 0,00%`), `TemperatureScaler` e **gerador *out-of-fold* causal** (`hydrology/oof.py`) |
| **F04 — Demanda de Internações e Tempo de Permanência (`demand`)** | [04-demanda-e-los.md](features/04-demanda-e-los.md) | ✅ Fase 3 V1.1 Concluída | `QuantileGBDT`, `NegBinHurdle`, **arquitetura ortogonal 2 estágios (`two_stage.py`)**, **`EVT-GPD` + `Mondrian Conformal` + pooling hierárquico (`extreme_uncertainty.py`)** e ***Nowcasting* `Chain-Ladder` + LOS riscos competitivos (`nowcasting.py`)** |
| **F05 — Prontidão, Filas Hospitalares e Gatilhos (`readiness`)** | [05-prontidao-e-filas-simpy.md](features/05-prontidao-e-filas-simpy.md) | ✅ Concluída (V1 RS) | `ConformalFloodCalibrator` (`2023`, `0,00%` viol.), simulador de eventos discretos `SimPy` (estresse pôlder `15%`, redução de **`-36,8%`** no pico de fila e **`16`** dias de sobrecarga evitados em POA), gatilhos observáveis e `Fig. C5` |
| **F06 — Auditoria Estatística, Análise Contrafactual e Relatório (`audit`)** | [06-auditoria-contrafactual-e-relatorio.md](features/06-auditoria-contrafactual-e-relatorio.md) | ✅ Fase 1 V1.1 Concluída | Controle sintético, placebo, `do()`, `Fig. C1–C7`, sensores `G1–G8` (`data_hygiene.py`), catraca *Champion-Challenger* (`hypothesis_ledger.py`) e **dossiê automático `.md` + `.pdf`** (`dossier_compiler.py`) |
| **F07 — CLI e Orquestração Reprodutível (`cli`)** | [07-cli-e-governanca.md](features/07-cli-e-governanca.md) | ✅ Concluída (V1 RS) | Comando instalável `previsao-leitos` (`doctor`, `verify-rs`), logging JSON estruturado, Net Benefit (`BSS=+0,088`, `ECE=0,049`), seção TRIPOD+AI/PROBAST+AI/RECORD e bloqueio de `2025` |

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
  - [x] Aposentar `legacy/previsao_leitos_colab.py` em commit dedicado após paridade modular (`F00–F07`) e cobertura $\ge 80\%$ (`86,8%`)

### ✅ Etapa 2 — Pivô Científico RS: Demanda Semanal Multi-Horizonte, LOS e Hidrologia Calibrada
- [x] Congelar splits temporais (`<=2022` treino, `2023` calibração, `2024` auditoria RS, **`2025` holdout intocado** com bloqueio físico)
- [x] Implementar Feature **F04 (`demand`)**: `QuantileGBDTForecaster` monotônico exato (`0,00%` viol.), `NegativeBinomialHurdleForecaster` (`statsmodels` GLM, `MASE 0,710` em Leptospirose Onda 2), `PooledRidgeDeltaForecaster` (`MASE 0,817` em Diárias UTI Adulto) e `SpecialtyLosEstimator` causal (`<= 2022`)
- [x] Acoplar análise contrafactual estruturada e geração de relatório (`F06`: `NoFloodSyntheticControl`, placebo, decomposição `do()` e `Fig. C1–C4`) sobre RS `2024`
- [x] Implementar Feature **F03 (`hydrology`)**: `HydroFiLMMonotonicNet` em PyTorch Lightning (`20,6s` CPU, `ROC-AUC=0,720`, `ECE=0,052`, `BSS=+0,076`, monotonicidade exata via `torch.autograd.grad` em ambas as cabeças) + `TemperatureScaler`
- [ ] *(Opcional V1.1 / V2)* Evoluir LOS para riscos competitivos (`scikit-survival`) e benchmarks `aeon` MiniRocket / `TimesFM 2.5`

### 🔄 Etapa 3 — Incerteza Calibrada, Prontidão (`SimPy`), Análise Contrafactual e Auditoria RS (2024/2025)
- [x] Implementar calibração conformal heterocedástica em `2023` (`ConformalFloodCalibrator`: eleva cobertura no choque UTI de `34,1% → 61,4%` mantendo `0,00%` violações monotônicas)
- [x] Construir simulador de filas hospitalares (`SimPy` em `src/previsao_leitos/readiness/`: emergência/UTI, adiamento de eletivas, *lead time* `24–72h`, evacuação em áreas aterradas/pôlderes e tabela de gatilhos observáveis — redução de `-36,8%` no pico de fila em POA)
- [x] Implementar análise contrafactual estruturada completa (`src/previsao_leitos/audit/`: controle sintético "sem enchente", testes placebo, acurácia de excesso/timing, decomposição `do()` e **todas as 5 figuras `Fig. C1–C5`**)
- [x] Implementar sensores automáticos de higiene de dados cruzados (`src/previsao_leitos/audit/data_hygiene.py`, **Gates `G1–G8`**: VIF, perda de join, censo vs. CNES, censura à direita, MAUP, colisão de grade, pureza `SUTVA` e trava anti-`0 >= 0`)
- [x] Implementar registro versionado de hipóteses e catraca científica *Champion-Challenger* (`src/previsao_leitos/audit/hypothesis_ledger.py`, persistindo `hypothesis_ledger.json` e `ablation_delta.json`)
- [x] Implementar compilador automático `.md` + `.pdf` (`src/previsao_leitos/audit/dossier_compiler.py`, `Fig. C6–C7`, espelhamento em `~/Downloads/`), corrigir `0.0 >= 0.0`, `ddof=1` e `eval_year` dinâmico na `Fig. C1`
- [x] **Fase 2 / Task 2.1 (`F01.1`):** implementar censo diário/semanal ativo SIH-SUS (`GENERATE_DATE_ARRAY` + todas as diárias UTI + `ano_cmpt_max`), `W_ref` externa agregada no SQL antes da LGPD (`origem != destino`, `<= 2022`) e `join_asof` temporal causal `CNES` sem vazamento `.any()`
- [x] **Fase 2 / Task 2.2 (`F01.2`):** implementar conectores e agregadores LGPD de `GERINT-POA` (fila reprimida), `SINAN` Leptospirose (1ºs sintomas `DT_SIN_PRI`) e `SIVEP-Gripe` (SRAG viral) em `ingestion/epidemiology_sources.py`
- [x] **Fase 2 / Task 2.3 (`F02.2`):** corrigir `BasinTopology.propagate_weighted_upstream` para convolução transitiva multi-salto completa, ponderar `MapBiomas` por população/leitos anti-MAUP e adicionar agregadores `ANA HidroWeb` + `MERGE-CPTEC 0,1°` + `SAR` 2024 (`geo/fine_hydro_sources.py`)
- [x] **Fase 2 / Task 2.4 (`F02.3`):** implementar conector causal de teleconexões `ENSO / El Niño` (`geo/enso_teleconnections.py`: `ONI`, `Niño 3.4`, `SOI`, `MEI.v2`, `TSA`, `SALLJ` com atraso de publicação e lags `1..24` semanas)
- [x] **Fase 3 / Task 3.1 (`F03.1`):** evoluir `HydroFiLMMonotonicNet` para **`HydroFiLM-ENSO`** (`enso_dim`, `autograd 0,00%`) e implementar `generate_causal_hydro_oof_predictions` (`hydrology/oof.py`)
- [x] **Fase 3 / Task 3.2 (`F04.1`):** implementar arquitetura ortogonal em 2 estágios em taxa de ocupação relativa (`demand/two_stage.py`) acoplada às saídas OOF da `F03.1` + roteador automático `NegBinHurdle` para contagens esparsas
- [x] **Fase 3 / Task 3.3 (`F04.2`):** implementar `ExtremeValueGpdTail` (`EVT-GPD`), `MondrianEnsoConformalCalibrator` (porte × regime `ENSO`) e `HierarchicalBasinBayesShrinkage` (`Hospital ⊂ Bacia`) em `demand/extreme_uncertainty.py`
- [x] **Fase 3 / Task 3.4 (`F04.3`):** implementar `ChainLadderNowcaster` (correção de censura à direita `2025–2026+`) e `CompetingRisksLosEstimator` (CIF alta viva vs. óbito) em `demand/nowcasting.py`
- [x] **Fase 4 / Task 4.1 (`F06.2`):** implementar `PreShockBstsCounterfactual` (controle estrutural Bayesiano pré-choque com `SIVEP-Gripe`) e `recursive_dynamic_do_intervention` (`do()` recursivo sem contaminação pós-tratamento) em `audit/bsts_counterfactual.py`
- [x] **Fase 4 / Task 4.2 (`F06.2`):** implementar bateria de rebate contrafactual duplo 2024 e oráculo prospectivo de cenários El Niño (`audit/enso_what_if.py`: `run_dual_counterfactual_rebuttal_2024` + `run_enso_what_if_oracle`)
- [x] **Fase 4 / Task 4.3:** implementar `basin_cluster_bootstrap_ci` (14 clusters de bacia) e `scripts/run_rs_v1_1_dossier.py`, emitindo o Dossiê Final V1.1 RS (`.md` + `.pdf` em `reports/runs/rs_v1_1_final_2024/` e espelhado em `~/Downloads/`) com `2025` preservado intocado
- [ ] Ajustar modelo hierárquico Bayesiano (`PyMC`: `Hospital-Escola ⊂ Macrorregião RS`) + explicabilidade `SHAP` auditada
- [ ] Executar avaliação final única no holdout `2025` e emitir o **Relatório Final de Verificação Científica V1 RS** (`verificacao_cientifica.md` + `run_manifest.json` + `Fig. C1–C5` + Net Benefit)

### ⏳ Etapa 4 — Escala Nacional Brasil (27 UFs e Rede Nacional de Hospitais-Escola)
- [ ] Escalar ingestão e cubos espaciais para 5.570 municípios e 27 UFs (`polars` + `dask`)
- [ ] Treinar `HydroTFTNet` multi-escala (janelas 15d–180d) + Hetero-GNN (`torch-geometric`) acoplando bacias e fluxos inter-hospitalares
- [ ] Executar validação espacial *Leave-UF-Out* e *Leave-Basin-Out* por bioma/porte/SUS e otimização multiobjetivo (`pymoo` NSGA-II/III)

### ⏳ Etapa 5 — Governança TRIPOD+AI / RECORD e Publicação
- [ ] Completar checklists **TRIPOD+AI**, **PROBAST+AI** e **RECORD** com diagrama de atrito amostral e tabelas por subgrupo
- [ ] Documentar plano de monitoramento de *drift*, recalibração, *rollback* e aposentadoria; publicar release `v1.0.0`
