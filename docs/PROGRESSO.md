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
| **Etapa 3** | Incerteza Bayesiana, Simulação de Filas (`SimPy`) e Auditoria RS (2024/2025) | 🔄 Em andamento (`V1.2 RS`) | `80%` | V1.1 RS auditada + **`Task 1.2.0` concluída** (Bugs A–J saneados, `CanonicalPanelSchema`, 8/8 Gates, PDF editorial); seguem `1.2.1–1.2.6` + Holdout `M2` (`2025` + `2026`) |
| **Etapa 4** | Escala Nacional Brasil (27 UFs, ST-GNN, `HydroTFTNet` e Otimização Pareto) | ⏳ Planejada | `0%` | **V2 Brasil validada** via *Leave-UF-Out* e *Leave-Basin-Out* em todos os biomas (Workstation Arch/ROCm + expansão opcional Google Colab NVIDIA) |
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
| **F08 — Evolução `V1.2 RS → V2 Brasil`, Transição Diária `F03.1` e Arquitetura Híbrida** | [08-plano-v1-2-v2-workstation-colab.md](features/08-plano-v1-2-v2-workstation-colab.md) | 🔄 Em andamento | Relatório de Estado Atual ([RELATORIO_ESTADO_ATUAL.md](RELATORIO_ESTADO_ATUAL.md)), roteiro `1.2.0–2.4`, empacotamento `cpu`/`rocm`/`cuda` e expansão opcional Google Colab NVIDIA |
| **F09 — Saneamento Estrutural `V1.2.0` (`Task 1.2.0`)** | [09-saneamento-v1-2-0.md](features/09-saneamento-v1-2-0.md) | ✅ Concluída | Correção dos Bugs A/B/C/D/F/G/H/J, `CanonicalPanelSchema`, `UnobservedComponents` BSTS (`viés W15–17 = +8,10`), Oráculo El Niño crescente, 8/8 Gates (`PASS=3, FAIL=5`) e PDF editorial HTML5/CSS3 Chromium (`136` testes, `87,45%` cov.) |

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
- [x] **Padrão Editorial de Relatório Técnico PDF:** publicar diretriz canônica em [docs/guides/padrao-relatorio-tecnico-pdf.md](guides/padrao-relatorio-tecnico-pdf.md) (HTML5/CSS3 → Chromium headless, zero Markdown cru, 6 capítulos obrigatórios e simulações em leitos ocupados/dia)
- [x] **Auditoria Forense V1.1 e Relatório de Estado Atual (`docs/RELATORIO_ESTADO_ATUAL.md`):** Mapeados quantitativamente os `10` defeitos estruturais da V1.1 (`RS <= 2024`, `2025` intocado), desenhada a arquitetura em `3` camadas (`CI CPU` + `Workstation Arch/i5-12400F/32 GB/RX 7500` + `Google Colab NVIDIA`) e publicado o plano `F08` ([08-plano-v1-2-v2-workstation-colab.md](features/08-plano-v1-2-v2-workstation-colab.md))
- [x] **Task 1.2.0 — Saneamento Estrutural V1.1 + `CanonicalPanelSchema` + PDF Editorial + 8/8 Gates ([09-saneamento-v1-2-0.md](features/09-saneamento-v1-2-0.md)):** Corrigidos Bugs A/B/C (`two_stage.py`: `MASE 3,22 → 0,935`, `Pinball95 19,76 → 10,14`, `Cov 80,6%`, `0,00%` viol.), Bug D (`enso_what_if.py`: `230,89 → 234,57`), Bug F (`run_rs_v1_1_dossier.py`: `CNES` mensal, `W18–30`, timing V1 real `2,10` sem., `enso_regime` em $t_0$), Bug G (`UnobservedComponents` BSTS com espaçamento exato de horizonte e proteção pré-intervenção no `do()` recursivo: `viés W15–17 = +8,10`, `z=1,6449`), Bug H (máscara anti-G8), Bug J (PDF HTML5/CSS3 Chromium `--no-pdf-header-footer`, tabelas largas sem corte, trava `CAMPEAO_*` sob `REJECT`, `SimPy` `em_cheia_hidro` exógeno, pré-ativação `lead_time` e limpeza de pedido pendente) e scorecard **8/8 Gates (`G1–G8`: `PASS=3, FAIL=5`)** (`136` testes, `87,45%` cov.)
- [ ] **Ciclo `V1.2 RS` (`1.2.1–1.2.6` → `Marco M1`):** Reextração BigQuery `v3` + fontes ao vivo Onda 2A/2B + transição `F03.1` semanal → diária (`60–90d`) + calibração de surtos (`BSS_pos >= +0,10`, `AgACI`/`SPCI`) + Pareto `NSGA-II` (`SimPy` CRN 2 estágios) + `SHAP 0.54.1` + experimento `PyMC 6.3.2` + `nutpie` + PDF Editorial #1
- [ ] **Marco `M2` (`1.3.0`):** Executar avaliação final única congelada (`v1.2.0-rs-frozen`) no holdout (`2025` classificado sob PROBAST+AI + `2026` prospectivo virgem com `ChainLadderNowcaster`) e emitir PDF Editorial #2

### ⏳ Etapa 4 — Escala Nacional Brasil (27 UFs e Rede Nacional de Hospitais-Escola)
- [ ] **Task 2.1:** Escalar ingestão e cubos espaciais para `5.570` municípios e `27` UFs exclusivamente com **`Polars 2.1.0`** em 2 níveis (passe global interestadual `< 16 GiB` + modo streaming portátil `< 6 GiB`; remover `dask` de `pyproject.toml`)
- [ ] **Task 2.2:** Implementar LOS dinâmico com riscos competitivos via **Hazard Discreto Agregado no BigQuery (`statsmodels` MIT)** preservando LGPD (`ADR-0005`) e removendo `scikit-survival` (GPL-3.0) de `pyproject.toml`
- [ ] **Task 2.3:** Executar validação espacial *Leave-UF-Out* (`27` folds) e *Leave-Basin-Out* (`12` folds / `6` biomas) comparando `F04.1`, `Hetero-ST-GNN` (`PyG 2.8.1`), `HydroTFTNet` (`ADR-0004`) e `TimesFM 2.5` (benchmark externo isolado), com suporte opcional ao Tier 2 **Google Colab NVIDIA**
- [ ] **Task 2.4:** Otimização multiobjetivo nacional (`pymoo` `NSGA-III` 4 objetivos + `SimPy` CRN 2 estágios) + `SHAP 0.54.1` nacional + PDF Editorial #3 (`Marco M3`)

### ⏳ Etapa 5 — Governança TRIPOD+AI / RECORD e Publicação
- [ ] Completar checklists **TRIPOD+AI**, **PROBAST+AI** e **RECORD** com diagrama de atrito amostral e tabelas por subgrupo
- [ ] Documentar plano de monitoramento de *drift*, recalibração, *rollback* e aposentadoria; publicar release `v1.0.0`
