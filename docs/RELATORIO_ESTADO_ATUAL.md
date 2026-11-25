# Relatório de Estado Atual, Diagnóstico Forense e Guia de Continuidade (`Previsao-de-leitos`)

> **Data de emissão:** 2026-11-25 | **Commit base avaliado:** `master@304dfd5` (`121` testes passando, `86%` de cobertura)
> **Fronteira Clínica e Operacional:** Projeto de pesquisa aberta e avaliação metodológica — **"Not for patient care or live clinical use"**.
> **Privacidade (LGPD):** Zero PII de pacientes; agregação na fronteira de extração (BigQuery/OpenDataSUS); supressão primária e complementar ($n < 5$ — [ADR-0005](adr/0005-data-layer-and-lgpd.md)).
> **Integridade Temporal:** Todas as auditorias quantitativas abaixo filtraram estritamente `ano_epi <= 2024`; nenhuma linha ou métrica do holdout `2025` foi lida ou utilizada.

---

## 1. Resumo Executivo do Estado Atual do Repositório

O repositório modular `src/previsao_leitos/` encontra-se íntegro, reproduzível e verificado mecanicamente (`ruff`, `mypy --strict`, `check_constraints.py`, `pytest --cov`, `pip-audit`). A primeira arquitetura completa da V1.1 RS (`F00` a `F07` + `F03.1`, `F04.1–F04.3`, `F06.1–F06.2`) foi implementada, mas a auditoria forense quantitativa sobre os dados reais de `RS 2024` revelou **10 defeitos estruturais** que explicam a rejeição da `F04.1` na catraca *Champion-Challenger*, a queda do `BSS_surge` para `-0,150` e as falhas nos sensores de higiene `G1`, `G5`, `G6` e `G7`.

| Dimensão | Estado Verificado (`master@304dfd5`) | Observação |
| :--- | :--- | :--- |
| **Código e testes** | `121 passed` (`19,6 s`), `86%` cobertura | Módulos `domain`, `ingestion`, `geo`, `hydrology`, `demand`, `readiness`, `audit`, `cli` |
| **Dados locais (`data/`)** | `20 MB` (`data/raw/` `18 MB`, `data/processed/` `1,7 MB`) | `rs_painel_semanal_regioes_v2.parquet` (`23.460` linhas `<= 2024`, `30` regiões) — fora do Git |
| **Relatórios (`reports/`)** | `19 MB` | Inclui `reports/runs/rs_v1_1_final_2024/` e diretórios prévios `reports/holdout_2025_*` (`2026-11-24 14:40`, não lidos) |
| **Gates de Higiene (`G1–G8`)** | `G4 PASS` (`7,2%`), `G8 PASS` (rótulos); **`G1, G5, G6, G7 FAIL`**; `G2, G3` não executados | Exige saneamento `V1.2.0` + reextração `v3` (`1.2.1`) + fontes ao vivo (`1.2.2–1.2.3`) |
| **Segurança operacional** | `.env` fora do Git; expurgo histórico de `b3ffad6:.env` pendente | Runbook em [docs/runbooks/git-history-secret-purge.md](runbooks/git-history-secret-purge.md) |

---

## 2. Diagnóstico Forense Quantitativo da Base `V1.1` (`RS <= 2024`)

### 2.1 Os 10 Defeitos Estruturais Confirmados em Código

1. **Bug A (`demand/two_stage.py` — `basal_cols`):** o modelo busca `sin_target_wk`/`cos_target_wk`, mas `demand/datasets.py` produz `target_week_sin`/`target_week_cos`. Em `fit()`, subtrai-se `mean(y_rate)` ($\approx 1,0$), mas em `predict_quantiles()` o fallback devolve zeros (`basal_hat = 0`), gerando viés de `-190` diárias (`MASE = 3,2217`, `mean P50 = 67,74` vs. real `258,53`).
2. **Bug B (`demand/two_stage.py` — vazamento AR + ausência de `CNES` mensal + escala homoscedástica):** apenas `y_origin` é filtrado do Estágio 2, deixando `y_roll4`, `y_roll12` e `y_dev_roll12` vazarem. Ao removê-los de fato (ortogonalidade real), `MASE` sobe para `1,1900` com capacidade estática e cai para `0,8840` ao integrar `cnes_leitos_especialidade_v2` mensal (expansão de `+32,4%` nos leitos UTI SUS no RS entre 2020 e 2024), mas `_resid_sd * cap` homoscedástico infla `Pinball95` (`19,76`) e `Cov` (`98,8%`).
3. **Bug C (`demand/two_stage.py` — quebra de monotonicidade):** `_DEFAULT_MONO_POS` procura `"chuva_montante_pond"`, mas o script gera `"chuva_montante_multihop"`, zerando `monotonic_cst`.
4. **Bug D (`audit/enso_what_if.py` — Oráculo plano `72,14 → 72,14`):** perturba colunas sem sufixo `_lag*`, enquanto o modelo consome `*_lag0..2`.
5. **Bug E (`hydrology/oof.py` — OOF circular sem `HydroFiLM-ENSO`):** `generate_causal_hydro_oof_predictions()` não chama `HydroFiLMMonotonicNet`; ajusta `LogisticRegression + Ridge` sem monotonicidade prevendo `chuva_montante_multihop / p95` a partir da própria variável (`corr = 1,000000`).
6. **Bug F (`scripts/run_rs_v1_1_dossier.py` — métricas hardcoded e vazamento *ex-post*):** `peak_timing_error_weeks = 2.0` do Campeão V1 é fixo no código; `shock_weeks = (18, 22)` exclui o pico real de UTI (`W26–28`, `POA W27 = 3.743`, excesso `+182` leitos/dia); `enso_regime` é juntado na semana-alvo futura ($t_0 + 2$) e `cal_year = 2023` contém apenas o regime `el_nino`.
7. **Bug G (`audit/bsts_counterfactual.py` — contrafactual estático viesado):** regressão linear global `2010–2024` superestima o período pré-enchente (`W15–17/2024`) em `+19..+32` diárias no RS (`+187..+329` em POA), gerando excesso observado médio negativo (`-6,1`); usa `1.96*sd` para `p05/p95`; `recursive_dynamic_do_intervention` não atualiza `y_roll4/12` e `y_dev_roll12`.
8. **Bug H (`BSS_surge = -0,150` após remoção do bug `0 >= 0`):** `ConformalFloodCalibrator` gera `p_surge = 0,7385` nas `104` linhas sem UTI (`y_roll12 == 0`), penalizando o BSS global em `-0,182`; nas 28 regiões ativas (`y_roll12 > 0`), a interpolação linear sobre 4 quantis conformais tem skill nulo (`Pos BSS = +0,0001`), e na janela de choque (`W18–28`) até um classificador monotônico direto ainda tem `Shock BSS < 0` sobre o painel `v2` (Rolling Platt `2019–2023` melhora `-0,0641 → -0,0177`).
9. **Bug I (`ingestion/queries.py` e `pipeline.py` — SQLs `v3` incompletos e desconectados):** `build_sih_active_census_query` e `build_sih_external_referral_matrix_query` não estão ligadas ao `extract_rs_core_health_pack()`, usam `DATE_DIFF + 1` (double-count de diárias), não têm folga de borda anual (`ano_inicio-1..ano_fim+1`) nem query de triângulo de atraso `(ano_epi, semana_epi, delay_weeks, cumulative_count)` para `ChainLadderNowcaster`. *(Atenção: conforme `CHANGELOG.md`, `sigla_uf` já se mostrou nulo na base SIH da Base dos Dados; validar obrigatoriamente via `inspect_columns()` + `dry_run=True` antes de alterar o filtro `STARTS_WITH(id_municipio_estabelecimento, @uf_code)`.)*
10. **Bug J (`audit/dossier_compiler.py` e `readiness/simulator.py`):** `_render_html_from_dossier` embute Markdown cru em `<pre>` (violando [padrao-relatorio-tecnico-pdf.md](guides/padrao-relatorio-tecnico-pdf.md)), omite `--print-to-pdf-no-header` e sobrescreve `~/Downloads/CAMPEAO_*` mesmo sob `REJECT`; `simulator.py` ativa perda de pôlder por proxy circular de demanda (`arrivals > 1.25 * median`), não pré-ativa leitos *surge* em $h=2$, e cria `~30.000` processos `_capacity_blocker` por execução (`256 ms/run`).

### 2.2 Status dos 8 Gates de Higiene (`G1–G8` em `RS <= 2024`)

- **`G1 (VIF < 5)` — FAIL:** `VIF = 7.181,48` (chuva) e `6.488,59` (API); `80,9%` de `sih_fluxos_referencia_semanal_v2` suprimido e `81,60%` do volume `<= 2024` intra-municipal.
- **`G2 (Perda de join)` e `G3 (Censo vs. CNES)` — NÃO EXECUTADOS:** dependem da reextração `v3` e integração de `cnes_leitos_especialidade_v2`.
- **`G4 (Censura à direita)` — PASS (`7,2%` em `<= 2024`):** exige triângulo de atraso para `2025–2026+`.
- **`G5 (MAUP <= 2,0x)` — FAIL:** distorção máxima `20,42x` (mediana `2,26x`).
- **`G6 (Colisão de grade)` — FAIL:** `5/30` regiões (`16,7%`) colidem na grade `NASA POWER 0,5°` (`07/08/23` e `25/29`).
- **`G7 (Pureza SUTVA)` — FAIL:** `5/5` doadores em `cli/main.py` receberam `147,4–280,4 mm/sem` em `W18–22/2024`.
- **`G8 (Anti `0 >= 0`)` — PASS** nos rótulos, mas contornado nas probabilidades (`p_surge = 0,74` quando `y_roll12 == 0`).

---

## 3. Arquitetura de Execução em 3 Camadas (CI + Workstation Arch/AMD + Google Colab)

Para alinhar rigor de CI, produtividade local na nova estação de trabalho e escalabilidade nacional na V2 Brasil, o projeto adota **3 camadas de execução** com os mesmos contratos de dados e manifestos:

| Camada | Hardware / Ambiente | Extra `uv` | Papel no Projeto |
| :--- | :--- | :--- | :--- |
| **Tier 0 — CI Contínuo** | GitHub Actions Linux (`CPU-only`) | `--extra cpu` | `ruff`, `mypy`, `check_constraints.py`, `pytest -m "not slow"`, `pip-audit`, determinismo estrito (`deterministic=True`) |
| **Tier 1 — Workstation Local** | **Arch Linux (Ryoku)**, **Intel Core i5-12400F** (`6C/12T`), **32 GB RAM**, **AMD RX 7500** (`RDNA 3 / gfx1102`, ROCm), **1 TB NVMe SSD** | `--extra rocm` (fallback `--extra cpu`) | 100% do **Ciclo V1.2 RS** (`1.2.0–1.2.6`), Holdout **`M2`**, cache local `MERGE`/`ERA5`/`DEM-HAND` no NVMe de `1 TB`, ETL nacional `Polars` (`< 16 GiB`), `F03.1` semanal/diária, `Hetero-ST-GNN` puro PyTorch, `PyMC + nutpie`, `SimPy + NSGA-II/III` |
| **Tier 2 — Expansão Google Colab** | **NVIDIA T4 (`16 GB`) / L4 (`24 GB`) / A100 (`40–80 GB` VRAM)** | `--extra cuda` | Treinos pesados da **V2 Brasil**: `HydroTFTNet` diário (`L=180d`, `39` folds), `Hetero-ST-GNN` em `5.570` municípios com extensões CUDA (`torch-scatter`/`pyg-lib`), `XGBoost` CUDA (`device="cuda"`), `TimesFM 2.5` + `XReg` em larga escala e ablações multi-seed extensas |

### Peculiaridades Técnicas Confirmadas da Workstation Arch + AMD RX 7500 (`gfx1102`)
1. **ROCm / HIP no Arch Linux (`gfx1102`):** PyTorch roda sobre AMD via HIP (`torch.cuda.is_available() == True`). Caso a wheel ROCm não reconheça `gfx1102` nativamente, configurar `export HSA_OVERRIDE_GFX_VERSION=11.0.0` (ou `11.0.2`) e adicionar o usuário aos grupos `video` e `render`. Verificar a VRAM física reportada no sistema real via `rocminfo` / `amd-smi` (placas Navi 33 `gfx1102` variam entre `6 GB` e `8 GB`; o orçamento de `F03.1` `0,1–1,2 GB` e `Hetero-ST-GNN` regional `0,5–1,5 GB` cabe em ambos).
2. **`PyTorch Geometric` em ROCm:** usar apenas operadores nativos de PyTorch (`HeteroConv`, `SAGEConv` com `scatter_add_`/`index_add_`) localmente, dispensando compilação manual de `torch-scatter`/`torch-sparse` (reservando extensões compiladas para o Tier 2 CUDA no Google Colab).
3. **`XGBoost`:** wheels oficiais PyPI são compiladas para NVIDIA CUDA. Na workstation local (`i5-12400F`, `6C/12T`, `32 GB RAM`), usar `tree_method="hist", device="cpu"` (extremamente rápido em tabular); usar `device="cuda"` apenas no Tier 2 (Google Colab NVIDIA).
4. **SSD NVMe `1 TB`:** permite cachear localmente `MERGE-CPTEC 0,1°`, `ERA5-Land`, `DEM/HAND 30 m` (inclusive tiling nacional `60–300 GB` por Ottobacia) e o Data Lake Hive Parquet (`data/processed/br_v1/`) sem pressão de disco.

---

## 4. Matriz Crítica de Decisão (`V1.2 RS → V2 Brasil`)

| # | Frente | Decisão | Tier Principal |
| :-: | :--- | :--- | :--- |
| **0** | **Saneamento V1.1** (10 bugs + PDF editorial + 8/8 Gates + `run_manifest.json`) | **Adotar Já (`1.2.0`)** | Tier 0 / Tier 1 |
| **1** | **Reextração BigQuery `v3`** (Censo, `W_ref`, `CNES` mensal, Triângulo Nowcasting) | **Adotar Já (`1.2.1`)** | Tier 1 + BQ (`< 2000 MB`) |
| **2** | **Ingestão Ao Vivo RS (Onda 2A + Onda 2B) + `F03.1` Semanal → Diária** | **Adotar em 2 Ondas (`1.2.2–1.2.3`)** | Tier 1 (ROCm/CPU) |
| **3** | **Calibração de Surtos (`BSS_pos > +0,10`, `ECE < 0,04`) + `AgACI` vs. `SPCI`** | **Adotar Já (`1.2.4`)** | Tier 1 |
| **8** | **Fronteira de Pareto (`pymoo` NSGA-II/III + `SimPy` CRN 2 Estágios + Fix Blocker)** | **Adotar Já (`1.2.5`, `2.4`)** | Tier 1 (`i5-12400F` `6C/12T`) |
| **9** | **Explicabilidade Auditada (`SHAP 0.54.1` 2 Estágios + Conformal)** | **Adotar Já (`1.2.6`)** | Tier 1 |
| **6** | **Inferência Bayesiana Hierárquica (`PyMC 6.3.2` + `nutpie` + `ArviZ 1.3.0`)** | **Experimento Controlado RS (`1.2.6`) + Por Bioma V2** | Tier 1 (`nutpie` CPU) |
| **7** | **LOS com Riscos Competitivos** (`Hazard Agregado BQ + statsmodels` MIT) | **Adaptar na V2 (`2.2`) — Descartar e remover `scikit-survival` (GPL-3.0 / `ADR-0005`)** | Tier 1 + BQ |
| **4** | **`Hetero-ST-GNN` (`PyG 2.8.1`) + `HydroTFTNet` (`ADR-0004`)** | **Explorar na V2 (`2.3`, `~450` Regiões) — Fora do RS (`30` nós)** | Tier 1 (regional) / Tier 2 Colab (diário `180d` & `5.570` mun.) |
| **5** | **`TimesFM 2.5` (200M, Apache-2.0) + `XReg` Ex-Ante** | **Benchmark Externo Não-Promovível em Venv Isolado (`3.0` rejeitado)** | Tier 1 / Tier 2 Colab |
| **10** | **Escala Nacional (`Polars 2.1.0` 2 Níveis) + *Leave-UF/Basin-Out*** | **Adotar `Polars 2.1.0` (`2.1`) — Descartar e remover `dask`** | Tier 1 (`< 16 GiB` global / `< 6 GiB` streaming) |

---

## 5. Roteiro de Execução (`V1.2 RS → M2 Holdout → V2 Brasil`)

Detalhamento completo em [docs/features/08-plano-v1-2-v2-workstation-colab.md](features/08-plano-v1-2-v2-workstation-colab.md) e rastreamento de progresso em [docs/PROGRESSO.md](PROGRESSO.md):

- **Ciclo `V1.2 RS` (`Marco M1` — PDF Editorial #1):**
  - [ ] `1.2.0` Saneamento dos 10 bugs V1.1 + `CanonicalPanelSchema` + PDF editorial real + 8/8 Gates.
  - [ ] `1.2.1` Reextração BigQuery `v3` (`inspect_columns()` + `dry_run=True`) + `rs_painel_semanal_regioes_v3.parquet`.
  - [ ] `1.2.2` Ingestão Ao Vivo Onda 2A (`NOAA`, `Zenodo 16929471`, `GERINT-POA`, `SIVEP/SINAN`) + extras `cpu`/`rocm`/`cuda` + `F03.1` semanal real (`13×3` OOF).
  - [ ] `1.2.3` Ingestão Ao Vivo Onda 2B (`MERGE-CPTEC 0,1°` + `ANA API v2`) + transição controlada `F03.1` semanal → diária (`seq_len=60–90` dias).
  - [ ] `1.2.4` Calibração probabilística de surtos (`BSS_pos >= +0,10`, `BSS_shock > 0`, `ECE <= 0,04`) + `AgACI` vs. `SPCI`.
  - [ ] `1.2.5` Otimização `_capacity_blocker` + Fronteira de Pareto (`pymoo` NSGA-II + `SimPy` CRN 2 estágios).
  - [ ] `1.2.6` Explicabilidade `SHAP 0.54.1` (2 estágios) + experimento controlado `PyMC 6.3.2` + `nutpie` + `ArviZ 1.3.0`.
- **Marco `M2` (`1.3.0` — PDF Editorial #2):** Avaliação única congelada (`v1.2.0-rs-frozen`) no holdout (`2025` documentado sob PROBAST+AI + `2026` prospectivo virgem com `ChainLadderNowcaster`) + checklists TRIPOD+AI / PROBAST+AI / RECORD.
- **Ciclo `V2 Brasil` (`Marco M3` — PDF Editorial #3):**
  - [ ] `2.1` Data Lake Nacional `Polars 2.1.0` (global `< 16 GiB` + streaming `< 6 GiB`, remoção de `dask`).
  - [ ] `2.2` LOS Dinâmico com Riscos Competitivos Agregado no BQ (`statsmodels` MIT, remoção de `scikit-survival`).
  - [ ] `2.3` Benchmark Nacional *Leave-UF-Out* (`27` folds) e *Leave-Basin-Out* (`12` folds / `6` biomas): `F04.1` vs. `Hetero-ST-GNN` vs. `HydroTFTNet` vs. `TimesFM 2.5` (+ expansão Tier 2 Google Colab).
  - [ ] `2.4` Alocação Nacional `NSGA-III` (4 objetivos incl. equidade SUS) + monitoramento de *drift* + Release `v1.0.0`.

---

## 6. Guia Passo a Passo de Migração para a Nova Workstation (Arch Linux Ryoku)

### Passo 1 — Sincronizar o Repositório Git
Este diretório já é um repositório Git com todo o histórico em `master` (atualmente à frente de `origin/master`). Escolha uma das duas opções:

- **Opção A (via GitHub `origin`):**
  ```bash
  # Na máquina atual:
  git push origin master

  # Na nova máquina Arch Linux:
  git clone https://github.com/GuilhermeSPedroso07/Previsao-de-leitos.git
  cd Previsao-de-leitos
  ```
- **Opção B (offline via pendrive/SCP sem enviar nada para a nuvem):**
  ```bash
  # Na máquina atual:
  git bundle create ../previsao-de-leitos.bundle --all

  # Na nova máquina Arch Linux:
  git clone previsao-de-leitos.bundle Previsao-de-leitos
  cd Previsao-de-leitos
  ```

### Passo 2 — Transferir Arquivos Locais Fora do Git (Via Pendrive/SCP Seguro — Nunca Commitar!)
Copie separadamente estes diretórios/arquivos locais (total aproximado: `~40 MB`):
1. `.env` (`329 B` — variáveis locais GCP/BigQuery; jamais enviar ao Git).
2. `data/` (`20 MB` — Parquets brutos `data/raw/` e painel `data/processed/` já extraídos, evitando reextração desnecessária).
3. *(Opcional)* `reports/` (`19 MB` — dossiês e figuras anteriores para comparação histórica).

### Passo 3 — Configurar o Ambiente no Arch Linux (`i5-12400F` + `AMD RX 7500`)
```bash
# 1. Pacotes de sistema essenciais (Arch Linux)
sudo pacman -S --needed git python base-devel chromium uv

# 2. (Opcional para aceleração GPU AMD RDNA 3 / ROCm)
# Instalar runtime ROCm e adicionar seu usuário aos grupos video e render:
sudo pacman -S --needed rocm-hip-runtime rocm-smi-lib
sudo usermod -aG video,render "$USER"
# Faça logout/login após alterar grupos.

# 3. Verificar GPU e VRAM reais no Arch Linux:
rocminfo | grep -E "Name:|Marketing Name:|gfx"
amd-smi static --vram 2>/dev/null || rocm-smi --showmeminfo vram

# 4. Instalar dependências Python 3.12 do projeto (base CPU idêntica ao CI):
uv sync --python 3.12 --extra dev --extra data --extra stats --extra hydro --extra sim

# 5. Validar toda a suíte mecânica e os 121 testes:
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/check_constraints.py src tests
uv run pytest -m "not slow" --cov=src/previsao_leitos --cov-fail-under=80
uv run pip-audit
```

### Passo 4 — Autenticar BigQuery (ADC sem arquivo `.json`)
Siga [docs/runbooks/setup-bigquery.md](runbooks/setup-bigquery.md):
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```
