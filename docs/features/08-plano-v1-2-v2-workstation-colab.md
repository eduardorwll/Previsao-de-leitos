# Feature F08 — Plano de Evolução `V1.2 RS → V2 Brasil`, Transição Diária `F03.1` e Arquitetura Híbrida (`Workstation Arch/ROCm` + `Google Colab`)

> **Status:** ⏳ Planejada (aguardando início da `Task 1.2.0`) | **Etapas cobertas:** Fechamento Etapa 3 (`V1.2 RS` + `M2`) → Etapas 4 e 5 (`V2 Brasil` + `v1.0.0`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

Consolidar o saneamento estrutural da V1.1 RS, planejar com rigor a transição da rede hidrológica `F03.1 HydroFiLM-ENSO` da grade semanal para a grade diária (`seq_len = 60–90` dias) e estruturar o empacotamento portável em **3 camadas de execução** (CI `CPU-only`, Workstation local `Arch Linux / i5-12400F / 32 GB RAM / AMD RX 7500 / 1 TB NVMe`, e expansão opcional em **Google Colab NVIDIA T4 / L4 / A100** para a escala nacional `V2 Brasil`).

---

## 2. Transição Detalhada: `F03.1 HydroFiLM-ENSO` Semanal (`1.2.2`) → Diária (`1.2.3`)

A migração para janelas diárias de `60–90` dias captura memória de solo saturado (*Antecedent Precipitation Index* de longo alcance), ondas de cheia sucessivas (como set/nov/2023 → mai/2024 no RS) e tempo real de trânsito fluvial (Taquari/Jacuí → Guaíba → Lagoa dos Patos) sem introduzir vazamento temporal.

### Fase A — Saneamento Semanal GPU/ROCm (`Task 1.2.2`, Onda 2A)
- [x] **Empacotamento portável (`pyproject.toml`):** remover `torch>=2.5.0` duplicado de `hydro`/`ml` e criar extras mutuamente exclusivos `cpu`, `rocm` e `cuda` (`[tool.uv] conflicts`), mantendo `.github/workflows/ci.yml` em `--extra cpu`.
- [x] **Correção dos 4 bloqueios de GPU/Dataset:**
  1. Criar `resolve_accelerator(prefer_gpu: bool = True)` estritamente *lazy* (zero importação em `src/previsao_leitos/__init__.py`).
  2. Mover tensores para `module.device` em `_predict_logits` (`scripts/run_rs_f03_hydrology.py`) e em `audit_autograd_monotonicity` (`src/previsao_leitos/hydrology/oof.py`).
  3. Forçar `fp32` + `torch.amp.autocast(..., enabled=False)` na auditoria `autograd` sobre **100% do split** (eliminando o recorte `batch0`).
  4. Adicionar suporte a `enso_cols` (`x_enso`) e pré-carga opcional em VRAM no `IndexedHydroSequenceDataset` (`src/previsao_leitos/hydrology/dataset.py`).
- [x] **OOF Causal Real Semanal (`13` folds × `3` seeds):** substituir a regressão linear circular de `hydrology/oof.py` (`corr = 1,000000`) pelo treino real de `HydroFiLMMonotonicNet` prevendo alvo hidrológico externo (`Zenodo 10.5281/zenodo.16929471` / cota), com determinismo `deterministic="warn"` em GPU e `deterministic=True` no CI CPU.

### Fase B — Transição Controlada para Grade Diária `60–90` Dias (`Task 1.2.3`, Onda 2B)
- [x] **Pré-requisito de dados diários reais:** ingerir `MERGE-CPTEC 0,1°` diário (`MergeCptecClient`, amostra ao vivo `450 KB`) + grade `0,1°` ERA5-Land + `V10M` (zerando `Gate G6`: `0/30`) e cliente OAuth v2 `ANA HidroWeb API v2` com fallback automático Zenodo `16929471`.
- [x] **Evolução do `IndexedHydroSequenceDataset` para suporte dual (`freq="weekly" | "daily"`):**
  - Remover a asserção fixa `days == 7` (`dataset.py`) e parametrizar `step_days = 1` (diário) ou `7` (semanal), validando estritamente continuidade temporal sem cruzar lacunas.
  - Projetar variáveis lentas (`MapBiomas`, `ENSO` com `publication_lag_days`) sem vazamento futuro.
- [x] **Alinhamento Causal Diário → Semanal (`F03.1 → F04.1`):**
  - Para cada semana epidemiológica de origem $t_0$ (encerrada no sábado ISO), o modelo diário lê exclusivamente os dias $[t_0 - L + 1, \dots, t_0]$ ($L = 60$ dias) e retorna saídas OOF alinhadas à chave canônica `(regiao_id, ano_epi, semana_epi)`.
- [x] **Orçamento e Guardrails na RX 7600 (`gfx1102`) / Colab:**
  - Medido em [PERF.md](../../PERF.md): `163.800` linhas diárias × `L = 60` dias (`13` folds × `3` seeds), `batch_size = 256`, pico **`251,3 MB` VRAM** e **`342,9 s`** (`0,0000%` viol. `autograd fp32`).
- [x] **Critério de Catraca (*Champion-Challenger*):** avaliado em `RS 2024` — semanal `0,1°` (`ROC-AUC = 0,7888`, `BSS = +0,1561`, `ECE = 0,0475`, `99,2 s`) superou diário `60d` (`ROC-AUC = 0,7299`, `BSS = +0,0899`, `ECE = 0,0589`) → **`KEEP_WEEKLY_CHAMPION`** (retestável com cotas diárias `ANA API v2`).

---

## 3. Arquitetura Aberta para Expansão no Google Colab (Tier 2 — `V2 Brasil`)

O projeto permanece 100% executável localmente na workstation Arch Linux, mas mantém **encaixe imediato para GPUs NVIDIA no Google Colab** (`T4 16 GB`, `L4 24 GB`, `A100 40–80 GB`) quando a escala nacional exigir maior largura de banda/VRAM:

1. **Interface única via CLI / Scripts:** todo treino pesado aceita `--device auto|cpu|rocm|cuda`, `--precision 32|bf16-mixed|16-mixed`, `--batch-size` e `--output-dir`, lendo Parquets Hive-particionados (`data/processed/br_v1/`) verificados por hash SHA-256.
2. **Empacotamento limpo:** notebooks enxutos em `notebooks/colab/` (a criar na V2) apenas montam o Drive/storage, instalam `uv sync --extra cuda --extra hydro --extra scale` (sem duplicar lógica de modelagem fora de `src/previsao_leitos/`) e executam os comandos oficiais da CLI.
3. **Cargas prioritárias para delegação opcional ao Google Colab (Tier 2):**
   - `HydroTFTNet` diário (`L = 180` dias, `~2,63M` dias-região nos `39` folds nacionais);
   - `Hetero-ST-GNN` em resolução municipal completa (`5.570` municípios, `50k–200k` arestas `W_ref`) com extensões CUDA nativas (`torch-scatter`/`pyg-lib`);
   - `XGBoost` CUDA (`tree_method="hist", device="cuda"`, `monotone_constraints`) como challenger tabular acelerado;
   - Benchmark externo `TimesFM 2.5` (200M, Apache-2.0) + `XReg` em múltiplos horizontes nacionais.
4. **Rastreabilidade de retorno:** toda execução no Colab exporta `run_manifest.json` (`git_sha`, `gpu_name`, `cuda_version`, `seed`, `data_sha256`), `ablation_delta.json` e métricas de auditoria `autograd` para validação na catraca local antes de qualquer promoção.

---

## 4. Checklist de Progresso da Frente

- [x] **1.2.0** Saneamento dos 10 bugs V1.1 + `CanonicalPanelSchema` + PDF editorial + 8/8 Gates ([09-saneamento-v1-2-0.md](09-saneamento-v1-2-0.md))
- [x] **1.2.1** Reextração BigQuery `v3` + `rs_painel_semanal_regioes_v3.parquet` ([10-painel-v3-bigquery.md](10-painel-v3-bigquery.md))
- [x] **1.2.2** Ingestão Ao Vivo Onda 2A + extras `cpu`/`rocm`/`cuda` + `F03.1` semanal real em GPU ([11-onda2a-rocm-f03-oof-real.md](11-onda2a-rocm-f03-oof-real.md))
- [x] **1.2.3** Ingestão Ao Vivo Onda 2B (`MERGE 0,1°` + `ANA API v2`), grade `0,1°` anti-G6 e ablação diária `F03.1` ([12-onda2b-grade-01deg-f03-diaria.md](12-onda2b-grade-01deg-f03-diaria.md))
- [x] **1.2.4** Calibração de surtos (`MonotonicSurgeClassifier` + `RollingPlatt`/`Beta`, `BSS_pos +0,2370`, `BSS_shock +0,0097`) + `AgACI` vs. `SPCI` ([13-calibracao-surtos-agaci-spci.md](13-calibracao-surtos-agaci-spci.md))
- [ ] **1.2.5** Otimização `_capacity_blocker` + Pareto `pymoo` NSGA-II (`SimPy` CRN 2 estágios)
- [ ] **1.2.6** `SHAP 0.54.1` (2 estágios) + experimento `PyMC 6.3.2` + `nutpie` + `ArviZ 1.3.0`
- [ ] **M1** Emissão do Dossiê Master `V1.2 RS` (PDF Editorial #1)
- [ ] **M2 (`1.3.0`)** Auditoria Final Única no Holdout (`2025` + `2026` virgem com `ChainLadderNowcaster`) + PDF Editorial #2
- [ ] **2.1–2.4 (`M3`)** Escala Nacional `V2 Brasil` (`Polars 2.1.0`, Hazard Agregado BQ, `Hetero-ST-GNN` + `HydroTFTNet` + expansão Colab, `NSGA-III`) + Release `v1.0.0` + PDF Editorial #3
