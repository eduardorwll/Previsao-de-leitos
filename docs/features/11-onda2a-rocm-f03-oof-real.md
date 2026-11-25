# Feature F11 — Ingestão Ao Vivo Onda 2A, Extras `cpu`/`rocm`/`cuda` e `F03.1 HydroFiLM-ENSO` Real em GPU (`Task 1.2.2`)

> **Status:** ✅ Concluída (`RS <= 2024`, holdout `2025` e `2026` preservados intocados) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

Eliminar o **Bug E** (OOF hidrológico linear circular com `corr = 1,000000` e `Gate G1 FAIL VIF = 11,20`), configurar o empacotamento portável em **3 camadas** (`cpu` no CI, `rocm` na Workstation Arch/RX 7600, `cuda` para expansão Google Colab), corrigir os bloqueios de GPU/Dataset e introduzir os clientes HTTP ao vivo da **Onda 2A** com cache atômico SHA-256.

---

## 2. Alterações Implementadas

- **Empacotamento Portável (`pyproject.toml`, `.github/workflows/ci.yml`, `uv.lock`):**
  - Criados extras mutuamente exclusivos `cpu` (`torch 2.15.0+cpu`), `rocm` (`torch 2.14.1+rocm7.2` + `triton-rocm 3.8.0`) e `cuda` (`torch 2.11.0+cu128`) sob `[tool.uv] conflicts`, mantendo o CI estritamente em `--extra cpu --frozen`.
- **Resolução Lazy de Acelerador e 4 Correções GPU/Dataset (`hydrology/device.py`, `dataset.py`, `oof.py`, `scripts/run_rs_f03_hydrology.py`):**
  - `resolve_accelerator()` *lazy* (zero importação de `torch` em `previsao_leitos/__init__.py`, preservando `C-07`) com suporte automático a `rocm`, `cuda` e `cpu` e guarda `torch.backends.cudnn.enabled = False` em ROCm para executar `nn.GRU` nativamente em `gfx1102` (evitando falha JIT `MIOpenDropoutHIP` sem headers `rocrand`).
  - `IndexedHydroSequenceDataset` atualizado com `enso_cols` (trava `ADR-0001` contra sobreposição com ramos monotônicos) e pré-carga opcional em VRAM (`preload_device`).
  - `audit_autograd_monotonicity` e `audit_dataset_autograd_monotonicity` forçados em **`fp32` sem `autocast`** sobre **100% do split** no dispositivo do modelo; `_predict_logits` movendo tensores para `module.device`.
  - `generate_neural_hydro_oof_predictions` treinando **`HydroFiLM-ENSO` real em `13` folds anuais expansivos × `3` seeds (`42, 137, 2024`)** com liberação de VRAM entre folds (`release_gpu_memory()`).
- **Clientes HTTP Ao Vivo Onda 2A (`ingestion/wave2a_clients.py`):**
  - `fetch_noaa_enso_monthly` (NOAA CPC ONI + PSL MEI.v2: `900` meses reais cacheados em `data/raw/wave2a/`), `fetch_zenodo_iph_rs_metadata` (Zenodo `10.5281/zenodo.16929471`: `9` arquivos catalogados), `discover_gerint_poa_csv_urls`, `ingest_gerint_poa_csv_lgpd`, `ingest_opendatasus_sivep_csv_lgpd` e `ingest_opendatasus_sinan_csv_lgpd` — todos com `HttpFetcher` injetável para testes offline e cache verificado por SHA-256.

---

## 3. Resultados Reais Auditados na RX 7600 (`RS 2024`, `rs_v1_2_2_f03_real_2024`)

| Indicador Auditado | Antes (`V1.2.1`, OOF Linear Circular) | Depois (`V1.2.2`, `HydroFiLM-ENSO` 13×3 GPU) |
| :--- | :---: | :---: |
| **Execução GPU (AMD RX 7600 `8 GB`, ROCm 7.2)** | CPU-only (`20,6 s` single-fit) | **Single-fit `11,8 s` (`1,75×`); OOF `13×3` `98,6 s` (`87,9 MB` pico VRAM)** |
| **Auditoria `autograd fp32` (100% splits)** | `batch0` apenas | **`0,0000%` violações em 100% dos 13 folds × 3 seeds** |
| **Circularidade `corr(chuva, sev_oof)`** | `1,000000` (Bug E) | **`0,6609`** (`< 0,95` — circularidade eliminada) |
| **`F03.1` Flood `ROC-AUC` (`RS 2024`)** | `0,7199` (single-seed) | **`0,7274`** (ensemble `3` seeds) |
| **`Gate G1` (VIF Covariáveis)** | `FAIL (11,20)` | **`WARN (3,98 <= 5,00)` — `FAIL` eliminado** |
| **Campeão `V1` MAE / `MASE` (`IC 95%`)** | `3,52` leitos/dia / `0,9886` | **`3,47` leitos/dia / `0,9732` (`[0,9443; 0,9994] < 1,0`)** |
| **Campeão `V1` `Pinball95` / Timing `W18–30`** | `4,901` / `2,17` sem. | **`4,785` / `1,87` sem.** (`-0,30` sem., `0,00%` viol.) |
| **Oráculo El Niño (`P50` médio)** | `235,40 → 235,80` | **`230,57 → 231,93`** (`+10,2` em `P90`, crescente) |

- **Próximo passo:** `Task 1.2.3` (Ingestão Ao Vivo Onda 2B — `MERGE-CPTEC 0,1°` + `ANA API v2` assim que o token for liberado — + ablação diária `F03.1` `60–90` dias na RX 7600 para zerar `Gate G6`).
