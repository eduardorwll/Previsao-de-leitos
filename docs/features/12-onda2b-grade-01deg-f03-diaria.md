# Feature F12 — Onda 2B (`MERGE-CPTEC 0,1°` + `ANA API v2`), Grade `0,1°` Anti-G6 e Ablação Diária `F03.1` em GPU (`Task 1.2.3`)

> **Status:** ✅ Concluída (`RS <= 2024`, holdout `2025` e `2026` preservados intocados) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

Eliminar a colisão de grade climática (`Gate G6 FAIL 5/30 = 16,7%`) migrando o painel `RS v3` para grade fina `0,1°` (`ERA5-Land` diário + `V10M`), introduzir os clientes ao vivo da **Onda 2B** (`MERGE-CPTEC 0,1°` e `ANA HidroWeb API v2` OAuth com fallback automático Zenodo `10.5281/zenodo.16929471`), evoluir o pipeline `F03.1` para suporte dual (`freq="weekly" | "daily"`) e executar na **AMD RX 7600 (ROCm 7.2)** a ablação controlada **`F03.1` Diário (`60` dias) vs. Semanal (`8` semanas)** sob catraca científica objetiva.

---

## 2. Alterações Implementadas

- **Clientes HTTP Ao Vivo Onda 2B (`src/previsao_leitos/ingestion/wave2b_clients.py`, `.env.example`):**
  - `MergeCptecClient` e `build_merge_cptec_daily_url`: download autenticado por SHA-256 em cache atômico dos arquivos diários `MERGE-CPTEC 0,1°` (`GRIB2`, CPTEC/INPE) sem dependências nativas quebráveis (`cfgrib`/`eccodes`) no `uv.lock`. Validado ao vivo com `MERGE_CPTEC_20240501.grib2` (`450.085 B`, SHA-256 `c01a7a1151a0...`).
  - `AnaHidrowebCredentials` e `AnaHidrowebClient`: autenticação OAuth v2 parametrizada via `ANA_HIDROWEB_USER` / `ANA_HIDROWEB_PASSWORD` (placeholders vazios em `.env.example`) com fallback determinístico para o repositório aberto `Zenodo 10.5281/zenodo.16929471` enquanto a solicitação institucional à ANA (`telemetria@ana.gov.br`) aguarda liberação.
- **Grade Hidrológica Fina `0,1°` Anti-G6 (`scripts/build_rs_panel_v3.py`):**
  - Adicionado `--hydro-source era5_01deg`, combinando `OpenMeteoEra5Adapter` (`0,1°`, `175.320` dias para as `30` Regiões de Saúde do RS em `2010–2025`, cacheados em `data/raw/era5_land_daily_2010_2025_*.parquet`) com `vento_sul_ms` (`NASA POWER V10M`) na reconstrução de `data/processed/rs_painel_semanal_regioes_v3.parquet`.
- **Suporte Dual Semanal/Diário em `F03.1` (`hydrology/dataset.py`, `hydrology/oof.py`, `scripts/run_rs_f03_daily_ablation.py`):**
  - `IndexedHydroSequenceDataset` e `generate_neural_hydro_oof_predictions` evoluídos com `freq: Literal["weekly", "daily"] = "weekly"`, `date_col`, `sample_mask_col` e guarda estrita `step_days = 1` (diário) ou `7` (semanal) para impedir cruzamento de lacunas calendário.
  - Orquestrador `scripts/run_rs_f03_daily_ablation.py` treinando `HydroFiLM-ENSO` diário (`seq_len = 60` dias encerrados no sábado ISO `t_0`, `163.800` dias / `23.400` amostras semanais, `13` folds × `3` seeds) na GPU e aplicando catraca automática contra o OOF semanal `0,1°`.
  - `scripts/run_rs_v1_1_dossier.py` atualizado para detectar `rs_f03_ablation_daily_60d_vs_weekly.json` e emitir o dossiê `rs_v1_2_3_wave2b_01deg_2024` (`V1.2.3`).
- **Testes Unitários Offline (`tests/unit/test_v1_2_3_wave2b_and_daily_f03.py`):**
  - Cobertura de URL/cache SHA-256 `MERGE-CPTEC`, OAuth + fallback Zenodo `ANA`, guarda anti-lacuna diária (`step_days=1`) e geração OOF diária com retorno alinhado à grade semanal.

### 2.1 Correção de Contrato da ANA API v2 (2026-11-25)

Com as credenciais institucionais em mãos, a validação ao vivo expôs três divergências do cliente em relação ao manual oficial `HidroWebService`:

| Item | Antes | Agora (manual oficial) |
| :--- | :--- | :--- |
| Rota de autenticação | `/EstacoesTelemetricas/OAuth/v1` (`HTTP 404`) | `/EstacoesTelemetricas/OAUth/v1` |
| Código da estação | `Codigo_Estacao` | `CodigoDaEstacao` |
| Filtro de data | `Tipo_Filtro_Data` | `TipoFiltroData` |
| Janela | `Range_Intervalo` | `RangeIntervaloDeBusca` |

O parser (`parse_telemetric_series_json`) já estava correto (`items`, `Data_Hora_Medicao`, `Cota_Adotada`, `Vazao_Adotada`) e foi mantido. Foi adicionado o teste de contrato `test_ana_hidroweb_uses_documented_routes_and_query_params` (sem rede) e a flag `--require-live-ana` na ablação diária, que impede o fallback Zenodo de mascarar falha de credencial e registra `ana_provenance` no JSON.

> **Bloqueio operacional remanescente:** a autenticação responde `HTTP 401 — "Usuário sem permissão de acesso ao módulo. Verifique!"`. As credenciais trafegam corretamente (headers `Identificador`/`Senha` aceitos pela rota), mas o usuário ainda não possui permissão no módulo `HidroWebService` do lado da ANA. Enquanto isso, o pipeline segue usando o fallback aberto `Zenodo 16929471`, e a ablação diária continua classificada como `KEEP_WEEKLY_CHAMPION`.

---

## 3. Resultados Reais Auditados na RX 7600 (`RS 2024`, `rs_v1_2_3_wave2b_01deg_2024`)

### 3.1 Eliminação do `Gate G6` e Salto da Grade `0,1°` no `F03.1`

| Indicador Auditado | Antes (`V1.2.2`, Grade `0,5°` NASA) | Depois (`V1.2.3`, Grade `0,1°` ERA5-Land + V10M) |
| :--- | :---: | :---: |
| **`Gate G6` (Colisão de Grade Espacial)** | `FAIL — 5/30 (16,7%)` | **`PASS — 0/30 (0,0%)`** |
| **`F03.1` Semanal `ROC-AUC` (`RS 2024`, `13×3`)** | `0,7274` | **`0,7888`** (`+0,0614`) |
| **`F03.1` Semanal `BSS` (`RS 2024`, `13×3`)** | `+0,0956` | **`+0,1561`** (`+0,0605`) |
| **`F03.1` Semanal `ECE` (`RS 2024`, `13×3`)** | `0,0521` (ref. F03) | **`0,0475`** |
| **Campeão `V1` `Pinball P95` / `ECE` (`RS 2024`)** | `4,785` / `0,0447` | **`4,740` / `0,0423`** (`0,00%` viol. monot.) |
| **Scorecard `8/8 Gates`** | `PASS=3, WARN=2, FAIL=3` | **`PASS=4 (G2, G5, G6, G8), WARN=2 (G1, G4), FAIL=2 (G3, G7)`** |

### 3.2 Ablação Controlada na RX 7600: `F03.1` Diário (`60d`) vs. Semanal (`8 sem.`, `0,1°`)

| Configuração (`13` folds × `3` seeds, RX 7600 ROCm 7.2) | Tempo | Pico VRAM | `Autograd fp32` | `ROC-AUC` (`2024`) | `ECE` (`2024`) | `BSS` (`2024`) | Decisão Catraca |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`F03.1` Semanal (`8 sem.`, `0,1°`, multi-salto + remanso)** | **`99,2 s`** | **`87,9 MB`** | **`0,0000%`** | **`0,7888`** | **`0,0475`** | **`+0,1561`** | **MANTIDO (`CHAMPION`)** |
| **`F03.1` Diário (`60d`, `0,1°`, local sem cotas ANA)** | `342,9 s` | `251,3 MB` | `0,0000%` | `0,7299` | `0,0589` | `+0,0899` | **`KEEP_WEEKLY_CHAMPION`** |

- **Diagnóstico científico da ablação:** A versão semanal `0,1°` superou a diária `60d` em discriminação (`+0,0589` ROC-AUC), habilidade probabilística (`+0,0662` BSS) e calibração (`-0,0114` ECE), rodando `3,5×` mais rápido e usando `2,9×` menos VRAM. A diferença explica-se porque o painel semanal já agrega propagação hidrológica multi-salto (`chuva_montante_multihop`) e remanso lagunar (`remanso_lagunar_idx`), enquanto a sequência diária atual ainda aguarda as séries diárias de cota/vazão da `ANA API v2` para acoplamento fluvial diário. O código dual diário permanece validado e pronto para reavaliação assim que as credenciais ANA forem liberadas.
- **Próximo passo:** `Task 1.2.4` — Calibração probabilística de surtos (`MonotonicSurgeClassifier` + Rolling Platt/Beta `2019–2023` visando `BSS_pos >= +0,10` e `ECE <= 0,04`) + ablação `AgACI` vs. `SPCI`.
