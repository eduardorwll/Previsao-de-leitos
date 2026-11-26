# Feature F13 — Calibração Probabilística de Surtos (`Platt`/`Beta`) e Ablação Conformal `AgACI` vs. `SPCI` (`Task 1.2.4`)

> **Status:** ✅ Concluída (`RS <= 2024`, holdout `2025` e `2026` preservados intocados) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

Substituir a estimativa indireta de probabilidade de surto por interpolação de quantis (que apresentava `BSS_pos = -0,0301`, `BSS_shock = -0,3343` e `ECE = 0,0874` no Challenger `V1.2.3`) por uma **cabeça classificadora probabilística monotônica dedicada** calibrada de forma estritamente causal em janela expansiva `2019–2023`, e conduzir a ablação controlada **`AgACI` vs. `SPCI` (QRF) normalizados por capacidade CNES** sob atraso causal real de `h = 2` semanas.

---

## 2. Alterações Implementadas

- **Calibração Probabilística Monotônica de Surtos (`src/previsao_leitos/demand/surge_calibration.py`):**
  - `RollingPlattCalibrator` (`a >= 0`) e `RollingBetaCalibrator` (Kull et al., AISTATS 2017, `a >= 0, b >= 0`): mapeamentos estritamente não-decrescentes com pesos de recência temporal (`decay = 0,80/ano`) e regularização L2 branda.
  - `MonotonicSurgeClassifier`: `HistGradientBoostingClassifier(loss="log_loss", monotonic_cst=...)` garantindo **`0,00%` violações monotônicas** sob aumento hidrológico/ENSO (`ADR-0001`), máscara estrutural anti-G8 (`y_roll12 <= 1e-9 -> p = 0.0`), suporte a definições de surto relativo (`>= 1,10 × y_roll12`), por ocupação CNES (`>= 90%`) ou composto (`either`), e ajuste *out-of-fold* expansivo estritamente causal em `2019–2023` (`n = 7.233`).
- **Calibração Conformal Adaptativa Normalizada por CNES (`src/previsao_leitos/demand/adaptive_conformal.py`):**
  - `compute_cnes_capacity_scale`: normaliza os escores não-conformes por `capacidade_sem` (homogeneizando metrópole vs. interior).
  - `AgACIConformalCalibrator` (Zaffran et al., ICML 2022): agrega `5` especialistas ACI ($\gamma \in \{0,005; 0,01; 0,025; 0,05; 0,10\}$) via pesos exponenciais de perda pinball com atraso causal exato `feedback_lag_weeks = 2` (modo `predict_quantiles_online`) e modo congelado `predict_quantiles` para cenários contrafactuais `do()`.
  - `SPCIConformalCalibrator` (Xu & Xie, ICML 2023): Quantile Random Forest sobre resíduos normalizados por CNES defasados (`residual_lags = 4`, `feedback_lag_weeks = 2`), preservando `0,00%` de violações monotônicas hidrológicas por construção.
- **Ablação, Dossiê e Testes (`scripts/run_rs_surge_and_conformal_ablation.py`, `scripts/run_rs_v1_1_dossier.py`, `tests/unit/test_v1_2_4_surge_and_adaptive_conformal.py`):**
  - Suíte TDD e *Prove-It* validando monotonicidade `0,00%`, máscara anti-G8, invariância bit-idêntica sob corrupção de rótulos futuros (`2024`) e atraso estrito `h = 2` no `AgACI`/`SPCI` online.

---

## 3. Resultados Reais Auditados (`RS 2024`, Dossiê `V1.2.4`, `7,5 s` CPU)

### 3.1 Calibração Probabilística de Surtos (`y >= 1,10 × y_roll12`, Máscara Anti-G8)

| Modelo / Calibrador (`RS 2024`) | `BSS_all` | `BSS_pos` | `BSS_shock` (`W18–28`) | `ECE_all` | `ROC-AUC_pos` | `PR-AUC_pos` | Espec. Interior | `NB (pt=0,30)` | Viol. Monot. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `V1.2.3` Quantis `TwoStage` (anterior) | `-0,0064` | `-0,0301` | `-0,3343` | `0,0874` | — | — | `93,04%` | — | `0,00%` |
| `V1_QuantileInterp_Baseline` | `+0,2357` | `+0,2178` | `+0,0198` | **`0,0363`** | `0,8009` | `0,5866` | `93,92%` | `+0,1025` | `0,00%` |
| `MonotonicSurge` (sem Platt/Beta) | `+0,2520` | `+0,2344` | `-0,0246` | `0,0420` | `0,8121` | `0,6098` | `96,35%` | `+0,1020` | `0,00%` |
| **`MonotonicSurge + RollingPlatt (2019–2023)`** | **`+0,2545`** | **`+0,2370`** | **`+0,0097`** | `0,0411` | **`0,8121`** | **`0,6098`** | **`95,95%`** | **`+0,1116`** | **`0,00%`** |
| `MonotonicSurge + RollingBeta (2019–2023)` | `+0,2524` | `+0,2348` | `+0,0081` | `0,0437` | `0,8121` | `0,6098` | `95,95%` | `+0,1107` | `0,00%` |

- No OOF causal `2019–2023` (`n = 7.233`), `RollingPlatt` reduziu o `ECE` de `0,0389 → 0,0246` e elevou `BSS` de `+0,1471 → +0,1529`. Em `RS 2024`, superou amplamente a meta de `BSS_pos >= +0,10` (`+0,2370`) e reverteu `BSS_shock` de `-0,3343` (e `-0,0246` raw) para **`+0,0097`** mantendo `95,95%` de especificidade no interior e `0,00%` de violações monotônicas.

### 3.2 Ablação Conformal Normalizada por CNES sobre o Campeão `V1` (`RS 2024`)

| Calibrador Conformal (`h = 2` sem.) | `Cov P10–P90` | `Cov Shock W18–28` | `Pinball P90` | `Pinball P95` | Largura Média (leitos/dia) | Viol. Monot. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ConformalFlood_2023` (baseline V1) | `68,01%` | `64,94%` | `6,8208` | `4,6918` | `6,98` | `0,00%` |
| `MondrianEnso_2023` | `69,17%` | `67,53%` | `6,5657` | `4,4401` | `7,59` | `0,00%` |
| **`AgACI_CNES_Online_Lag2`** | **`79,36%`** (`+11,4 pp`) | **`69,48%`** (`+4,5 pp`) | `7,1283` | `4,5811` | `12,86` | **`0,00%`** |
| **`SPCI_QRF_CNES_Online_Lag2`** | `71,73%` (`+3,7 pp`) | `68,18%` (`+3,2 pp`) | **`6,6639`** | **`4,3584`** (`-7,1%`) | `10,99` | **`0,00%`** |

- **Trade-off científico claro:** `AgACI_CNES_Online_Lag2` entregou a melhor cobertura global (`79,36%`, praticamente no alvo nominal `80%`) e no choque de maio/2024 (`69,48%`), enquanto `SPCI_QRF_CNES_Online_Lag2` alcançou o melhor escore próprio de cauda extrema (`Pinball P95 = 4,3584`, `-7,1%` sobre o baseline), ambos respeitando atraso estrito `h = 2` semanas e `0,00%` de violações monotônicas.
- **Próximo passo:** `Task 1.2.5` — Fronteira de Pareto (`pymoo` NSGA-II + `SimPy` CRN em 2 estágios + refatoração medida de `_capacity_blocker`).
