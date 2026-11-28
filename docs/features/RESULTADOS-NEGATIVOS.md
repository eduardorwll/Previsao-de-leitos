# Resultados Negativos e Higienização do Repositório

> **Status:** Consolidado da higienização (v1.5) | **Fronteira:** *Not for patient care or live clinical use.*

Este documento consolida **todos os experimentos rejeitados pela régua** e explica **o que foi
removido** do código na higienização. O objetivo é manter só os **melhores métodos** (o campeão
`gbdt_conformal` e o pipeline que o sustenta) e preservar a memória científica das hipóteses que
não funcionaram.

## 1. Experimentos rejeitados (e removidos)

Todos foram testados na **mesma régua de 6 folds** (`2019–2024`, `train ≤ y−2`, calibração `y−1`)
e **rejeitados** por não atingirem `MASE ≤ −2%` **ou** `erro de fase ≤ −1 semana`, com `0` violações
monotônicas.

| Experimento | Método | Resultado na régua | Veredito | Por quê |
| :--- | :--- | :--- | :--- | :--- |
| **F20 — Memória do choque** | `shock_cum_4w/8w`, `shock_max_8w`, `shock_ewm_4w` | `MASE ±0,3%`, fase `0,1–0,3` sem. | ❌ | O pico é sazonal; o choque muda amplitude, não fase |
| **F22 — Ensemble calibrado** | `ensemble_top3` (pesos ∝ 1/MAE) | Perde em 4/5 alvos | ❌ | Membros correlacionados → sem diversificação |
| **F25 — Dengue (SINAN)** | `dengue_casos` | `+0,2%` a `+3,7%` (pior) | ❌ | Correlação era sazonalidade compartilhada |
| **F26 — RONI (ENSO relativo)** | `roni_c`/`roni_regime` (lag 2 meses) | `+0,13%` a `+0,86%` (pior) | ❌ | Redundante com `el_nino_intensity` (ONI) |
| **F27 — SOI (ENSO atmosférico)** | `soi_std` (lag 1 mês) | `-0,09%` a `+1,04%` | ❌ | Redundante com ONI |
| **F28 — BOM Niño 3.4 semanal** | `bom_rnino` (lag 1 sem.) | `-0,27%` a `+0,63%` | ❌ | Redundante com ONI |
| **F31 — Temperatura média** | `temp_media_sem_c` | `-0,18%` máx., fase `-0,45` sem. | ❌ | Redundante com harmônicos sazonais |

> **F23 (momentum de internações)** foi **parcialmente aceito** (ganho em respiratório, `−5,7%`
> no F04.1) e **permanece** como flag opt-in `--admissions-momentum`.

## 2. O que foi removido do código (higienização v1.5)

| Removido | Onde estava | Motivo |
| :--- | :--- | :--- |
| Flags `--dengue`, `--shock-memory`, `--roni`, `--soi`, `--bom`, `--temperature` | `cli/main.py`, `modeling.py`, `backtest.py` | Experimento rejeitado |
| `DENGUE_FEATURE_COLS`, `SHOCK_MEMORY_COLS`, `RONI/SOI/BOM/TEMPERATURE_FEATURE_COLS` | `demand/modeling.py` | Constantes órfãs |
| `parse_noaa_roni_table`, `parse_noaa_soi_table`, `parse_bom_rnino_table` | `geo/enso_teleconnections.py` | Código morto (só usavam as flags) |
| `build_causal_roni/soi/bom_*` | `geo/enso_teleconnections.py` | Idem |
| `fetch_noaa_roni_monthly`, `fetch_noaa_soi_monthly`, `fetch_bom_rnino_weekly` | `ingestion/wave2a_clients.py` | Idem |
| Joins RONI/SOI/BOM/dengue + colunas de memória do choque | `scripts/build_rs_modeling_panel.py` | Painel mais limpo |

**Mantido (melhores métodos):** `gbdt_conformal` (campeão), `parse_noaa_oni_table` /
`build_causal_enso_weekly_features` (ENSO **ONI**, que alimenta `el_nino_intensity`), hidrologia de
enchente, conformal (`Flood`/`AgACI`/`SPCI`), `PyMC` hierárquico, `SimPy`/`pareto`, LOS com riscos
competitivos agregado, régua de backtest, catraca e dossiê M1.

## 3. Por que a redução é segura

- O campeão **nunca** usou as flags opcionais (default **off**); o comportamento observável não muda.
- Os 6 folds da régua dependiam só de `DYNAMIC_FEATURE_COLS` + `admissions_momentum`.
- A infraestrutura ENSO **ONI** (a única com sinal) foi mantida.

## 4. Referências preservadas

Os estudos analíticos que valem a leitura continuam em:
- [F24 — Fontes ENSO e vintage](24-enso-fontes-e-vintage.md) (por que ONI × RONI discordam).
- [F29 — Diagnóstico de higiene/fase](29-diagnostico-higiene-fase-e-v2.md) (G1/G4 e persistência).
- [F30 — Estratégias de fase](30-estrategias-fase-analise.md) (o que não funciona e por quê).
- [F17 — Régua de backtest](17-regua-backtest-origem-movel.md) e [docs/benchmark/](../benchmark/README.md).
