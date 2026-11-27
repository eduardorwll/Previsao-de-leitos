# Feature F27 — SOI (índice atmosférico CPC) como Covariável: resultado negativo (`E3`)

> **Status:** ⚠️ Implementado e **rejeitado** pela régua | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*
> **Estudo precedente:** [F24 — ENSO, fontes e vintage](24-enso-fontes-e-vintage.md) (`Task 1.3.10`)

## 1. O que foi implementado

| Item | Valor |
| :--- | :--- |
| Fonte primária | CPC SOI (`https://www.cpc.ncep.noaa.gov/data/indices/soi`, bloco `STANDARDIZED`, 1951–presente) |
| Cache | `data/raw/wave2a/noaa_cpc_soi.txt` + manifesto SHA-256 |
| Publication lag | **1 mês** (30 dias) |
| Feature semanal | `soi_std` (nível, componente atmosférico Tahiti−Darwin) |
| Flag | `--soi` (default **off**) em `backtest` e `benchmark`/`run` |
| Guarda | coluna ausente → `DataContractError` (teste dedicado) |

O parser ignora o primeiro bloco (`ANOMALY`) e lê o segundo (`STANDARDIZED DATA`), tratando
`-999.9` como dado faltante (inclusive o caso colado `-1.1-999.9-999.9` na borda do mês corrente).

## 2. Hipótese

O SOI é a **metade atmosférica** do ENSO, complementar ao ONI (oceânico). A hipótese era que ele
carregaria sinal de circulação atmosférica (Walker) útil além do `el_nino_intensity` derivado do ONI.

## 3. Resultado (régua de 6 folds, 14 modelos, `--exclude hierarchical_bayes`)

| Alvo | MASE base | MASE `--soi` | Δ MASE | Fase base | Fase `--soi` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UTI adulto | **0,9434** | 0,9508 | +0,79% (pior) | 5,52 | 5,87 |
| Clínico | **0,9243** | 0,9340 | +1,04% (pior) | 4,52 | 4,33 |
| Respiratório | **0,8662** | 0,8654 | -0,09% | 6,22 | 6,53 |

Violações monotônicas: **0** em ambas as rodadas.

**Veredito: rejeitado.** Nenhum alvo melhorou em MASE de forma material (melhor caso `-0,09%`, muito
abaixo do critério `>= 2%`); a fase melhorou apenas em clínico (`-0,19` sem.) e piorou em UTI adulto
e respiratório.

## 4. Por que falhou

Assim como no E1 (RONI), o `soi_std` é redundante com a teleconexão já representada por
`el_nino_intensity` (ONI): o acoplamento oceano-atmosfera do ENSO já está, em grande parte, embutido
no índice oceânico que o modelo consome. Em `h = 2` semanas, a informação atmosférica de baixa
frequência não adiciona poder preditivo além do sinal oceânico contemporâneo.

## 5. O que fica

- Fetcher CPC + parser + projeção semanal causal preservados (reutilizável para E4, índice consenso).
- Coluna `soi_std` no painel de modelagem (join opcional no builder).
- Flag `--soi` default **off**; **sem impacto no campeão** `gbdt_conformal`.
- Hipótese rejeitada no ciclo (alvo, memória do choque, ensemble, dengue, RONI e SOI; momentum de
  internações segue como ganho parcial).

## 6. Arquivos

- `src/previsao_leitos/geo/enso_teleconnections.py` (`parse_noaa_soi_table`, `build_causal_soi_weekly_features`)
- `src/previsao_leitos/ingestion/wave2a_clients.py` (`fetch_noaa_soi_monthly`, `NOAA_SOI_URL`)
- `scripts/build_rs_modeling_panel.py` (join opcional de `soi_std`)
- `src/previsao_leitos/demand/modeling.py` (`SOI_FEATURE_COLS`, guarda `soi_features`)
- `src/previsao_leitos/audit/backtest.py` + `src/previsao_leitos/cli/main.py` (`--soi`)
- `tests/unit/test_geo_enso_teleconnections.py`, `test_v1_2_2_wave2a_and_gpu_oof.py`,
  `test_demand_registry_and_benchmark.py`
