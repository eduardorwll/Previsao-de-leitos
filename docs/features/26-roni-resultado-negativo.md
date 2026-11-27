# Feature F26 — RONI (índice relativo CPC) como Covariável: resultado negativo (`E1`)

> **Status:** ⚠️ Implementado e **rejeitado** pela régua | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*
> **Estudo precedente:** [F24 — ENSO, fontes e vintage](24-enso-fontes-e-vintage.md) (`Task 1.3.10`)

## 1. O que foi implementado

| Item | Valor |
| :--- | :--- |
| Fonte primária | CPC RONI (`https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt`, 1950–presente) |
| Cross-check | JSON `https://www.elninolive.com/api/enso/roni` — **apenas dev, nunca runtime** |
| Cache | `data/raw/wave2a/noaa_cpc_roni.ascii.txt` + manifesto SHA-256 |
| Publication lag | **2 meses** (60 dias) — o RONI revisa por até 2 meses (filtro alta-frequência ERSSTv6) |
| Features semanais | `roni_c` (nível) + `roni_regime` (±0,5 / ±1,5, 4 classes) |
| Flag | `--roni` (default **off**) em `backtest` e `benchmark`/`run` |
| Guarda | coluna ausente → `DataContractError` (teste dedicado) |

## 2. Hipótese

O estudo F24 mostrou que **ONI × RONI discordam de regime em 33,3% dos meses desde 2015**
(14,8% no total). Como `enso_regime` entra no modelo, a hipótese era que o RONI — que isola o
aquecimento do Pacífico da tendência global — corrigiria 1/3 dos rótulos de regime e reduziria o
MASE mediano ou o erro de fase.

## 3. Resultado (régua de 6 folds, 14 modelos, `--exclude hierarchical_bayes`)

| Alvo | MASE base | MASE `--roni` | Δ MASE | Fase base | Fase `--roni` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UTI adulto | **0,9434** | 0,9515 | +0,86% (pior) | 5,52 | 4,98 |
| Clínico | **0,9243** | 0,9290 | +0,51% (pior) | 4,52 | 4,87 |
| Respiratório | **0,8662** | 0,8673 | +0,13% (pior) | 6,22 | 6,37 |

Violações monotônicas: **0** em ambas as rodadas (critério de higiene preservado).

**Veredito: rejeitado.** Nenhum alvo melhorou em MASE (todos pioraram `+0,13%` a `+0,86%`); a fase
melhorou apenas em UTI adulto (`-0,54` sem.) e piorou em clínico e respiratório. Não atende ao
critério de aceite (ganho `>= 2%` no MASE **ou** melhora consistente de fase).

## 4. Por que falhou

O `roni_c` é altamente redundante com o ONI (`r = 0,958`; F24), e o modelo **já** consome
`el_nino_intensity` (derivado do ONI). Adicionar o RONI não injeta sinal novo — e o
*publication lag* de 2 meses **descarta** o ENSO mais recente que o `el_nino_intensity`
(lag de 32 dias) ainda retém, explicando a piora marginal. A discordância de regime de 33,3%
concentra-se em `|ONI| < 0,5` (fronteira de rótulo), que o classificador GBDT já suaviza ao usar a
intensidade contínua em vez do rótulo.

## 5. O que fica

- Fetcher primário CPC + parser + projeção semanal causal preservados (custo já pago; reutilizável
  para E3/E4 se necessário).
- Colunas `roni_c`/`roni_regime` no painel de modelagem (join opcional no builder).
- Flag `--roni` default **off**; **sem impacto no campeão** `gbdt_conformal`.
- Hipótese rejeitada no ciclo, juntando-se a alvo, memória do choque, ensemble e dengue; momentum
  de internações segue como ganho parcial.

## 6. Arquivos

- `src/previsao_leitos/geo/enso_teleconnections.py` (`parse_noaa_roni_table`, `build_causal_roni_weekly_features`)
- `src/previsao_leitos/ingestion/wave2a_clients.py` (`fetch_noaa_roni_monthly`, `NOAA_RONI_URL`)
- `scripts/build_rs_modeling_panel.py` (join opcional de `roni_c`/`roni_regime`)
- `src/previsao_leitos/demand/modeling.py` (`RONI_FEATURE_COLS`, guarda `roni_features`)
- `src/previsao_leitos/audit/backtest.py` + `src/previsao_leitos/cli/main.py` (`--roni`)
- `tests/unit/test_geo_enso_teleconnections.py`, `test_v1_2_2_wave2a_and_gpu_oof.py`,
  `test_demand_registry_and_benchmark.py`
