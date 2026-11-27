# Feature F28 — BOM Niño 3.4 relativo semanal como Covariável: resultado negativo (`E2`)

> **Status:** ⚠️ Implementado e **rejeitado** pela régua | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*
> **Estudo precedente:** [F24 — ENSO, fontes e vintage](24-enso-fontes-e-vintage.md) (`Task 1.3.10`)

## 1. O que foi implementado

| Item | Valor |
| :--- | :--- |
| Fonte primária | BOM Niño 3.4 relativo semanal (`https://www.bom.gov.au/clim_data/IDCK000072/rnino_3.4.txt`, 2008–presente) |
| Acesso | BOM bloqueia automação sem `User-Agent`; o fetcher padrão do projeto funciona (HTTP 200) |
| Cache | `data/raw/wave2a/bom_rnino_3_4.txt` + manifesto SHA-256 |
| Publication lag | **1 semana** (valor da semana W disponível na semana W+1) |
| Feature semanal | `bom_rnino` (nível), ancorada no `week_start` (segunda) → semana ISO |
| Flag | `--bom` (default **off**) em `backtest` e `benchmark`/`run` |
| Guarda | coluna ausente → `DataContractError` (teste dedicado) |

## 2. Hipótese

O BOM publica o índice relativo em **frequência semanal** (o ONI/RONI/SOI são mensais), com o
próprio limiar El Niño de **+0,8 °C** (não o ±0,5 °C do NOAA). A hipótese era que esse sinal de
maior frequência, alinhado ao painel `h = 2` semanas, capturaria variação ENSO que os índices
mensais suavizam.

## 3. Resultado (régua de 6 folds, 14 modelos, `--exclude hierarchical_bayes`)

| Alvo | MASE base | MASE `--bom` | Δ MASE | Fase base | Fase `--bom` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UTI adulto | **0,9434** | 0,9409 | -0,27% | 5,52 | 4,92 |
| Clínico | **0,9243** | 0,9301 | +0,63% (pior) | 4,52 | 4,23 |
| Respiratório | **0,8662** | 0,8667 | +0,06% (pior) | 6,22 | 6,48 |

Violações monotônicas: **0** em ambas as rodadas.

**Veredito: rejeitado.** O melhor ganho de MASE foi `-0,27%` (UTI adulto), muito abaixo do critério
`>= 2%`; a fase melhorou em UTI adulto (`-0,60` sem.) e clínico (`-0,29` sem.) mas piorou em
respiratório.

## 4. Por que falhou

Encerra a trilha E1/E2/E3 com **3 de 3 índices ENSO alternativos rejeitados**. RONI, SOI e BOM
semanal são todos redundantes com a teleconexão oceânica já representada por `el_nino_intensity`
(ONI): em `h = 2` semanas, nenhum deles injeta sinal preditivo além do índice oceânico que o modelo
já consome. Isso corrobora a conclusão do handoff `V1.3` — o teto desta base de features é
`MASE 0,86–0,96`, e ganho adicional exige **dado novo** ou **classe de modelo diferente**.

## 5. O que fica

- Fetcher BOM + parser + projeção semanal causal preservados (reutilizável para E4, índice consenso).
- Coluna `bom_rnino` no painel de modelagem (join opcional no builder).
- Flag `--bom` default **off**; **sem impacto no campeão** `gbdt_conformal`.
- Hipóteses ENSO (E1/E2/E3) rejeitadas no ciclo, juntando-se a alvo, memória do choque, ensemble
  e dengue; momentum de internações segue como ganho parcial.

## 6. Arquivos

- `src/previsao_leitos/geo/enso_teleconnections.py` (`parse_bom_rnino_table`, `build_causal_bom_rnino_weekly`)
- `src/previsao_leitos/ingestion/wave2a_clients.py` (`fetch_bom_rnino_weekly`, `BOM_RNINO_URL`)
- `scripts/build_rs_modeling_panel.py` (join opcional de `bom_rnino`)
- `src/previsao_leitos/demand/modeling.py` (`BOM_RNINO_FEATURE_COLS`, guarda `bom_features`)
- `src/previsao_leitos/audit/backtest.py` + `src/previsao_leitos/cli/main.py` (`--bom`)
- `tests/unit/test_geo_enso_teleconnections.py`, `test_v1_2_2_wave2a_and_gpu_oof.py`,
  `test_demand_registry_and_benchmark.py`
