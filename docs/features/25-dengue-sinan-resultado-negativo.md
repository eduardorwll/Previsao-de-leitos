# Feature F25 — Dengue (SINAN) como Covariável: resultado negativo (`Task 1.3.8b`)

> **Status:** ⚠️ Ingerido e **rejeitado** pela régua | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

## 1. Ingestão (feita e preservada)

| Item | Valor |
| :--- | :--- |
| Tabela | `basedosdados.br_ms_sinan.microdados_dengue` |
| Extração | `532,7 MB` faturados, `24.563` linhas, `499` municípios, `2010–2024` |
| Agregação | `(região de saúde, ano_epi, semana_epi)` — `9.595` linhas, `370.914` casos, 30/30 regiões |
| Arquivos | `data/raw/sinan/rs_dengue_semanal.parquet` + manifesto SHA-256 |
| LGPD | somente contagens; nenhum registro individual |

O `week` da fonte vem como `"YYYY-WW"`; o parsing usa `SPLIT` no SQL (documentado no runbook).

## 2. Hipótese e teste

Correlação de dengue com os alvos cresce com o lag (`+0,21` em UTI no lag 8; `+0,17` no clínico),
mais forte que o choque de enchente (`|r| ≈ 0,1`). A hipótese era que a dengue carregaria sinal
sazonal antecipado útil.

Feature `dengue_casos` com defasagens `0,1,2`, ativada por `--dengue` (default **off**).

## 3. Resultado

| Alvo | Modelo | MASE off | MASE on | Δ |
| :--- | :--- | ---: | ---: | ---: |
| UTI adulto | `gbdt_v1` | **0,9434** | 0,9487 | +0,6% (pior) |
| UTI adulto | `f041_local_level` | **1,1348** | 1,1766 | +3,7% (pior) |
| Clínico | `gbdt_v1` | **0,9243** | 0,9300 | +0,6% (pior) |
| Respiratório | `gbdt_v1` | **0,8662** | 0,8683 | +0,2% (pior) |
| Nec. clínico | `gbdt_v1` | **0,8842** | 0,9041 | +2,2% (pior) |
| Nec. respiratório | `f041_local_level` | **0,9566** | 0,9578 | +0,1% (pior) |

**Veredito: rejeitado.** A correlação era **sazonalidade compartilhada** (dengue no verão, leitos
no inverno), já capturada por harmônicos e `y_roll*`. Nenhum alvo melhorou.

## 4. O que fica

- Extração e série preservadas (custo já pago, `~533 MB` de 1 TB/mês).
- Flag `--dengue` default **off**; sem impacto no campeão.
- Quinta hipótese rejeitada no ciclo (alvo, memória do choque, ensemble, dengue; momentum de
  internações foi ganho parcial).

## 5. Arquivos

- `scripts/sql/sinan_dengue_rs_semanal.sql`
- `scripts/build_rs_modeling_panel.py` (join opcional)
- `src/previsao_leitos/demand/modeling.py` (`DENGUE_FEATURE_COLS`)
- `tests/unit/test_demand_registry_and_benchmark.py` (guarda + lags)
- `docs/runbooks/ingest-s2id-sinan.md`
