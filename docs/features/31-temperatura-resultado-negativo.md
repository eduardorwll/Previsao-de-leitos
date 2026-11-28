# Feature F31 — Temperatura média semanal como Covariável de Demanda: resultado negativo

> **Status:** ⚠️ Implementado e **rejeitado** pela régua | **Etapa:** Transição V1.4 → V2
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*
> **Estudo precedente:** [F30 — estratégias de fase](30-estrategias-fase-analise.md)

## 1. O que foi testado

Flag `--temperature` (default **off**) inclui `temp_media_sem_c` como covariável dinâmica
(defasagens `0/1/2` semanas). Hipótese: o frio anteciparia a subida respiratória, adiantando o
pico (o único líder exógeno já disponível no painel, sem depender de acesso externo).

## 2. Resultado (régua de 6 folds, 14 modelos, `--exclude hierarchical_bayes`)

| Alvo | MASE base | MASE `--temperature` | Δ MASE | Fase base | Fase `--temperature` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UTI adulto | **0,9434** | 0,9417 | -0,18% | 5,52 | 5,07 |
| Clínico | **0,9243** | 0,9300 | +0,61% (pior) | 4,52 | 4,67 |
| Respiratório | **0,8662** | 0,8649 | -0,16% | 6,22 | 6,47 |

Violações monotônicas: **0** em ambas as rodadas.

**Veredito: rejeitado.** O ganho de MASE é desprezível (`-0,18%` no melhor caso, muito abaixo do
critério `-2%`); a fase melhora apenas em UTI adulto (`-0,45` sem., abaixo de `-1` sem.) e piora em
clínico/respiratório.

## 3. Por que falhou

A temperatura média semanal é **redundante com os harmônicos sazonais** (`target_week_sin/cos`), que
já capturam o ciclo de inverno. Ela não carrega *timing* epidêmico além do calendário — confirmando
o F30: o que precede a internação não é o frio médio, mas a **onda viral** (vigilância
laboratorial/wastewater) e o seu *onset*, que não estão no painel.

## 4. O que fica

- Flag `--temperature` default **off** (sem impacto no campeão `gbdt_conformal`).
- Coluna `temp_media_sem_c` já existente no painel; nenhuma dependência nova.
- Hipótese rejeitada no ciclo, somando-se a alvo, memória do choque, ensemble, dengue, RONI, SOI,
  BOM e temperatura.

## 5. Arquivos

- `src/previsao_leitos/demand/modeling.py` (`TEMPERATURE_FEATURE_COLS`, guarda)
- `src/previsao_leitos/audit/backtest.py` + `src/previsao_leitos/cli/main.py` (`--temperature`)
- `tests/unit/test_demand_registry_and_benchmark.py` (guarda + lags)
