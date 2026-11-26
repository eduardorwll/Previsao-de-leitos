# Feature F20 — Memória Causal do Choque (resultado negativo) (`Task 1.3.5`)

> **Status:** ⚠️ Implementada e **rejeitada** pela régua | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

---

## 1. Hipótese

O pico de ocupação ocorre `6–13` semanas depois da enchente, mas o modelo recebe apenas o
choque contemporâneo em `h=2`. A hipótese era que a **exposição acumulada** corrigiria o erro
de fase de `8–9` semanas do `F04.1` e as `4–6` do GBDT.

## 2. Implementação

Quatro colunas causais (janela móvel por região, sem olhar o futuro):

| Coluna | Definição |
| :--- | :--- |
| `shock_cum_4w` | soma do choque nas últimas 4 semanas |
| `shock_cum_8w` | soma do choque nas últimas 8 semanas |
| `shock_max_8w` | máximo do choque nas últimas 8 semanas |
| `shock_ewm_4w` | média exponencial (α = 0,16 ≈ meia-vida 4 semanas) |

Ativadas por `--shock-memory` (default **off**, para comparação honesta), entram no desenho
sem defasagem adicional e recebem restrição monotônica positiva (`shock_*` no `ADR-0001`).

## 3. Resultado (`bt_shock_*` e `bt_need_shock_*`)

| Alvo | Modelo | MASE off | MASE on | Pico off | Pico on |
| :--- | :--- | ---: | ---: | ---: | ---: |
| UTI adulto | `gbdt_v1` | 0,9434 | **0,9428** | 5,52 | **5,40** |
| UTI adulto | `f041_local_level` | 1,1348 | 1,1351 | 8,65 | 8,62 |
| Necessidade — UTI adulto | `gbdt_v1` | 0,9592 | 0,9599 | 4,68 | **4,33** |
| Necessidade — clínico | `gbdt_v1` | 0,8842 | 0,8865 | 4,28 | 4,33 |
| Necessidade — clínico | `ridge_pooled` | 0,9546 | 0,9566 | 4,77 | **4,53** |

**Veredito:** ganho nulo em acurácia (variação `±0,3%`, dentro do ruído) e `0,1–0,3` semana de
melhora de fase no melhor caso. **Hipótese rejeitada.**

## 4. Por que falhou (e o que isso ensina)

1. O sinal de choque tem correlação fraca com ocupação (`|r| ≈ 0,1`) em **todos** os lags: a
   acumular não cria informação que não exista.
2. O pico de ocupação é **sazonal** (inverno, semanas 28–31), não um pico induzido pelo evento.
   O choque muda **amplitude**, não **fase** — por isso a memória do choque não desloca o pico.
3. O erro de pico medido por `argmax` semanal é instável em topos achatados: parte das
   `4–6` semanas pode ser ruído de medição. **Correção de métrica** (pico em série suavizada de
   3 semanas + tolerância ±1) entra no escopo do benchmark documentado (`Task 1.3.6`).

**Consequência:** duas hipóteses foram rejeitadas (alvo e memória do choque). O teto desta
base de features está em `MASE ≈ 0,88–0,96`. Ganhos adicionais exigem **dados novos**
(alvo diário, capacidade real, rótulos de evento) ou **classe de modelo** — não mais engenharia
de features sobre os mesmos sinais.

## 5. O que fica

- As colunas e a flag `--shock-memory` permanecem disponíveis (custo zero, default off).
- O resultado está documentado para evitar que a ideia seja retentada sem contexto.
- A priorização do ciclo muda: `1.3.6` (benchmark documentado) → `1.3.8` (dados) → `1.3.9`
  (arquitetura), com a promoção final só depois.

## 6. Arquivos

- `scripts/build_rs_modeling_panel.py` (4 colunas de memória)
- `src/previsao_leitos/demand/datasets.py` (`extra_feature_cols`)
- `src/previsao_leitos/demand/modeling.py` (`SHOCK_MEMORY_COLS`, `shock_memory`)
- `src/previsao_leitos/demand/two_stage.py` (prefixos monotônicos)
- `src/previsao_leitos/audit/backtest.py` e `cli/main.py` (`--shock-memory`)
- `reports/runs/bt_shock_*`, `reports/runs/bt_need_shock_*`
