# Feature F14 — Fronteira de Pareto Multiobjetivo (`pymoo` NSGA-II + `SimPy` CRN 2 Estágios) e Otimização `_capacity_blocker` (`Task 1.2.5`)

> **Status:** ✅ Concluída (`RS <= 2024`, holdout `2025` e `2026` preservados intocados) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

1. Eliminar o gargalo de recriação diária de corrotinas em `_capacity_blocker` no simulador `SimPy` (`src/previsao_leitos/readiness/simulator.py`), medindo antes e depois em [PERF.md](../../PERF.md).
2. Substituir a política fixa única da V1 (`trigger = 0,90`, `deferral = 0,75`, `surge = 20%`) por uma **fronteira de Pareto multiobjetivo (`pymoo` NSGA-II)** auditada em **2 estágios CRN independentes** (`R_search = 6` na busca → `R_audit = 30` replicações disjuntas na auditoria), testando a hipótese `H-1.2.5` (`>= 20%` menos eletivas adiadas mantendo redução de fila `>= 35%` sem degradar o pico de fila da V1).

---

## 2. Alterações Implementadas

- **Refatoração Medida de `_capacity_blocker` e Suporte CRN (`src/previsao_leitos/readiness/simulator.py`):**
  - Substituídos os bloqueios temporários diários por **bloqueadores persistentes diferenciais** (`_persistent_capacity_blocker` com `simpy.Event`): tokens só são adquiridos/liberados quando a capacidade indisponível efetivamente muda entre dias consecutivos.
  - Ganho medido em `10×` CRN POA (`431` base + `86` surge, `182` dias sob estresse): **`3,476 s → 1,707 s` (`2,04×` mais rápido)** com saídas **bit-idênticas** (`react_q = 68, pro_q = 0, def = 2196, ov_saved = 43`).
  - Adicionados `trigger_occ_threshold`, `hysteresis_close_threshold` em `HospitalReadinessConfig`, `surge_bed_days_active` em `PolicySimulationResult` e utilitários CRN (`CrnScenarioStream`, `generate_crn_scenarios`, `evaluate_policy_on_crn`).
- **Otimizador Pareto NSGA-II em 2 Estágios CRN (`src/previsao_leitos/readiness/pareto.py`):**
  - `optimize_contingency_pareto_front` otimiza `4` alavancas (`trigger_occ`, `close_occ`, `elective_deferral_frac`, `surge_beds_frac`) minimizando `3` objetivos (`pico_fila_medio`, `eletivas_adiadas`, `leitos_dia_surge`).
  - Execução paralela multi-core segura com Polars/Rayon no Linux via `multiprocessing.get_context("spawn")` + inicializador de worker (IPC de apenas 4 floats por candidato, evitando deadlock `fork()`).
  - Auditoria Estágio 2 sobre `R_audit = 30` replicações CRN estritamente disjuntas + seleção de joelho seguro (`pico_fila <= V1`).
- **Orquestração, Dossiê `V1.2.5` e Testes (`scripts/run_rs_pareto_contingency.py`, `scripts/run_rs_v1_1_dossier.py`, `tests/unit/test_v1_2_5_pareto_simpy.py`):**
  - Integração automática da tabela Pareto no Dossiê `rs_v1_2_5_pareto_2024` (`V1.2.5`) e cobertura unitária validando determinismo CRN, não-dominância estrita e histerese válida.

---

## 3. Resultados Reais Auditados (`RS 2024`, `pop=32, gen=16, R_search=6 → R_audit=30`, 8 workers)

| Região de Saúde RS (`CNES 2024`) | Política | `trigger` / `close` / `defer` / `surge` | Pico Fila (médio) | Red. Fila vs. `H0` | Red. Eletivas vs. `V1` | Red. Leitos-Dia Surge vs. `V1` | Tempo (8 cores) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Região 10 — Capital e Vale do Gravataí** (`431` UTI SUS) | `Reactive_H0` | — | `58,00` | `0,0%` | `+53,5%` | `+66,9%` | — |
| **Região 10 — Capital e Vale do Gravataí** | `V1_Fixed_H2` | `0,90 / 0,85 / 0,75 / 86` | `20,53` | `64,6%` | `0,0%` | `0,0%` | — |
| **Região 10 — Capital e Vale do Gravataí** | **`Pareto_03` (Joelho)** | **`0,799 / 0,769 / 0,177 / 125`** | **`1,00`** | **`98,3%`** | **`+65,0%` menos** | `-139,2%` | **`86,1 s`** (`32` sol.) |
| **Região 21 — Sul** (`110` UTI SUS) | `Reactive_H0` | — | `15,10` | `0,0%` | `+68,0%` | `+81,3%` | — |
| **Região 21 — Sul** | `V1_Fixed_H2` | `0,90 / 0,85 / 0,75 / 22` | `8,13` | `46,1%` | `0,0%` | `0,0%` | — |
| **Região 21 — Sul** | **`Pareto_17` (Joelho)** | **`0,795 / 0,756 / 0,214 / 29`** | **`1,30`** | **`91,4%`** | **`+55,8%` menos** | `-139,7%` | **`37,6 s`** (`30` sol.) |

- **Veredito científico (`H-1.2.5` confirmada em 2 objetivos críticos):** Em ambas as regiões, a fronteira Pareto dominou estritamente a política fixa V1 em **segurança assistencial** (`pico de fila 20,53 → 1,00` em POA e `8,13 → 1,30` no Sul, superando `>= 35%` de redução vs. `H0`) **e** em **continuidade de cirurgias eletivas** (`-65,0%` e `-55,8%` menos adiamentos, superando amplamente a meta `>= 20%`). O terceiro objetivo (`leitos_dia_surge`) explicita honestamente o custo físico real de mobilização necessária para preservar eletivas sem formar filas.
- **Próximo passo:** `Task 1.2.6` — Explicabilidade auditada (`SHAP 0.54.1` 2 Estágios) + experimento Bayesiano hierárquico (`PyMC` + `nutpie` + `ArviZ`) + fechamento do `Marco M1`.
