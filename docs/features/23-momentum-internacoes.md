# Feature F23 — Momentum de Internações por Causa (`Task 1.3.9b`)

> **Status:** ⚠️ Ganho específico (não promove) | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

## 1. Contexto

Com BigQuery desbloqueado (projeto `previsao-leitos`), a busca por S2iD/SIVEP esbarrou em
`403/404` sem a API key da Base dos Dados. Em vez de gastar orçamento às cegas, testamos uma
fonte **já presente no painel e não usada como feature**: internações por causa.

## 2. Hipótese

As internações da semana `t` (total, respiratória, leptospirose, trauma) carregam *momentum*
assistencial que antecipa a ocupação em `t+2` — informação que o GBDT talvez já capture, mas
que o F04.1 (harmônicos + choque) não tem.

Ativação: `--admissions-momentum` (default off), com defasagens `0,1,2` (causal).

## 3. Resultado

| Alvo | Modelo | MASE off | MASE on | Δ |
| :--- | :--- | ---: | ---: | ---: |
| UTI adulto | `gbdt_v1` | 0,9434 | **0,9406** | −0,3% |
| Clínico | `gbdt_v1` | **0,9243** | 0,9314 | +0,8% (pior) |
| Respiratório | `gbdt_v1` | **0,8662** | 0,8676 | +0,2% (pior) |
| Respiratório | `f041_local_level` | 0,9396 | **0,8860** | **−5,7%** |
| Necessidade — clínico | `gbdt_v1` | **0,8842** | 0,8891 | +0,6% (pior) |
| Necessidade — respiratório | `f041_local_level` | 0,9566 | **0,9026** | **−5,6%** |

## 4. Leitura

1. **O F04.1 ganha onde mais precisa.** No alvo respiratório — o de pior erro percentual
   (`wMAPE ≈ 35%`) — o momentum reduz o MASE em `5,6–5,7%` e o `wMAPE` de `40,4%` para `37,0%`.
   É o primeiro feature que move o challenger de forma material.
2. **O GBDT é saturado.** Ganha `0,3%` em UTI e perde `0,2–0,8%` nos demais: a informação já
   está capturada por `y_roll*` e pelas covariáveis climáticas.
3. **Não promove.** O GBDT segue à frente em todos os alvos; o F04.1 melhora, mas não alcança.

## 5. Decisão

- Flag mantida (**default off**) para não alterar o campeão nem os runs comparáveis.
- Registrada como evidência de que o gargalo do F04.1 é **sinal de dinâmica**, não calibração.
- Reprodução: `backtest --admissions-momentum` nos alvos de interesse.

## 6. Arquivos

- `src/previsao_leitos/demand/modeling.py` (`ADMISSIONS_MOMENTUM_COLS`, flag)
- `src/previsao_leitos/audit/backtest.py`, `src/previsao_leitos/cli/main.py` (`--admissions-momentum`)
- `reports/runs/bt_adm_*`, `reports/runs/bt_need_adm_*`
