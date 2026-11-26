# Feature F17 — Régua de Backtest de Origem Móvel (`Task 1.3.2`)

> **Status:** ✅ Concluída | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo de dados:** `RS`, folds `2019–2024` (holdout `2025` e `2026` intocados)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.*

---

## 1. Motivo

Até a `v1.3.0`, a escolha de modelo era feita em **um único ano** (`2024`). Com enchente sendo
evento raro, um fold favorável podia eleger o modelo errado. Esta feature troca o julgamento
pontual por uma **régua de origem móvel**: treino expansivo até `y-2`, calibração em `y-1`,
avaliação em `y`, para `y = 2019…2024`.

O ranking passa a ser **MASE mediano entre folds**, com pior fold explícito e número de vitórias —
nunca a média, que é sensível a um único ano.

---

## 2. Comando

```bash
uv run --no-sync previsao-leitos backtest \
  --target all --folds 2019-2024 \
  --exclude hierarchical_bayes \
  --run-id backtest_rs_2024
```

Saídas em `reports/runs/<run-id>/`:

| Arquivo | Conteúdo |
| :--- | :--- |
| `backtest_folds.csv` | uma linha por modelo × fold (14 métricas) |
| `backtest_summary.csv` | mediana, pior fold, rank mediano, vitórias, cobertura, pico |
| `<alvo>/relatorio_backtest_<alvo>.pdf` | diagnóstico de 5 seções |
| `<alvo>/fig_backtest_mase.png` | MASE macro por fold, com mediana marcada |
| `<alvo>/fig_error_decomposition.png` | MASE mediano vs. erro de pico (nível × fase) |

---

## 3. Métricas da régua

| Métrica | Definição | Por que |
| :--- | :--- | :--- |
| `mase_pooled` | MAE do modelo / MAE da persistência, todas as linhas | visão global |
| `mase_macro` | média **não ponderada** das razões por região (`n >= 5`, persistência `>= 1` leito-dia) | 30 regiões não têm o mesmo peso; protege contra séries quase constantes |
| `mae_macro` | média não ponderada do MAE por região | idem, em unidade clínica |
| `pinball_mean` | média dos quatro quantis | erro probabilístico agregado |
| `peak_signed_weeks` | semana do pico previsto − observado | direção do erro de fase |
| `peak_abs_weeks` | módulo do anterior | meta operacional ≤ 2 semanas |
| `shape_corr` | correlação das séries semanais agregadas | forma da curva |
| `event_mae` | MAE restrito a semanas com `prob_enchente_f03_oof >= 0,50` | desempenho no que importa |
| `bias_relative` | viés médio / média observada | nível |

**Correção registrada:** a primeira versão do MASE macro explodia (`10^6`) em regiões com
persistência quase perfeita. O denominador passou a exigir `MAE_persistencia >= 1` leito-dia e
o número de regiões usadas é reportado por fold (`26–29` de 30).

---

## 4. Resultados (`backtest_rs_2024`)

### 4.1 Ranking por mediana

| Alvo | Líder | MASE mediano | Pior fold | Cobertura mediana | Pico mediano |
| :--- | :--- | ---: | ---: | ---: | ---: |
| UTI adulto | `gbdt_v1` / `gbdt_agaci` | **0,943** | 1,108 | 71,2% / 81,4% | 5,5 sem. |
| UTI total | `gbdt_v1` | **0,943** | 1,108 | 71,2% | 5,5 sem. |
| Clínico | `gbdt_v1` | **0,924** | 1,030 | 72,9% | 4,5 sem. |
| Respiratório | `gbdt_v1` | **0,866** | 0,878 | 77,8% | 6,2 sem. |

### 4.2 Estabilidade (τ de Kendall entre folds)

`UTI adulto +0,73` · `UTI total +0,72` · `clínico +0,85` · `respiratório +0,82`

O ranking é estável o suficiente para decidir (τ ≥ 0,72 em todos os alvos). A régua substitui
o julgamento por 2024.

### 4.3 Três achados que redirecionam o esforço

1. **O modelo falha nos anos que importam.** Em `2020` e `2021` o GBDT perde da persistência
   (`MASE 1,09` e `1,06` em UTI adulto) e a cobertura cai para `49–60%`. Fora desses anos, ganha
   com folga (`0,82–0,90`). O problema não é o modelo médio: é o comportamento em choque.
2. **Erro de fase é estrutural.** O pico é acertado com `4,5–6,2` semanas de erro, enquanto a
   persistência erra `1,9–3,6`. Ou seja: ganhamos nível e perdemos fase — exatamente o inverso
   do que o planejamento de leitos precisa.
3. **O erro dobra nas semanas de choque.** `event_mae` chega a `288` pacientes-dia (clínico)
   contra `127` de MAE global, e a cobertura nessas semanas é a pior do ano.

### 4.4 Consequência para o plano

- A hipótese de **alvo errado** (ocupação realizada vs. necessidade) segue de pé e é o próximo
  passo (`Task 1.3.3`).
- A hipótese de **lag distribuído** ganha força: o pico de ocupação ocorre `6–13` semanas depois
  da enchente, mas o modelo recebe o choque contemporâneo.
- Nenhuma arquitetura nova deve ser avaliada antes de corrigir alvo e lag — a régua mostraria
  apenas ruído.

---

## 5. Limitações

- `2020` e `2021` são anos atípicos (COVID) e entram na régua; eles dominam o pior fold. A
  alternativa — excluí-los — esconderia justamente o regime que queremos prever.
- A régua mede ocupação realizada, não necessidade; será reexecutada sobre a série de
  necessidade quando ela existir.
- Bayes hierárquico foi excluído do backtest por custo (`~4 min × 6 folds × 4 alvos`); ele
  permanece avaliado no run anual.

---

## 6. Arquivos

- `src/previsao_leitos/audit/backtest.py` (folds, métricas, sumário)
- `src/previsao_leitos/audit/backtest_report.py` (figuras, HTML, PDF)
- `src/previsao_leitos/cli/main.py` (`backtest`)
- `tests/unit/test_demand_registry_and_benchmark.py` (folds, bloqueio de holdout)
- `reports/runs/backtest_rs_2024/`
