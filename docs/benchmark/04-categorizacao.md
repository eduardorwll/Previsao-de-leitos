# 4. Categorização (rubrica v1)

Cada modelo recebe uma **letra por quesito** e um **selo geral**. A rubrica está congelada:
mudar limiares exige `v2` e reexecução completa.

## 4.1 Tabela de limiares

| Critério | Métrica | A | B | C | D |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Acurácia | `wmape` | ≤ 8% | ≤ 12% | ≤ 20% | > 20% |
| Fase | `peak_abs_weeks` | ≤ 2 | ≤ 4 | ≤ 6 | > 6 |
| Calibração | `\|cobertura − 80%\|` | ≤ 3 p.p. | ≤ 7 p.p. | ≤ 12 p.p. | > 12 p.p. |
| Robustez | `mase_worst` | ≤ 1,00 | ≤ 1,10 | ≤ 1,25 | > 1,25 |
| Consistência | `mase_cv` (desvio/média entre folds) | ≤ 0,05 | ≤ 0,10 | ≤ 0,20 | > 0,20 |
| Custo | `fit_seconds` | ≤ 1 s | ≤ 10 s | ≤ 60 s | > 60 s |

## 4.2 Regra do selo

1. **Selo = pior** entre Acurácia, Fase e Calibração. Não há compensação: acurácia alta não
   salva fase ruim.
2. **Violação monotônica > 0** ⇒ selo `Reprovado`, independentemente das notas.
3. **Dependência ausente** ⇒ `N/A`. O modelo permanece listado.
4. Robustez, consistência e custo **informam**, mas não puxam o selo.

## 4.3 Exemplos comentados

| Caso | Acurácia | Fase | Calibração | Selo | Leitura |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Modelo preciso e pontual | A | A | A | **A** | pronto para disputar promoção |
| Preciso, mas atrasado | A | D | B | **D** | inútil para planejar pico |
| Preciso e pontual, intervalo otimista | A | A | D | **D** | cobertura 60%: falso conforto |
| Monotonicidade violada | A | A | A | **Reprovado** | quebra `ADR-0001` |

## 4.4 Política de versão

- `v1` (atual): limiares acima.
- Qualquer alteração: registrar `v2`, justificar antes de rodar e reexecutar **todos** os runs
  de referência. Nunca ajustar limiar para fazer um modelo passar.
