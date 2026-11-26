# Benchmark de Modelos de Demanda de Leitos

Documentação do benchmark que compara modelos de previsão de leitos no Rio Grande do Sul,
com avaliação causal (origem móvel), métricas em unidades clínicas e **rubrica congelada** de
categorização.

> **Fronteira:** *Not for patient care or live clinical use.* Pesquisa e avaliação
> metodológica apenas. LGPD: zero PII, agregação na fronteira, supressão `n < 5`.

---

## Por onde começar

| Se você quer… | Leia |
| :--- | :--- |
| entender **como** os modelos são avaliados | [01-metodologia.md](01-metodologia.md) |
| saber o que cada número significa | [02-metricas.md](02-metricas.md) |
| entender o erro em **% do real** | [03-erro-percentual.md](03-erro-percentual.md) |
| saber por que um modelo tirou nota C | [04-categorizacao.md](04-categorizacao.md) |
| encontrar um arquivo de saída | [05-artefatos.md](05-artefatos.md) |
| adicionar modelo, métrica ou alvo | [06-como-estender.md](06-como-estender.md) |
| interpretar um cartão de modelo | [07-leitura-de-cards.md](07-leitura-de-cards.md) |
| ver os resultados atuais | [RESULTADOS.md](RESULTADOS.md) |

---

## Princípios

1. **A régua não se move.** Limiares e fórmulas são versionados (`rubrica v1`, pré-registros
   `v1→v3`). Mudar depois de ver o resultado exige nova versão e reexecução completa.
2. **Nível e fase são critérios separados.** Acertar a média e errar a semana do pico reprova
   para planejamento de capacidade.
3. **Nada é omitido.** Modelo sem dependência aparece como `N/A`; modelo que falha aparece
   com o erro, nunca desaparece da tabela.
4. **O holdout é sagrado.** `2025` e `2026` não entram em nenhum fold; a validação final é o
   `Marco M2`.
5. **Toda decisão é reproduzível.** `run_manifest.json` guarda git SHA, seed e SHA-256 do
   painel; o comando está no cartão.

---

## Comandos

```bash
# benchmark de um ano (rápido) e relatório
uv run --no-sync previsao-leitos run --target all --models all --eval-year 2024 --jobs 4

# régua de origem móvel (6 folds) + diagnóstico
uv run --no-sync previsao-leitos backtest --target all --folds 2019-2024

# regerar figuras, cartões e PDF sem retreinar
uv run --no-sync previsao-leitos report --run-id <run-id>
```

---

## Versões

| Item | Versão | Alteração exige |
| :--- | :--- | :--- |
| Rubrica de categorização | `v1` | nova rubrica `v2` + rerun |
| Série de necessidade | `v3` | novo pré-registro `v4` |
| Métricas | `v1` | documentar e versionar |

