# 3. Erro percentual vs. dados reais

## 3.1 Definição

Para cada linha `i` (região × semana):

```
erro_pct_i = 100 × |pred_i − obs_i| / max(obs_i, piso)
piso      = 5% da média observada do conjunto
```

O piso existe porque `obs ≈ 0` transforma qualquer erro em percentual infinito — o defeito
clássico do MAPE. Com piso, uma região de 2 leitos não domina a métrica de 300.

## 3.2 Por que não MAPE

| Métrica | Comportamento em `obs → 0` | Ponderação |
| :--- | :--- | :--- |
| MAPE clássico | explode; média inútil | igual por linha |
| **wMAPE** | estável | pondera por volume |
| **mediana do erro %** | estável | robusta a outliers |
| **P90 do erro %** | estável | captura cauda |

## 3.3 Como ler

- **Mediana**: "em metade das semanas o erro fica abaixo de X%".
- **P90**: "em 1 a cada 10 semanas o erro supera Y%" — é a margem de segurança.
- **wMAPE**: "no agregado, erramos Z% do volume" — compara modelos sem ser enganado por
  regiões minúsculas.
- **`subestimacao_pct`**: fração de semanas em que o modelo previu **menos** que o real.
  Para leitos, subestimar é mais grave que superestimar; valores acima de ~55% acendem alerta.
- **`erro_pct_evento`**: mesma mediana, restrita às semanas com `prob_enchente >= 0,5`. Se for
  muito maior que a mediana geral, o modelo piora justamente no evento.

## 3.4 Exemplo de leitura

```
Modelo A:  wMAPE 9,1% · mediana 8,4% · p90 18,3% · subestimação 47% · evento 14,2%
Modelo B:  wMAPE 8,7% · mediana 12,1% · p90 31,0% · subestimação 61% · evento 22,5%
```

`B` tem wMAPE menor (melhor no agregado), mas **erra mais na mediana, tem cauda pior e
subestima 61% das semanas**. Para capacidade hospitalar, `A` é a escolha defensável.

## 3.5 Onde aparece

- `benchmark.csv` e `backtest_folds.csv`: colunas `wmape`, `erro_pct_mediana`,
  `erro_pct_p90`, `subestimacao_pct`, `erro_pct_evento`.
- `backtest_summary.csv`: medianas entre folds (`wmape_median`, `erro_pct_mediana`, …).
- PDF: seção "4. Erro percentual vs. dados reais", com tabela e figura de distribuição.
- Cartões: colunas wMAPE, P90 e subestimação por alvo.
