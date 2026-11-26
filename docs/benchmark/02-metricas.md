# 2. Catálogo de Métricas

Cada métrica responde a uma pergunta específica. Nenhuma delas, isolada, autoriza promoção.

## 2.1 Métricas principais (aparecem no cartão)

### wMAPE
- **Pergunta:** qual fração do volume observado o modelo erra?
- **Fórmula:** `100 × Σ|pred − obs| / Σ|obs|`
- **Unidade:** % do volume observado. **Direção:** menor melhor.
- **Casos-limite:** regiões pequenas não distorcem (pondera por volume).
- **Não diz:** nada sobre fase nem sobre calibração do intervalo.

### MASE
- **Pergunta:** o modelo bate a persistência (repetir a semana de origem)?
- **Fórmula:** `MAE_modelo / MAE_persistência`
- **Unidade:** adimensional. **Direção:** menor melhor; `< 1,00` é o mínimo para valer.
- **Casos-limite:** empate exato com a persistência = `1,00`; não usar entre alvos.
- **Não diz:** se o intervalo é honesto nem se o pico está na semana certa.

### Fase (erro de pico)
- **Pergunta:** o modelo avisa **quando** o pico chega?
- **Fórmula:** `|semana(argmax previsto) − semana(argmax observado)|`, média entre regiões.
- **Unidade:** semanas. **Direção:** menor melhor; meta operacional ≤ 2.
- **Casos-limite:** topo achatado torna `argmax` instável — leia junto com o MASE.
- **Não diz:** magnitude do erro.

### Cobertura P10–P90
- **Pergunta:** o intervalo previsto contém o valor real com que frequência?
- **Fórmula:** `média( P10 <= obs <= P90 )`
- **Unidade:** fração. **Direção:** alvo `80%`; desvio para cima = conservador, para baixo =
  confiança excessiva.
- **Não diz:** se o P50 está bom (um intervalo largo cobre com P50 ruim).

### P90 do erro percentual
- **Pergunta:** qual o pior caso típico?
- **Fórmula:** percentil 90 de `100 × |pred − obs| / max(obs, piso)`
- **Unidade:** %. **Direção:** menor melhor.
- **Uso:** dimensiona margem de segurança; complementa a mediana (que esconde a cauda).

## 2.2 Métricas de apoio

| Métrica | Fórmula | Unidade | Direção | Pergunta |
| :--- | :--- | :--- | :--- | :--- |
| `erro_pct_mediana` | mediana do erro % por linha | % | menor | "metade das semanas erra menos que X%" |
| `subestimacao_pct` | fração de `pred < obs` | fração | ~50% | "erra para menos?" |
| `erro_pct_evento` | mediana do erro % em semanas de choque ≥ 0,5 | % | menor | "e na enchente?" |
| `pinball_mean` | média dos quatro quantis | unidade do alvo | menor | erro probabilístico agregado |
| `pinball_p95` | perda no quantil 95 | unidade do alvo | menor | cauda superior |
| `mae_pooled` / `mae_macro` | MAE global / média por região | unidade do alvo | menor | magnitude |
| `bias_relative` | `média(pred − obs) / média(obs)` | fração | ~0 | viés sistemático |
| `shape_corr` | correlação das séries semanais | adimensional | maior | forma da curva |
| `monotonic_violations` | quedas de P50 quando o choque sobe | contagem | 0 | ADR-0001 |
| `fit_seconds` | tempo de ajuste | s | menor | custo |

## 2.3 Regras de leitura

1. **MASE sozinho não promove.** Exige fase, cobertura e robustez.
2. **Cobertura fora de 78–82%** pede recalibração antes de qualquer conclusão.
3. **`subestimacao_pct` muito acima de 50%** indica viés estrutural de subprevisão — o pior
   erro para capacidade hospitalar.
4. **Comparação entre alvos é inválida** (unidades diferentes).
