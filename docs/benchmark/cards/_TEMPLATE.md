# <modelo>

**Selo geral:** `<A|B|C|D|Reprovado|N/A>` · **Rubrica:** `v1` · **Família:** `<família>`

**Como ler estas métricas**

- **wMAPE** — erro percentual ponderado: soma dos erros ÷ soma dos leitos observados.
  Responde *quanto do volume real se erra*; robusto a regiões pequenas; menor é melhor.
- **MASE** — erro do modelo ÷ erro da persistência (repetir a semana de origem).
  `1,00` = empatar com o ingênuo; `0,90` = 10% melhor. Abaixo de `1,00` é o mínimo.
- **Fase** — distância, em semanas, entre o pico previsto e o observado. Mede se o modelo
  avisa *quando* o pico chega, não só o tamanho. Alvo operacional: ≤ 2 semanas.
- **Cobertura P10–P90** — fração de semanas com o valor real dentro do intervalo previsto.
  Nominal 80%: muito acima = intervalo conservador; muito abaixo = confiança excessiva.
- **P90 do erro** — erro percentual superado em apenas 10% das semanas: o *pior caso típico*.

## Resultados por alvo

| Alvo | wMAPE | MASE | Fase (sem.) | Cobertura | P90 do erro | Subestimação |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| <alvo> | <%> | <mase> | <sem> | <%> | <%> | <%> |

## Categorias

| Alvo | Acurácia | Fase | Calibração | Robustez | Consistência | Custo | Selo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| <alvo> | <A-D> | <A-D> | <A-D> | <A-D> | <A-D> | <A-D> | **<selo>** |

## Leitura

- Selo geral (pior caso entre alvos): **<selo>**.
- Ponto mais fraco: <justificativa>.
- wMAPE e MASE medem tamanho do erro; fase mede o *quando*; cobertura e P90 medem a confiança.

## Reprodutibilidade

- Run: `<run-id>`
- Git: `<sha>` · seed `<seed>`
- Painel SHA-256: `<sha256>`

> Not for patient care or live clinical use. LGPD: agregação e supressão n < 5.
