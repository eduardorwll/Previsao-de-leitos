# Feature F30 — Estratégias não ortodoxas para adiantar a previsão: estudo aprofundado

> **Status:** 📋 Estudo concluído (com evidência empírica) | **Etapa:** Transição V1.4 → V2
> **Fronteira:** *Not for patient care or live clinical use.* | **Holdout:** `2025`/`2026` intocados

## 1. O que a observação "formato perfeito, só atrasado" realmente é

O campeão `gbdt_conformal` correlaciona `0,99` com o observado quando deslocado **+2 semanas**
(h vs. `persistencia` = `1,000`). Ou seja, ele **ecoa o valor atual** — é uma persistência com o
horizonte `h = 2`. O erro de fase decompõe-se **integralmente nas features** (benchmark `RS 2024`):

| Alvo | pico observado | `y_origin` | `y_roll4` | `y_roll12` |
| :--- | ---: | ---: | ---: | ---: |
| UTI adulto | W30 | W32 (**+2**) | W34 (**+4**) | W36 (**+6**) |
| Clínico | W31 | W33 (**+2**) | W34 (**+3**) | W36 (**+5**) |
| Respiratório | W30 | W32 (**+2**) | W34 (**+4**) | W38 (**+8**) |

**Conclusão:** o atraso não é "epidemia imprevisível" — é **engenharia de features**. `y_origin`
é o valor atual (defasado em `h=2`); `y_roll4`/`y_roll12` são médias móveis, cujo pico chega 4–8
semanas depois (centroide da janela). O modelo prevê o futuro com features do passado → prevê o
passado.

## 2. Evidência: o "delta" é ruído (por que modelar a variação não salva)

`target_delta = y_{t+2} − y_t` (a variação que o modelo precisaria antecipar):

| Alvo | `std(delta)/std(nível)` | corr com `y_origin` | corr com `chuva` | corr com `prob_enchente` | corr com ENSO |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UTI adulto | 7,7% | −0,03 | +0,03 | +0,00 | +0,00 |
| Clínico | 7,7% | −0,04 | +0,04 | +0,00 | +0,00 |
| Respiratório | 42,7% | −0,21 | +0,02 | +0,01 | +0,00 |

O delta é **quase ortogonal às features** (corr `< 0,05` em UTI/clínico). Por isso o
`PooledRidgeDelta` (que reconstrói `y_origin + Δ̂`) empata com a persistência. **A variação futura
não é previsível com o que temos** — só um **indicador líder exógeno** pode quebrar a persistência.

## 3. Estratégias não ortodoxas — análise honesta (impacto × risco × teste)

| # | Estratégia | Mecanismo | Impacto esperado | Risco | Veredito |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Modelar o delta** | Prever `Δ = y_{t+h} − y_t` e reconstruir | Baixo — `Δ` é ruído (corr < 0,05) | Regressão se o delta-amostra for enviesado | **Descartar** (evidência acima) |
| 2 | **Extrapolação de tendência** | `y_origin + k·(y_origin − y_roll4)` como feature | Nulo — testado: pico fica em `+2` para todo `k` | GBDT não extrapola (árvores) | **Descartar** |
| 3 | **Correção de fase *a posteriori*** | Estimar lag na calibração e deslocar `−k` | **Negativo** — deslocar `−2` usa dado *mais antigo* = pior | Vazamento se estimado no eval | **Descartar** (é red herring) |
| 4 | **Nowcast assumido** | Apresentar como estimativa do *estado atual*, não previsão | Honesto — é o que o modelo faz | Muda a promessa (não é previsão `h=2`) | **Adotar na comunicação** |
| 5 | **Indicador líder exógeno** (vigilância viral/wastewater/temperatura) | Sinal que **precede** a internação (não a acompanha) | **Alto** — único caminho real | Depende de acesso/qualidade do dado | **Único caminho para previsão** |
| 6 | **Momentum/derivada** | Usar `Δy_t` para detectar subida | Baixo (≈ delta, ruído) | Idem | **Descartar** |
| 7 | **Ensemble de fases** | Combinar modelos deslocados | Baixo — membros correlacionados (F22) | Sem diversificação | **Já rejeitado (F22)** |

> **Achado central:** `SIVEP-Gripe` **não é** indicador líder por si só (SRAG = a própria
> internação). O que precede a demanda é a **vigilância viral/laboratorial** e o **frio** (temperatura).
> A temperatura (`temp_media_sem_c`) já está no painel, mas **não** é feature de demanda — é um
> candidato barato e ainda não testado.

## 4. Métodos de teste (para nunca commitar uma versão pior)

Toda estratégia nova entra **opt-in (default off)** e passa pela mesma régua, sem exceção:

1. **Régua de 6 folds** (`2019–2024`, `train ≤ y−2`, calibração `y−1`): MASE mediano, erro de pico
   (sinal + módulo), MASE macro, cobertura `P10–P90`.
2. **Catraca *Champion–Challenger*** (`audit/hypothesis_ledger.py`): promoção **só** se
   `MASE mediano ≤ campeão −2%` **ou** `erro de fase ≤ campeão −1 semana`, **com** `0` violações
   monotônicas (`ADR-0001`) e **nenhum fold** com regressão `> +5%`.
3. **Pré-registro da hipótese** (`docs/preregistro/` + `hypothesis_ledger.json`): hipótese, teste
   discriminante e catraca escritos **antes** de rodar (como E1/E2/E3 do ENSO).
4. **Ablação A/B** com a flag on/off no **mesmo painel e folds**, registrando `ablation_delta.json`.
5. **Bloqueio de regressão**: se qualquer critério falhar, a hipótese é **rejeitada e documentada**
   (padrão F20/F22/F23/F26/F27/F28) — nunca promovida por "parecer razoável".

## 5. Recomendação priorizada

1. **Temp. média como feature** (`--temperature`): barato, já no painel, testável na régua — checar se
   o frio antecipa a subida respiratória.
2. **Vigilância viral** (laboratório/wastewater) e **SIVEP** quando liberados — único caminho para
   previsão real `h=2`.
3. **Reenquadrar o produto**: enquanto não houver líder, o modelo é **nowcast agregado** (estado
   atual), não previsão de pico — comunicar isso no README e nos relatórios.
4. **V2**: `HydroTFTNet`/`Hetero-ST-GNN` com covariáveis epidêmicas (atenção aprende *timing*).

## 6. Arquivos

- Evidências: `reports/runs/rs_v1_3_benchmark_2024/predictions.parquet`, `rs_modeling_panel_v3.parquet`
- Modelo delta existente: `src/previsao_leitos/demand/models.py` (`PooledRidgeDeltaForecaster`)
- Catraca: `src/previsao_leitos/audit/hypothesis_ledger.py`
- Diagnóstico de higiene/fase: [29-diagnostico-higiene-fase-e-v2.md](29-diagnostico-higiene-fase-e-v2.md)
