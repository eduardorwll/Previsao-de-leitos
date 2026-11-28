# Feature F29 — Diagnóstico de higiene/fase e início da V2 (LOS com riscos competitivos)

> **Status:** 📋 Diagnóstico + Task 2.2 implementada | **Etapa:** Transição V1.4 → V2
> **Fronteira:** *Not for patient care or live clinical use.* | **Holdout:** `2025`/`2026` intocados

## 1. Por que os Gates de higiene falham

Os dois `FAIL` do dossiê M1 têm causas **distintas e diagnosticadas**:

### G1 — VIF 5,31 (`chuva_sem_mm` × `chuva_catchment_mm`)
- **Causa:** `chuva_catchment_mm` (chuva da área de captação = local + montante) **contém**
  `chuva_sem_mm` (chuva local). Correlação `r = 0,901` (`R² = 0,812`) → `VIF = 1/(1-R²) = 5,31`,
  logo acima do limite `5,0`. É colinearidade **mecânica**, não erro de dados.
- **Correção:** (a) usar apenas `chuva_catchment_mm` (remover a redundante local), ou
  (b) ortogonalizar em `local` + `montante_puro` (`catchment − local`), ou (c) aceitar `WARN`.
  Recomendação: **(b)** — preserva o sinal local e o de montante sem colinearidade.

### G4 — Censura à direita 37,9% (últimas 4 semanas)
- **Causa:** o painel completo vai até **2025**, e a cauda de 2025 (semanas 49–52) cai de
  `6.336 → 4.136` diárias por **atraso de processamento do SIH** (censura à direita). No painel
  `<= 2024`, o G4 cai para **`WARN` (11,5%)** — ou seja, o `FAIL` é **artefato do holdout 2025**,
  não do modelo.
- **Correção:** (a) calcular G4 apenas em `<= 2024` (o holdout já é lacrado), ou (b) aplicar o
  `ChainLadderNowcaster` (`demand/nowcasting.py`) para reconstruir a cauda. Recomendação: **(a)
  para o dossiê + (b) para uso prospectivo no M2**.

## 2. Por que a previsão parece "perfeita, mas atrasada"

Análise de **correlação cruzada** entre o `P50` previsto e o observado (benchmark `RS 2024`):

| Alvo | corr no alvo (lag 0) | melhor lag | corr no melhor lag |
| :--- | ---: | ---: | ---: |
| UTI adulto | 0,842 | **+2 sem.** | 0,989 |
| Clínico | 0,804 | **+2 sem.** | 0,975 |
| Respiratório | 0,796 | **+2 sem.** | 0,924 |
| `persistencia` (baseline) | — | +2 sem. | **1,000** |

**Diagnóstico:** o campeão é **praticamente persistência com lag de 2 semanas** (o horizonte
`h = 2`): ele "ecoa" o valor atual 2 semanas atrás (corr `0,989` vs. `1,000` da persistência). O
`MASE` de `0,94–0,98` confirma que o ganho sobre a persistência é marginal. O formato é perfeito
porque **é a curva atual deslocada** — não há antecipação real do pico.

## 3. Ideias (não ortodoxas) para adiantar a previsão

1. **Modelar o delta, não o nível.** Treinar o alvo `y_{t+h} − y_t` (variação) e reconstruir
   `ŷ_{t+h} = y_t + Δ̂`; penalizar/remover `y_origin`/`y_roll*` do ramo principal força o modelo a
   usar sinal de antecipação. *(testável na régua; alto potencial se o Δ for previsível.)*
2. **Correção de fase causal (cross-correlation).** Estimar o lag `k` que maximiza a correlação
   previsto×observado no ano de **calibração (2023)** e deslocar a previsão de `−k` semanas no ano
   de avaliação. *(barato; só funciona se o lag for estável entre anos.)*
3. **Indicador líder epidêmico (SIVEP-Gripe).** A admissão por SRAG antecede o pico de leitos em
   1–3 semanas; quebra a persistência ao trazer sinal *ex-ante*. *(bloqueado no acesso; alavanca nº 1.)*
4. **Derivada/momentum como detecção de subida.** Usar a taxa de variação recente (`Δy_t`) para
   detectar a fase de subida e **adiantar o pico** por extrapolação monotônica restrita.
5. **Ensemble de modelos com deslocamento de fase.** Combinar previsões em `+0` e `+k` semanas,
   ponderando pela estabilidade do lag observada em validação.

> **Prioridade:** (3) SIVEP-Gripe quando liberar → (1) delta → (2) correção de fase causal. Cada
> uma entra na régua de 6 folds com critério `MASE −2%` ou `erro de fase −1 sem.`, `0` violações.

## 4. V2 — Task 2.2 implementada (LOS com riscos competitivos, sem GPL)

- **`AggregatedCompetingRisksLosEstimator`** (`demand/los.py`): hazard causa-específico
  não-paramétrico por bucket de dia (`h_k = eventos_k / em_risco`) + incidência acumulada de
  **Aalen-Johansen** (`CIF_k(d) = Σ S(s−1)·h_k(s)`), separando **alta viva vs. óbito**. Entrada
  **agregada** (`bucket_dia_internacao, em_risco, alta_viva, obito`) → **LGPD-safe** (sem AIH
  individual), acoplável ao `SimPy` via Lei de Little.
- **`scikit-survival` (GPL-3.0) removido** de `pyproject.toml` e `uv.lock` (e deps transitivas
  `ecos`, `numexpr`, `osqp`), conforme `ADR-0005`.
- **Testes:** `tests/unit/test_demand_competing_risks.py` (CIF, LOS condicional, normalização de
  hazard, filtro causal, colunas ausentes).

## 5. Próximos passos V2

- **Task 2.1:** remover `dask` do extra `scale` (verificar dependência transitiva de `aeon`) e
  desenhar o data lake `Polars` em 2 níveis (`< 16 GiB` global / `< 6 GiB` streaming).
- **Task 2.2 (query BQ):** materializar o agregado `(cnes, semana_epi, bucket, em_risco, alta, obito)`
  com supressão `n < 5` e acoplar ao `SimPy`.
- **Task 2.3:** benchmark nacional *Leave-UF/Basin-Out* (`F04.1` vs. `Hetero-ST-GNN`/`HydroTFTNet`).

## 6. Arquivos

- `src/previsao_leitos/demand/los.py` (`AggregatedCompetingRisksLosEstimator`)
- `src/previsao_leitos/demand/__init__.py` (export)
- `pyproject.toml` + `uv.lock` (remoção do `scikit-survival`)
- `tests/unit/test_demand_competing_risks.py`
