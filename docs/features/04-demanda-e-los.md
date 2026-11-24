# Feature F04 — Demanda Semanal Multi-Horizonte, Contagem (Onda 2) e LOS

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída (Etapa 2 — Modelos Globais RS 2024 + LOS Causal) |
| **Etapa** | Etapa 2 — Pivô Científico RS |
| **Especificação** | [spec/SPEC-demand-models.md](../../spec/SPEC-demand-models.md) |
| **Pacote Implementado** | `src/previsao_leitos/demand/` + `scripts/run_rs_etapa2_verification.py` |
| **Verificação** | `ruff` OK · `mypy --strict` OK · `check_constraints.py` OK · `66` testes (`89%+` cov.) · `pip-audit` OK |

> **Fronteira Científica:** Pesquisa metodológica e avaliação *ex-ante* (*Not for patient care or live clinical use*). O *holdout* **`2025`** permanece 100% intocado.

---

## 1. Arquitetura Entregue (`src/previsao_leitos/demand/`)

- **[datasets.py](../../src/previsao_leitos/demand/datasets.py) (`94%` cov.):** painel causal global (`~18.000` semanas-região de treino `<= 2022` nas 30 Regiões do RS), unindo lags hidrológicos distribuídos (`0..4` sem.), roteamento ponderado + remanso lagunar (`V10M`), atributos estruturais de áreas aterradas/pôlderes (`MapBiomas <= 2022`), rede `W_ref` e estatísticas autorregressivas causais (`y_roll4`, `y_roll12`).
- **[models.py](../../src/previsao_leitos/demand/models.py) (`90%` cov.):**
  - `QuantileGBDTForecaster` (Location-Scale GBDT com `loss='squared_error'` + `monotonic_cst` — garantindo **`0,00%` de violações monotônicas** e não-cruzamento $0 \le P_{10} \le P_{50} \le P_{90} \le P_{95}$, evitando a quebra de monotonicidade do `update_leaves_values` de `loss='quantile'` no `scikit-learn`);
  - `NegativeBinomialHurdleForecaster` (`statsmodels` GLM Binomial + Binomial Negativa com quantis analíticos `scipy.stats.nbinom` para contagens esparsas de cauda pesada na Onda 2);
  - `PooledRidgeDeltaForecaster` (baseline linear global).
- **[los.py](../../src/previsao_leitos/demand/los.py):** `SpecialtyLosEstimator` causal (`<= 2022`) com conversão para leitos ocupados/dia pela Lei de Little (`L = D / 7`): `uti_adulto = 12,74 dias` (`6,46 diárias UTI`), `uti_neonatal = 17,01 dias`, `uti_pediatrica = 15,67 dias`, `clinico = 6,61 dias`, `cirurgico = 3,42 dias`.

---

## 2. Benchmark Real RS 2024 (`h = 2` semanas, Treino `<= 2022`, *Block-Bootstrap* 95% CI)

| Alvo (30 Regiões RS) | Melhor Modelo F04 | MAE F02 | MAE F04 | MASE vs. Pers. (IC 95%) | MASE vs. Sazonal | Pinball $P_{95}$ | Cobertura $P_{10}–P_{90}$ | Viol. Monot. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diárias UTI Adulto** | `PooledRidge` / `QuantileGBDT` | `55,82` | **48,86 / 49,01** | **0,817** `[0,753; 0,878]` / **0,820** `[0,770; 0,865]` | **0,741 / 0,743** | **8,97** *(GBDT)* | `72,7%` *(GBDT)* | **`0,00%`** |
| **Leptospirose (Onda 2)** | **`NegBinHurdle (GLM)`** | `0,37` | **0,279** | **0,710** `[0,653; 0,761]` | **0,763** | `0,139` | **`88,2%`** | **`0,00%`** |
| **Respiratório / SRAG** | **`QuantileGBDT (Monotonic)`** | `12,16` | **12,11** | **0,871** | **0,699** | `2,621` | `67,6%` | **`0,00%`** |

---

## 3. Checklist da Feature F04

- [x] Redigir especificação [spec/SPEC-demand-models.md](../../spec/SPEC-demand-models.md)
- [x] Implementar `datasets.py` (painel global causal multi-região + bloqueio físico do holdout `2025`)
- [x] Implementar `models.py` (`QuantileGBDTForecaster` monotônico exato, `NegativeBinomialHurdleForecaster` e `PooledRidgeDeltaForecaster`)
- [x] Implementar `los.py` (`SpecialtyLosEstimator` causal + Lei de Little)
- [x] Validar com testes *Prove-It* e acoplar à auditoria contrafactual `F06` no RS 2024
- [x] **Evolução `F04.1` (`demand/two_stage.py`):** implementar `OrthogonalTwoStageOccupancyForecaster` (Estágio 1 basal lento sem `y_origin` curto + Estágio 2 de choque ex-ante monotônico em taxa de ocupação relativa `ocupacao / capacidade_cnes` acoplado às saídas OOF da `F03.1`) e roteador automático `select_demand_forecaster` para `NegativeBinomialHurdleForecaster` em contagens esparsas
- [ ] **Incerteza de Cauda Extrema (`F04.2`):** implementar hierarquia Bayesiana (`Hospital-Escola ⊂ Bacia`), ajuste de cauda `EVT-GPD` e `Mondrian Conformal` estratificado por porte e regime `ENSO`
- [ ] ***Nowcasting* (`F04.3`):** implementar correção *Chain-Ladder* para censura à direita (`2025–2026+`) e benchmark `TimesFM 2.5`
