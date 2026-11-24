# Especificação V1 — Módulo `demand`: Modelos Globais/Quantílicos, Contagem (Onda 2) e LOS (`spec/SPEC-demand-models.md`)

- **Status:** `approved` *(implementação TDD na Etapa 2 acoplada ao módulo `audit` F06)*
- **Etapa / Feature:** Etapa 2 — Feature `F04 — Demanda Semanal Multi-Horizonte e Tempo de Permanência (LOS)`
- **ADRs de Referência:** [ADR-0001](../docs/adr/0001-structural-monotonicity.md), [ADR-0002](../docs/adr/0002-target-teaching-hospital-demand.md), [ADR-0003](../docs/adr/0003-causal-ex-ante-splits-and-calibration.md), [ADR-0006](../docs/adr/0006-landfill-polders-and-intermunicipal-routing.md), [ADR-0007](../docs/adr/0007-counterfactual-analysis-and-verification-report.md)

---

## 1. Problema que Esta Especificação Resolve

No *Tracer Bullet* linear da F02 (`Ridge` independente por região), identificamos três limitações estruturais que impedem capturar toda a física do RS:
1. **Sem *pooling* entre regiões:** atributos estruturais causais (`indice_aterro_varzea`, `frac_area_polder_regiao`, `frac_banhado_agua`) são constantes dentro de uma única região e só ganham poder preditivo quando aprendidos **conjuntamente entre as 30 Regiões de Saúde** (~`18.000` semanas-região de treino `<= 2022`).
2. **Sem quantis calibrados nem restrição monotônica:** `Ridge` simétrico não estima caudas superiores ($P_{90}/P_{95}$) para dimensionamento de contingência (*surge capacity*) nem impõe $\partial \hat{y}/\partial x_{\text{risco}} \ge 0$.
3. **Contagens esparsas de cauda pesada (Onda 2 — Leptospirose):** um modelo gaussiano não modela excesso de zeros nem sobredispersão (`6 → 42` casos na Semana 20 de 2024).

---

## 2. Estrutura do Módulo `src/previsao_leitos/demand/`

```text
src/previsao_leitos/demand/
├── __init__.py          # Superfície pública
├── datasets.py          # Construção do painel causal global multi-região (<= 2022 treino, 2025 bloqueado)
├── models.py            # DemandForecaster: QuantileGBDTForecaster (monotônico), NegativeBinomialHurdleForecaster, PooledRidgeDeltaForecaster
└── los.py               # Estimador causal de LOS por especialidade + conversão para leitos ocupados/dia (Lei de Little)
```

---

## 3. Contratos e Garantias (*Prove-It*)

1. **`QuantileGBDTForecaster` (`scikit-learn` `HistGradientBoostingRegressor`):**
   - Estima diretamente quantis $P_{10}, P_{50}, P_{90}, P_{95}$ sobre o incremento $\Delta y_{t+h} = y_{t+h} - y_t$ (ou nível escalado por porte regional `y_roll12`), com **restrição monotônica estrita (`monotonic_cst = +1`)** nas variáveis de risco físico (`chuva_montante_pond`, `api_montante_pond`, `remanso_lagunar_idx`, `exposicao_aterro_ativa`, `indice_aterro_varzea`).
   - Pós-processamento determinístico garante não-negatividade ($\hat{y} \ge 0$) e **não-cruzamento de quantis** ($P_{10} \le P_{50} \le P_{90} \le P_{95}$).
2. **`NegativeBinomialHurdleForecaster` (`statsmodels` GLM + `scipy.stats.nbinom`):**
   - Modelo em duas partes para agravos raros/esparsos (ex.: **Leptospirose — Onda 2**): probabilidade logística de ocorrência $P(Y > 0 \mid X)$ acoplada a intensidade Binomial Negativa/Poisson com link logarítmico e quantis analíticos $P_{10}, P_{50}, P_{90}, P_{95}$.
3. **`PooledRidgeDeltaForecaster`:**
   - Baseline linear global padronizado exclusivamente no treino para ablação honesta.
4. **`SpecialtyLosEstimator` (`los.py`):**
   - Calcula estatísticas causais de LOS (`dias_permanencia_total / internacoes` e `diarias_uti_total / internacoes` em `<= train_end_year`) por especialidade (`uti_adulto`, `uti_pediatrica`, `uti_neonatal`, `clinico`, `cirurgico`) e converte diárias semanais previstas em **leitos ocupados/dia** (`diarias_semanais / 7.0`).
5. **Testes *Prove-It* obrigatórios:**
   - **Não-cruzamento e não-negatividade** ($0 \le P_{10} \le P_{50} \le P_{90} \le P_{95}$);
   - **Monotonicidade verificada:** aumentar `remanso_lagunar_idx`, `chuva_montante_pond` ou `indice_aterro_varzea` nunca reduz a previsão $P_{50}/P_{90}$ do `QuantileGBDTForecaster`;
   - **Bloqueio de vazamento (`2025`):** `prepare_global_demand_splits` lança `LeakageViolationError` se `eval_year >= 2025` sem `allow_holdout=True`.
