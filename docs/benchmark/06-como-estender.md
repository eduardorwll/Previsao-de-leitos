# 6. Como estender

## 6.1 Adicionar um modelo

1. Implemente `DemandForecaster` (`model_name`, `fit`, `predict_quantiles`).
2. Registre em `demand/registry.py` com `fit_mode` correto:
   - `plain`: `fit(train, features)`;
   - `train_cal`: `fit` + `fit_calibration(train_df=…, cal_df=…)`;
   - `cal_only`: `fit` + `fit_calibration(cal_df=…)`;
   - `online=True` se existir `predict_quantiles_online` (recalibração causal).
3. Declare `requires` para extras opcionais (aparece como `N/A` se ausente).
4. Escreva teste de contrato: P10 ≤ P50 ≤ P90 ≤ P95, shape = `eval_df`, determinismo.
5. Rode a régua: `backtest --models <novo>,champion` e compare medianas.

**Critério de entrada:** o modelo precisa aparecer em `MASE`, `wMAPE`, fase e cobertura — sem
métrica nova escondida.

## 6.2 Adicionar uma métrica

1. Implemente em `audit/metrics.py` com docstring de **seis campos**: pergunta, fórmula,
   unidade, direção, casos-limite, o que **não** diz.
2. Propague em `model_benchmark._metric_row` e `backtest.evaluate_fold` / `summarize_backtest`.
3. Adicione teste de fronteira e de valor ausente (`NaN`).
4. Documente em [02-metricas.md](02-metricas.md) e, se entrar no cartão, em
   [03-erro-percentual.md](03-erro-percentual.md).
5. Só entra na rubrica com nova versão (`v2`).

## 6.3 Adicionar um alvo

1. Crie `TargetSpec` em `demand/modeling.py` (coluna, unidade, capacidade, escala, limiar de
   porte).
2. Garanta a coluna no painel (`build_rs_modeling_panel.py`) e documente a origem.
3. Rode `backtest --target <novo>` nos 6 folds; verifique `regions_used`.
4. Nunca compare MASE entre alvos de unidades diferentes.

## 6.4 Adicionar um calibrador

Calibradores são wrappers; declare `fit_mode` e, se houver versão online, `online=True`.
Registre no cartão que o P50 não deve mudar (a calibração move P10/P90/P95) — se mudar,
documente por quê.

## 6.5 Checklist de PR

- [ ] Testes de contrato e de fronteira verdes
- [ ] `ruff`, `mypy`, `check_constraints`, cobertura ≥ 80%
- [ ] run de referência regerado (`benchmark` + `backtest`)
- [ ] cartões e `RESULTADOS.md` atualizados
- [ ] nenhum segredo, nenhum dado `>= 2025`
