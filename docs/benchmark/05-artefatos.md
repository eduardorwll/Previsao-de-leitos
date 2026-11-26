# 5. Artefatos e esquemas

## 5.1 Estrutura de um run

```
reports/runs/<run-id>/
├── benchmark.csv              # 1 linha por modelo × alvo × fold? (não: por modelo × alvo)
├── predictions.parquet        # tidy: região × semana × modelo × observado × P10..P95
├── comparison.csv             # observado vs predito por região (n >= 5)
├── decisions.csv              # catraca por alvo
├── eligibility.csv            # cobertura >= 80% e 0 violações
├── benchmark_cards.md         # índice dos cartões
├── cards/<modelo>.md          # um cartão por modelo
├── fig_*.png                  # 12 figuras (3 por alvo) + erro percentual
├── relatorio_benchmark_leitos.pdf
└── run_manifest.json
```

No backtest, o diretório tem `backtest_folds.csv`, `backtest_summary.csv`, figuras e
`relatorio_backtest_<alvo>.pdf` (um por alvo quando há vários).

## 5.2 `benchmark.csv`

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `candidate`, `family`, `status` | str | identidade e estado (`ok`, `failed`, `unavailable`) |
| `is_champion` | bool | campeão registrado |
| `mae_p50`, `mae_persistencia`, `mae_sazonal` | f64 | MAE em unidade do alvo |
| `mase_vs_pers`, `mase_vs_saz` | f64 | MASE contra baselines |
| `mase_ci95_low/high` | f64 | IC por cluster de bacia |
| `pinball_p10/p50/p90/p95` | f64 | perda por quantil |
| `cobertura_p10_p90` | f64 | cobertura empírica |
| `wmape` | f64 | erro percentual ponderado |
| `erro_pct_mediana`, `erro_pct_p90` | f64 | distribuição do erro % |
| `subestimacao_pct` | f64 | fração `pred < obs` |
| `erro_pct_evento` | f64 | erro % nas semanas de choque |
| `peak_timing_error_weeks` | f64 | erro de fase |
| `monotonic_violations` | i64 | quedas de P50 sob choque |
| `fit_seconds`, `predict_seconds` | f64 | custo |
| `n_obs` | i64 | linhas avaliadas |

## 5.3 `backtest_folds.csv`

Uma linha por **modelo × fold**, com as mesmas métricas acima mais:

| Coluna | Descrição |
| :--- | :--- |
| `eval_year` | fold |
| `mase_pooled` / `mase_macro` | visão global / média por região (piso 1 leito-dia) |
| `mae_macro` | MAE médio por região |
| `pinball_mean` | média dos 4 quantis |
| `peak_signed_weeks`, `peak_abs_weeks` | fase com sinal e em módulo |
| `shape_corr` | correlação da curva semanal |
| `bias_relative` | viés relativo |
| `event_mae`, `event_share` | erro nas semanas de choque |
| `regions_used` | regiões incluídas no macro (26–29 de 30) |

## 5.4 `backtest_summary.csv`

| Coluna | Descrição |
| :--- | :--- |
| `mase_median`, `mase_mean`, `mase_worst`, `mase_best` | dispersão entre folds |
| `rank_median`, `fold_wins`, `n_folds` | ranking e vitórias |
| `coverage_median` | cobertura mediana |
| `wmape_median`, `erro_pct_mediana`, `erro_pct_p90`, `subestimacao_pct` | erro % |
| `mase_cv` | consistência (desvio/média) |
| `fit_seconds_median` | custo |

## 5.5 `run_manifest.json`

`run_id`, `generated_at_utc`, `git_sha`, `targets`, `models`, `seed`, `n_boot`, `jobs`,
`parallel`, `sampler`, `device`, `accelerator`, `panel_path`, `panel_sha256`, `elapsed_seconds`,
`candidates_ok`, `candidates_total`, `notes`, `holdout_policy`, `disclaimer`.
