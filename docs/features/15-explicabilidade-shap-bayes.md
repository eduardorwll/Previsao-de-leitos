# Feature F15 — Explicabilidade Auditada (`SHAP` 2 Estágios) e Pooling Hierárquico Bayesiano (`Task 1.2.6`)

> **Status:** ✅ Concluída (módulos + gates) | **Promoção na catraca `RS 2024`:** ⏳ pendente (avaliada no `Marco M1`)
> **Etapa:** Etapa 3 (`Ciclo V1.2 RS`) | **Escopo de dados:** `RS`, `ano_epi <= 2024` (holdout `2025` e `2026` intocados)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

1. Tornar a predição pontual da arquitetura ortogonal `F04.1` (`OrthogonalTwoStageOccupancyForecaster`) **auditável por estágio**, decompondo-a em bloco basal (Estágio 1) e contribuições de choque (Estágio 2), com verificação mecânica de erro aditivo `< 1e-6`.
2. Separar explicitamente a **margem conformal** (que desloca apenas `P10`/`P90`/`P95`) da decomposição do ponto (`P50`), impedindo que incerteza de intervalo seja lida como atribuição de feature.
3. Testar a hipótese `H-1.2.6`: **pooling parcial bayesiano** (`Região ⊂ Bacia`) reproduz a estrutura hierárquica com posteriores honestos (`R-hat < 1,01`, `ESS >= 400`, `0` divergências, `PSIS-LOO` e `PPC`), ficando elegível para a catraca *Champion–Challenger* — **sem** substituir o campeão `V1` por decisão implícita.

---

## 2. Alterações Implementadas

- **`src/previsao_leitos/demand/two_stage.py` (refatoração de estrutura, comportamento preservado):**
  - Novo `TwoStageComponents` (dataclass imutável) com `capacity`, `basal_rate`, `shock_rate`, `shock_matrix`, `shock_cols` e as propriedades `point_rate` / `point_beds`.
  - Novo método público `predict_stage_components(df)` — extrai a álgebra que antes vivia duplicada dentro de `predict_quantiles`, que agora apenas consome `point_beds`. `point_beds` reproduz o `P50` **bit a bit** (teste dedicado).
  - Nova propriedade `shock_estimator` (somente leitura) para auditoria externa do GBDT do Estágio 2.
- **`src/previsao_leitos/audit/shap_audit.py` (novo):**
  - `decompose_two_stage_point(model, df, *, background, conformal=None) -> ShapPointDecomposition`.
  - `ConformalMargins` (`low_10`, `high_90`, `high_95`) e `ShapPointDecomposition` (`basal_contribution`, `base_value`, `shap_values`, `pred_p50`, `median_shift`, `conformal_margins`, `rate_residual`, `beds_residual`).
  - Identidade publicada e verificada: `pred_p50 = basal_contribution + base_value + Σ shap_values + median_shift`, com `median_shift` (intercepto de mediana dos resíduos padronizados) reportado à parte porque **não é atribuição de feature**.
  - `additivity_residual(dec)` e `assert_additive(dec)` com tolerância `1e-6` em taxa relativa.
  - Import de `shap` estritamente **lazy** (após as validações de vazamento), preservando `C-07`.
- **`src/previsao_leitos/demand/hierarchical_bayes.py` (novo):**
  - `validate_hierarchical_panel(...)` — contrato de dados e anti-vazamento: colunas obrigatórias, `capacidade_sem > 0`, `target_y >= 0`, rejeição de `ano_epi >= 2025` (`LeakageViolationError`) e filtro `ano_epi <= 2022`.
  - `build_hierarchical_model(...)` — interceptos **aninhados** (`u_entity = u_basin[entidade] + desvio`), sazonalidade de Fourier (`k = 1..2`), `delta_shock` global monotônico-consistente (opção hierárquica por bacia via `hierarchical_shock_slope=True`), tendência opcional (`include_trend`, desligada por padrão por não ser identificável em 2 ciclos anuais) e verossimilhança **Negative Binomial** (contagem com sobredispersão).
  - `fit_hierarchical_bayes(...)` — amostragem `nutpie` em CPU, `mp_ctx="spawn"` (seguro com `Polars`), `compute_log_likelihood` explícito (o `nutpie` não persiste o grupo) e seed determinística.
  - `summarize_diagnostics(idata) -> HierarchicalBayesDiagnostics` (`r_hat_max`, `ess_bulk_min`, `divergences`, `loo_elpd`, `pareto_k_max`, `n_obs`, propriedade `passes`).
  - `ppc_tail_probability(model, idata, observed)` — p-valor preditivo posterior por estatística de discrepância `T(y) = mean((y - média_preditiva)²)`, testando vício e dispersão simultaneamente.
- **`pyproject.toml` / `uv.lock`:**
  - `nutpie>=0.13.0` e `numba>=0.68.0,<0.69` no extra `bayes`; `shap.*`, `pymc.*`, `arviz.*` nos overrides de `mypy`.
  - Novo fork de conflito `bayes` × `scale`: `nutpie` exige `numba >= 0.68`, enquanto `aeon` (extra `scale`) exige `numba < 0.64`. Sem o fork, o resolvedor escolhia `numba==0.63.0b1` (pré-release), cujo `numba.np.arraymath` referencia `np.trapz` (removido no NumPy 2.4) e quebrava qualquer amostragem.

---

## 3. Invariantes e Gates

| Gate | Critério | Resultado |
| :--- | :--- | :---: |
| Aditividade (taxa relativa) | `max_i ǀshock_hat_i − (φ0 + Σ_j φ_ij)ǀ < 1e-6` | ✅ `5,3e-16` |
| Reconstrução em leitos | `P50 = basal + φ0·cap + Σφ·cap + median_shift` | ✅ `atol = 1e-6·cap` |
| Margem conformal | fora do SHAP; `P90 = P50 + high_90` | ✅ |
| Vazamento de background | `background` com `ano_epi > 2022` ⇒ `LeakageViolationError` | ✅ |
| Vazamento de holdout | `ano_epi >= 2025` ⇒ `LeakageViolationError` | ✅ |
| Features proibidas | nenhuma chave em `y_origin/y_roll*/target_y` | ✅ |
| `C-07` | `import previsao_leitos` sem `shap`/`pymc`/`arviz` no topo | ✅ |
| Convergência | `R-hat < 1,01`, `ESS >= 400`, `0` divergências | ✅ `R-hat 1,0048`, `ESS 761`, `0` |
| Calibração preditiva | `PSIS-LOO` finito, `pareto-k < 0,7`, `PPC p ∈ (0,02, 0,98)` | ✅ `k = 0,18–0,31` |
| Monotonicidade (`ADR-0001`) | `0,00%` de violações | ✅ (inalterado) |

---

## 4. Evidências Medidas (fixture determinística)

| Medição | Condição | Resultado |
| :--- | :--- | :--- |
| Custo do SHAP | fit `61 ms` + decomposição `79 ms` (240 linhas de treino, 60 de auditoria) | `< 0,15 s` |
| Custo do MCMC | `960` observações, 12 entidades ⊂ 6 bacias, `4` cadeias × `3.000` draws (`1.500` tune), `nutpie` CPU, `target_accept=0,99` | **`~43–55 s`** |
| Diagnósticos | mesma execução | `R-hat 1,0048`, `ESS_bulk 761`, `0` divergências, `elpd_loo = −5.064,6` |

> **Limitação honesta:** os gates acima foram exercitados na fixture determinística da suíte (bem especificada: ruído Negative Binomial `alpha = 40`, efeitos de entidade e bacia presentes). O número em `RS 2024` real e a decisão de catraca contra o campeão `V1` são produzidos no **`Marco M1`**, quando o Dossiê Master roda com o painel `v3` completo.

---

## 5. Riscos e Limitações

1. **Não-promoção implícita:** `hierarchical_bayes` é *challenger*. Nada no módulo altera o campeão `V1`; a promoção exige vitória na catraca em `RS 2024` (pendente).
2. **Identificação de variâncias:** com poucas bacias, a decomposição `sigma_basin` vs. `sigma_entity` é fracamente identificada (fato conhecido de modelos aninhados). Por isso os interceptos são aninhados (`entidade` em torno da bacia) e não cruzados.
3. **`nutpie` e `numba`:** o wheel pré-release de `numba` quebra com NumPy 2.4; o lock fixa `numba >= 0.68` no fork `bayes`.
4. **`background` do SHAP:** o baseline é intrínseco às árvores (`tree_path_dependent`). Forçar *masker* interventional degrada a aditividade medida para `~2e-1` em taxa — por isso a escolha é documentada e testada.
5. **`median_shift`:** é calibração do `P50`, não atribuição. Somá-lo ao SHAP sem separá-lo produziria a leitura errada de que o modelo "explica" o deslocamento de mediana.

---

## 6. Como Reproduzir

```bash
uv sync --extra rocm --extra dev --extra data --extra stats --extra sim --extra hydro --extra bayes --extra ml
uv run --no-sync pytest tests/unit/test_audit_shap_two_stage.py tests/unit/test_demand_hierarchical_bayes.py -q
uv run --no-sync pytest tests/unit/test_demand_hierarchical_bayes.py -m slow -q   # ~55 s
```

---

## 7. Arquivos

- `src/previsao_leitos/audit/shap_audit.py`
- `src/previsao_leitos/demand/hierarchical_bayes.py`
- `src/previsao_leitos/demand/two_stage.py` (extração de `predict_stage_components`)
- `tests/unit/test_audit_shap_two_stage.py`, `tests/unit/test_demand_hierarchical_bayes.py`
- `pyproject.toml`, `uv.lock` (extras `bayes`, fork `bayes` × `scale`, overrides `mypy`)
