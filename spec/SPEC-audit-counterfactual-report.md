# Especificação V1 — Módulo `audit`: Análise Contrafactual Estruturada e Relatório Final de Verificação (`spec/SPEC-audit-counterfactual-report.md`)

- **Status:** `proposed` *(planejamento aprovado antes da implementação nas Etapas 2–3)*
- **Etapa / Feature:** Etapa 3 (com núcleo antecipado na Etapa 2 para auditar `F04`) — Feature `F06 — Auditoria Estatística, Análise Contrafactual e Relatório de Verificação`
- **ADR de Referência:** [ADR-0007](../docs/adr/0007-counterfactual-analysis-and-verification-report.md), [ADR-0003](../docs/adr/0003-causal-ex-ante-splits-and-calibration.md), [ADR-0006](../docs/adr/0006-landfill-polders-and-intermunicipal-routing.md)

---

## 1. Objetivo

Fornecer uma camada reprodutível e auditável que:
1. Execute **análise contrafactual estruturada** (linha de base "sem enchente" com controle sintético + testes placebo + intervenções físicas em áreas aterradas, remanso, escoamento a montante e rede `W_ref`);
2. Meça a **acurácia real dos modelos durante choques extremos** (além de métricas médias anuais);
3. Gere automaticamente o **Relatório Final de Verificação Científica** (`reports/<run_id>/verificacao_cientifica.md` + `run_manifest.json` + 5 gráficos contrafactuais padronizados) documentando metodologia, ferramentas, datasets e evidências visuais.

> **Fronteira Obrigatória:** Todo relatório e gráfico gerado carrega explicitamente o aviso de pesquisa metodológica (*"Not for patient care or live clinical use"*), respeita supressão LGPD ($n < 5$) e protege o *holdout* `2025` contra acesso acidental (`allow_holdout=False` por padrão).

---

## 2. Estrutura do Módulo `src/previsao_leitos/audit/`

```text
src/previsao_leitos/audit/
├── __init__.py          # Superfície pública
├── metrics.py           # MAE, MASE, Pinball Loss (P50/P90/P95), Cobertura, ECE/BSS, Block-Bootstrap 95% CI, Net Benefit
├── counterfactual.py    # Controle sintético "sem enchente", validação placebo, intervenções do() e acurácia de excesso
├── plots.py             # 5 gráficos contrafactuais + confiabilidade (paleta Okabe-Ito WCAG 2.2, 300 DPI, PNG/SVG)
└── report_builder.py    # Construtor determinístico do Relatório de Verificação (metodologia, ferramentas, datasets, figuras)
```

---

## 3. Contratos Principais

### 3.1 `counterfactual.py`
- **`NoFloodCounterfactualBaseline`:**
  - Ajustado estritamente em `<= train_end_year` (`2022`) e séries de regiões-controle não inundadas (ex.: Planalto/Serra para RS 2024; doadores fora da bacia atingida na V2).
  - Produz `y_cf_no_flood`, `y_cf_p05`, `y_cf_p95` e **excesso observado** `excess_obs = y_true - y_cf_no_flood`.
- **`evaluate_counterfactual_accuracy(...) -> CounterfactualAccuracyReport`:**
  - Compara `excess_obs` contra o **excesso previsto *ex-ante*** pelo modelo avaliado (`excess_pred = y_pred(X_real) - y_pred(X_climatologia_p50)`).
  - Calcula: `shock_excess_mae`, `shock_excess_bias`, `peak_timing_error_weeks`, `interval_coverage_90`, `placebo_bias`, `placebo_mae` e `monotonic_violation_rate`.
- **`decompose_intermunicipal_and_landfill_effect(...) -> pl.DataFrame`:**
  - Executa intervenções sequenciais controladas:
    1. `base_climatologia` (todas as covariáveis climáticas/hidrológicas no `P50` histórico);
    2. `+ chuva_local` (ativa apenas precipitação/solo local observado);
    3. `+ escoamento_montante` (ativa `chuva_montante_pond` e `api_montante_pond` com `travel_weeks`);
    4. `+ remanso_e_aterro` (ativa `remanso_lagunar_idx` sob vento Sul e `exposicao_aterro_ativa`);
    5. `+ transbordamento_w_ref` (ativa pressão hidrológica das cidades de captação via `W_ref`).

### 3.2 `plots.py` (5 Gráficos Contrafactuais Obrigatórios)
1. **`fig_c1_trajetoria_contrafactual`:** séries semanais (`Sem. 10–35`) comparando Real Observado, Contrafactual Sem Enchente (IC 95%) e Previsão *Ex-Ante* ($P_{50}$ e banda $P_{10}–P_{90}$) para POA, Pelotas/Sul, Vale do Taquari e Vale do Caí.
2. **`fig_c2_acuracia_excesso_choque`:** excesso observado ($\Delta^{\text{obs}}_t$) vs. excesso previsto ($\Delta^{\text{mod}}_t$) com destaque para Onda 1 (`D+0–5`) e Onda 2 (`D+14` leptospirose/SRAG) + erro de timing de pico ($\Delta t_{\text{pico}}$).
3. **`fig_c3_validacao_placebo`:** distribuição do erro contrafactual em semanas/regiões placebo (sem enchente), demonstrando ausência de falsos picos.
4. **`fig_c4_decomposicao_intermunicipal_aterro`:** gráfico *waterfall* decompondo o pico previsto em POA e Pelotas entre chuva local, cidades a montante, remanso lagunar, áreas aterradas e rede `W_ref`.
5. **`fig_c5_contingencia_simpy`:** comparação contrafactual de ocupação e fila de UTI no `SimPy` sob acionamento antecipado (`h = 2` sem.) vs. reação tardia (`h = 0`).

### 3.3 `report_builder.py` (Estrutura Obrigatória do Relatório Final)
Gera `reports/<run_id>/verificacao_cientifica.md` + `run_manifest.json` com 6 seções fixas:
1. **Resumo Executivo e Fronteira Clínica**;
2. **Metodologia Científica, Splits Congelados e Contratos Causais** (`ADR-0001` a `ADR-0007`);
3. **Ferramentas, Bibliotecas, Versões e Reprodutibilidade** (`python`, `polars`, `statsmodels`, `scikit-learn`, `pymc`, `scikit-survival`, `pytorch-lightning`, `simpy`, `git_sha`, `seeds`);
4. **Datasets Utilizados, Cobertura e Auditoria LGPD** (fonte, período, linhas, nulos, supressão $n < 5$, bytes BigQuery);
5. **Resultados de Acurácia Preditiva e Calibração** (tabelas por especialidade e região com *block-bootstrap* 95% CI vs. Persistência e Sazonal-Ingênuo);
6. **Análise Contrafactual Estruturada e Figuras (`Fig. C1` a `Fig. C5`)** + checklists **TRIPOD+AI / PROBAST+AI / RECORD**.

---

## 4. Testes *Prove-It* Obrigatórios na Implementação

- **Zero vazamento no contrafactual:** perturbar `2024–2025` não altera pesos do controle sintético nem perfis climatológicos `P50` ajustados em `<= 2022`.
- **Sanidade placebo:** em série sintética sem choque de enchente, `|placebo_bias|` permanece dentro da tolerância estatística e a taxa de violação monotônica é `0.0`.
- **Reprodutibilidade do relatório:** executar `build_verification_report` duas vezes com a mesma seed gera `verificacao_cientifica.md`, tabelas e manifesto determinísticos com todas as 5 figuras presentes e não vazias.
- **Bloqueio de holdout:** tentar gerar relatório sobre `2025` com `allow_holdout=False` lança `LeakageViolationError`.
