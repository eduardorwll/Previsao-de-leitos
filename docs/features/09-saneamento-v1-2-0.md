# Feature F09 — Saneamento Estrutural `V1.2.0` (`Task 1.2.0`: Bugs A–J, Contrato Canônico, 8/8 Gates e PDF Editorial)

> **Status:** ✅ Concluída (`RS <= 2024`, holdout `2025` preservado intocado) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

Eliminar os defeitos estruturais identificados na auditoria forense da base `V1.1`, formalizar o contrato canônico de colunas do painel causal (`CanonicalPanelSchema`), restaurar a emissão de relatórios em **PDF editorial HTML5/CSS3** conforme [padrao-relatorio-tecnico-pdf.md](../guides/padrao-relatorio-tecnico-pdf.md) e habilitar o scorecard completo de **8/8 Gates de Higiene (`G1–G8`)** antes da reextração BigQuery `v3` (`Task 1.2.1`).

---

## 2. Correções Implementadas

| Código | Módulo | Defeito Auditado (`V1.1`) | Correção Aplicada (`V1.2.0`) |
| :---: | :--- | :--- | :--- |
| **Bug A** | `demand/two_stage.py` | Buscava `sin_target_wk`/`cos_target_wk` em vez de `target_week_sin`/`target_week_cos`; fallback devolvia `basal_hat = 0` (`MASE = 3,22`, viés `-190`) | Harmônicos canônicos `target_week_sin`/`target_week_cos` como padrão (aliases legados apenas como fallback) e persistência de `_basal_mean` em `predict_quantiles()` |
| **Bug B** | `demand/two_stage.py` | `y_roll4`, `y_roll12` e `y_dev_roll12` vazavam no Estágio 2; incerteza homoscedástica `_resid_sd * cap` inflava `Pinball95` (`19,76`) e `Cov` (`98,8%`) | Bloqueio explícito de todas as colunas AR (`y_origin`, `y_roll*`, `y_dev_roll*`) + quantis empíricos padronizados por $\sqrt{\text{cap}}$ preservando `0,00%` de violações monotônicas |
| **Bug C** | `demand/two_stage.py` | `_DEFAULT_MONO_POS` omitia `chuva_montante_multihop`, zerando restrições monotônicas | Inclusão de `chuva_montante_multihop`, `chuva_sem_mm`, `api_max_sem`, `indice_aterro_varzea`, `oni_c` e exposição auditável de `monotone_constraints` |
| **Contrato** | `demand/datasets.py` | Ausência de contrato único entre produtores e consumidores de features | Criação de `CanonicalPanelSchema` e `validate_canonical_panel_frame` (chaves canônicas, validação numérica e trava anti-PII LGPD) |
| **Bug D** | `audit/enso_what_if.py` | Oráculo El Niño perturbava apenas colunas sem `_lag*`, gerando saída plana (`72,14 → 72,14`) | Perturbação consistente de colunas-base e defasadas (`*_lag*`) de ENSO, chuva e probabilidade/severidade `F03.1` |
| **Bug F** | `scripts/run_rs_v1_1_dossier.py` | `peak_timing_error_weeks = 2.0` hardcoded para V1, janela `W18–22` cortando o pico de UTI (`W26–28`) e `enso_regime` juntado em $t_0 + 2$ | Integração causal de `CNES` mensal (`<= 2024`), junção de `enso_regime` em $t_0$, janela `W18–30`, cálculo real do timing de pico para V1 e V1.2.0, e reporte completo dos **8/8 Gates (`G1–G8`)** |
| **Bug G** | `audit/bsts_counterfactual.py` | Regressão linear global `2010–2024` superestimava `W15–17/2024`, `1.96` em vez de `1.6449` para `p05/p95` e `do()` não atualizava `y_roll4/12` | Modelo estrutural de espaço de estados `UnobservedComponents` (`local level` + harmônicos + `SIVEP-Gripe`), quantil exato `1.6449` e atualização recursiva de `y_origin`, `y_roll4`, `y_roll12` e `y_dev_roll12` |
| **Bug H** | `audit/metrics.py` | `estimate_surge_probability_from_quantiles` atribuía `p_surge = 0,74` a linhas sem UTI (`y_roll12 == 0`) | Máscara estrutural anti-G8 (`baseline <= 1e-9 -> p = 0.0`) e reporte segregado de `BSS_all`, `BSS_pos` e `BSS_shock` |
| **Bug J** | `audit/dossier_compiler.py`, `report_builder.py`, `readiness/simulator.py` | HTML embutia Markdown cru em `<pre>`, omitia `--print-to-pdf-no-header`, sobrescrevia `CAMPEAO_*` sob `REJECT`; `SimPy` ativava pôlder por proxy circular de demanda e defasava `eff_beds` em 1 dia | Renderização editorial HTML5/CSS3 (Capa, 6 Capítulos, cartões KPI, tabelas HTML, 300 DPI Okabe-Ito, `--print-to-pdf-no-header`), trava `CAMPEAO_*` apenas sob `PROMOTE`, alinhamento de `run_manifest.json`, suporte exógeno `em_cheia_hidro` e pré-ativação `lead_time` no `SimPy` |

---

## 3. Resultados Reais Auditados em `RS 2024` (`rs_v1_2_0_sanitized_2024`, Holdout `2025` Intocado)

| Indicador Auditado | Antes (`V1.1`) | Depois (`V1.2.0`) | Critério da Catraca `1.2.0` |
| :--- | :---: | :---: | :---: |
| **`F04.1` MASE (vs. Persistência)** | `3,2217` | **`0,9353`** (`7,99` leitos/dia MAE) | Saneado (aguarda painel `v3` em `1.2.1`) |
| **`F04.1` Média $P_{50}$ (diárias/sem)** | `67,74` (viés `-190`) | **`231,00`** (`33,00` vs. real `36,93` leitos/dia) | Viés estrutural eliminado |
| **`F04.1` Pinball Loss $P_{95}$ / Cobertura** | `19,76` / `98,8%` | **`10,14` / `80,6%`** (alvo `80%`) | Calibração heterocedástica restaurada |
| **Violação Monotônica (`ADR-0001`)** | — | **`0,00%`** (`V1` e `V1.2.0`) | ✅ `0,00%` |
| **Oráculo El Niño ($P_{50}$ médio)** | `72,14 → 72,14` (plano) | **`230,89 → 234,57`** (`P90`: `+92,1`) | ✅ Estritamente crescente |
| **Viés BSTS Pré-Choque (`W15–17/2024`)** | `+19..+32` (`excesso -6,1`) | **`+8,10`** (`excesso W18–30 = +55,83`) | ✅ `\|viés\| < 10` |
| **Scorecard de Higiene (`G1–G8`)** | 3/8 reportados | **8/8 reportados** (`G2,G4,G8 PASS`; `G1,G3,G5,G6,G7 FAIL` expostos honestamente: `G1=11,18` OOF circular, `G6=5/30` = `16,7%`) | ✅ 8/8 auditados |
| **PDF Editorial + Trava `CAMPEAO_*`** | `<pre>` Markdown / sobrescrevia `CAMPEAO_*` | **HTML5/CSS3 Chromium (`--no-pdf-header-footer` + `--print-to-pdf-no-header`, tabelas largas compactas sem corte, zero cabeçalho de navegador/Markdown cru, `CAMPEAO_*` protegido sob `REJECT`)** | ✅ Conforme |

---

## 4. Verificação Mecânica e Próxima Etapa

- **Suíte automatizada:** `136` testes unitários passando (`15` testes de regressão em `tests/unit/test_v1_2_0_sanitization.py`), cobertura global **`87,45%`** ($\ge 80\%$), `ruff`, `mypy --strict` (`src/` e `scripts/`), `check_constraints.py` e `pip-audit` 100% verdes.
- **Próximo passo:** `Task 1.2.1` (reextração BigQuery `v3` com censo ativo sem double-count, `W_ref` externa e triângulo de atraso para zerar `G3` e `G5`).
