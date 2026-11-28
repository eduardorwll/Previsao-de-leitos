# Relatório Completo de Execução Científica — rs_v1_4_0_m1

> **Aviso Obrigatório:** Pesquisa aberta e avaliação metodológica. Not for patient care or live clinical use.
> **Alvo:** `uti_adulto_diarias (RS 2024)` | **Ano Avaliado:** `2024` | **Commit:** `cf18ef6` | **Gerado em (UTC):** `2026-11-28T14:17:20.211257+00:00` | **Seed:** `42`

---

## 1. Scorecard de Higiene de Dados Cruzados (Gates G1-G8)
- **Status Global:** `FAIL` (PASS=1, WARN=0, FAIL=2)

| gate | indicador | status | observado | limite | resumo |
| --- | --- | --- | --- | --- | --- |
| G8 | Trava Anti-Rótulo Degenerado de Surto (0 >= 0) | PASS | 0 casos | <= 0 casos | 0 rótulo(s) de surto degenerados com y <= 1e-09 (0.00% das observações). |
| G1 | Colinearidade de Covariáveis (VIF) | FAIL | 5.314 VIF | <= 5 VIF | VIF máximo observado = 5.31 (limite <= 5.00). |
| G4 | Detector de Censura à Direita (Atraso de Processamento) | FAIL | 37.95 % | <= 25 % | Queda máxima nas últimas 4 semanas = 37.9% vs. mediana de referência (6665.0; limite <= 25.0%). |

## 2. Desempenho Preditivo e Calibração (Benchmark)

| modelo | mae_p50 | mae_persistencia | mae_sazonal | mase_vs_pers | mase_ci95_low | mase_ci95_high | mase_vs_saz | pinball_p50 | pinball_p90 | pinball_p95 | cobertura_p10_p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QuantileGBDT (Monotonic) | 24.3432 | 24.9492 | 40.2751 | 0.9757 | 0.9513 | 0.9975 | 0.6044 | 12.1716 | 6.8156 | 4.7370 | 0.7115 |
| QuantileGBDT (Monotonic) + Conformal(2023) | 24.3432 | 24.9492 | 40.2751 | 0.9757 | 0.9513 | 0.9975 | 0.6044 | 12.1716 | 6.8156 | 4.7370 | 0.7115 |

## 3. Catraca Científica Champion-Challenger

_Sem comparação Champion-Challenger registrada nesta execução._


## 4. Notas Metodológicas e Governança (TRIPOD+AI / RECORD)

- ECE_surge=0.0941 | BSS_surge=-0.0108
- Violação monotônica = 0.0000

## 5. Figuras Geradas (Fig. C1-C7)

- `fig_c1_trajetoria_contrafactual.png`
- `fig_c2_acuracia_excesso_choque.png`
- `fig_c3_validacao_placebo.png`
- `fig_c4_decomposicao_intermunicipal_aterro.png`
- `fig_c5_contingencia_simpy.png`
- `fig_c6_higiene_dados.png`
