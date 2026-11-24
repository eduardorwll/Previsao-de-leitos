# Feature F06 — Auditoria Estatística, Análise Contrafactual e Relatório de Verificação

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída V1 RS (`Fig. C1–C5` completas + Relatório de Verificação + `run_manifest.json`) |
| **Etapa** | Etapa 2–3 (F06 integrada à validação da F04) |
| **Especificação** | [spec/SPEC-audit-counterfactual-report.md](../../spec/SPEC-audit-counterfactual-report.md) · [ADR-0007](../adr/0007-counterfactual-analysis-and-verification-report.md) |
| **Pacote Implementado** | `src/previsao_leitos/audit/` (`metrics.py`, `counterfactual.py`, `plots.py`, `report_builder.py`) |

---

## 1. Objetivo

Garantir que toda verificação dos modelos produza um **Relatório Final de Verificação Científica** padronizado contendo:
1. **Metodologia completa** (contratos causais, splits congelados, monotonicidade e acoplamento intermunicipal/aterros);
2. **Ferramentas, versões e manifesto reproduzível** (`run_manifest.json`);
3. **Inventário auditado de datasets utilizados** (cobertura, qualidade e prova LGPD $n < 5$);
4. **Análise Contrafactual Estruturada + 5 Gráficos Contrafactuais** medindo a acurácia real dos modelos em choques extremos (excesso observado vs. excesso previsto, timing de pico +2 semanas, validação placebo, decomposição intermunicipal/aterros e contingência `SimPy`).

---

## 2. Checklist da Feature F06

- [x] Formalizar estratégia causal e de relatório em [ADR-0007](../adr/0007-counterfactual-analysis-and-verification-report.md)
- [x] Redigir especificação [spec/SPEC-audit-counterfactual-report.md](../../spec/SPEC-audit-counterfactual-report.md)
- [x] Implementar `metrics.py` (MAE, MASE, Pinball Loss $P_{50}/P_{90}/P_{95}$, Cobertura, *block-bootstrap* 95% CI)
- [x] Implementar `counterfactual.py` (`NoFloodSyntheticControl`, validação placebo, acurácia de excesso/timing e decomposição `do()` intermunicipal + aterros/remanso)
- [x] Implementar `plots.py` (gráficos contrafactuais `Fig. C1–C4` em paleta *Okabe-Ito* WCAG 2.2, 300 DPI)
- [x] Implementar `report_builder.py` (gerador automático de `reports/<run_id>/verificacao_cientifica.md` + `run_manifest.json`) com testes *Prove-It* e executar sobre RS 2024
- [x] Acoplar **`Fig. C5`** (curva contrafactual de filas `SimPy` com acionamento `h=2` vs. `h=0`) e calibrar bandas sob choque em `2023` (`34,1% → 61,4%` em UTI 2024)
