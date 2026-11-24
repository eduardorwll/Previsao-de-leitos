# Feature F06 — Auditoria Estatística, Análise Contrafactual e Relatório de Verificação

| Campo | Valor |
| :--- | :--- |
| **Status** | ⏳ Especificação pronta (`spec/SPEC-audit-counterfactual-report.md`) |
| **Etapa** | Etapa 3 (núcleo antecipado acoplado à Etapa 2 para auditar `F04`) |
| **Especificação** | [spec/SPEC-audit-counterfactual-report.md](../../spec/SPEC-audit-counterfactual-report.md) · [ADR-0007](../adr/0007-counterfactual-analysis-and-verification-report.md) |
| **Pacote Alvo** | `src/previsao_leitos/audit/` |

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
- [ ] Implementar `metrics.py` (MAE, MASE, Pinball Loss $P_{50}/P_{90}/P_{95}$, ECE/BSS, *block-bootstrap* 95% CI, Net Benefit)
- [ ] Implementar `counterfactual.py` (controle sintético "sem enchente", testes placebo, acurácia de excesso/timing e decomposição `do()` intermunicipal + aterros/remanso)
- [ ] Implementar `plots.py` (5 gráficos contrafactuais `Fig. C1–C5` em paleta *Okabe-Ito* WCAG 2.2, 300 DPI)
- [ ] Implementar `report_builder.py` (gerador automático de `reports/<run_id>/verificacao_cientifica.md` + `run_manifest.json`) com testes *Prove-It*
