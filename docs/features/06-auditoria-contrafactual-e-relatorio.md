# Feature F06 — Auditoria Estatística, Análise Contrafactual e Relatório de Verificação

| Campo | Valor |
| :--- | :--- |
| **Status** | 🔄 Em evolução V1.1 (`Fig. C1–C5` + Sensores de Higiene `G1–G8` implementados) |
| **Etapa** | Etapa 2–3 (F06 e evolução F06.1 — Higiene de Dados e Dossiê Automático) |
| **Especificação** | [spec/SPEC-audit-counterfactual-report.md](../../spec/SPEC-audit-counterfactual-report.md) · [ADR-0007](../adr/0007-counterfactual-analysis-and-verification-report.md) |
| **Pacote Implementado** | `src/previsao_leitos/audit/` (`metrics.py`, `counterfactual.py`, `plots.py`, `report_builder.py`, `data_hygiene.py`) |

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
- [x] Implementar `data_hygiene.py` (**Gates `G1–G8`**): sensores determinísticos de colinearidade (`VIF`), perda/explosão de join, conservação de censo diário vs. capacidade CNES, censura à direita nas semanas finais, diluição espacial MAUP, colisão de grade climática, pureza `SUTVA` de controles e trava anti-rótulo degenerado (`0 >= 0`), com scorecard `data_hygiene_scorecard.json`
- [ ] Implementar `hypothesis_ledger.py` e catraca científica *Champion-Challenger* (`hypothesis_ledger.json`)
- [ ] Implementar `dossier_compiler.py` (relatório completo `.md` + `.pdf` automático ao fim de cada treino + `Fig. C6–C7`)
