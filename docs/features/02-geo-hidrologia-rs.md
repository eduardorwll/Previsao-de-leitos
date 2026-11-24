# Feature F02 — Geoespacial, Hidrologia 100% RS e *Tracer Bullet* Semanal

| Campo | Valor |
| :--- | :--- |
| **Status** | 🔄 Especificação pronta (aguardando aprovação) |
| **Etapa** | Etapa 1 — Trilha 1A (Fatia 1A.3) |
| **Especificação** | [spec/SPEC-geo-hydro-etl.md](../../spec/SPEC-geo-hydro-etl.md) |
| **Pacote Alvo** | `src/previsao_leitos/geo/` |

---

## 1. Objetivo

Conectar os dados reais de saúde do RS (já extraídos em `data/raw/` pela feature **F01**) a um painel hidrológico semanal cobrindo **100% dos 497 municípios do RS**, com propagação montante→jusante (`Ottobacias ANA`) e grade calendário livre de vazamento, entregando o primeiro modelo *ex-ante* com defasagens distribuídas ($h \in \{1,2,3,4\}$ semanas, foco em $h=2$ semanas) por **UTI específica e Região de Saúde**.

---

## 2. Checklist da Feature F02

- [x] Redigir especificação [spec/SPEC-geo-hydro-etl.md](../../spec/SPEC-geo-hydro-etl.md)
- [ ] Aprovar `spec/SPEC-geo-hydro-etl.md`
- [ ] Implementar `calendar_grid.py` (grade diária/semanal completa + API vetorizado sem vazamento) com testes *Prove-It*
- [ ] Implementar `basin_graph.py` (DAG de bacias RS montante→jusante) e `hydro_sources.py` (`MERGE-CPTEC`, `ANA HidroWeb`, `S2iD`)
- [ ] Implementar `panel_builder.py` + primeiro *Tracer Bullet* semanal ($h=1..4$ sem.) vs. Persistência e Sazonal-Ingênuo no RS
