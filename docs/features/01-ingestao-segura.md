# Feature F01 — Ingestão Segura, Granularidade de Hospitais-Escola e Contratos LGPD

| Campo | Valor |
| :--- | :--- |
| **Status** | 🔄 Especificação pronta (aguardando aprovação) |
| **Etapa** | Etapa 1 — Trilha 1A (Núcleo Semanal RS) |
| **Especificação** | [spec/SPEC-ingestion.md](../../spec/SPEC-ingestion.md) |
| **Guia de Setup** | [docs/runbooks/setup-bigquery.md](../runbooks/setup-bigquery.md) |
| **Pacote Alvo** | `src/previsao_leitos/ingestion/` |

---

## 1. Por Que Esta Feature Existe

Para responder à pergunta central do projeto — *"diante de uma enchente nesta semana $t$, quantos leitos de UTI Adulto, UTI Pediátrica/Neonatal, Clínico e Cirúrgico cada hospital-escola e macrorregião do RS precisará em $t+1..t+4$ semanas (foco em $h=2$ semanas)?"* — precisamos corrigir simultaneamente quatro limitações do protótipo legado:

1. **Segurança e controle de custo no BigQuery:** substituir SQL por f-string por `ScalarQueryParameter` + checagem prévia `dry_run=True` contra teto configurável (`PREVISAO_MAX_BQ_MB`).
2. **Hospitais-escola e UTI específica:** extrair CNES no grão de estabelecimento cruzando `estabelecimento`, `leito` e `estabelecimento_ensino`, separando tipos de UTI e leitos SUS vs. privados.
3. **Demanda semanal real sem PII:** agregar internações (`br_ms_sih.aihs_reduzidas`) e agravos pós-enchente (`SINAN` Leptospirose) diretamente na origem em granularidade semanal, aplicando supressão primária e complementar de pequenas células ($n < 5$).
4. **Testabilidade offline e cache determinístico:** isolar acesso externo atrás de `TabularSourcePort` (`BigQueryAdapter` + `ParquetFixtureAdapter`) com cache atômico indexado por SHA-256.

---

## 2. Checklist de Progresso da Feature F01

- [x] Definir escopo, contratos `TabularSourcePort`, queries parametrizadas e regra LGPD em [spec/SPEC-ingestion.md](../../spec/SPEC-ingestion.md)
- [x] Criar guia passo a passo de configuração segura do BigQuery sem chaves `.json` em [docs/runbooks/setup-bigquery.md](../runbooks/setup-bigquery.md)
- [ ] Aprovar `spec/SPEC-ingestion.md`
- [ ] Implementar `ports.py`, `adapters.py` (`BigQueryAdapter` + `ParquetFixtureAdapter`), `cache.py`, `lgpd.py`, `schemas.py` e `queries.py` em TDD
- [ ] Validar 100% da suíte offline (`pytest -m "not slow and not bigquery"`) com cobertura $\ge 80\%$
- [ ] Executar extração inicial RS (`2010–2025`) com `dry_run=True` e registrar manifesto de auditoria
