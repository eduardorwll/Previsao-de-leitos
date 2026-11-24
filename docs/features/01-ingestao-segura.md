# Feature F01 — Ingestão Segura, Granularidade de Hospitais-Escola e Contratos LGPD

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída e Validada em Dados Reais RS (`2010–2025`) |
| **Etapa** | Etapa 1 — Trilha 1A (Núcleo Semanal RS) |
| **Especificação** | [spec/SPEC-ingestion.md](../../spec/SPEC-ingestion.md) |
| **Guia de Setup** | [docs/runbooks/setup-bigquery.md](../runbooks/setup-bigquery.md) |
| **Pacote** | `src/previsao_leitos/ingestion/` |
| **Testes** | `39/39` offline (`88%+` cobertura) + `1/1` live BigQuery 0-byte |

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
- [x] Aprovar `spec/SPEC-ingestion.md`
- [x] Implementar `ports.py`, `adapters.py` (`BigQueryAdapter` + `ParquetFixtureAdapter`), `cache.py`, `lgpd.py`, `schemas.py`, `queries.py` e `pipeline.py` em TDD
- [x] Validar 100% da suíte offline (`pytest -m "not slow and not bigquery"`) com cobertura $\ge 80\%$ e validar sintaxe/colunas reais das 5 queries RS via `dry_run` gratuito (`0 bytes` cobrados) no BigQuery
- [x] Corrigir códigos oficiais via `br_ms_cnes.dicionario` (`'5'` = hospital de ensino, `'86'` = UCO III adulto, `'64'/'65'` = UCI) e `br_ms_sih.dicionario` (`STARTS_WITH(id_municipio_estabelecimento, @uf_code)`, `tipo_uti` por tipo de UTI e `COALESCE(cid_principal_subcategoria, cid_principal_categoria)`)
- [x] Executar extração real RS (`2010–2025`) em cache local `data/raw/` (497 mun., 30 Regiões de Saúde, 44 hospitais-escola ativos em mai/2024, cobertura integral 2010–2025 no CNES e SIH-SUS) e implementar `aggregate_regional_weekly_demand` com supressão LGPD ($n < 5$)
