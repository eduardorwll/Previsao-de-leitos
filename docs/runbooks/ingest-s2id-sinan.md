# Runbook — Ingestão S2iD (desastres) e SINAN (agravos)

> Objetivo: obter **desabrigados/desalojados** (S2iD) e **notificações por agravo** (SINAN)
> para enriquecer o painel semanal, respeitando LGPD (agregados, sem registro individual).
> Fronteira: *Not for patient care or live clinical use.*

---

## Parte A — Base dos Dados (SINAN)

### A.1 O que já está resolvido

Os links fornecidos apontam para o dataset **SINAN** (`f51134c2-5ab9-4bbc-882f-f1034603147a`);
a tabela acessível é:

```
basedosdados.br_ms_sinan.microdados_dengue
```

| Item | Valor |
| :--- | :--- |
| Linhas | `33.462.203` |
| Colunas | `152` |
| Partição | **nenhuma** (todo filtro de ano varre a coluna inteira) |
| Acesso | ✅ funciona com o projeto `previsao-leitos` (ADC atual) |
| API key da BD | **não é necessária para consultar**; só para descobrir metadados no site |
| Leptospirose | ❌ não existe como tabela `microdados_leptospirose`; verificar dataset de agravos ou usar S2iD |

### A.2 Custo medido (dry run, 0 B)

```sql
SELECT ano, semana_sintomas, id_municipio_residencia, COUNT(*) AS casos
FROM `basedosdados.br_ms_sinan.microdados_dengue`
WHERE ano BETWEEN 2010 AND 2024 AND sigla_uf_residencia = 'RS'
GROUP BY ano, semana_sintomas, id_municipio_residencia
```

**`532,4 MB`** — acima dos `170 MB` originalmente autorizados (≈ `US$ 0,003`, dentro do
free tier de 1 TB/mês). Como a tabela não é particionada, não há como reduzir sem cortar
colunas — e essas quatro já são o mínimo.

### A.3 Passo a passo

1. **Confirmar autorização de custo** (532 MB vs. 170 MB estimados).
2. **Rodar com teto rígido** (`maximum_bytes_billed=650_000_000`) para nunca estourar:

```bash
uv run --no-sync --env-file .env python - <<'PY'
from google.cloud import bigquery
client = bigquery.Client(project="previsao-leitos")
sql = open("scripts/sql/sinan_dengue_rs_semanal.sql").read()
cfg = bigquery.QueryJobConfig(maximum_bytes_billed=650_000_000, use_query_cache=True)
df = client.query(sql, job_config=cfg, location="US").to_dataframe()
df.to_parquet("data/raw/sinan/rs_dengue_semanal.parquet", index=False)
print(df.shape)
PY
```

3. **Validar**: `n = 0` nulos em `ano/semana`, cobertura `2010–2024`, 497 municípios do RS.
4. **Agregar para região de saúde** (nunca manter município no painel publicado):

```python
# contagens por (região de saúde, ano, semana) — supressão n<5 na publicação
```

5. **Registrar proveniência**: SHA-256 do parquet, consulta, bytes faturados, data.

### A.4 O que fazer com o dado

- Dengue é **sazonal e sensível a água parada** — entra como covariável defasada
  (`0..8` semanas) no Estágio 2, com restrição monotônica positiva.
- Testar na régua (`backtest`) antes de qualquer promoção; se não mover MASE/fase, fica
  documentado como rejeitado (mesmo padrão de F20/F22).

---

## Parte B — Portal S2iD (MIDR)

### B.1 Diagnóstico do portal

| Item | Resultado |
| :--- | :--- |
| URL | `https://s2id.mi.gov.br/paginas/relatorios/` → **HTTP 200** |
| Tecnologia | JSF/PrimeFaces (`.xhtml`, `jsessionid`, `ViewState`) |
| API pública | ❌ `/api` e `/api/` retornam **404** |
| Automação por `curl` | ❌ inviável (exige sessão + `javax.faces.ViewState`) |

### B.2 Exportação manual (passo a passo)

1. Abrir `https://s2id.mi.gov.br/paginas/relatorios/`.
2. No menu, escolher **Relatórios → Reconhecimentos/Desastres** (o relatório que lista
   portarias de reconhecimento e o FIDE).
3. Filtros:
   - **UF:** `RS`
   - **Período:** `01/01/2010` a `31/12/2024`
   - **COBRADE:** `12100` (inundação), `12200` (enxurrada), `12300` (alagamento)
4. Clicar em **Gerar/Pesquisar** e aguardar a tabela.
5. Clicar em **Exportar** (CSV/XLSX) e salvar em `data/raw/s2id/`.
6. Conferir se as colunas incluem: `município`, `data do desastre`, `COBRADE`,
   `decreto`, `desabrigados`, `desalojados`, `enfermos`, `feridos`, `mortos`.

### B.3 Se precisar automatizar

- Requer **navegador headless com sessão** (Playwright/Selenium), não `curl`.
- Receita: abrir a página, selecionar filtros por `id` do componente JSF, capturar o
  download via `page.expect_download()`.
- Alternativa institucional: pedir o extrato ao MIDR via e-SIC, ou usar o conjunto
  "S2iD — Dados Abertos" no `dados.gov.br`.

### B.4 Integração

1. Salvar em `data/raw/s2id/s2id_rs_2010_2024.csv`.
2. Normalizar colunas e mapear município → região de saúde (tabela já usada no painel).
3. Agregar por `(região, ano_epi, semana_epi)` somando desabrigados/desalojados.
4. Entrar como covariável **ex-ante** (semana do evento e defasagens), com teste na régua.

---

## Parte C — Guardas obrigatórias

| Guarda | Regra |
| :--- | :--- |
| LGPD | somente contagens agregadas; nenhum registro individual em disco |
| Supressão | células com `n < 5` não são publicadas |
| Custo | `dry_run` antes; `maximum_bytes_billed` no job real |
| Holdout | anos `>= 2025` nunca incluídos |
| Pré-registro | nova feature precisa de hipótese e teste na régua antes de promoção |
