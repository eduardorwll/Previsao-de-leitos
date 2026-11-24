# Feature F02 — Geoespacial, Hidrologia 100% RS e *Tracer Bullet* Semanal

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída (30/30 Regiões de Saúde do RS — *Tracer Bullet* 2023–2024) |
| **Etapa** | Etapa 1 — Trilha 1A (Fatia 1A.3 — conclusão da Trilha 1A) |
| **Especificação** | [spec/SPEC-geo-hydro-etl.md](../../spec/SPEC-geo-hydro-etl.md) |
| **Pacote Implementado** | `src/previsao_leitos/geo/` + `scripts/run_rs_tracer_bullet.py` |
| **Verificação** | `ruff` OK · `mypy --strict` OK · `check_constraints.py` OK · `54` testes (`88,2%` cov.) · `pip-audit` OK |

> **Fronteira Científica:** Todos os resultados abaixo são exclusivamente de pesquisa e avaliação metodológica (*Not for patient care or live clinical use*). O ano **`2025`** permanece **100% intocado** como *holdout* final, protegido por exceção explícita (`LeakageViolationError`).

---

## 1. Objetivo e Arquitetura Entregue

Conectar os dados reais de saúde do RS (extraídos em `data/raw/` na feature **F01**) a um painel hidrológico diário→semanal cobrindo **100% das 30 Regiões de Saúde (497/497 municípios do RS)**, com propagação estritamente montante→jusante (`RS_BASIN_EDGES`) e grade calendário à prova de vazamento temporal, entregando o primeiro baseline *ex-ante* com **Defasagens Distribuídas** ($h \in \{1,2,3,4\}$ semanas, foco em $h=2$ semanas / D+14).

### Módulos em `src/previsao_leitos/geo/`

| Arquivo | Responsabilidade | Cobertura |
| :--- | :--- | :---: |
| [`calendar_grid.py`](../../src/previsao_leitos/geo/calendar_grid.py) | Grade diária e semanal ISO sem compressão de lacunas (`null` != `0.0 mm`), `API` estritamente causal ($t-1$ e anteriores) e *distributed lags* | `84%` |
| [`basin_graph.py`](../../src/previsao_leitos/geo/basin_graph.py) | Grafo dirigido acíclico (`DAG`) de bacias hidrográficas do RS montante→jusante (`RS_BASIN_EDGES`) e mapeamento 30/30 das Regiões de Saúde | `93%` |
| [`hydro_sources.py`](../../src/previsao_leitos/geo/hydro_sources.py) | Interface `HydroSourcePort` com `NasaPowerHydroAdapter` (MERRA-2/IMERG `PRECTOTCORR`, `GWETTOP`, `T2M`, sentinela `-999.0` → `null`), `OpenMeteoEra5Adapter` (lote + checkpoint + retry 429) e `HydroFixtureAdapter` offline | `88%` |
| [`panel_builder.py`](../../src/previsao_leitos/geo/panel_builder.py) | Painel semanal causal, política congelada de splits (`SplitPolicy`, bloqueio físico de `2025`) e avaliador *Tracer Bullet* vs. Persistência e Sazonal-Ingênuo | `90%` |

---

## 2. Resultados Reais do *Tracer Bullet* Semanal RS (30/30 Regiões de Saúde)

- **Dados integrados:** `175.320` registros diários NASA POWER (`2010–2025`, `0` nulos, `0,94 MB` em cache atômico) + `1,93M` linhas `SIH-SUS v3` + `CNES v2` + `IBGE v1` → painel de **`25.020` semanas-região** construído em **`1,18s`** em cache (`19,2s` a frio).
- **Regime estritamente *ex-ante*:** treino congelado em `<= 2022`; avaliação em `2023` (calibração) e `2024` (auditoria de desenvolvimento — ano da enchente histórica); **`2025` bloqueado**.
- **Features causais (`lags 0..4` semanas):** `chuva_sem_mm`, `chuva_montante_mm`, `api_max_sem`, `api_montante_max`, `umidade_solo_sem`, `temp_media_sem_c`, além de nível de origem `y_origin` e sazonalidade cíclica da semana-alvo.

### Avaliação 2023 (Calibração) e 2024 (Auditoria Dev — Enchente Histórica)

| Alvo (30 Regiões RS) | Ano | $h$ (sem.) | MAE Modelo | MAE Persist. | MAE Sazonal | MASE vs. Pers. | MASE vs. Saz. | $R^2_\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diárias UTI Adulto** | 2023 | `h=2` | **44,52** | 49,85 | 53,39 | **0,893** | **0,834** | `+0,183` |
| **Diárias UTI Adulto** | 2024 | `h=1` | **53,89** | 56,68 | 65,92 | **0,951** | **0,818** | `+0,029` |
| **Diárias UTI Adulto** | 2024 | `h=2` | **55,82** | 59,77 | 65,92 | **0,934** | **0,847** | `+0,065` |
| **Diárias UTI Adulto** | 2024 | `h=4` | **56,58** | 60,77 | 65,92 | **0,931** | **0,858** | `+0,094` |
| **Respiratório / SRAG** | 2023 | `h=2` | **9,47** | 10,80 | 11,57 | **0,877** | **0,819** | `+0,232` |
| **Respiratório / SRAG** | 2024 | `h=2` | **12,16** | 13,90 | 17,32 | **0,875** | **0,702** | `+0,239` |
| **Respiratório / SRAG** | 2024 | `h=4` | **14,01** | 17,45 | 17,32 | **0,803** | **0,808** | `+0,386` |
| **Leptospirose (Onda 2)** | 2024 | `h=2` | **0,37** | 0,39 | 0,37 | **0,930** | `1,000` | `+0,429` |
| **Internações Clínicas** | 2024 | `h=2` | `25,22` | **24,71** | 34,45 | `1,021` | **0,732** | `-0,032` |
| **Internações Clínicas** | 2024 | `h=4` | **29,59** | 31,85 | 34,45 | **0,929** | **0,859** | `+0,248` |

### Recorte Regional — Heterogeneidade Física Validada (`h=2` sem., 2024)

| Região de Saúde (RS) | Posição na Bacia | Diárias UTI Adulto (MASE vs. Pers.) | Respiratório / SRAG (MASE vs. Pers.) |
| :--- | :--- | :---: | :---: |
| **Região 10 — Capital e Vale do Gravataí (POA)** | Jusante (`guaiba_capital`) | **0,895** (`-10,5%` erro) | **0,855** (`-14,5%` erro) |
| **Região 21 — Sul (Pelotas / Rio Grande)** | Jusante (`lagoa_patos_sul`) | **0,909** (`-9,1%` erro) | **0,876** (`-12,4%` erro) |
| **Região 29 — Vales e Montanhas (Taquari)** | Vale receptor (`taquari_antas`) | **0,918** (`-8,2%` erro) | **0,842** (`-15,8%` erro) |
| **Região 17 — Planalto (Norte alto)** | Cabeceira (`planalto_alto_jacui`) | `0,950` | `0,908` |
| **Região 23 — Caxias e Hortênsias (Serra)** | Cabeceira (`serra_taquari_antas`) | `1,001` | `0,850` |

### Diagnóstico Honesto que Guia a Etapa 2
1. **Onde o baseline linear já funciona:** em **Diárias de UTI Adulto** e **Respiratório/SRAG**, a combinação de grade contínua + propagação de chuva montante→jusante + defasagens distribuídas supera ambos os baselines em todos os horizontes ($h=1..4$ sem.), com os maiores ganhos exatamente nas bacias receptoras de jusante (Capital/POA, Sul e Vale do Taquari).
2. **Onde o baseline linear mostra seu limite estrutural (motivação direta da Etapa 2):**
   - **Leptospirose (Onda 2 esparsa):** supera Persistência em 2024 (`MASE = 0,930`, $R^2_\Delta = +0,429$), mas um modelo gaussiano simétrico (`Ridge`) não captura excesso de zeros nem cauda extrema (`6 → 42` casos na Sem. 20) — exigindo **Binomial Negativa / Hurdle (`statsmodels`)** e **Regressão Quantil ($P_{90}/P_{95}$)**.
   - **Internações Clínicas / UTI em `h=1` de 2024:** a queda abrupta de acesso nas Semanas 18–19 (`-22%`) gera assimetria negativa que penaliza modelos lineares sem informação de bloqueio viário (`OSMnx`) e fila reprimida (`GERINT-POA`).

---

## 3. Checklist da Feature F02

- [x] Redigir e aprovar especificação [spec/SPEC-geo-hydro-etl.md](../../spec/SPEC-geo-hydro-etl.md)
- [x] Implementar `calendar_grid.py` (grade diária/semanal completa + API causal + lags distribuídos) com testes *Prove-It*
- [x] Implementar `basin_graph.py` (DAG de bacias RS montante→jusante + mapeamento 30/30 Regiões de Saúde)
- [x] Implementar `hydro_sources.py` (`NasaPowerHydroAdapter`, `OpenMeteoEra5Adapter` com batch/retry 429/checkpoint e `HydroFixtureAdapter`)
- [x] Implementar `panel_builder.py` + executar *Tracer Bullet* semanal real RS (`2023` e `2024`, $h=1..4$ sem.) com bloqueio físico do holdout `2025`
