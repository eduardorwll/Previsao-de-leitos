# Feature F02 — Geoespacial, Hidrologia 100% RS e *Tracer Bullet* Semanal

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída (F02 + F02.1: 30/30 Regiões RS, Aterros/Pôlderes, Remanso e Grafo `W_ref`) |
| **Etapa** | Etapa 1 — Trilha 1A + Evolução Intermunicipal F02.1 |
| **Especificação** | [spec/SPEC-geo-hydro-etl.md](../../spec/SPEC-geo-hydro-etl.md) · [ADR-0006](../adr/0006-landfill-polders-and-intermunicipal-routing.md) |
| **Pacote Implementado** | `src/previsao_leitos/geo/` + `scripts/run_rs_tracer_bullet.py` |
| **Verificação** | `ruff` OK · `mypy --strict` OK · `check_constraints.py` OK · `60` testes (`88,5%` cov.) · `pip-audit` OK |

> **Fronteira Científica:** Todos os resultados abaixo são exclusivamente de pesquisa e avaliação metodológica (*Not for patient care or live clinical use*). O ano **`2025`** permanece **100% intocado** como *holdout* final, protegido por exceção explícita (`LeakageViolationError`).

---

## 1. Objetivo e Arquitetura Entregue

Conectar os dados reais de saúde do RS (extraídos em `data/raw/` na feature **F01**) a um painel hidrológico diário→semanal cobrindo **100% das 30 Regiões de Saúde (497/497 municípios do RS)**, com propagação estritamente montante→jusante (`RS_BASIN_EDGES`) e grade calendário à prova de vazamento temporal, entregando o primeiro baseline *ex-ante* com **Defasagens Distribuídas** ($h \in \{1,2,3,4\}$ semanas, foco em $h=2$ semanas / D+14).

### Módulos em `src/previsao_leitos/geo/`

| Arquivo | Responsabilidade | Cobertura |
| :--- | :--- | :---: |
| [`calendar_grid.py`](../../src/previsao_leitos/geo/calendar_grid.py) | Grade diária e semanal ISO sem compressão de lacunas (`null` != `0.0 mm`), `API` estritamente causal ($t-1$ e anteriores) e *distributed lags* | `84%` |
| [`basin_graph.py`](../../src/previsao_leitos/geo/basin_graph.py) | Grafo dirigido acíclico (`DAG`) de bacias RS montante→jusante (`RS_BASIN_EDGES` + `RS_BASIN_WEIGHTED_EDGES` com `weight` e `travel_weeks`), remanso estuarino por vento Sul (`compute_estuarine_backwater_index`) e mapeamento 30/30 | `92%` |
| [`hydro_sources.py`](../../src/previsao_leitos/geo/hydro_sources.py) | Interface `HydroSourcePort` com `NasaPowerHydroAdapter` (`PRECTOTCORR`, `GWETTOP`, `T2M`, **`V10M` vento Sul**, sentinela `-999.0` → `null`), `OpenMeteoEra5Adapter` (lote + checkpoint + retry 429) e `HydroFixtureAdapter` offline | `89%` |
| [`landfill_exposure.py`](../../src/previsao_leitos/geo/landfill_exposure.py) | Exposição causal em áreas aterradas/várzeas (`MapBiomas <= ref_year`: Urbano `24`, Banhado `11`, Marisma `84`, Água `33`) combinada a fração de municípios com sistemas de pôlderes/diques (`RS_POLDER_MUNICIPALITIES`) | `91%` |
| [`referral_graph.py`](../../src/previsao_leitos/geo/referral_graph.py) | Grafo assistencial intermunicipal causal $W_{\text{ref}}$ (`MUNIC_RES → CNES`, congelado em `<= train_end_year`) calculando pressão hidrológica da área de captação | `90%+` |
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

### Ablação F02 vs. F02.1 — Aterros, Remanso (`V10M`) e Acoplamento Intermunicipal (`h=2` sem., 2024)

| Alvo (30 Regiões RS, Treino `<= 2022`) | MAE F02 | MAE F02.1 | MASE F02 | MASE F02.1 | $R^2_\Delta$ F02 | $R^2_\Delta$ F02.1 | Destaque Regional F02.1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Diárias UTI Adulto** | `55,82` | **55,70** | `0,934` | **0,932** | `+0,065` | `+0,053` | **Sul (Pelotas/RG):** `0,909 → 0,900` · **Vale do Caí:** `0,997 → 0,990` |
| **Leptospirose (Onda 2)** | `0,37` | **0,36** | `0,930` | **0,923** | `+0,429` | **+0,437** | **Capital/POA:** **`0,715 → 0,704`** (`-29,6%` erro vs. Persistência) |
| **Internações Clínicas** | `25,22` | **24,97** | `1,021` | **1,011** | `-0,032` | **-0,013** | Redução de erro sob choque de acesso via pressão da rede `W_ref` |
| **Respiratório / SRAG** | **12,16** | `12,22` | **0,875** | `0,879` | `+0,239` | `+0,223` | Leve variância no Ridge por região (exige seleção/pooling na Etapa 2) |

> **Nota técnica importante:** No *Tracer Bullet* linear (`Ridge` ajustado separadamente por região), atributos estruturais estáticos (como `indice_aterro_varzea`) atuam como fator de escala constante por entidade e só expressam seu ganho completo em **modelos globais/hierárquicos entre regiões** (`PyMC`, `HistGradientBoosting` Quantílico e `HydroFiLMMonotonicNet` na Etapa 2). Ainda assim, o índice causal (`MapBiomas <= 2022`) ranqueou **Capital/POA (`#1`)**, **Vale dos Sinos (`#2`)**, **Vale do Caí/Canoas (`#4`)** e **Sul/Pelotas-RG (`#6`)** no topo de vulnerabilidade física **sem tocar nenhum dado de 2024**.

---

## 3. Checklist da Feature F02

- [x] Redigir e aprovar especificação [spec/SPEC-geo-hydro-etl.md](../../spec/SPEC-geo-hydro-etl.md)
- [x] Implementar `calendar_grid.py` (grade diária/semanal completa + API causal + lags distribuídos) com testes *Prove-It*
- [x] Implementar `basin_graph.py` (DAG ponderado de bacias RS montante→jusante com `travel_weeks` + remanso estuarino `V10M`)
- [x] Implementar `hydro_sources.py` (`NasaPowerHydroAdapter` com `V10M`, `OpenMeteoEra5Adapter` e `HydroFixtureAdapter`)
- [x] Implementar `landfill_exposure.py` (exposição causal `MapBiomas <= 2022` de áreas aterradas/banhados + sistemas de pôlderes) e `referral_graph.py` (matriz intermunicipal causal `W_ref` sobre `1,78M` fluxos `SIH-SUS`)
- [x] Implementar `panel_builder.py` + executar ablação real RS (`F02` vs. `F02.1`, `2023` e `2024`, `2025` bloqueado)
- [x] **Física Fina V1.1 (`F02.2`):** corrigir `BasinTopology.propagate_weighted_upstream` para convolução transitiva multi-salto completa (`transitive_upstream_paths`), adicionar ponderação populacional/leitos anti-MAUP em `build_causal_landfill_exposure` e implementar agregadores de `ANA HidroWeb`, `MERGE-CPTEC 0,1°` e `SAR` 2024 em `geo/fine_hydro_sources.py`
- [ ] **Teleconexões El Niño / ENSO (`F02.3`):** implementar conector causal de índices `ONI`, `Niño 3.4`, `SOI`, `MEI.v2`, `TSA` e `SALLJ` com lags `1..24` semanas
