# Especificação V1 — Módulo `geo` + ETL Hidrológico RS (`spec/SPEC-geo-hydro-etl.md`)

- **Status:** `implemented` *(implementado em TDD, verificado em 30/30 Regiões de Saúde do RS no Tracer Bullet 2023–2024)*
- **Etapa / Feature:** Etapa 1 — Trilha 1A (Fatia 1A.3: Feature `F02 — Geoespacial, Hidrologia 100% RS e Tracer Bullet Semanal`)
- **Escopo V1:** 497 municípios e 30 Regiões de Saúde do Rio Grande do Sul (`2010–2025`), com grade calendário diária/semanal sem lacunas e primeiro baseline *ex-ante* com defasagem distribuída ($h \in \{1, 2, 3, 4\}$ semanas, foco em $h = 2$ semanas / D+14).

---

## 1. Problema que Esta Especificação Resolve

1. **Fim da exclusão silenciosa de ~90% dos municípios:** no protótipo legado, o *inner join* com estações INMET deixava de fora ~90% dos municípios do RS. Passamos a usar **fontes contínuas em grade atrás da interface `HydroSourcePort`** (`NasaPowerHydroAdapter` — `PRECTOTCORR`/`GWETTOP`/`T2M` sem chave e `OpenMeteoEra5Adapter` — `ERA5-Land` com cache por entidade e retry HTTP 429) cobrindo **30/30 Regiões de Saúde / 497/497 municípios (100% do RS)** sem lacunas silenciosas.
2. **Sinal físico real e propagação montante→jusante:** acoplamos **cotas fluviométricas reais (ANA HidroWeb / SGB-SACE)** e o **grafo dirigido de Ottobacias (ANA)**, capturando explicitamente como a chuva nas regiões altas do Norte/Serra escoa para os vales (Taquari-Antas, Jacuí, Caí, Sinos) e o Guaíba/Lagoa dos Patos.
3. **Integridade de grade calendário e zero vazamento:** substituímos `rolling`/`shift` por índice de linha e `dropna()` destrutivo por **grade calendário completa** (diária → semana epidemiológica ISO), garantindo que nenhum dado futuro vaze para o treino (`<= 2022`), calibração (`2023`) ou holdout intocado (`2025`).
4. ***Tracer Bullet* Semanal (*Abordagem Mista*):** acoplamos o painel hidrológico semanal 100% RS à demanda real `SIH-SUS` (já em `data/raw/`) para rodar o **primeiro baseline *ex-ante* com Defasagens Distribuídas (*Distributed Lags*, lags $0..4$ semanas)** prevendo internações e diárias por especialidade (**UTI Adulto, UTI Pediátrica, UTI Neonatal, Clínico, Cirúrgico**) e por **Região de Saúde / Hospital-Escola** — comparado explicitamente contra **Persistência** e **Sazonal-Ingênuo**.

---

## 2. Estrutura do Módulo e Contratos

```text
src/previsao_leitos/geo/
├── __init__.py          # Superfície pública
├── calendar_grid.py     # Grade calendário diária/semanal completa + API vetorizado + janelas por data real
├── basin_graph.py       # Grafo dirigido de bacias (Ottobacias ANA -> agregação chuva/cota a montante)
├── hydro_sources.py     # Adaptadores para MERGE-CPTEC, ANA HidroWeb, S2iD FIDE + Fixture offline
└── panel_builder.py     # Cruzamento causal semanal (Hidrologia montante + Oferta CNES + Demanda SIH)
```

### Garantias Obrigatórias (Testes *Prove-It*)
- **Zero vazamento temporal (`test_leakage.py`):** perturbar qualquer valor em `2024–2025` deve deixar features, escaladores e previsões de treino (`<= 2022`) e calibração (`2023`) **bit-idênticos**.
- **Integridade de lacunas calendário (`test_calendar_grid.py`):** inserir um buraco de 10 dias na série de chuva não pode comprimir janelas móveis nem aplicar decaimento `API` como se os dias fossem adjacentes; ausência de leitura fica explicitamente como `null` (jamais confundida com `0.0 mm`).
- **Direcionalidade hidrológica (`test_basin_graph.py`):** o grafo de bacias é estritamente acíclico (`DAG`); chuva na sub-bacia a montante eleva o acumulado a montante na sub-bacia a jusante, mas chuva a jusante **nunca** altera a montante.
- **Métricas honestas no *Tracer Bullet*:** reportar MAE, MASE, $R^2_\Delta$ e *pinball loss* ($P_{90}$) por **Região de Saúde** e por **UTI específica**, sempre lado a lado com **Persistência** ($\hat{y}_{t+h} = y_t$) e **Sazonal-Ingênuo** ($\hat{y}_{t+h} = y_{t+h-52}$).

---

## 3. Splits Temporais Congelados (Conforme `ADR-0003`)

| Partição | Período | Uso Permitido |
| :--- | :---: | :--- |
| **Treino** | `2010–2022` | Ajuste de parâmetros, escaladores e imputadores |
| **Calibração / Seleção** | `2023` | Seleção de hiperparâmetros, *early stopping* e calibração probabilística |
| **Auditoria de Desenvolvimento RS** | `2024` | Diagnóstico detalhado da catástrofe de maio/2024 (Onda 1 + Onda 2 leptospirose/SRAG) |
| **Holdout Intocado** | `2025` | Avaliação final única na conclusão da V1 RS (proibido usar para tuning) |

---

## 4. Escopo Adiado (*Out of Scope — Deferred* para Trilha 1B e Etapa 2)

- Rasters `Sentinel-1 SAR`, `HAND` (`Copernicus DEM`), `ERA5-Land`, `GERINT-POA` e malha viária `OSMnx` → **Trilha 1B** (acoplados sobre o mesmo `panel_builder` sem quebrar contratos).
- Redes neurais `HydroFiLMMonotonicNet` (PyTorch Lightning), `PyMC` hierárquico, `scikit-survival` LOS e `SimPy` → **Etapas 2 e 3**.
