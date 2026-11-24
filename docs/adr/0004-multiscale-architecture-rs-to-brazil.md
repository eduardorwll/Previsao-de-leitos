# ADR-0004: Seleção Arquitetural por Escala (RS vs. Brasil) e Incerteza Hierárquica

- **Status:** PROPOSED
- **Data:** 2026-11-24

## Contexto e Decisão

O histórico do projeto mostrou que `HydroFiLMMonotonicNet` (Bi-GRU + FiLM + ramo monotônico) generaliza melhor em recorte estadual menor (RS), enquanto `HydroTFTNet` (VSN + LSTM + atenção) aproveita o volume nacional (~2,5M linhas). Entretanto, o código atual não inclui `HydroFiLMMonotonicNet`, não modela acoplamento montante→jusante nem transbordamento entre hospitais, e produz apenas estimativas pontuais descalibradas.

**Decisão:** Adotar uma **abordagem mista (híbrida)** em duas escalas sob um mesmo harness `pytorch-lightning` + baselines estatísticos/tabulares (`statsmodels` com *Distributed Lags*, `scikit-learn` quantil, `aeon`, `TimesFM 2.5`):
- **V1 (RS Hospitais-Escola — Abordagem Mista Semanal $h=1..4$ semanas):**
  1. *Camada Hidrológica:* `HydroFiLMMonotonicNet` (RS) com propagação montante→jusante (`Ottobacias ANA` + `MERGE-CPTEC` + `HAND/SAR`) explicando o escoamento das regiões altas do Norte/Serra para os vales e o Guaíba.
  2. *Camada de Demanda por Especialidade (UTI Adulto, UTI Ped/Neo, Clínico, Cirúrgico):* modelos de **Defasagem Distribuída** (`statsmodels` NegBin/Hurdle + `HistGradientBoosting` Quantílico $P_{50}/P_{90}/P_{95}$) evoluindo para **PyMC Hierárquico** (`Hospital-Escola ⊂ Macrorregião RS`) + **`scikit-survival`** para LOS.
  3. *Camada de Prontidão:* simulador de filas **`SimPy`** para leitos de contingência e adiamento de eletivas.
- **V2 (Escala Brasil — 27 UFs):** `HydroTFTNet` multi-escala (janelas 15d–180d estratificadas por bioma/regime hidrológico) + **Hetero-GNN (`torch-geometric`)** acoplando grafos dirigidos de bacias (`Ottobacias`) e de referência/transbordamento hospitalar (`MUNIC_RES → CNES`), integrado a **`pymoo`** (NSGA-II/III) para alocação nacional de contingência.

## Alternativas Rejeitadas

- **Usar apenas `HydroTFTNet` e GBDT L1 em todas as escalas:** sofre overfitting no RS, ignora topologia de bacias/transferências e não quantifica incerteza de cauda para eventos extremos raros.
- **Adotar TimesFM 3.0 como núcleo:** pesos têm restrição não-comercial/não-produção; mantemos apenas **TimesFM 2.5 (Apache-2.0)** como benchmark probabilístico zero-shot.

## Consequências

Cada salto de complexidade arquitetural (ex.: PyMC vs. GLM, PyG vs. `HydroTFTNet`) só é promovido se superar os baselines mais simples na validação temporal/espacial sob critério pré-definido.
