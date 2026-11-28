# ADR-0008: Processamento de escala em Polars (2 níveis) e remoção do Dask

- **Status:** ACCEPTED
- **Data:** 2026-11-28
- **Contexto:** `Task 2.1` (Escala Nacional — Brasil V2)

## Contexto

A expansão para **27 UFs / 5.570 municípios / ~4,73M linhas** exige processar um painel semanal
que não cabe confortavelmente em uma única passada de memória barata, mantendo a propagação
interestadual exata das Ottobacias e da matriz de referência `W_ref`. O projeto declarava o extra
`scale` com `dask[dataframe,distributed]` como candidato a motor distribuído.

Evidências contra o Dask neste projeto:
- **Nenhum import de `dask`** em `src/` (verificado por `grep`); ele era dependência declarada e não usada.
- O Dask arrasta `distributed`, `tornado`, `toolz`, `partd`, `zict`, `locket`, `tblib` — um grafo
  de dependências grande para um ganho que o `polars` já entrega em nó único.
- O `Polars` 2.x oferece **streaming/lazy** (`scan_parquet`, `sink_parquet`, `collect(engine="streaming")`)
  com uso de memória menor que o Dask para o mesmo volume neste hardware (32 GB RAM).

## Decisão

**Adotar `Polars` 2 níveis** e **remover o `dask`**:
1. **Passe global nacional** (`< 16 GiB`): processamento exato da propagação interestadual e `W_ref`
   em `polars` lazy, sem truncamento de bacias.
2. **Modo streaming portátil** (`< 6 GiB`): `scan_parquet` + `sink_parquet` por partição
   Hive (`uf=.../bacia=...`), executável em máquinas modestas.

O extra `scale` passa a conter apenas `aeon` e `torch-geometric` (benchmarks/modelagem de escala),
sem `dask`.

## Alternativas rejeitadas

- **Manter o Dask para "futuro":** dependência declarada, não usada, com grafo pesado e sem ganho
  demonstrado neste hardware nem neste volume. Removida para reduzir superfície de manutenção e
  o `pip-audit`.
- **Spark/pyarrow dataset como motor primário:** mais infraestrutura operacional do que o necessário
  para um nó único com `Polars`.

## Consequências

- `dask`/`distributed` (e transitivas `tornado`, `toolz`, `partd`, `zict`, `locket`, `tblib`)
  removidos de `pyproject.toml`/`uv.lock`; **235 pacotes** no lock (vs. 243).
- **Data lake 2 níveis implementado** em `src/previsao_leitos/ingestion/data_lake.py`:
  - `write_hive_partitioned` (escrita atômica `.tmp` + `os.replace`, partições `uf=`/`bacia=`),
  - `scan_hive_partitioned` (nível 1, lazy com `hive_partitioning=True`),
  - `sink_hive_partitioned` (nível 2, streaming por partição via `sink_parquet`).
- Se um dia o volume exigir processamento verdadeiramente distribuído, a decisão pode ser
  reavaliada com benchmark medido (registrado em `PERF.md`), não por precaução.
