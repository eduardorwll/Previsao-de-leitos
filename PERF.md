# PERF.md — Diário de Otimizações e Experimentos de Desempenho

Regra obrigatória: **Medir baseline → Alterar 1 variável → Re-medir sob condições idênticas → Manter apenas se superar a variância (neutro = REVERTER).** Todas as tentativas — inclusive as revertidas — ficam registradas abaixo.

## Baseline Histórico Registrado (Protótipo Legado)

| Data | Escopo | Mudança | Baseline | Resultado | Decisão | Observações |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 2026-11-23 | Dataset PyTorch (`previsao_leitos.py`) | Tensor 3D materializado → `IndexedFloodDataset` 2D sob demanda | ~1,7 GB RAM | ~110 MB RAM | **KEPT** | Documentado no `README.md`; precisa preservar integridade de lacunas calendário na Etapa 1 |
| 2026-11-24 | Ambiente local de desenvolvimento | Diagnóstico inicial da máquina de desenvolvimento | AMD Ryzen 7 7735HS (16t), 6,9 GiB RAM (~2,2 GiB livre), sem GPU NVIDIA | Confirmado limite de RAM local | **BASELINE** | Define uso de `polars` streaming + fixtures leves locais e GPU remota (`modal`/Colab) para treino pesado |
| 2026-11-24 | Download BigQuery (`sih_demanda_semanal_agg`, 1,93M linhas) | Adicionar `google-cloud-bigquery-storage` (Arrow gRPC) em vez de paginação REST | ~287,3 s (REST, 1,16M + 1,78M linhas) | **30,6 s** (gRPC, 1,93M linhas) | **KEPT** | ~9,4× mais rápido no streaming para `pl.from_arrow()`; mantido no extra `[data]` |
| 2026-11-24 | Extração hidrológica 100% RS (`30` Regiões × 16 anos = `175.320` dias) | `OpenMeteoEra5Adapter` sequencial (600 u/min -> HTTP 429) vs. `NasaPowerHydroAdapter` paralelo (`max_workers=4`, cache Parquet atômico) | ~6–7 min (Open-Meteo sob rate-limit 600 u/min) | **19,23 s** a frio (`0,94 MB`, `0` nulos) / **1,18 s** em cache (painel completo com `1,93M` linhas SIH) | **KEPT** | Ambos os adaptadores mantidos atrás de `HydroSourcePort`; `NasaPowerHydroAdapter` padrão na Trilha 1A e `OpenMeteoEra5Adapter` (com lote/checkpoint) para refinamento 9 km na Trilha 1B |
| 2026-11-24 | Painel RS v2 com MapBiomas (`232,6k` linhas), `W_ref` (`1,78M` fluxos) e NASA POWER `v2` (`+V10M` vento Sul) | Reconstrução completa `30` Regiões × `2010–2025` (`rs_painel_semanal_regioes_v2.parquet`) | `19,23 s` a frio (sem `V10M`/MapBiomas/`W_ref`) | **22,89 s** a frio (`+V10M`, MapBiomas, DAG ponderado + `W_ref`) | **KEPT** | Overhead mínimo (`+3,6s`) para incluir remanso lagunar, exposição causal de áreas aterradas e grafo assistencial intermunicipal |
| 2026-11-24 | `QuantileGBDTForecaster` (`demand/models.py`) + Suite Etapa 2 (`3` alvos RS + `12` gráficos contrafactuais 300 DPI) | Substituir `4× HistGradientBoosting(loss='quantile')` (com quebra de monotonicidade em `1,44%` das folhas) por Location-Scale `loss='squared_error'` + `monotonic_cst` | `~4×` custo + `1,44%` violação monotônica | **`0,00%` violação monotônica** e pipeline completo (3 alvos + bootstrap + 12 gráficos) em **11,9 s** na CPU | **KEPT** | Garante matematicamente `ADR-0001` e acelera ~4× o treino/avaliação CPU-only |
| 2026-11-24 | Pipeline Etapa 3 (`Conformal 2023` + `SimPy` POA 182 dias + Relatório `Fig. C1–C5`) | Execução completa `run_rs_etapa3_readiness.py` com Common Random Numbers e 5 gráficos 300 DPI | N/A (primeira medição) | **3,9 s** na CPU (`89,1%` cobertura de testes) | **BASELINE** | Permite iterar cenários de estresse composto em hospitais-escola em segundos sem GPU |

## Fila de Gargalos para Medição na Etapa 1

1. Loop Python puro `api_indice_saturacao` (2,5M iterações) → medir vetorização `polars`/Numba.
2. `groupby().transform(lambda s: s.rolling(...))` em pandas → medir `polars.rolling_sum_by("data", ...)`.
3. Transferências redundantes `y_cls.to(device)` (5x/batch) e `sigmoid(logits)` (2x/batch) no treino PyTorch → medir no `LightningModule`.
4. Escaneamento BigQuery `br_ms_cnes.estabelecimento` (teto 12 GB) → medir redução de bytes com `dry_run=True` + filtro `sigla_uf = 'RS'` + `br_ms_cnes.leito`.
