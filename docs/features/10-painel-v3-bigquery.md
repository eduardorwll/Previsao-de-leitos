# Feature F10 — Reextração BigQuery `v3` e Painel Causal `RS v3` (`Task 1.2.1`)

> **Status:** ✅ Concluída (`RS <= 2024`, holdout `2025` e `2026` preservados intocados) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** Zero PII, agregação na fronteira SQL, supressão $n < 5$ ([ADR-0005](../adr/0005-data-layer-and-lgpd.md)).

---

## 1. Objetivo

Substituir a variável-alvo semanal por data de admissão (`v2`) pelo **censo ativo semanal real de pacientes-dia e diárias de UTI (`v3`)**, corrigir a diluição espacial `MAUP` via ponderação populacional causal (`MapBiomas`), extrair a matriz de referência externa pré-LGPD (`W_ref` inter-regional `<= 2022`) e gerar o **triângulo de atraso de processamento (`sih_delay_triangle`)** para uso do `ChainLadderNowcaster` no Marco `M2` (`2026` prospectivo).

---

## 2. Descobertas Empíricas no BigQuery (`0 bytes` na inspeção)

- **`basedosdados.br_ms_sih.aihs_reduzidas` (`215,2M` linhas, `131,24 GB`):** a inspeção via `inspect_columns()` e a checagem controlada em `2024-05` (`64.387` AIHs do RS) comprovaram que a coluna `sigla_uf` é `INT64` e **100% `NULL`** na tabela da Base dos Dados. O filtro seguro por UF permanece `STARTS_WITH(id_municipio_estabelecimento, @uf_code)`.
- **Extração única cacheada em `data/raw/` (`21,65 s`, `~26,3 GB` lidos dentro do *free tier* mensal de `1 TB`):**
  - `sih_censo_ativo_semanal` (`v2`, janela `2010–2025` com folga de borda `ano BETWEEN 2009 AND 2026`, sem double-count `+1` no dia da alta e sem leitura de `cid_principal_subcategoria`): **`1.305.728` linhas**.
  - `sih_w_ref_externa_treino` (`v2`, `origem != destino`, `<= 2022`, folga `ano <= 2023`, `HAVING COUNT(*) >= 5`): **`11.190` arestas municipais** (`778.705` internações inter-regionais de urgência/UTI).
  - `sih_delay_triangle` (`v1`, `2020–2026`, `delay_weeks in 0..12` com `cumulative_count` via janela SQL): **`1.929` células**.

---

## 3. Alterações de Engenharia Implementadas

- **Consultas e Contratos (`ingestion/queries.py`, `schemas.py`, `pipeline.py`):**
  - Saneado `_SQL_SIH_ACTIVE_CENSUS` (`v2`) com cálculo exato de `n_dias_ativos` (eliminando o `+1` no dia da alta e garantindo `diaria_uti_dia <= 1,0` por paciente-dia) e folga de borda `ano BETWEEN @ano_inicio - 1 AND @ano_fim + 1`.
  - Adicionada `build_sih_delay_triangle_query` + `validate_sih_delay_triangle_frame` + `extract_rs_v3_sih_pack` (e flag opcional `include_v3_sih` em `extract_rs_core_health_pack`).
- **Construtor Canônico Causal `RS v3` (`geo/v3_panel.py`, `geo/referral_graph.py`, `scripts/build_rs_panel_v3.py`):**
  - `build_rs_v3_panel_from_frames` integra de forma pura e determinística: censo ativo SIH, `CNES` mensal causal (`join_asof backward`), `MapBiomas` ponderado por população `<= 2022` (`mun_weights`), propagação multi-salto (`multi_hop=True`) e `W_ref` estritamente inter-regional (`exclude_self_region=True`).
  - Gerado `data/processed/rs_painel_semanal_regioes_v3.parquet` (`25.020` linhas, `30` regiões, `45` colunas, `0,59 s` no rebuild offline).
- **Testes TDD e *Prove-It* (`tests/unit/test_v1_2_1_bigquery_v3.py`):**
  - Prova determinística de **zero vazamento futuro** (perturbar `2024+` em todas as tabelas mantém `100%` das linhas `<= 2022` bit-a-bit idênticas), integração do triângulo com `ChainLadderNowcaster`, extração offline com fixtures e verificação anti-auto-referência regional.

---

## 4. Resultados Auditados em `RS 2024` (`rs_v1_2_1_panel_v3_2024`, Holdout `2025` Intocado)

| Indicador Auditado | `V1.2.0` (Painel `v2` Admissão) | `V1.2.1` (Painel `v3` Censo Ativo) | Diagnóstico Científico |
| :--- | :---: | :---: | :--- |
| **`Gate G5` (MAUP)** | `FAIL (7,90x)` | **`PASS (1,00x)`** | Diluição espacial eliminada no painel `v3` |
| **`Gate G3` (Censo vs. CNES SUS)** | `FAIL (316,3%)` | `FAIL (230,1%)` (`p99 = 130,0%` vs. CNES existente) | Queda de `-86 p.p.`; picos remanescentes refletem leitos de contingência/contratualizados em W29–33/2024 |
| **`Gate G2` / `G8` / `G4`** | `PASS` / `PASS` / `PASS` | **`PASS (0,00%)` / `PASS (0)` / `WARN (11,5%)`** | `G4` reflete atraso real de faturamento no fim de 2024 |
| **Viés BSTS Pré-Choque (`W15–17/2024`)** | `+8,10` diárias/sem | **`+2,21` diárias/sem (`0,32` leitos/dia)** | Série suave de censo ativo quase zera viés pré-choque |
| **Campeão `V1` MAE / `Pinball95` (`leitos/dia`)** | `6,91` / `8,69` (`MASE 0,810`) | **`3,52` / `4,90`** (`MASE 0,989`, `MASE_saz 0,612`) | Erro real cai à metade; persistência fica mais forte (`24,95` vs `59,77`) na série contínua |
| **`F04.1` Ortogonal MAE / `Pinball95`** | `7,99` / `10,14` | **`5,78` / `8,46`** (`MASE 1,623`, `0,00%` viol.) | Catraca `REJECT` honesta; exige OOF real (`1.2.2`) e basal regional lento |
| **Oráculo El Niño (`P50` médio)** | `230,89 → 234,57` | **`235,40 → 235,80`** (`+21,6` em `P90`) | Estritamente crescente |

- **Próximo passo:** `Task 1.2.2` (Ingestão Ao Vivo Onda 2A — `NOAA`, `Zenodo 16929471`, `GERINT-POA`, `SIVEP/SINAN` — + empacotamento `cpu`/`rocm`/`cuda` + `F03.1 HydroFiLM-ENSO` OOF causal real em GPU para zerar `G1`).
