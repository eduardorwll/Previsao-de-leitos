# Previsão de Leitos em Hospitais-Escola sob Risco de Enchentes

Projeto de **pesquisa aberta** voltado à previsão *ex-ante* da **necessidade de leitos hospitalares (demanda, ocupação e *surge capacity*) em hospitais-escola** frente a eventos de enchente — com foco inicial no **Rio Grande do Sul (V1)** e expansão planejada para **escopo nacional — Brasil (V2)**.

> **Aviso de Escopo Científico e Operacional:** Este repositório destina-se exclusivamente à pesquisa metodológica e ao planejamento agregado de capacidade hospitalar. Seus modelos e relatórios **não** constituem recomendação clínica, triagem assistencial nem sistema de alerta operacional ao vivo (*Not for patient care or live clinical use*).

---

## Navegação da Documentação

| Documento | Finalidade |
| :--- | :--- |
| **[Relatório de Estado Atual e Migração](docs/RELATORIO_ESTADO_ATUAL.md)** | Diagnóstico forense quantitativo (`V1.1`), arquitetura em 3 camadas (`CI` / `Workstation Arch + AMD ROCm` / `Google Colab`) e guia passo a passo para continuar em outra máquina |
| **[Mapa de Progresso (Etapas e Features)](docs/PROGRESSO.md)** | Acompanhamento consolidado etapa por etapa e feature por feature com marcação de progresso |
| **[Feature F08 — Evolução V1.2 RS → V2 Brasil](docs/features/08-plano-v1-2-v2-workstation-colab.md)** | Transição detalhada `F03.1` semanal → diária (`60–90d`), empacotamento `cpu`/`rocm`/`cuda` e expansão Google Colab |
| **[Feature F00 — Domínio e Fundação](docs/features/00-dominio-e-fundacao.md)** | Tipos validados (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`), exceções e qualidade base |
| **[Feature F01 — Ingestão Segura e LGPD](docs/features/01-ingestao-segura.md)** | Módulo `ingestion` validado em dados reais RS (`2010–2025`): CNES UTI específica + hospitais-escola, SIH semanal (`COALESCE` CID-10) e LGPD |
| **[Feature F02 — Geo, Hidrologia 100% RS e Tracer Bullet](docs/features/02-geo-hidrologia-rs.md)** | Módulo `geo` concluído: grade calendário 30/30 Regiões RS (`NASA POWER` + `ERA5-Land` + DAG de bacias + aterros/pôlderes + `W_ref`) e ablação 2024 ($h=1..4$ sem.) |
| **[Feature F03 — Hidrologia Calibrada (HydroFiLMMonotonicNet)](docs/features/03-hidrologia-calibrada-film.md)** | Módulo `hydrology` concluído (PyTorch Lightning CPU): `ROC-AUC=0,720`, `ECE=0,052`, `BSS=+0,076` e monotonicidade `autograd 0,00%` |
| **[Feature F04 — Demanda Global, Contagem e LOS](docs/features/04-demanda-e-los.md)** | Módulo `demand` concluído (V1 RS): `QuantileGBDT` monotônico exato (`0,00%` viol.), `Conformal(2023)`, `NegBinHurdle` Onda 2 (`MASE 0,710`) e LOS causal |
| **[Feature F05 — Prontidão e Filas SimPy](docs/features/05-prontidao-e-filas-simpy.md)** | Módulo `readiness` concluído (V1 RS): simulador `SimPy` com estresse pôlder (`-36,8%` pico de fila em POA `h=2` vs. `h=0`), gatilhos observáveis e `Fig. C5` |
| **[Feature F06 — Auditoria Contrafactual e Relatório](docs/features/06-auditoria-contrafactual-e-relatorio.md)** | Módulo `audit` concluído V1 RS (`Fig. C1–C5`): controle sintético "sem enchente", placebo, decomposição `do()`, `SimPy`, Net Benefit e Relatório de Verificação |
| **[Feature F07 — CLI e Governança TRIPOD+AI/RECORD](docs/features/07-cli-e-governanca.md)** | Comando `previsao-leitos` (`doctor`, `verify-rs`), logging JSON estruturado, `ECE`/`BSS`, Net Benefit e conformidade TRIPOD+AI / PROBAST+AI / RECORD |
| **[Guia de Setup do BigQuery](docs/runbooks/setup-bigquery.md)** | Passo a passo para obter `GCP_PROJECT_ID`, autenticar via ADC sem chaves `.json` e testar com custo `0 bytes` |
| **[Catálogo de Bases de Dados](docs/data-catalog.md)** | 14 fontes oficiais (Saúde, Hidrologia, Observação da Terra, Mobilidade e Demografia) e critérios de seleção |
| **[Mapa de Capacidades](spec/capability-map.md)** | Arquitetura modular alvo (`domain` → `ingestion` → `geo` → `hydrology` → `demand` → `readiness` → `audit` → `cli`) |
| **[Glossário de Domínio](GLOSSARY-MAP.md)** | Terminologia canônica dividida em Hidrologia, Capacidade Hospitalar e Previsão/Auditoria |
| **[Decisões Arquiteturais (ADRs)](docs/adr/)** | Registros das decisões técnicas e científicas centrais (`0001`–`0007`: monotonicidade, demanda real, splits, escala, LGPD, aterros/intermunicipal e contrafactual) |
| **[Padrões de Qualidade](CONSTRAINTS.md)** | Regras mecânicas de lint, tipagem, testes, segurança e privacidade (LGPD) |
| **[Diário de Desempenho](PERF.md)** & **[Changelog](CHANGELOG.md)** | Medições de performance e histórico de mudanças por versão |

---

## Início Rápido (Ambiente Local — Python 3.12 + `uv`)

```bash
# 1. Instalar dependências de desenvolvimento (Python 3.12 pinado)
uv sync --python 3.12 --extra dev

# 2. Executar verificação completa (lint, tipagem, regras de qualidade e testes)
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/check_constraints.py src tests
uv run pytest -m "not slow" --cov=src/previsao_leitos --cov-fail-under=80
uv run pip-audit
```

> Copie `.env.example` para `.env` ao executar extrações no Google BigQuery localmente (`.env` nunca deve ser commitado).

---

## Benchmark de Modelos em Um Comando

O benchmark compara **leitos ocupados observados vs. preditos** por modelo, alvo e região,
com calibração, catraca e relatório editorial. Tudo roda em CPU e nunca toca o holdout.

```bash
# 1. Montar o painel de modelagem (rede só aqui; grava apenas anos <= 2024)
uv run --no-sync python scripts/build_rs_modeling_panel.py

# 2. Rodar os 13 modelos nos 4 alvos e emitir o PDF comparativo
uv run --no-sync previsao-leitos run --target all --models all --eval-year 2024 \
  --run-id rs_v1_3_benchmark_2024
```

Saídas em `reports/runs/<run-id>/`: `predictions.parquet`, `benchmark.csv`, `comparison.csv`,
`decisions.csv`, `run_manifest.json`, três figuras por alvo e `relatorio_benchmark_leitos.pdf`
(espelhado em `~/Downloads/` com SHA-256 conferido).

Opções úteis: `--models champion`, `--models persistencia,f041_core`, `--no-report`,
`--target uti_adulto` e `--seed`. O ano `2025` é bloqueado por padrão (`--allow-holdout`
existe apenas para o `Marco M2`).

**Desempenho.** `--jobs N` roda um processo por alvo (padrão: um por alvo, limitado a
`núcleos/2`, porque cada amostragem NUTS usa 2 cadeias). Os modelos tabulares são de CPU;
a GPU acelera apenas o amostrador bayesiano, via `--sampler numpyro --device gpu`
(requer `numpyro` + `jax` com plugin ROCm/CUDA; sem a stack, o comando falha com instrução
de instalação em vez de cair silenciosamente para CPU). O `run_manifest.json` registra
`jobs`, `sampler`, `device` e o acelerador detectado.

> **Fronteira clínica:** todos os artefatos são de pesquisa e avaliação metodológica —
> *Not for patient care or live clinical use*.

---

## Histórico Experimental Inicial (Protótipo Legado — `legacy/previsao_leitos_colab.py`)

As seções abaixo documentam a evolução exploratória das **3 fases iniciais** executadas em protótipo Colab (2010–2025), que motivaram a reestruturação modular e científica registrada em [docs/PROGRESSO.md](docs/PROGRESSO.md) e nos [ADRs 0001–0005](docs/adr/).

### Variáveis Utilizadas na Previsão de Enchentes (Protótipo Inicial)

Os modelos exploratórios utilizaram uma janela temporal de **15 dias** composta por:

* **Clima e Hidrologia (dinâmicas):** precipitação diária, precipitação máxima horária, precipitação acumulada (3, 7 e 15 dias), índice de saturação do solo (`API`), escoamento efetivo, temperatura (média e máxima), umidade relativa média, pressão atmosférica média e sazonalidade cíclica ($\sin/\cos$ do dia do ano, na escala nacional).
* **Uso do Solo e Geografia (estáticas):** índice de impermeabilização urbana, índice de retenção vegetal, altitude da estação e, na escala nacional, coordenadas geográficas (`latitude`/`longitude`) + `UF Embedding`.

### 1ª Fase — Combate ao *Overfitting* e à Inversão Causal (Recorte RS)

Nos experimentos iniciais no Rio Grande do Sul, foram identificados dois gargalos estruturais:

* **Memorização espacial (`mun_embedding`):** Cada município possuía um vetor treinável próprio, que a rede usava como atalho para memorizar o histórico de alagamentos da cidade. A solução inicial foi remover o `mun_embedding`.
* **Inversão causal (`lambda_pinn`):** Penalidades suaves na função de perda degradavam o desempenho preditivo sem garantir coerência física. A solução foi substituí-las por uma **restrição estrutural rígida** com pesos estritamente positivos (`F.softplus`).

#### Arquitetura desenvolvida: `HydroFiLMMonotonicNet`

* **Hydro:** **Bi-GRU** de 2 camadas + **Multi-Head Self-Attention** sobre janela de 15 dias gerando *logit* climático base (`base_logits`) e estimativa de severidade (`severity_head`).
* **FiLM (*Feature-wise Linear Modulation*):** Modula as ativações da Bi-GRU a partir do contexto topográfico sem decorar municípios.
* **Monotonic (Ramo Monotônico Estrutural):** Pesos estritamente positivos via `F.softplus` para uso do solo (refinado no [ADR-0001](docs/adr/0001-structural-monotonicity.md) para eliminar vazamento via `escoamento_efetivo_mm`).

#### Resultados Exploratórios — RS (Treino: `2010–2023` | Calibração: `2024` | Teste: `2025`)

| Modelo (RS — Teste 2025) | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: |
| **HydroFiLMMonotonicNet (PyTorch)** | **0.7585** | **0.0508** | **0.00766** |
| TabFM (MITRA) | 0.6218 | 0.0170 | 0.00781 |

### 2ª Fase — Escala Nacional com `HydroTFTNet`

Na segunda fase, o experimento exploratório foi expandido para `~2,5 milhões` de dias-município (`571` municípios com estação INMET, `27` UFs), comparando **`HydroTFTNet`** contra **`TabFM (MITRA)`** (limitado a `8.000` amostras de contexto por restrição quadrática de memória).

| Modelo (Brasil — Teste 2025) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| **HydroTFTNet Nacional (Monotônico)** | **0.8834** | **0.0328** |
| TabFM (MITRA) | 0.8762 | 0.0255 |

### 3ª Fase — Cruzamento Inicial com Leitos CNES + IBGE e Diagnóstico Metodológico

O cruzamento exploratório com oferta mensal cadastrada no CNES (2010–2023 → 2024) revelou limitações críticas agora formalizadas para correção nas Etapas 1–3 ([ADR-0002](docs/adr/0002-target-teaching-hospital-demand.md) e [ADR-0003](docs/adr/0003-causal-ex-ante-splits-and-calibration.md)):

1. **Oferta cadastrada (CNES) vs. Demanda real:** O modelo L1 sobre variação mensal de estoque cadastral tende à persistência ($\hat{\Delta}_t = 0$) e não captura demanda clínica, tempo de permanência (LOS) nem filas reprimidas.
2. **Granularidade municipal vs. Hospitais-Escola:** A soma municipal não isolava hospitais de ensino nem suas macrorregiões de referência.
3. **Cobertura e causalidade:** A dependência exclusiva de estações INMET excluía ~90% dos municípios brasileiros, e o uso de chuva/decretos do próprio mês-alvo impedia validação prospectiva *ex-ante*.

<img width="1289" height="440" alt="Comparativo exploratório RS 2024: proxy de alagamento e leitos complementares CNES" src="https://github.com/user-attachments/assets/f04dc47f-636d-49a3-8bd4-55a2b1c5cf1c" />
