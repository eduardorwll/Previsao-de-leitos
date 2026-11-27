# Previsão de Leitos Hospitalares sob Risco de Enchentes

**Pesquisa aberta** que estima, *ex-ante*, quantos **leitos hospitalares** serão necessários
(demanda, ocupação e *surge capacity*) em hospitais-escola durante eventos de enchente —
começando pelo **Rio Grande do Sul** e escalando para o **Brasil**.

O repositório entrega um pipeline reproduzível de ponta a ponta: ingestão com LGPD,
hidrologia calibrada, modelos probabilísticos de demanda, simulação de filas, auditoria
contrafactual e — desde a `v1.3.0` — **um comando único** que roda 13 modelos em 4 alvos,
compara leitos observados e preditos e emite um relatório em PDF.

[![CI](https://github.com/eduardorwll/Previsao-de-leitos/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/eduardorwll/Previsao-de-leitos/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v1.3.0-0f766e)](https://github.com/eduardorwll/Previsao-de-leitos/releases/tag/v1.3.0)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-2A6DB2)](https://mypy-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-84.9%25-brightgreen)](#qualidade-e-reprodutibilidade)
[![License: MIT](https://img.shields.io/badge/license-MIT-3DA639)](LICENSE)

> **Fronteira clínica e operacional.** Este repositório é de **pesquisa e avaliação
> metodológica**. Modelos e relatórios **não** são recomendação clínica, triagem,
> regulação de vagas ou alerta em tempo real — *Not for patient care or live clinical use*.

---

## Sumário

- [Destaques](#destaques)
- [Modelo vs. realidade](#modelo-vs-realidade)
- [Início rápido](#início-rápido)
- [Benchmark em um comando](#benchmark-em-um-comando)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Dados, privacidade e holdout](#dados-privacidade-e-holdout)
- [Qualidade e reprodutibilidade](#qualidade-e-reprodutibilidade)
- [Documentação](#documentação)
- [Citação](#citação)
- [Licença](#licença)

---

## Destaques

| Frente | Resultado auditado |
| :--- | :--- |
| **Demanda de leitos (`F04.1`)** | `MAE 3,51` leitos/dia em UTI adulto (`RS 2024`), `MASE 0,984` vs. persistência |
| **Hidrologia (`F03.1`)** | `ROC-AUC 0,789`, `BSS +0,156` em grade `0,1°` anti-*MAUP* (OOF real na RX 7600 / ROCm) |
| **Calibração de surtos** | `BSS_pos +0,237`, `ECE 0,041`, cobertura `AgACI` `79,4%` (P10–P90) |
| **Prontidão e filas (`SimPy`)** | `-65%` de eletivas adiadas e pico de fila `20,5 → 1,0` (POA) na fronteira de Pareto |
| **Benchmark multi-modelo (`v1.3.0`)** | 13 modelos × 4 alvos em **434 s** (4 processos), PDF comparativo com 30 regiões |
| **Explicabilidade e Bayes (`1.2.6`)** | SHAP 2 estágios com erro aditivo `5,3e-16`; MCMC com `R-hat 1,005` / `ESS 761` |

---

## Modelo vs. realidade

Comparação direta entre o **campeão `gbdt_conformal`** (linha laranja, com banda de incerteza
P10–P90) e a **medição factual observada** (linha preta, SIH-SUS) nas 30 regiões de saúde do RS
no ano de auditoria (`2024`):

<p align="center">
  <img src="docs/figures/comparativo_uti_adulto.png" alt="UTI adulto — observado vs. predito" width="92%">
</p>

<p align="center">
  <img src="docs/figures/comparativo_clinico.png" alt="Clínico — observado vs. predito" width="92%">
</p>

**Como ler:** o eixo vertical é o total regional por semana (UTI adulto em diárias semanais;
clínico em pacientes-dia). O campeão acompanha a forma e o patamar da demanda — `MASE ≈ 0,94`
(UTI adulto) e `≈ 0,92` (clínico) frente à persistência — mas ainda **erra o pico de inverno em
~4–6 semanas**, uma limitação de causa epidêmica conhecida e registrada em
[pontos pendentes](docs/runbooks/acesso-bases-pendentes.md).

Relatórios completos com tabelas por região, calibração, catraca e contrafactuais:
[Benchmark](docs/benchmark/README.md) ·
[Padrão de relatório PDF](docs/guides/padrao-relatorio-tecnico-pdf.md) ·
[Mapa de Progresso](docs/PROGRESSO.md) ·
[CHANGELOG](CHANGELOG.md).

---

## Início rápido

Requisitos: **Python 3.12** e [`uv`](https://docs.astral.sh/uv/). CPU é suficiente; GPU
AMD/NVIDIA é opcional (ROCm/CUDA) e acelera apenas o treino da rede hidrológica e o
amostrador bayesiano.

```bash
git clone https://github.com/eduardorwll/Previsao-de-leitos.git
cd Previsao-de-leitos

# ambiente de desenvolvimento (CPU) + extras de ciência de dados
uv sync --python 3.12 --extra dev --extra data --extra stats --extra sim --extra hydro --extra cpu

# sanidade do ambiente e das bases locais
uv run previsao-leitos doctor

# bateria de qualidade
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/check_constraints.py src tests
uv run pytest -m "not slow" --cov=src/previsao_leitos --cov-fail-under=80
```

Com GPU AMD (ROCm 7.2) ou NVIDIA (CUDA 12.8), troque `--extra cpu` por `--extra rocm` ou
`--extra cuda`. As credenciais opcionais (BigQuery, ANA HidroWeb) ficam em `.env`, nunca no Git.

---

## Benchmark em um comando

O fluxo completo — treinar, avaliar, comparar e relatar — em dois passos:

```bash
# 1. monta o painel de modelagem (rede só aqui; grava apenas anos <= 2024)
uv run --no-sync python scripts/build_rs_modeling_panel.py

# 2. roda 13 modelos nos 4 alvos, em paralelo, e emite o PDF comparativo
uv run --no-sync previsao-leitos run --target all --models all --eval-year 2024 \
  --run-id rs_v1_3_benchmark_2024 --jobs 4
```

Saídas em `reports/runs/<run-id>/`:

| Arquivo | Conteúdo |
| :--- | :--- |
| `predictions.parquet` | região × semana × modelo × observado × `P10/P50/P90/P95` |
| `benchmark.csv` | uma linha por modelo × alvo, com todas as métricas |
| `comparison.csv` | observado vs. predito por região (viés, MAE, picos), `n ≥ 5` |
| `decisions.csv` | catraca *Champion–Challenger* por alvo |
| `eligibility.csv` | quem cumpre cobertura `≥ 80%` e zero violações monotônicas |
| `fig_regions_<alvo>.png` | grade de 30 regiões: leitos reais vs. previstos |
| `fig_observed_vs_predicted_<alvo>.png` | série agregada com banda `P10–P90` |
| `relatorio_benchmark_leitos.pdf` | relatório editorial (espelhado em `~/Downloads/` com SHA-256) |
| `run_manifest.json` | git SHA, seed, SHA-256 do painel, `jobs`, `sampler`, acelerador |

Opções úteis: `--models champion`, `--models persistencia,gbdt_agaci`, `--target uti_adulto`,
`--sampler numpyro --device gpu` (amostrador em GPU via JAX), `--no-report`.

---

## Arquitetura

```
domain → ingestion → geo → hydrology → demand → readiness → audit → cli
```

| Camada | Responsabilidade |
| :--- | :--- |
| **`domain`** | Tipos nominais validados (`CodigoCNES`, `CompetenciaMensal`), hierarquia de exceções |
| **`ingestion`** | BigQuery parametrizado com *dry-run*, CNES/SIH, clientes NOAA/ANA/MERGE-CPTEC, LGPD |
| **`geo`** | Grade semanal `0,1°`, DAG de bacias, `W_ref` intermunicipal, anti-*MAUP* |
| **`hydrology`** | `HydroFiLM-ENSO` com monotonicidade por `autograd`, OOF causal, device ROCm/CUDA/CPU |
| **`demand`** | `QuantileGBDT`, `NegBinHurdle`, 2 estágios ortogonais, conformais (Flood/AgACI/SPCI), bayes hierárquico |
| **`readiness`** | Simulador `SimPy` de filas e `NSGA-II` (`pymoo`) para fronteira de Pareto |
| **`audit`** | Contrafactual BSTS, higiene de dados, SHAP, catraca, dossiê editorial, benchmark |
| **`cli`** | `doctor`, `verify-rs`, `benchmark`, `run`, `report` |

---

## Tecnologias

![Polars](https://img.shields.io/badge/Polars-1.x-0B5B8C)
![PyArrow](https://img.shields.io/badge/PyArrow-18%2B-1F4E79)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20ROCm%2FCUDA-EE4C2C?logo=pytorch&logoColor=white)
![PyMC](https://img.shields.io/badge/PyMC-6.3-1f425f)
![nutpie](https://img.shields.io/badge/nutpie-NUTS-4B8BBE)
![ArviZ](https://img.shields.io/badge/ArviZ-1.3-1f77b4)
![SimPy](https://img.shields.io/badge/SimPy-event%20sim-6A5ACD)
![pymoo](https://img.shields.io/badge/pymoo-NSGA--II%2FIII-8B4513)
![pandera](https://img.shields.io/badge/pandera-validation-2E8B57)
![pytest](https://img.shields.io/badge/pytest-8-0A9EDC?logo=pytest&logoColor=white)

| Área | Ferramentas |
| :--- | :--- |
| Dados | `polars`, `pyarrow`, `pandera`, `google-cloud-bigquery` (Storage API) |
| Modelagem tabular | `scikit-learn` (GBDT monotônico, Ridge), `statsmodels` |
| Deep learning | `torch` (ROCm/CUDA/CPU), `lightning`, `torchmetrics` |
| Bayes | `pymc`, `nutpie`, `arviz` |
| Simulação e otimização | `simpy`, `pymoo` |
| Explicabilidade | `shap` (TreeSHAP, 2 estágios) |
| Qualidade | `ruff`, `mypy` (strict), `pytest`, `pytest-cov`, `pip-audit`, `gitleaks` |
| Relatórios | HTML5/CSS3 → Chromium headless (PDF editorial, figuras 300 DPI Okabe-Ito) |

---

## Dados, privacidade e holdout

- **Zero PII.** Nenhum CPF, CNS, nome, nascimento ou prontuário é ingerido ou armazenado.
- **Agregação na fronteira** e **supressão de células com `n < 5`** em toda tabela publicada.
- **Fontes abertas e oficiais:** CNES, SIH/SUS, IBGE, MapBiomas, ERA5-Land, NASA POWER,
  MERGE-CPTEC, NOAA CPC, ANA HidroWeb e Zenodo (IPH-UFRGS).
- **Holdout lacrado.** `2025` (PROBAST+AI) e `2026` (prospectivo virgem) só são avaliados no
  `Marco M2`; o ano `2025` é fisicamente removido do painel de modelagem e bloqueado na CLI.
- **Credenciais** (BigQuery, ANA) vivem em `.env` local, fora do Git.

---

## Qualidade e reprodutibilidade

| Gate | Critério |
| :--- | :--- |
| `ruff` | lint + regras de segurança (`S`) em `src`, `tests`, `scripts` |
| `mypy` | tipagem estrita em 100% de `src` |
| `check_constraints.py` | sem `warnings.filterwarnings("ignore")`, `except: pass`, `allow_pickle`, supressões |
| `pytest` | `183` testes rápidos, cobertura `84,9%` (piso `80%`) |
| `C-07` | `import previsao_leitos` sem rede, GPU ou variáveis de ambiente |
| `pip-audit` / `gitleaks` | zero vulnerabilidades conhecidas e zero segredos no histórico |
| `uv sync --frozen` | lock reproduzível em CPU (`--extra cpu`) |

Cada execução registra `run_manifest.json` com git SHA, seed, SHA-256 do painel, hash das
entradas e versões de ambiente.

---

## Documentação

| Documento | Finalidade |
| :--- | :--- |
| **[Mapa de Progresso](docs/PROGRESSO.md)** | Etapas, features e status com marcação de progresso |
| **[Release v1.3.0](docs/RELEASES/v1.3.0.md)** | Notas da release: benchmark, paralelismo e GPU |
| **[Benchmark — Documentação](docs/benchmark/README.md)** | Metodologia, catálogo de métricas, erro percentual, rubrica A–D e cartões |
| **[Feature F24 — Fontes ENSO e vintage](docs/features/24-enso-fontes-e-vintage.md)** | RONI, BOM semanal, SOI e o risco de revisão (`33,3%` de discordância de regime desde 2015) |
| **[Feature F25 — Dengue (SINAN)](docs/features/25-dengue-sinan-resultado-negativo.md)** | Ingestão de `532,7 MB` e teste na régua — rejeitada |
| **[Feature F23 — Momentum de internações](docs/features/23-momentum-internacoes.md)** | `−5,7%` no F04.1 respiratório |
| **[Feature F22 — Ensemble calibrado](docs/features/22-ensemble-resultado-negativo.md)** | Pesos do ano de calibração — rejeitado |
| **[Feature F20 — Memória do choque](docs/features/20-memoria-choque-resultado-negativo.md)** | Resultado negativo documentado |
| **[Runbook — Ingestão S2iD/SINAN](docs/runbooks/ingest-s2id-sinan.md)** | Passo a passo de extração e LGPD |
| **[Feature F19 — Nível Local no Estágio 1](docs/features/19-nivel-local-estagio-1.md)** | `−14%` a `−41%` de MASE no F04.1 com âncora de patamar |
| **[Feature F18 — Série de Necessidade](docs/features/18-serie-necessidade-leitos.md)** | Alvo pré-registrado, três iterações auditadas e lacuna necessidade × ocupação |
| **[Feature F17 — Régua de Backtest](docs/features/17-regua-backtest-origem-movel.md)** | 6 folds, ranking por mediana, nível × fase e métrica evento-condicional |
| **[Feature F16 — Benchmark, CLI e Relatório](docs/features/16-benchmark-cli-e-relatorio.md)** | Comando único, métricas, elegibilidade e PDF |
| **[Feature F15 — SHAP e Bayes Hierárquico](docs/features/15-explicabilidade-shap-bayes.md)** | Explicabilidade auditada e pooling `Região ⊂ Bacia` |
| **[Feature F14 — Pareto NSGA-II e SimPy](docs/features/14-pareto-nsga2-simpy.md)** | Fronteira de contingência e otimização de filas |
| **[Feature F13 — Calibração de Surtos](docs/features/13-calibracao-surtos-agaci-spci.md)** | Platt/Beta, `AgACI` vs. `SPCI` |
| **[Feature F12 — Onda 2B e Grade 0,1°](docs/features/12-onda2b-grade-01deg-f03-diaria.md)** | MERGE-CPTEC, ANA API v2, ablação diária |
| **[Feature F11 — Onda 2A e F03.1 em GPU](docs/features/11-onda2a-rocm-f03-oof-real.md)** | ROCm, OOF real, `HydroFiLM-ENSO` |
| **[Feature F10 — Painel v3 (BigQuery)](docs/features/10-painel-v3-bigquery.md)** | Reextração causal do painel semanal |
| **[Feature F09 — Saneamento V1.2](docs/features/09-saneamento-v1-2-0.md)** | Correção dos bugs estruturais e contrato canônico |
| **[Features F00–F08](docs/features/)** | Fundação, ingestão, geo, hidrologia, demanda, prontidão, auditoria, CLI e plano V2 |
| **[Relatório de modelagem (narrativa)](docs/RELATORIO_MODELAGEM_ENCHENTES.md)** | Fases, variáveis e experimentos na visão da equipe de modelagem |
| **[Catálogo de Dados](docs/data-catalog.md)** | 14 fontes oficiais e critérios de seleção |
| **[ADRs](docs/adr/)** | Decisões científicas e arquiteturais (`0001`–`0007`) |
| **[Glossário](GLOSSARY-MAP.md)** | Terminologia canônica por contexto delimitado |
| **[Padrões de Qualidade](CONSTRAINTS.md)** | Regras mecânicas de engenharia, segurança e privacidade |
| **[PERF.md](PERF.md)** · **[CHANGELOG.md](CHANGELOG.md)** | Medições antes/depois e histórico de mudanças |

---

## Citação

Se este material apoiar sua pesquisa, cite a release:

```bibtex
@software{previsao_leitos_2026,
  title        = {Previsão de Leitos Hospitalares sob Risco de Enchentes},
  author       = {{Equipe Previsão de Leitos}},
  year         = {2026},
  version      = {1.3.0},
  license      = {MIT},
  url          = {https://github.com/eduardorwll/Previsao-de-leitos}
}
```

Metadados completos em [`CITATION.cff`](CITATION.cff).

---

## Licença

Código sob **MIT** — veja [LICENSE](LICENSE). Dependências mantêm suas próprias licenças
(`scikit-survival` é GPL-3.0 e está isolado no extra opcional `survival`, com remoção
planejada na `Task 2.2`).

---

<sub>Projeto de pesquisa aberta. **Not for patient care or live clinical use.** LGPD: zero PII,
agregação na fronteira, supressão `n < 5`.</sub>
