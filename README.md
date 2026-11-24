# Previsão de Leitos em Hospitais-Escola sob Risco de Enchentes

Projeto de **pesquisa aberta** voltado à previsão *ex-ante* da **necessidade de leitos hospitalares (demanda, ocupação e *surge capacity*) em hospitais-escola** frente a eventos de enchente — com foco inicial no **Rio Grande do Sul (V1)** e expansão planejada para **escopo nacional — Brasil (V2)**.

> **Aviso de Escopo Científico e Operacional:** Este repositório destina-se exclusivamente à pesquisa metodológica e ao planejamento agregado de capacidade hospitalar. Seus modelos e relatórios **não** constituem recomendação clínica, triagem assistencial nem sistema de alerta operacional ao vivo (*Not for patient care or live clinical use*).

---

## Navegação da Documentação

| Documento | Finalidade |
| :--- | :--- |
| **[Mapa de Progresso (Etapas e Features)](docs/PROGRESSO.md)** | Acompanhamento consolidado etapa por etapa e feature por feature com marcação de progresso |
| **[Feature F00 — Domínio e Fundação](docs/features/00-dominio-e-fundacao.md)** | Tipos validados (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`), exceções e qualidade base |
| **[Feature F01 — Ingestão Segura e LGPD](docs/features/01-ingestao-segura.md)** | Módulo `ingestion` validado em dados reais RS (`2010–2025`): CNES UTI específica + hospitais-escola, SIH semanal (`COALESCE` CID-10) e LGPD |
| **[Feature F02 — Geo, Hidrologia 100% RS e Tracer Bullet](docs/features/02-geo-hidrologia-rs.md)** | Módulo `geo` concluído: grade calendário 30/30 Regiões RS (`NASA POWER` + `ERA5-Land` + DAG de bacias) e *Tracer Bullet* 2023–2024 ($h=1..4$ sem.) |
| **[Guia de Setup do BigQuery](docs/runbooks/setup-bigquery.md)** | Passo a passo para obter `GCP_PROJECT_ID`, autenticar via ADC sem chaves `.json` e testar com custo `0 bytes` |
| **[Catálogo de Bases de Dados](docs/data-catalog.md)** | 14 fontes oficiais (Saúde, Hidrologia, Observação da Terra, Mobilidade e Demografia) e critérios de seleção |
| **[Mapa de Capacidades](spec/capability-map.md)** | Arquitetura modular alvo (`domain` → `ingestion` → `geo` → `hydrology` → `demand` → `readiness` → `audit` → `cli`) |
| **[Glossário de Domínio](GLOSSARY-MAP.md)** | Terminologia canônica dividida em Hidrologia, Capacidade Hospitalar e Previsão/Auditoria |
| **[Decisões Arquiteturais (ADRs)](docs/adr/)** | Registros das decisões técnicas e científicas centrais (`0001`–`0005`) |
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
