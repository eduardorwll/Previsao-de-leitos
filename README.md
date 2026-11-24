# Previsão de Enchentes e Impacto em Leitos Hospitalares

Relatório de evolução da aplicação de **Deep Learning** e **Modelos Tabulares** na previsão de eventos de enchente e na análise de impacto sobre a oferta de leitos hospitalares no Brasil (2010–2025).

O projeto evoluiu em **4 fases principais**, partindo de um protótipo regional no Rio Grande do Sul até uma auditoria nacional *out-of-time* cruzando dados meteorológicos (INMET), uso do solo (MapBiomas), decretos de desastres (SEDEC), demografia (IBGE), capacidade hospitalar por especialidade (CNES) e número de internações reais (SIH/SUS).

---

## Variáveis Utilizadas na Previsão de Enchentes

Os modelos de previsão hidrológica utilizam uma janela temporal de **15 dias** composta por:

* **Clima e Hidrologia (dinâmicas):** precipitação diária, precipitação máxima horária, precipitação acumulada (3, 7 e 15 dias), índice de saturação do solo (`API`), escoamento efetivo (relação entre chuva, impermeabilização e saturação), temperatura (média e máxima), umidade relativa média, pressão atmosférica média e sazonalidade cíclica ($\sin/\cos$ do dia do ano, na escala nacional).
* **Uso do Solo e Geografia (estáticas):** índice de impermeabilização urbana, índice de retenção vegetal, altitude da estação e, na escala nacional, coordenadas geográficas (`latitude`/`longitude`) + `UF Embedding`.

---

## 1ª Fase — Combate ao *Overfitting* e à Inversão Causal (Recorte RS)

Nos experimentos iniciais no Rio Grande do Sul, foram identificados dois gargalos estruturais:

* **Memorização espacial (`mun_embedding`):** Cada município possuía um vetor treinável próprio, que a rede usava como atalho para memorizar o histórico de alagamentos da cidade, elevando o risco mesmo sem indícios climáticos. A solução foi remover o `mun_embedding`.
* **Inversão causal (`lambda_pinn`):** Penalidades suaves adicionadas à função de perda para punir inversões na impermeabilização degradavam o desempenho preditivo sem garantir coerência física. A solução foi substituir a penalidade por uma **restrição estrutural rígida** com pesos estritamente positivos (`F.softplus`).

### Arquitetura desenvolvida: `HydroFiLMMonotonicNet`

Arquitetura em PyTorch projetada especificamente para o projeto, dividida em três blocos:

* **Hydro:** Uma **Bi-GRU** de 2 camadas lê a janela de 15 dias nos dois sentidos para capturar a evolução da tempestade e o encharcamento do solo. Em seguida, uma camada de **Multi-Head Self-Attention** pondera quais dias mais contribuem para o risco atual, gerando o *logit* climático base (`base_logits`) e a estimativa de severidade (`severity_head`).
* **FiLM (*Feature-wise Linear Modulation*):** Usa a altitude da estação para gerar dois vetores ($\gamma$ e $\beta$) que modulam as ativações da Bi-GRU, adaptando a leitura da chuva ao relevo sem decorar municípios.
* **Monotonic (Ramo Monotônico Estrutural):** Em vez de penalidades na *loss*, o uso do solo entra em um ramo paralelo com pesos forçados a serem estritamente positivos via `F.softplus`. Isso garante matematicamente que aumentar a impermeabilização nunca reduza o risco e aumentar a vegetação nunca o eleve.

### Resultados — RS (Treino: `2010–2023` | Calibração: `2024` | Teste: `2025`)

A `HydroFiLMMonotonicNet` superou o `TabFM (MITRA)` em todas as métricas factuais:

| Modelo (RS — Teste 2025) | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: |
| **HydroFiLMMonotonicNet (PyTorch)** | **0.7585** | **0.0508** | **0.00766** |
| TabFM (MITRA) | 0.6218 | 0.0170 | 0.00781 |

---

## 2ª Fase — Escala Nacional com `HydroTFTNet`

Na segunda fase, o experimento foi expandido para **todo o Brasil** (`~2,5 milhões` de dias-município, `571` municípios e `27` UFs), comparando o **`HydroTFTNet`** (com o mesmo Ramo Monotônico Estrutural) contra o **`TabFM (MITRA)`**.

O `HydroTFTNet` foi adotado nessa escala por combinar:
* **Variable Selection Network (VSN):** calcula pesos dinâmicos para selecionar quais variáveis climáticas mais importam em cada região/bioma.
* **LSTM condicionada geograficamente:** mantém um estado de memória separado (`cell state`) inicializado a partir do contexto geográfico (`UF Embedding`, latitude, longitude e altitude).

### Principais otimizações de engenharia

* **Limite de contexto do TabFM (MITRA):** Por utilizar atenção quadrática $O(N^2)$ entre linhas, o MITRA precisou ser limitado a **8.000 amostras** de contexto (selecionadas com *Hard Negative Mining*) para não estourar a VRAM da GPU.
* **Dataset 2D Indexado:** Em vez de materializar um tensor 3D completo em memória (~1,7 GB), os dados passaram a ser armazenados em uma matriz 2D (~110 MB) e fatiados em janelas de 15 dias sob demanda no `__getitem__`.
* **Sazonalidade cíclica (`doy_sin`, `doy_cos`):** Permitiu diferenciar chuvas em estações chuvosas de chuvas isoladas em períodos secos nos diferentes biomas brasileiros.
* **Balanceamento e Early Stopping:** Uso de `WeightedRandomSampler` (~8% de positivos por *batch*), `OneCycleLR` e **Early Stopping** monitorando o PR-AUC de validação (2024) para conter o *overfitting* do Transformer.

### Resultados — Brasil Inteiro (Teste Out-of-Time 2025)

| Modelo (Brasil — Teste 2025) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| **HydroTFTNet Nacional (Monotônico)** | **0.8834** | **0.0328** |
| TabFM (MITRA) | 0.8762 | 0.0255 |

> **Conclusão arquitetural:** A `HydroFiLMMonotonicNet` (Bi-GRU) performa melhor em datasets menores (recorte estadual), evitando *overfitting*; já o `HydroTFTNet` se sobressai em escala nacional, aproveitando o grande volume de dados e a diversidade climática.

---

## 3ª Fase — Cruzamento com Leitos Hospitalares (CNES + IBGE) e Auditoria de 2024

Na terceira fase, os dados climáticos e as previsões do `HydroTFTNet` (treinado estritamente em `2010–2023`) foram cruzados com a oferta mensal de leitos por especialidade (**CNES**, removendo colunas binárias de indicadores), população municipal (**IBGE**) e área territorial (**MapBiomas**).

### Principais avanços

* **Normalização demográfica e territorial:** Cálculo de leitos por **1.000 habitantes** e por **100 km²**, eliminando distorções entre cidades pequenas e polos regionais.
* **Auditoria da enchente de 2024 (`HydroTFTNet`):** O modelo previu com precisão o pico da catástrofe de maio de 2024 no RS (**ROC-AUC `0.8196`** e **PR-AUC `0.3133`** em maio/2024). Em meses sem decretos oficiais da SEDEC, o índice observado (`Nivel_Alag_Real`) permanece em `0.000`, enquanto o modelo continua sinalizando severidade contínua quando há chuvas fortes.
* **Previsão de leitos via Delta Mensal (`HistGradientBoostingRegressor`):** Um modelo tabular dedicado foi treinado em `2010–2023` para prever a **variação mensal de leitos** ($\Delta_t = y_t - y_{t-1}$) com perda L1 (`absolute_error`), eliminando o viés de subestimação nos polos regionais ($R^2 > 0.9995$ e MAE entre `0.15` e `0.31` leitos). Comparar a oferta prevista com a real isolou o choque emergencial de **+163 UTIs** em maio/2024 e a desmobilização de **-111 UTIs** em julho/2024 no RS.

### RS 2024 Mês a Mês: Enchente e Leitos (Previsto vs. Real)

| Mês | Prob. Enchente (Prev) | Nível Alag. (Prev) | Nível Alag. (Real) | Leitos Clín. (Prev) | Leitos Clín. (Real) | Leitos UTI (Prev) | Leitos UTI (Real) | Leitos Clín. / 1.000 hab |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.006 | 1.845 | 0.489 | 4990.0 | 4997 | 1955.0 | 1955 | 1.108 |
| 2 | 0.006 | 1.074 | 0.000 | 5172.0 | 5216 | 2036.0 | 2145 | 1.119 |
| 3 | 0.008 | 2.251 | 0.425 | 5272.0 | 5229 | 2203.3 | 2193 | 1.108 |
| 4 | 0.012 | 4.806 | 4.157 | 5229.0 | 5407 | 2193.0 | 2204 | 1.114 |
| **5** | **0.021** | **6.578** | **5.470** | **5493.0** | **5550** | **2224.0** | **2387** | **1.159** |
| 6 | 0.010 | 3.883 | 0.060 | 5550.0 | 5649 | 2388.5 | 2388 | 1.173 |
| 7 | 0.005 | 0.156 | 0.000 | 5649.0 | 5624 | 2388.0 | 2277 | 1.166 |
| 8 | 0.010 | 3.154 | 0.000 | 5624.0 | 5666 | 2277.0 | 2282 | 1.165 |
| 9 | 0.013 | 3.739 | 0.691 | 5584.0 | 5593 | 2254.0 | 2268 | 1.189 |
| 10 | 0.021 | 4.435 | 0.487 | 5593.0 | 5573 | 2268.0 | 2278 | 1.183 |
| 11 | 0.013 | 2.322 | 0.000 | 5573.0 | 5624 | 2278.0 | 2241 | 1.186 |

---

## 4ª Fase — Roteamento de Bacias Hidrográficas e Demanda Real de Leitos (SIH/SUS vs. CNES-SUS no RS)

A auditoria da 3ª fase revelou duas limitações estruturais que motivaram uma quarta etapa focada nos **497 municípios do Rio Grande do Sul** (validação *out-of-time* em 2024):

- **Ponto cego hidrológico em cidades de jusante:** Tratar municípios isoladamente não capturava cheias em regiões como Porto Alegre e Pelotas, onde o alagamento decorre da água acumulada que desce de bacias a montante.
- **Rigidez da oferta no CNES e persistência ingênua:** Testes no histórico `2011–2023` mostraram que o cadastro de leitos do CNES tem elasticidade climática praticamente nula (`+0.00000`) em eventos comuns — hospitais absorvem a demanda elevando a taxa de ocupação sem abrir novos leitos no sistema. Além disso, prever o delta mensal usando `lag_1m` real com perda L1 induzia persistência ingênua (curva prevista deslocada 1 mês para a direita).

### Ferramentas, Arquitetura e Adições de Engenharia

- **Roteamento Hidrológico Causal (`HydroFiLMRegionalBasinNet`):** Os 497 municípios do RS foram agrupados em **8 regiões hidrográficas** conectadas por um grafo causal **Montante $\rightarrow$ Jusante** (`shift(1)`). A arquitetura em PyTorch (`Bi-GRU` + `FiLM` condicionado por região/relevo + `Multi-Head Self-Attention` + **Ramo Monotônico Estrutural** via `F.softplus`) passou a receber a chuva acumulada (`7d`, `15d`, `30d`) e o escoamento efetivo das bacias a montante, além da variação barométrica em 3 dias (`pressao_var_3d`, proxy de frentes frias/vento Sul), treinada sobre os decretos completos da SEDEC (sem filtro de chuva local).
- **Migração para Demanda Real (`SIH/SUS` via BigQuery / `basedosdados`):** A demanda hospitalar passou a ser medida diretamente pelas **diárias de internação do SIH/SUS**, convertidas em leitos necessários sob a taxa de ocupação segura recomendada pela OMS/MS (**85%**):

$$
\text{Leitos Necessários}_{r,m} = \frac{\text{Diárias de Internação}_{r,m}}{\text{Dias do Mês}_m \times 0{,}85}
$$

- **Saneamento de Microdados do DATASUS/IBGE:** Foram corrigidos quatro gargalos clássicos no ETL: (1) **deduplicação por `id_aih`** (`SELECT DISTINCT`), eliminando um pico artificial de 2x em agosto/2024 na tabela bruta; (2) **agrupamento por `data_internacao`** (e não pelo mês de faturamento/competência), evitando o deslocamento temporal de AIHs faturadas com atraso; (3) **padronização IBGE de 6 dígitos (`id_mun_6`)** para cruzar SIH e IBGE; e (4) **alinhamento SUS vs. SUS**, filtrando o CNES exclusivamente por `quantidade_sus` (Total e UTI `tipo_leito = 3`) para remover falsos superávits causados por leitos privados.
- **Modelagem de Demanda (`Ridge` + `StandardScaler`) com Separação de Coortes:** Treinado em `2011–2019` e `2023` (excluindo os anos de pandemia `2020–2022`) e ancorado exclusivamente na base de 2023 (*zero leakage* intra-anual). Foram adicionados parâmetros para capturar a dinâmica real de crises: `sev_log` (resposta côncava), `disrupcao_extrema` (curva em "U invertido" para bloqueio hospitalar), `sev_lag_1m`/`sev_lag_2m` (rebote em 30–60 dias), `sev_vizinhos_max` (transbordamento de pacientes entre regiões adjacentes) e `temp_frio_idx`.
  - **Coortes de Urgência SUS (Total e UTI) — Formulação Multiplicativa:** Remove cirurgias eletivas (`carater_internacao = 01`) e prevê a variação percentual sobre a base de dezembro do ano anterior

$$
\text{Base}_{\text{dez}} \times (1 + \hat{p}).
$$

  - **Coortes de CIDs Sensíveis a Enchentes (Total e UTI) — Formulação Aditiva/100k:** Filtra Leptospirose (`A27`), hídricas (`A00–A09`, `B15`), respiratórias agudas (`J00–J22`, `J45–J46`), pele (`L00–L08`) e traumas/hipotermia (`S00–T98`), **excluindo Dengue (`A90–A92`)** para evitar contaminação pela epidemia de outono de 2024. Por serem séries esparsas (onde dezembro pode ser `0`), prevê o acréscimo aditivo de leitos por 100 mil habitantes sobre a média do ano anterior, evitando o travamento em zero.

### Resultados — Estágio 1: Alagamento por Região Hidrográfica (RS — Teste Out-of-Time 2024)

| Região Hidrográfica | ROC-AUC | PR-AUC | Brier Score | MAE (% Mun.) |
| :--- | :---: | :---: | :---: | :---: |
| **Planalto Norte (Alto Jacuí/Uruguai)** | **0.8562** | **0.7532** | 0.1459 | 2.66% |
| **Fronteira Oeste e Campanha** | 0.8262 | 0.4518 | **0.1045** | 2.70% |
| **Depressão Central (Médio Jacuí)** | 0.7948 | 0.6336 | 0.1270 | 2.97% |
| **Vales (Taquari e Rio Pardo)** | 0.7704 | 0.4283 | 0.1303 | 3.54% |
| **Serra Gaúcha (Alto Taquari/Caí)** | 0.7357 | 0.5526 | 0.1321 | 2.95% |
| **Noroeste e Missões** | 0.7333 | 0.5392 | 0.1655 | **2.63%** |
| **Metropolitana e Bacia do Guaíba** | 0.7216 | 0.5671 | 0.1580 | **2.63%** |
| **Região Sul e Lagoa dos Patos** | 0.6981 | 0.5701 | 0.1798 | 4.33% |

### Resultados — Estágio 2: Demanda de Leitos SIH/SUS (2024: Alag. Previsto vs. Alag. Real)

| Coorte (SIH/SUS — RS 2024) | Formulação | MAE (Alag. Previsto) | $R^2$ (Alag. Previsto) | MAE (Alag. Real) | $R^2$ (Alag. Real) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Urgência — Total SUS** | Multiplicativa | **159.26 leitos** | **0.9719** | 157.88 leitos | 0.9723 |
| **Urgência — UTI SUS** | Multiplicativa | **20.09 leitos** | **0.9605** | 20.00 leitos | 0.9605 |
| **CIDs Enchente — Total SUS** | Aditiva / 100k | **2.49 leitos** | **0.7283** | 2.76 leitos | 0.6586 |
| **CIDs Enchente — UTI SUS** | Aditiva / 100k | **0.29 leito** | **0.4107** | 0.30 leito | 0.3467 |

### Principais Achados e Limitações Mapeadas

- **Equivalência Operacional (Rede Neural vs. SEDEC):** Alimentar o modelo de leitos com o alagamento *previsto* pela `HydroFiLMRegionalBasinNet` empatou com o uso dos decretos *reais* da SEDEC nas coortes de Urgência ($R^2 = 0.9719$ vs. $0.9723$) e os superou nas coortes de CIDs ($0.7283$ vs. $0.6586$), permitindo estimar a pressão hospitalar antes da emissão formal dos decretos.
- **Efeito "Bloqueio e Rebote" (Maio vs. Julho/2024):** As internações gerais no RS caíram **`-12,5%`** em maio (`71.877` $\rightarrow$ `62.903`) devido ao cancelamento de cirurgias eletivas, bloqueio de rodovias e evacuação de hospitais. Isolando apenas **Urgências**, a queda foi de apenas **`-5,1%`**, seguida por um **rebote recorde em julho** (`72.569` internações gerais, `52.021` urgências e `46.505` diárias de UTI), quando a **Depressão Central** atingiu **81,9%** de ocupação da oferta SUS apenas com UTIs de urgência.
- **Limitações conhecidas:** (1) Ausência de séries de **vento superficial** e **cotas fluviométricas (ANA)**, fatores críticos no represamento da Lagoa dos Patos e do Guaíba; e (2) preenchimento parcial da coluna `cid_principal_categoria` na tabela do BigQuery (`304–465` internações/mês no RS), motivo pelo qual as coortes de **Urgência SUS** constituem o resultado principal da auditoria.

> **Conclusão da 4ª Fase:** O acoplamento entre roteamento hidrológico de bacias e demanda real deduplicada do SIH/SUS elimina a persistência ingênua do CNES e demonstra que a rede neural já fornece sinal operacional suficiente para antecipar a pressão sobre leitos SUS (especialmente UTIs de urgência) e dimensionar o rebote pós-enchente.
