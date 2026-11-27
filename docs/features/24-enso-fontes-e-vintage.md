# Feature F24 — Avaliação: ElNinoLive, índices ENSO alternativos e risco de *vintage* (`Task 1.3.10`)

> **Status:** 📋 Estudo concluído (implementação proposta) | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Fronteira:** *Not for patient care or live clinical use.*

## 1. O que o site é (e o que não é)

`elninolive.com/data` declara explicitamente: **"Nothing on this site is our own measurement"**.
É um agregador independente que busca, parseia e plota dados primários de **NOAA, BOM e NASA**.
Não há modelo próprio, nem dado proprietário.

| Item | Constatação |
| :--- | :--- |
| Fontes | CPC (SST semanal, ONI, RONI, SOI), BOM (Niño 3.4 relativo semanal), NASA GIBS (imagens) |
| API JSON pública | `/api/enso/{status,weekly,oni,soi,roni,bom-rnino,race,probabilities}` |
| Licença | feeds públicos com atribuição; os dados subjacentes são NOAA (domínio público) e BOM |
| Natureza | projeto independente, **não** serviço oficial; sem SLA |

**Consequência direta:** não devemos depender do site em runtime. A jogada correta é **adotar a
metodologia e buscar as fontes primárias** (que já sabemos acessar), usando o JSON deles apenas
como *cross-check* de parsing em desenvolvimento.

## 2. O que já usamos vs. o que é novo

| Fonte | Nosso pipeline | Novo? |
| :--- | :--- | :--- |
| ONI mensal (CPC) | ✅ `fetch_noaa_enso_monthly` | — |
| MEI.v2 (PSL) | ✅ | — |
| Regime `el_nino/neutro/la_nina` | ✅ via ONI | — |
| **RONI** (ONI relativo) | ❌ | **sim** |
| **BOM Niño 3.4 relativo semanal** | ❌ | **sim** (frequência semanal) |
| **SOI** (componente atmosférico) | ❌ | **sim** |
| Probabilidades CPC/IRI | ❌ | **sim** (ex-ante, com risco de vintage) |

## 3. Achado empírico que justifica o estudo

Comparamos **ONI (o que usamos) × RONI (o padrão moderno)** em 912 meses (1950–2025):

| Métrica | Valor |
| :--- | ---: |
| Correlação | `0,958` |
| Diferença média (RONI − ONI) | `+0,007 °C` |
| **Discordância de regime** (limiar ±0,5 °C) | **`14,8%`** |
| **Discordância desde 2015** | **`33,3%`** |

Ou seja: em **um terço dos meses recentes** o rótulo de regime muda. Como `enso_regime` entra no
modelo (e no agrupamento do calibrador Mondrian), a escolha do índice não é cosmética.

## 4. Risco metodológico que o site expõe: revisões (*vintage*)

O próprio site alerta: **RONI pode revisar por até dois meses após a publicação**. Nosso pipeline
usa a série **final (revisada)** — ou seja, em `t` podemos estar usando um valor que só existiu em
`t+2`. É um vazamento leve, mas real, e do tipo que infla métricas.

**Correção proposta (independente de qualquer modelo novo):** aplicar *publication lag* explícito
— ONI/RONI com `2` meses, SOI com `1` mês, SST semanal da BOM com `1` semana. Isso deve **piorar**
ligeiramente as métricas atuais; é o preço da honestidade causal.

## 5. Experimentos propostos (mesma régua, opt-in, sem promoção automática)

| # | Experimento | Fonte primária | Custo | Hipótese |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | `roni_c` + regime RONI, lag 2 meses | CPC RONI (fallback: API do site) | baixo | corrige 1/3 dos rótulos de regime |
| **E2** | Niño 3.4 relativo **semanal** (BOM), lag 1 sem. | `bom.gov.au/.../rnino_3.4.txt` | baixo | sinal semanal alinhado ao painel `h=2` |
| **E3** | SOI mensal, lag 1 mês | `cpc.ncep.noaa.gov/data/indices/soi` | baixo | componente atmosférico complementar |
| **E4** | Índice consenso (média padronizada ONI/RONI/BOM) | derivado | baixo | reduz ruído de índice único |
| **E5** | Probabilidades CPC/IRI | arquivo histórico de *plumes* | médio | **só** com vintage; sem arquivo, descartar |

**Catraca de cada experimento:** ganho `>= 2%` no MASE mediano **ou** redução do erro de fase,
com `0` violações monotônicas; caso contrário, rejeitado e documentado (como F20, F22, F23).

> **Status dos experimentos:** **E1 (RONI) e E3 (SOI) foram executados e rejeitados** pela régua
> — ver [F26 — RONI](26-roni-resultado-negativo.md) e [F27 — SOI](27-soi-resultado-negativo.md).
> E2 (BOM semanal) permanece na fila.

## 6. O que **não** fazer

1. **Não** chamar a API do elninolive em produção (sem SLA, dependência de terceiro).
2. **Não** usar as probabilidades publicadas hoje para avaliar anos passados (vintage futuro).
3. **Não** misturar escalas: ONI, RONI e BOM têm limiares diferentes (BOM usa +0,8 °C).
4. **Não** promover modelo só porque o índice "faz sentido" — a régua decide.

## 7. Recomendação

Ordem: **E1 + E3** (menor custo, maior aderência ao padrão NOAA), depois **E2** (ganho de
frequência), **E4** (consenso) e **E5** apenas se houver arquivo de vintages. E, antes de tudo,
aplicar o *publication lag* para eliminar o vazamento por revisão.

## 8. Arquivos

- Estudo: este documento
- Pipeline atual: `src/previsao_leitos/geo/enso_teleconnections.py`, `ingestion/wave2a_clients.py`
- Fontes confirmadas: BOM `rnino_3.4.txt` (2008+), CPC `soi` (1951+), CPC RONI (1950+)
