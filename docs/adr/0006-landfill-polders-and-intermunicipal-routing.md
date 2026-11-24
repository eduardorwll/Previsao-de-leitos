# ADR-0006: Áreas Aterradas, Sistemas de Pôlderes/Diques e Acoplamento Intermunicipal Duplo ($G_{\text{hydro}} + G_{\text{health}}$)

- **Status:** ACCEPTED
- **Data:** 2026-11-24

## Contexto e Problema

A revisão física do desastre de 2024 no Rio Grande do Sul (e de grandes cheias urbanas brasileiras em geral) evidenciou dois mecanismos estruturais que modelos puramente locais ou baseados em médias simples de macro-bacias não conseguem representar adequadamente:

1. **Áreas aterradas sobre várzeas/banhados e sistemas de diques/pôlderes com bombeamento** (ex.: 4º Distrito, Humaitá, Sarandi e Praia de Belas em **Porto Alegre**; Laranjal, Navegantes e margens do Canal São Gonçalo em **Pelotas**; áreas baixas de **Rio Grande** e Canoas):
   - Tais zonas exibem comportamento **em degrau (*threshold failure*)** — permanecem artificialmente secas até o transbordamento/rompimento de dique ou falha de estações de bombeamento (EBAPs) — seguido por **alagamento estagnado prolongado (*ponding*)** durante semanas, o que amplifica drasticamente a **Onda 2** (leptospirose, SRAG em abrigos e descompensações crônicas) e paralisa vias/hospitais locais.
   - Em zonas estuarinas e lagunares de baixíssima declividade (Lago Guaíba ↔ Lagoa dos Patos ↔ Canal São Gonçalo), **ventos do quadrante Sul/Sudeste (`V10M > 0`)** combinados a níveis elevados na laguna criam **remanso (*backwater*)**, represando a descarga — uma exceção física ao escoamento estritamente unidirecional montante → jusante.
2. **Escoamento e transbordamento intermunicipal ("cidades afetam outras cidades")** por duas vias acopladas:
   - **Via Hidrológica ($G_{\text{hydro}}$):** a chuva e o *runoff* acelerado pela impermeabilização/aterro em cidades a montante (ex.: Serra/Taquari-Antas, Vale dos Sinos, Caí, Gravataí) propagam-se com **peso de contribuição de bacia ($w_{uv}$)** e **tempo de trânsito ($\tau_{uv}$ em dias/semanas)** até as cidades de vale e planície lagunar (`1–2d` no Vale do Taquari, `2–5d` no Guaíba/POA, `7–14d` em Pelotas/Lagoa dos Patos).
   - **Via Assistencial ($G_{\text{health}}$):** quando municípios periféricos ou hospitais em áreas aterradas ficam submersos ou sem acesso viário, a demanda hospitalar **transborda** para hospitais-escola em outros municípios (ex.: Canoas/Eldorado do Sul/Vale do Taquari → HCPA, Santa Casa, GHC em POA; região Sul → HE-UFPel/HU-FURG), padrão mensurável empiricamente pelos fluxos históricos `MUNIC_RES → CNES` do `SIH-SUS`.

## Decisão

Revisar a arquitetura modular em quatro frentes permanentes (V1 RS e V2 Brasil):

1. **Grafo Hidrológico Intermunicipal Ponderado com Tempo de Trânsito (`src/previsao_leitos/geo/basin_graph.py`):**
   - Evoluir `BasinTopology` para suportar arestas ponderadas e defasadas `(src, dst, weight, travel_days)`, propagando não apenas precipitação e `API`, mas também **pressão de impermeabilização/aterro das sub-bacias a montante**.
   - Incluir termo explícito de **remanso lagunar/estuarino (*backwater*)** para nós costeiros/estuarinos (`guaiba_capital`, `lagoa_patos_sul`), acoplando componente meridional de vento Sul (`V10M`) + acúmulo hídrico na laguna, preservando o grafo fluvial principal estritamente acíclico (`DAG`).
2. **Exposição em Áreas Aterradas e Pôlderes (`src/previsao_leitos/geo/landfill_exposure.py`):**
   - Combinar descritores físicos abertos e transferíveis nacionalmente:
     - `frac_urbano_hand_lt_3m` / `frac_urbano_hand_lt_5m` (interseção `Copernicus DEM / MERIT Hydro HAND` × área urbana);
     - `frac_expansao_urbana_sobre_varzea` e razão urbano/áreas úmidas derivadas de **MapBiomas (`1985–presente`, estritamente `<= ano - 1`)** + **JRC Global Surface Water**;
     - inventário oficial aberto de sistemas de proteção contra cheias (`GeoPOA/DMAE`, `GeoPelotas/SANEP`, `IPH/UFRGS Zenodo 10.5281/zenodo.16929471`, `SGB/CPRM`).
   - **Guardrail anti-memorização (complemento ao ADR-0001):** nenhum indicador binário de pôlder pode atuar sozinho como substituto de `mun_embedding`; deve sempre ser combinado com descritores físicos contínuos (`HAND`, declividade, vento Sul, área alagável) e auditado via validação espacial (*Leave-Basin-Out*).
3. **Grafo Assistencial Intermunicipal Causal (`src/previsao_leitos/geo/referral_graph.py`):**
   - Construir a matriz empírica de referência $W_{\text{ref}}(u \to v)$ a partir de `sih_fluxos_referencia_semanal` (`MUNIC_RES → CNES / Região`), **estritamente congelada em `<= train_end_year` (ou `<= ano - 1`)** para evitar vazamento temporal.
   - Gerar `pressao_hidro_catchment` (exposição hidrológica e de aterro ponderada pelos municípios de origem que drenam pacientes para cada Hospital-Escola / Região de Saúde).
4. **Extensão do Ramo Monotônico (ADR-0001) e Simulador `SimPy` (F05):**
   - O ramo monotônico `F.softplus` em `hydrology` (F03) passa a receber `(impermeab_local, frac_urbano_baixo_hand, frac_aterro_varzea, impermeab_montante_ponderada)` com derivada $\ge 0$ e retenção vegetal local/montante com derivada $\le 0$.
   - O simulador de filas `SimPy` (F05) incorpora atributos `hospital_hand_m` e `in_polder_zone` para simular evacuação/perda de capacidade em hospitais em áreas aterradas e redirecionamento de carga via $W_{\text{ref}}$ e malha viária (`OSMnx`).

## Alternativas Rejeitadas

- **Usar apenas altitude média municipal ou `lat/lon`:** falha em distinguir bairros aterrados/protegidos por diques (ex.: 4º Distrito em POA, Laranjal em Pelotas) de bairros altos no mesmo município e induz memorização espacial.
- **Permitir ciclos no grafo fluvial principal para modelar remanso:** quebraria a garantia estrutural acíclica e a propagação causal montante→jusante; modelar o remanso como condição de contorno explícita de jusante (`vento Sul + estoque lagunar`) mantém a física limpa e auditável.
- **Calcular $W_{\text{ref}}$ usando o ano de avaliação (ex.: 2024):** introduziria vazamento direto dos desvios emergenciais do próprio desastre avaliado.

## Consequências

- Exige versionamento de features causais `<= ano - 1` para `MapBiomas` e `W_ref` (`SIH-SUS`).
- Permite testes *Prove-It* dedicados: (i) impermeabilização/chuva a montante com `travel_days` altera jusante no lag correto e nunca altera montante; (ii) perturbar fluxos `SIH-SUS` de `2024–2025` deixa $W_{\text{ref}}$ de treino (`<= 2022`) bit-idêntico; (iii) vento Sul amplifica o termo de remanso apenas em bacias estuarinas/lagunares.
