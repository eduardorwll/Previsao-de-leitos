# Catálogo Detalhado de Bases de Dados — Seleção V1 (RS Hospitais-Escola) → V2 (Brasil)

> **Regime:** Pesquisa aberta | **Privacidade (LGPD):** Bases assistenciais (`SIH-SUS`, `SINAN`, `GERINT`, `SIVEP-Gripe`) são ingeridas **exclusivamente em forma agregada na consulta de origem** (zero PII, com verificação automática de supressão de pequenas células $n < 5$).

---

## 1. Resumo Comparativo (14 Fontes Oficiais e Abertas)

| # | Base de Dados | Grupo | Papel Principal no Projeto | Granularidade | Acesso Principal | Recomendação |
| :-: | :--- | :--- | :--- | :--- | :--- | :---: |
| **1** | **CNES** (`estabelecimento` + `leito` + `estabelecimento_ensino` + `equipamento`) | Saúde | Identificar hospitais-escola, estoque de leitos por especialidade (SUS vs. privado) e equipamentos críticos | Estabelecimento × Especialidade × Mês | BigQuery `basedosdados.br_ms_cnes` / DATASUS | **Obrigatória (V1)** |
| **2** | **SIH-SUS** (AIH Reduzida `rd` — agregada sem PII) | Saúde | **Variável-alvo real de demanda:** internações, tempo de permanência (LOS), diárias de UTI, urgência vs. eletiva, fluxos `MUNIC_RES → CNES` | Hospital-Escola × CID-10 × Dia/Semana | BigQuery `basedosdados.br_ms_sih` / FTP DATASUS | **Obrigatória (V1)** |
| **3** | **SINAN** (Leptospirose, Dengue, Hepatite A, Tétano) | Saúde | Ondas tardias (D+7 a D+30) de agravos hídricos pós-enchente que pressionam enfermarias e UTIs | Município × Semana Epidemiológica | DATASUS / Base dos Dados / OpenDataSUS | **Recomendada (V1)** |
| **4** | **GERINT — Dados Abertos POA / SES-RS** | Saúde | Demanda reprimida e solicitações reais de leitos/regulação na rede pública de Porto Alegre (HCPA, Santa Casa, HSL-PUCRS) | Solicitação agregada × Especialidade × Dia (POA/RS) | `dadosabertos.poa.br` (CSV/API CKAN) | **Recomendada (V1 RS)** |
| **5** | **SIVEP-Gripe / SIM** (SRAG e Óbitos Hospitalares) | Saúde | Picos respiratórios (SRAG em abrigos + inverno gaúcho pós-cheia) e auditoria de gravidade/óbitos | Município/Hospital agg × Semana | OpenDataSUS / Base dos Dados | **Opcional (V1/V2)** |
| **6** | **MERGE-CPTEC / INPE** | Hidrologia | Precipitação diária em grade (`0,1°` ≈ 10 km) cobrindo **100% dos 497 mun. RS e 5.570 do Brasil** (corrige cegueira de 90% do INMET) | Grade 0,1° × Dia (2000–presente) | FTP/HTTP CPTEC-INPE (GRIB2/NetCDF) | **Obrigatória (V1)** |
| **7** | **ANA HidroWeb + Ottobacias (BHO) + SGB/SACE** | Hidrologia | **Sinal físico real de enchente:** cotas fluviométricas (cm), vazões (m³/s), cotas de alerta/inundação e topologia montante→jusante | Estação/Trecho × Dia/Hora | API HidroWeb ANA / Portal SGB-SACE / Shapefiles BHO | **Obrigatória (V1)** |
| **8** | **ERA5-Land (Copernicus/ECMWF)** | Hidrologia | Umidade física do solo em 4 camadas (`swvl1–4`), escoamento (`ro`), temperatura e evapotranspiração em 100% do território | Grade ~9 km (`0,1°`) × Hora/Dia | Copernicus CDS API / Google Earth Engine | **Recomendada (V1/V2)** |
| **9** | **INMET BDMEP + CEMADEN** | Hidrologia | Chuva horária/sub-horária (10 min–1 h) para enxurradas urbanas rápidas e alertas oficiais de risco hidrológico | Estação/Pluviômetro × 10 min / Hora | BigQuery `br_inmet_bdmep` / CEMADEN Dados Abertos | **Complementar (V1/V2)** |
| **10** | **S2iD / SEDEC — FIDE (MIDR)** | Desastres | Decretos de Emergência/Calamidade + **danos humanos** (`desabrigados`, `desalojados`, `enfermos`, `feridos`, danos à saúde) | Município × Evento (FIDE) | Portal S2iD Dados Abertos / Base dos Dados | **Obrigatória (V1)** |
| **11** | **Sentinel-1 SAR + Copernicus DEM / HAND + IPH/UFRGS Zenodo** | EO / Relevo | Mancha física real de inundação (atravessa nuvens!), duração de *ponding* (`DOI 10.5281/zenodo.16929471`) e `HAND` de hospitais/áreas aterradas | Raster 10–30 m + GeoPackage MGB RS | Planetary Computer STAC + [Zenodo IPH/UFRGS-ANA](https://doi.org/10.5281/zenodo.16929471) + `rigeo.sgb.gov.br` | **Recomendada (V1/V2)** |
| **12** | **MapBiomas** (`cobertura_municipio_classe` + Rasters 30 m) | Uso do Solo / Aterros | Impermeabilização urbana (`24`), **banhados/áreas úmidas (`11`), marismas (`84`), água (`33`)** e floresta (`3`/`49`) desde `1985` (proxy causal de aterro/ocupação de várzeas) | Município/Pixel 30 m × Ano (`1985–2024`) | BigQuery `br_mapbiomas_estatisticas` / MapBiomas | **Obrigatória (V1)** |
| **13** | **OpenStreetMap (`OSMnx`) + DNIT / DAER-RS** | Mobilidade | Malha viária, pontes, tempo de ambulância e isolamento de hospitais-escola sob bloqueios rodoviários | Grafo viário (arestas/nós) | Overpass/OSMnx + boletins DNIT/DAER-RS | **Recomendada (V1/V2)** |
| **14** | **IBGE (`populacao`, Censo 2022, malha) + Macrorregiões SUS** | Demografia | População total e por faixa etária (idosos/crianças), área oficial em km² e fronteiras de Regiões/Macrorregiões de Saúde | Município / Setor Censitário × Ano | BigQuery `br_ibge_populacao` + `br_bd_diretorios_brasil` / IBGE | **Obrigatória (V1)** |
| **15** | **Sistemas de Proteção Contra Cheias (GeoPOA/DMAE, GeoPelotas/SANEP, IEDE-RS/SEMA, ANA BHO v6)** | Diques / Pôlderes / Bacias | Polígonos de pôlderes, traçado de diques, estações de bombeamento (23 EBAPs POA, Laranjal/São Gonçalo em Pelotas) e topologia `BHO v6 RS` (Ottotrechos/Ottobacias) | Vetorial (`GeoJSON`/`GeoPackage`/`SHP`) | `gis-smamus.portoalegre.rs.gov.br`, `geopelotas-pmpel.hub.arcgis.com`, `iede.rs.gov.br`, `dadosabertos.ana.gov.br` | **Recomendada (V1 RS)** |
| **16** | **NASA POWER (`PRECTOTCORR`, `GWETTOP`, `T2M`, `V10M`)** | Hidrologia / Remanso | Série diária contínua sem lacunas (`1981–presente`, 0 nulos) incluindo **vento meridional `V10M`** (vento Sul/Sudeste que gera remanso no Guaíba e Lagoa dos Patos) | Grade diária 100% território (sem API key) | `power.larc.nasa.gov/api/temporal/daily/point` | **Obrigatória (V1/V2)** |

---

## 2. Detalhamento Completo por Base (O que Possuem e Por Que Selecionar)

### Grupo A — Saúde, Capacidade Hospitalar e Demanda Clínica

#### 1. CNES — Cadastro Nacional de Estabelecimentos de Saúde (`basedosdados.br_ms_cnes`)
- **O que possui de detalhado:**
  - `estabelecimento`: código `id_estabelecimento_cnes` (7 dígitos), `id_municipio`, `sigla_uf`, `tipo_unidade` (Hospital Geral, Hospital Especializado, Pronto-Socorro, etc.), esfera administrativa, natureza jurídica, gestão e coordenadas/endereço.
  - `leito`: contagem mensal por estabelecimento e **especialidade/tipo de leito** (Clínico, Cirúrgico, Obstétrico, Pediátrico, **UTI Adulto/Pediátrica/Neonatal separadas de UCI/Isolamento**), distinguindo **`quantidade_existente`** vs. **`quantidade_sus`**.
  - `estabelecimento_ensino` *(e atributos de ensino no CNES)*: vínculo com universidades, residência médica/multiprofissional e certificação formal MEC/MS como **Hospital de Ensino** (identifica HCPA, HUSM/UFSM, HE-UFPel, HU-FURG, Santa Casa/UFCSPA, HSL-PUCRS no RS e toda a rede EBSERH/universitária no Brasil).
  - `equipamento` / `profissional` *(opcional V1/V2)*: respiradores/ventiladores mecânicos, tomógrafos, leitos com suporte avançado e carga horária de médicos intensivistas/enfermeiros (gargalos reais no simulador `SimPy`).
- **Por que selecionar:** Resolve imediatamente o erro `DB-001` e a confusão `leitos_complementar` vs. UTI; permite filtrar **hospitais-escola** e distinguir oferta SUS real de leitos privados.
- **Armadilhas conhecidas:** Mede **estoque cadastrado**, não ocupação real; pode ter atraso de atualização ou saltos administrativos de recadastramento (detectáveis via `aeon` ClaSP).

#### 2. SIH-SUS — Sistema de Informações Hospitalares do SUS (AIH Reduzida `rd` / `basedosdados.br_ms_sih`)
- **O que possui de detalhado (campos oficiais DATASUS/AIH Reduzida):**
  - `CNES` (7 dígitos): hospital exato onde ocorreu a internação (permite isolar **cada hospital-escola**!).
  - `MUNIC_RES` (6 dígitos IBGE) e `MUNIC_MOV`: município de residência do paciente vs. município do hospital — **constrói empiricamente a matriz de referência (*catchment*) e os fluxos inter-municipais (`networkx` / `torch-geometric`)**!
  - `DT_INTER` e `DT_SAIDA` + `DIAS_PERM`: datas e **duração exata da internação (LOS em dias)** para modelagem de ocupação diária e riscos competitivos (`scikit-survival`).
  - `UTI_MES_TO` (e diárias de UTI/UCI): dias efetivos ocupando leito de Terapia Intensiva.
  - `DIAG_PRINC` (CID-10): diagnóstico principal — permite separar internações sensíveis a enchentes (traumas `S00–T98`, leptospirose `A27`, doenças infecciosas intestinais `A00–A09`, respiratórias `J00–J99`, descompensações cardiovasculares/renais `I00–I99`/`N17–N19`) de causas não relacionadas.
  - `CAR_INT` (`01` Eletivo vs. `02` Urgência/Emergência): permite quantificar o **espaço real de contingência via adiamento de cirurgias eletivas** no `SimPy`!
  - `MORTE` (`0` alta viva, `1` óbito hospitalar) e `ESPEC` (especialidade do leito: Cirurgia, Clínica, Obstetrícia, Pediatria).
- **Por que selecionar:** É a **base transformadora do projeto**. Sem o SIH-SUS, o modelo só enxerga oferta cadastral (CNES). Com o SIH-SUS agregado (zero PII), passamos a medir **demanda real, tempo de permanência, uso real de UTI, adiamento de eletivas e fluxo de pacientes para hospitais-escola**.
- **Guardrail LGPD:** Agregar diretamente no SQL BigQuery por `(CNES, MUNIC_RES_ou_macrorregiao, grupo_cid, carater_internacao, data_ou_semana)` e aplicar supressão de pequenas células ($n < 5$). **Limitação:** cobre apenas internações financiadas pelo SUS (~70–80% do país e ~100% das vagas SUS dos hospitais universitários) e tem latência de processamento de 1–2 meses (excelente para treino/calibração/validação; não é feed em tempo real).

#### 3. SINAN — Sistema de Informação de Agravos de Notificação (DATASUS / Base dos Dados)
- **O que possui de detalhado:**
  - Notificações compulsórias georreferenciadas por município de residência/infecção, data dos primeiros sintomas, hospitalização (`sim/não`), evolução (cura/óbito) e **ficha específica de Leptospirose** com campo explícito de exposição: *"contato com água ou lama de enchente nos 30 dias anteriores aos sintomas"*.
  - Cobre também **Dengue/Arboviroses**, **Hepatite A**, **Tétano acidental** e **Acidentes por animais peçonhentos** (que disparam durante e após enchentes).
- **Por que selecionar:** O impacto hospitalar de uma enchente ocorre em **duas ondas**: Onda 1 (D+0 a D+5: trauma, hipotermia, evacuação de hospitais alagados, interrupção de diálise/crônicos) e **Onda 2 (D+7 a D+30: leptospirose grave com insuficiência renal/UTI, infecções hídricas e respiratórias)**. O SINAN modela explicitamente a Onda 2.

#### 4. GERINT — Dados Abertos de Porto Alegre / SES-RS (`dadosabertos.poa.br`)
- **O que possui de detalhado:**
  - Registros agregados e anonimizados de **solicitações de internação hospitalar** na rede pública regulada pelo GERINT em Porto Alegre (central que regula vagas para HCPA, Santa Casa/UFCSPA, HSL-PUCRS e GHC), incluindo tipo de leito solicitado (enfermaria/UTI), prioridade/risco, data da solicitação, especialidade e situação da regulação.
- **Por que selecionar:** Diferentemente do SIH-SUS (que registra apenas quem *conseguiu* internar — demanda atendida/censurada pela capacidade!), o GERINT mede **demanda solicitada e fila de espera (demanda reprimida)**! É um diferencial científico extraordinário para validar a V1 no RS.
- **Limitação:** Cobertura aberta detalhada concentrada na capital (Porto Alegre/região regulada), não replicada de forma padronizada nos 27 estados — por isso entra como **validador de demanda reprimida na V1 RS**, mantendo SIH-SUS como espinha dorsal escalável para o Brasil (V2).

#### 5. SIVEP-Gripe / SIM (OpenDataSUS / DATASUS)
- **O que possui de detalhado:**
  - `SIVEP-Gripe`: internações por Síndrome Respiratória Aguda Grave (SRAG), admissão em UTI, uso de suporte ventilatório (invasivo/não invasivo) e evolução — altamente sensível à combinação de **abrigos superlotados + queda de temperatura** (como ocorreu em maio–junho/2024 no RS).
  - `SIM`: óbitos por causa básica (CID-10) e local de ocorrência (hospitalar vs. domiciliar/via pública), útil para auditar gravidade e mortalidade evitável.
- **Por que selecionar:** Captura pressão respiratória/ventilatória aguda em UTIs durante enchentes de outono/inverno no Sul do Brasil.

---

### Grupo B — Hidrologia Física, Meteorologia e Desastres

#### 6. MERGE-CPTEC / INPE (Precipitação Diária em Grade)
- **O que possui de detalhado:**
  - Precipitação diária (`mm/dia`) em grade regular de **`0,1°` (~10 km × 10 km)** sobre toda a América do Sul (desde 2000 até D-1), combinando estimativas de satélite **IMERG/GPM (NASA)** corrigidas com estações de superfície brasileiras (INMET, ANA, ONS, redes estaduais).
- **Por que selecionar:** **Elimina o erro crítico `C7` (90% de exclusão municipal):** enquanto estações INMET cobrem apenas 571 de 5.570 municípios brasileiros (e ~40 de 497 no RS), o MERGE cobre **100% dos municípios e 100% das ottobacias** com viés corrigido para o clima brasileiro!

#### 7. ANA HidroWeb API + Ottobacias (BHO) + SGB/SACE
- **O que possui de detalhado:**
  - `HidroWeb (ANA)`: séries históricas e telemétricas de **cota fluviométrica (cm)** e **vazão (m³/s)** da Rede Hidrometeorológica Nacional.
  - `Ottobacias (BHO/ANA)`: polígonos e trechos de drenagem com codificação hierárquica de Otto Pfafstetter — permite calcular **chuva acumulada a montante** e construir o **grafo dirigido de propagação de cheias** (`networkx.DiGraph` / `torch-geometric`).
  - `SACE (SGB/CPRM)`: cotas oficiais de **atenção, alerta e inundação** para bacias críticas brasileiras (no RS: **Taquari-Antas, Jacuí, Sinos, Caí, Gravataí, Guaíba e Uruguai**).
- **Por que selecionar:** Substitui o proxy circular `nivel_alagamento_obs` por **medições hidrológicas físicas reais** e resolve o grande gargalo físico do RS (Porto Alegre alagou em maio/2024 pela chuva extrema nas bacias a montante do Taquari/Jacuí/Caí/Sinos, não apenas pela chuva local na capital!).

#### 8. ERA5-Land (Copernicus / ECMWF)
- **O que possui de detalhado:**
  - Reanálise física horária/diária em ~9 km (`0,1°`) com **umidade volumétrica do solo em 4 profundidades (`swvl1–swvl4`)**, escoamento superficial e subsuperficial físico (`sro`, `ssro`, `ro`), temperatura a 2 m (crucial para hipotermia e SRAG pós-enchente!), vento e evapotranspiração potencial.
- **Por que selecionar:** Complementa o MERGE com estado físico real de encharcamento do solo (substituindo/validando o `API` exponencial com `k_decay=0.88` fixo, que falha em biomas diferentes no Brasil).

#### 9. INMET BDMEP (`basedosdados.br_inmet_bdmep`) + CEMADEN
- **O que possui de detalhado:**
  - `INMET`: observações horárias (`precip_max_horaria_mm`, temperatura, umidade relativa, pressão atmosférica) nas estações automáticas/convencionais.
  - `CEMADEN`: milhares de pluviômetros automáticos em áreas urbanas de risco (acúmulos de 10 min a 1 h) e histórico de alertas de risco hidrológico/geológico em ~1.100–1.300 municípios prioritários.
- **Por que selecionar:** O MERGE é diário; `INMET` + `CEMADEN` trazem intensidade **sub-horária/horária**, fundamental para distinguir chuvas convectivas extremas (enxurradas urbanas rápidas) de chuvas estratiformes prolongadas.

#### 10. S2iD / SEDEC — FIDE (MIDR)
- **O que possui de detalhado:**
  - Histórico nacional de desastres por código COBRADE (separando **Inundação `12100`**, **Enxurrada `12200`** e **Alagamento `12300`**!), data do desastre, decretos de Emergência/Calamidade e **Formulário de Informações do Desastre (FIDE)** com danos humanos (`desabrigados`, `desalojados`, `enfermos`, `feridos`, `desaparecidos`, `mortos`) e danos materiais a **instalações públicas de saúde**.
- **Por que selecionar:** Não deve ser o único rótulo físico de enchente (devido a viés administrativo), mas fornece **sinal valioso de deslocamento populacional (`desabrigados`/`desalojados`)** e **danos diretos a hospitais**, fortemente correlacionados com a pressão sobre hospitais-escola!

---

### Grupo C — Observação da Terra, Relevo, Uso do Solo e Mobilidade

#### 11. Sentinel-1 SAR + Copernicus DEM GLO-30 / MERIT Hydro (HAND) via Planetary Computer STAC
- **O que possui de detalhado:**
  - `Sentinel-1 GRD (SAR banda C)`: imagens de radar de 10 m que **penetram nuvens** e detectam lâmina d'água real sobre áreas urbanas e hospitais (ex.: mapa exato da inundação de maio/2024 em Porto Alegre, Canoas, São Leopoldo, Vale do Taquari e Pelotas/Rio Grande).
  - `Copernicus DEM GLO-30` / `MERIT Hydro`: modelo digital de elevação (30 m–90 m), declividade, direção de fluxo e **HAND (*Height Above Nearest Drainage*)** em CRS métrico (`SIRGAS 2000`).
- **Por que selecionar:** Permite verificar objetivamente **quais hospitais periféricos e vias de acesso ficaram debaixo d'água** (perda de capacidade funcional + desvio de pacientes para hospitais-escola) e substituir `lat/lon` crus por atributos geomorfológicos transferíveis RS → Brasil (`ADR-0001`).

#### 12. MapBiomas (`basedosdados.br_mapbiomas_estatisticas` + Rasters)
- **O que possui de detalhado:**
  - Série anual (1985–presente, 30 m) de cobertura e uso da terra: infraestrutura urbana (impermeabilização), floresta/vegetação nativa (retenção), agricultura e dinâmica de superfície de água.
- **Por que selecionar:** Alimenta diretamente o ramo monotônico estrutural (`ADR-0001`), agora sem vazar anos futuros (`2020–2024` usados em `2010–2023`) e sem usar soma de classes como proxy da área territorial municipal.

#### 13. OpenStreetMap (`OSMnx`) + Boletins de Bloqueios DNIT / DAER-RS
- **O que possui de detalhado:**
  - Grafo viário completo (rodovias federais/estaduais, pontes, vias urbanas, tempos estimados de deslocamento de ambulâncias) + registros históricos de interdições totais/parciais em rodovias (ex.: BR-116, BR-290, RS-118, ERS-129/130 em maio/2024).
- **Por que selecionar:** Em maio/2024, o problema não foi só "quantos pacientes precisavam de leito", mas **quais hospitais-escola permaneceram acessíveis** quando pontes e rodovias ruíram — entrada direta para `networkx`, `SimPy` e `pymoo`.

---

### Grupo D — Demografia e Territorialização SUS

#### 14. IBGE (`br_ibge_populacao`, Censo 2022, `br_bd_diretorios_brasil.municipio`) + Macrorregiões de Saúde (MS/DATASUS)
- **O que possui de detalhado:**
  - População municipal anual + **Censo 2022 por setor censitário e faixa etária** (proporção de idosos $\ge 60$ anos e crianças $< 5$ anos em zonas de baixo HAND/risco de inundação!).
  - Área oficial municipal (km²) e malhas geográficas oficiais (`SIRGAS 2000`).
  - Tabela oficial de **Regiões de Saúde (CIR) e Macrorregiões de Saúde do SUS** (agrupamento assistencial real do RS — 7 Macrorregiões / 30 Regiões de Saúde — e do Brasil).
- **Por que selecionar:** Corrige a distorção ecológica (`leitos_por_1000hab` no município-sede do hospital-escola) ao normalizar a capacidade pela **população real da Macrorregião de Referência** e pela população vulnerável exposta à mancha de inundação.

---

## 3. Estratégia Mista de Ingestão Adotada (V1 RS → V2 Brasil)

Para evitar paralisia de ETL sem abrir mão do rigor científico (defasagem de 1–4 semanas, UTI específica e heterogeneidade regional POA vs. Norte/Serra), a seleção segue uma **estratégia mista em duas trilhas incrementais na V1 (RS)** antes da expansão nacional:

| Trilha / Fase | Bases Incluídas | Esforço | O que Entrega de Imediato |
| :--- | :--- | :---: | :--- |
| **Trilha 1A — Núcleo Semanal RS (*Tracer Bullet*)** *(Início Imediato)* | `#1 CNES (estab + leito por tipo de UTI + ensino)` + `#2 SIH-SUS (agg semanal sem PII)` + `#3 SINAN (Leptospirose/Onda 2)` + `#6 MERGE-CPTEC (100% RS)` + `#7 ANA HidroWeb/Ottobacias` + `#10 S2iD FIDE` + `#12 MapBiomas` + `#14 IBGE/Macrorregiões` | **Baixo–Médio** | Resolve 100% dos erros críticos e entrega rapidamente o primeiro modelo semanal *ex-ante* ($h=1..4$ semanas, foco $h=2$ semanas) por **UTI Adulto / UTI Ped-Neo / Clínico** e por **Macrorregião / Hospital-Escola do RS** |
| **Trilha 1B — Enriquecimento Físico & Fila Reprimida (V1 Completa)** | **Adiciona sobre a 1A:** `#4 GERINT-POA (demanda solicitada/fila na capital)` + `#8 ERA5-Land (umidade solo + temp)` + `#11 DEM/HAND & Sentinel-1 SAR (eventos RS)` + `#13 OSMnx (acesso/cortes viários)` | **Médio** | Valida demanda reprimida em Porto Alegre, incorpora física de relevo/inundação de hospitais periféricos e bloqueios de rodovias sem atrasar a validação inicial do *lag* de 2 semanas |
| **Trilha V2 — Escala Nacional Brasil (27 UFs)** | **Expande 1A + 1B para todo o Brasil** (mantendo SIH/CNES como espinha dorsal nacional) + `#5 SIVEP-Gripe (SRAG)` + `#9 INMET horário + CEMADEN` via `polars`/`dask` | **Alto** | Cobertura nacional (5.570 municípios, todos os biomas, cheias rápidas e lentas, Hetero-GNN e alocação Pareto federal) |
