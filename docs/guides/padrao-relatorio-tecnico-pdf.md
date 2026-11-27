# Diretriz Canônica — Modelo Oficial de Relatório Técnico-Científico em PDF

> **Status:** Padrão Oficial Vigente do Projeto `Previsao-de-leitos`  
> **Documento de Referência Aprovado:** `Relatorio_Completo_Previsao_de_Leitos_RS_2024.pdf` (13 páginas, 13 gráficos em 300 DPI, diagramação HTML/CSS editorial via Chromium headless)  
> **Fronteira Obrigatória:** Todo relatório emitido sob este padrão deve declarar explicitamente em destaque: **"Not for patient care or live clinical use"** e conformidade estrita **LGPD** (zero PII; supressão de pequenas células $n < 5$).

---

## 1. Princípio Editorial Inegociável

**Proibido embutir sintaxe Markdown crua em PDF.**  
Jamais gerar PDFs colando texto Markdown bruto (`#`, `**`, `| --- |`, backticks ou blocos `<pre>`) dentro do documento final. Todo relatório em PDF deve ser compilado a partir de **HTML5 + CSS3 editorial dedicado**, renderizado via **Chromium Headless** (`--headless=new --disable-gpu --no-sandbox --print-to-pdf=... --print-to-pdf-no-header`), com:

- **Paginação A4 limpa:** margens balanceadas (`15 mm`), rodapé contínuo com numeração (`Página X de Y`) e faixa fixa de fronteira clínica/pesquisa aberta;
- **Hierarquia tipográfica clara:** capa/cabeçalho institucional em gradiente sobrio (`#072a40 → #0b4a6f → #115e59`), banners de capítulo numerados, cartões de KPIs (6 indicadores executivos), caixas de destaque semânticas (alerta clínico laranja, insight analítico azul, conclusão verde) e tabelas HTML verdadeiras com cabeçalho escuro, linhas zebradas e alinhamento numérico tabular (`tabular-nums`);
- **Controle de quebras de página:** `page-break-inside: avoid` em figuras, cartões e tabelas; `page-break-after: avoid` em títulos; quebras explícitas entre capítulos maiores para evitar títulos órfãos ou tabelas partidas ao meio;
- **Gráficos científicos acessíveis:** todos os gráficos em **300 DPI**, embarcados em Base64 PNG, utilizando obrigatoriamente a **paleta Okabe-Ito (WCAG 2.2)** amigável a daltonismo, acompanhados de legenda numerada e explicação interpretativa logo abaixo da imagem;
- **Destino padrão de entrega:** salvo no diretório de execuções (`reports/runs/<run_id>/`) e espelhado em `~/Downloads/` com nome claro e autoexplicativo quando solicitado para leitura direta.

---

## 2. Estrutura Obrigatória (Capa + 6 Capítulos)

Todo relatório técnico-científico completo de avaliação de modelos ou resultados do projeto deve seguir esta arquitetura de conteúdo:

### Capa Editorial e Sumário Executivo
- Título, subtítulo, versão, escopo territorial, estoque real de referência CNES, partições temporais congeladas (com status do holdout `2025`) e metadados de reprodutibilidade (`git SHA`, suíte de testes, cobertura, vulnerabilidades).
- **Banner obrigatório de Fronteira Clínica e Privacidade (LGPD)**.
- **Mapa estrutural (índice)** dos 6 capítulos.
- **6 Cartões de KPIs (KPI Cards)** resumindo os principais números reais auditados (acurácia nos alvos primários, coerência física, impacto operacional simulado, higiene de dados e evolução de versão).

### Capítulo 1 — Introdução Conceitual Profunda (Modelos, Métodos e Siglas)
- Contextualização científica do problema (ex.: Onda 1 vs. Onda 2, riscos de modelos caixa-preta, necessidade de ecossistema modular).
- **Cartões detalhados de cada modelo utilizado**, cobrindo obrigatoriamente para cada um:
  1. Nome completo e significado de **todas as siglas**;
  2. Onde é usado no projeto;
  3. Como funciona matematicamente e operacionalmente (fórmulas/mecanismos-chave);
  4. Por que essa abordagem foi escolhida frente a alternativas.
- **Enciclopédia Conceitual** explicando com rigor e clareza didática todos os termos técnicos relevantes ao relatório (incluindo fórmulas quando útil): `MASE`, `MAE`, `Pinball Loss (P50/P90/P95)`, `Cobertura Empírica`, `Monotonicidade Estrutural & autograd`, `SimPy`, `VIF`, `Censura à Direita & Nowcasting`, `Rótulos Degenerados (0 >= 0 / Gate G8)`, `ECE`, `BSS`, `Net Benefit`, `MAUP`, `SUTVA`, `Ex-Ante vs. Ex-Post`, `Holdout Intocado`, `Catraca Champion-Challenger`, `LGPD (n < 5)` e teleconexões `ENSO`.

### Capítulo 2 — Significado Estatístico e Demonstração Visual das Métricas
- Tabela explicativa traduzindo **cada estatística real encontrada**:
  - Valor numérico exato (com intervalo de confiança 95% quando aplicável);
  - **Em comparação a quê** (Persistência, Sazonal, modelo linear, política reativa `h=0`, versão anterior);
  - **Relevância prática** no planejamento hospitalar (em linguagem humana).
- **Demonstrações gráficas didáticas (`Fig. D1`, `Fig. D2`, ...):**
  - Comparação visual de `MASE` frente à linha de empate ingênuo (`1,00`) + evolução da cobertura $P_{10}–P_{90}$ frente ao alvo nominal (`80%`);
  - Provas visuais sobre dados reais das propriedades críticas do sistema (curva real de monotonicidade, Curva de Decisão *Net Benefit* e demonstração visual de censura à direita + reconstrução *Nowcasting*).

### Capítulo 3 — Origem e Rastreabilidade Completa das Bases de Dados
- **Inventário real dos arquivos em disco** (nome, linhas exatas, colunas, contagem de nulos, tamanho em MB e diagnóstico de qualidade/LGPD).
- **Catálogo completo de todas as fontes de dados** (ativas, conectores prontos e planejadas), especificando obrigatoriamente:
  1. **Sigla e nome oficial por extenso** (ex.: `SIH-SUS`, `CNES`, `IBGE`, `NASA POWER`, `MapBiomas`, `ANA HidroWeb / BHO v6`, `MERGE-CPTEC`, `ERA5-Land`, `Sentinel-1 SAR / IPH-UFRGS`, `DMAE/SANEP`, `GERINT-POA`, `SINAN`, `SIVEP-Gripe`, `NOAA CPC ENSO`, `S2iD/FIDE`, `OSMnx/DNIT`);
  2. Origem / tabela oficial;
  3. Status de integração (`Ativa`, `Conector V1.1`, `Planejada V2`);
  4. **Onde** é utilizada no pipeline;
  5. **Por que** foi escolhida (valor científico e qual viés corrige).

### Capítulo 4 — Simulações Contrafactuais Reais (Previsto Ex-Ante vs. Real Observado)
- **Unidade operacional compreensível:** sempre converter diárias semanais para **Leitos Ocupados/Dia** (Lei de Little: `Diárias / 7`) e traçar a linha de **Capacidade Real SUS CNES** para comparação direta.
- **Gráficos descritivos obrigatórios sobre o ano de auditoria (ex.: 2024):**
  - Trajetórias *Previsto Ex-Ante ($P_{50}$ e banda $P_{10}–P_{90}$)* vs. *Real Observado SIH-SUS* vs. *Contrafactual Sem Enchente (`BSTS` Pré-Choque)* nas macrorregiões-chave;
  - **Tabelas semana a semana** na janela de crise para as regiões-âncora (ex.: Capital/POA e Sul/Pelotas), exibindo semana ISO, chuva a montante, contrafactual, previsto $P_{50}/P_{90}$, real observado em leitos/dia, erro e indicador de acerto na banda;
  - Ranking estadual comparativo das maiores regiões atingidas;
  - Simulações dedicadas para **Onda 2 (Leptospirose)** e **Respiratório / SRAG**;
  - Auditorias contrafactuais complementares: **Acurácia de Excesso na Janela de Choque**, **Validação Placebo**, **Decomposição Causal *Waterfall* (`do()`)**, **Simulação Operacional `SimPy`** (`h=2` proativo vs. `h=0` reativo), **Oráculo Prospectivo El Niño**, **Scorecard de Higiene (`G1–G8`)** e **Catraca *Champion-Challenger***.

### Capítulo 5 — Arquitetura Modular, Decisões Irreversíveis (ADRs) e Stack Tecnológica
- Diagrama visual em etapas do fluxo completo do sistema (Ingestão → Física/ENSO → Hidrologia → Demanda/Cauda → Prontidão/Auditoria).
- Tabela das **Decisões Arquiteturais (ADRs)** indicando *o que foi decidido*, *por quê*, *qual alternativa foi descartada* e *qual risco foi evitado*.
- Tabela da **Stack Tecnológica** detalhando cada biblioteca/ferramenta e a barra de verificação mecânica contínua (`ruff`, `mypy --strict`, `pytest-cov`, `pip-audit`, `gitleaks`).

### Capítulo 6 — Abordagens de Melhoria e Novos Métodos para Explorar
- **Melhorias Imediatas de Alta Alavancagem (curto prazo):** limitações honestamente diagnosticadas na versão atual + ação concreta e ganho esperado (ex.: reextração BigQuery `v3`, ingestão ao vivo de fontes finas, recalibração de surtos).
- **Pontos Pendentes para Estudo Posterior (acessos a dados e análises em aberto):** seção obrigatória listando o que ficou para ser destravado depois — acessos ainda não concedidos (`SIVEP-Gripe`, `SINAN`, `S2iD/FIDE`, `ANA HidroWeb`), o erro de fase epidêmico investigado e os experimentos rejeitados na régua (trilha ENSO E1/E2/E3). A fonte única de verdade é `src/previsao_leitos/audit/pending_items.py` (espelhada em `docs/runbooks/acesso-bases-pendentes.md`).
- **Novos Métodos para Explorar Posteriormente (fronteira de pesquisa / V2 Brasil):** cartões detalhando novos paradigmas promissores (`Hetero-ST-GNN`, `TimesFM 2.5 Apache-2.0`, `PyMC` MCMC/NUTS, otimização Pareto `pymoo`, *Adaptive Conformal Inference* + Fine-Gray, `SHAP` monotônico e validação *Leave-UF-Out*).
- **Box de Conclusão Geral** sintetizando maturidade atual e próximo marco.

---

## 3. Checklist Obrigatório Antes de Entregar Qualquer Relatório PDF

- [ ] **Zero sintaxe Markdown crua** (`#`, `**`, `| --- |`, `<pre>` com markdown) em todo o PDF.
- [ ] **100% dos números e gráficos extraídos de execuções reais** (jamais valores fictícios ou estimados manualmente; auditar saídas antes de compilar para evitar gráficos planos ou bugs silenciosos de coluna).
- [ ] **Aviso obrigatório** *"Not for patient care or live clinical use"* e conformidade **LGPD** visíveis na capa e no rodapé de todas as páginas.
- [ ] **Transparência científica integral:** reportar tanto vitórias quanto limitações, falhas de gates ou rejeições da catraca *Champion-Challenger*.
- [ ] **Verificação final do artefato PDF:** confirmar cabeçalho válido `%PDF-`, número de páginas coerente, tamanho esperado e ausência de gráficos cortados ou tabelas ilegíveis.
