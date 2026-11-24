# ADR-0007: Análise Contrafactual Estruturada e Relatório Final de Verificação Científica

- **Status:** ACCEPTED
- **Data:** 2026-11-24

## Contexto e Problema

Métricas globais médias anuais (como MAE/RMSE ao longo de 52 semanas) diluem o desempenho em eventos extremos raros: um modelo pode apresentar baixo MAE médio em semanas calmas e, ainda assim, errar drasticamente o **excesso de leitos**, o **atraso de pico (+2 semanas na Onda 2)** ou atribuir impacto às variáveis erradas (ex.: ignorar escoamento a montante, áreas aterradas ou remanso lagunar).

Além disso, a entrega científica final do projeto exige um **artefato de verificação auditável e reproduzível** que reúna, em um único relatório padronizado, a metodologia, as ferramentas, a linhagem de datasets e **gráficos contrafactuais objetivos para medir a acurácia real dos modelos**.

## Decisão

Incorporar uma camada formal de **Análise Contrafactual Estruturada** e **Geração Automática de Relatório de Verificação** no módulo `src/previsao_leitos/audit/` (Feature **F06**, acoplada ao simulador `SimPy` da **F05**), composta por dois eixos complementares e um contrato fixo de relatório:

### 1. Eixo A — Contrafactual Retrospectivo "Sem Enchente" (*No-Flood Baseline* + Validação Placebo)
- **Construção independente da linha de base contrafactual ($\hat{y}^{(0)}_t$):** estimar a trajetória esperada de demanda sem o choque climático utilizando **Controle Sintético / Séries Temporais Estruturais Bayesianas** ajustados em `<= 2022` e apoiados por regiões não inundadas de controle (ex.: Planalto/Serra em 2024; UFs/bacias não atingidas na V2), sempre com intervalo de credibilidade de `95%`.
- **Avaliação direta da acurácia no choque:** comparar o **excesso real observado** ($\Delta^{\text{obs}}_t = y_t - \hat{y}^{(0)}_t$) contra o **excesso previsto *ex-ante* pelo modelo** ($\Delta^{\text{mod}}_t = \hat{y}(X_{\text{real}}) - \hat{y}(X_{\text{climatologia P50}})$) na janela de desastre (ex.: Semanas 18–30 de 2024).
- **Testes Placebo obrigatórios:** aplicar o mesmo procedimento em períodos/regiões **sem enchente** (ex.: calibração `2023` em semanas secas e cabeceiras não inundadas), verificando que o viés placebo contém `0` no IC 95% (auditoria contra falsos alarmes).

### 2. Eixo B — Contrafactuais Intervencionistas Estruturais (*What-If* Físicos e de Gestão)
Avaliados sob as restrições físicas do [ADR-0001](0001-structural-monotonicity.md) e [ADR-0006](0006-landfill-polders-and-intermunicipal-routing.md):
1. **Decomposição Intermunicipal ($G_{\text{hydro}}$):** `do(chuva_montante_pond = P50)` mantendo chuva local observada — mede a fração da demanda causada por água vinda de cidades a montante.
2. **Áreas Aterradas, Pôlderes e Remanso Lagunar:** `do(remanso_lagunar_idx = 0)` e variação controlada de `indice_aterro_varzea` — quantifica a contribuição de áreas aterradas e vento Sul em Porto Alegre e Pelotas/Rio Grande.
3. **Transbordamento Assistencial ($G_{\text{health}}$):** `do(W_ref = I)` (ausência de desvio intermunicipal) vs. $W_{\text{ref}}$ real — estima o peso da rede de referência sobre hospitais-escola.
4. **Intervenção Operacional de Contingência (`SimPy` — F05):** acionamento antecipado de reserva de leitos/adiamento de eletivas com antecedência `h = 2` semanas vs. reação tardia (`h = 0`).

### 3. Métricas Contrafactuais de Acurácia
- **Shock-Window Excess MAE / Bias** (erro na magnitude do excesso durante o desastre);
- **Peak-Timing Error ($\Delta t_{\text{pico}}$ em semanas)** (acurácia temporal na Onda 1 e Onda 2);
- **Placebo Bias & Coverage (95%)** (ausência de efeito espúrio fora de choques);
- **Taxa de Violação Física/Monotônica** (alvo `0%` sob perturbações positivas de chuva a montante/aterro/remanso).

### 4. Contrato do Relatório Final de Verificação (`reports/<run_id>/verificacao_cientifica.md`)
Todo ciclo de avaliação gera automaticamente:
1. Relatório estruturado contendo: **Metodologia**, **Ferramentas/Versões/Manifesto (`run_manifest.json`)**, **Inventário e Auditoria de Datasets (com prova LGPD $n < 5$)**, **Benchmark de Acurácia com *block-bootstrap* 95% CI** e **Seção Contrafactual**;
2. Pacote padronizado de **5 gráficos contrafactuais acessíveis (paleta *Okabe-Ito*, 300 DPI, PNG/SVG)**:
   - `fig_c1_trajetoria_contrafactual`: Observado vs. Contrafactual Sem Enchente vs. Previsão *Ex-Ante* ($P_{50}$ e banda $P_{10}–P_{90}$);
   - `fig_c2_acuracia_excesso_choque`: Excesso real ($\Delta^{\text{obs}}_t$) vs. Excesso previsto ($\Delta^{\text{mod}}_t$) e erro de timing de pico;
   - `fig_c3_validacao_placebo`: Diagnóstico placebo em regiões/anos sem cheia;
   - `fig_c4_decomposicao_intermunicipal_aterro`: Atribuição contrafactual em cascata (*waterfall*) separando chuva local, escoamento a montante, remanso lagunar, áreas aterradas e rede `W_ref`;
   - `fig_c5_contingencia_simpy`: Curva contrafactual de ocupação e fila hospitalar (`h = 2` sem. vs. `h = 0`).

## Alternativas Rejeitadas

- **Reportar apenas MAE/RMSE anual médio:** oculta falhas graves exatamente durante a janela crítica de enchente e não testa coerência física.
- **Apresentar contrafactuais puramente baseados no mesmo modelo sem controle sintético independente nem teste placebo:** cria validação circular sem referência observacional externa.
- **Tratar saídas contrafactuais como prova causal clínica ou recomendação assistencial ao vivo:** violaria a fronteira de pesquisa aberta (*Not for patient care or live clinical use*).

## Consequências

- A especificação detalhada do módulo `audit` fica registrada em [spec/SPEC-audit-counterfactual-report.md](../../spec/SPEC-audit-counterfactual-report.md) e acompanhada como Feature **F06** em [docs/features/06-auditoria-contrafactual-e-relatorio.md](../features/06-auditoria-contrafactual-e-relatorio.md).
- Os gráficos e métricas contrafactuais entram já na **Etapa 2/3** (sobre `2023` e `2024` de desenvolvimento) e serão executados uma única vez no *holdout* intocado **`2025`** na auditoria final da V1 RS.
