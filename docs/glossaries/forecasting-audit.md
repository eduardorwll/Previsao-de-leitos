# Glossário — Contexto: Previsão, Calibração e Auditoria

### Origem da Previsão ($t_0$)
Instante exato em que uma previsão é emitida. Apenas informações já observadas e disponíveis até $t_0$ podem alimentar o modelo naquela previsão.

### Horizonte de Previsão ($h$)
Intervalo temporal futuro entre a origem da previsão ($t_0$) e o período previsto ($t_0 + h$), medido em dias calendário ou semanas epidemiológicas.

### Covariável *Ex-Ante* vs. *Ex-Post*
Uma variável é *ex-ante* quando já é conhecida ou prevista em $t_0$ (ex.: probabilidade de cheia emitida pelo modelo hidrológico); é *ex-post* quando só se torna conhecida após o desfecho (ex.: chuva total acumulada no mês-alvo ou decreto emitido posteriormente). Modelos prospectivos só podem utilizar covariáveis *ex-ante*.

### Baseline de Persistência
Referência mínima em que a previsão para o período seguinte repete o último valor observado ($\hat{y}_t = y_{t-1}$, isto é, variação prevista zero). Todo modelo de demanda precisa superar a persistência na escala de variação ($\Delta$) e em meses de choque.

### Holdout Intocado
Partição temporal ou espacial reservada exclusivamente para a avaliação final única, nunca utilizada para seleção de arquitetura, ajuste de hiperparâmetros, *early stopping* ou calibração.

### Calibração Probabilística
Grau de concordância entre as probabilidades previstas e as frequências reais observadas dos eventos, auditado por Erro Esperado de Calibração (ECE), *Brier Skill Score* (BSS) e diagramas de confiabilidade.

### Resíduo Contrafactual de Choque
Diferença entre o total observado durante um evento extremo e a trajetória esperada sem o choque. Serve como diagnóstico descritivo retrospectivo, não como prova causal automática nem como previsão prospectiva de demanda.

### Supressão de Pequenas Células (LGPD)
Regra de controle estatístico de divulgação que oculta (supressão primária e complementar) contagens agregadas abaixo de um limiar mínimo de pacientes (ex.: $n < 5$) para impedir reidentificação indireta em municípios ou estratos pequenos.
