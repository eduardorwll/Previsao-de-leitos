# Previsao-de-leitos
Relatório de evolução da aplicação de Deep Learning à previsão de leitos hospitalares em enchentes

A aplicação até o momento atual percorreu 3 principais fases, partindo de previsão simples de enchentes no RS até abrangência de previsão nacional e relação das previsões com número de leitos alocados para cada área hospitalar, comparando com o número real.
Os modelos usam para aprendizado: precipitação por dia, precipitação máxima em uma hora, precipitação acumulada(dividida entre, 3/7/15 dias, índice de encharcamento do solo, índice de escoamento (relação entre impermeabilização e encharcamento do solo), temperatura (média e máxima), umidade relativa, pressão média, sazonalidade, impermeabilidade, retenção vegetal e altitude.

- 1ª fase:
Mun_embedding e lambda_pinn
Os municípios originalmente tinha um valor fixo para identificar cada um, e o modelo guardava esse valor e aumentava a chance de enchente com base no histórico de alagamento, mesmo sem nenhum indicio de enchente.
O código originalmente causava uma penalidade na loss quando havia inversão de causalidade na impermeabilidade das cidades, mas isso afetava negativamente os resultados, a solução foi tornar os valores estritamente positivos.
Nessa primeira fase, os dados foram testados com HydroFiLMMonotonicNet e o TabFM. A HydroFiLMMonotonicNet é a arquitetura de DeepLearning em PyTorch desenvolvida para esse projeto para prever a probabilidade e severidade de enchentes, o nome descreve os três blocos que compõem o fluxo da rede:
Hydro:
- Uma BiGru de 2 camadas que lê a sequência nos dois sentidos para capturar a evolução da tempestade e do encharcamento do solo.
- Uma camada de Multi-Head Self-Attention pondera quais dias da janela de 15 dias mais contribuem para o risco no dia atual.
- A saída gera o logit base puramente climático (base_logits) e a estimativa de severidade (severity_head).
FiLM:
- Feature-wise Linear Modulation
- É um mecanismo que usa uma variável estática, nesse caso a altitude da estação, para gerar dois vetores que ajustam a Bi-GRU
Monotonic:
- Ramo Monotônico Estrutural
- Ao invés de usar penalidades suaves na função de perda (lambda-pinn), o uso do solo entra em um ramo separado, com pesos forçados a serem estritamente positivos, fazendo com que aumentar a impermeabilização do solo nunca diminua o risco de crises e diminuir a impermeabilização nunca aumente o risco.
Treino: 2010–2023 | Validação/Calibração: 2024 | Teste Factual e Contrafactual: 2025
Nesse primeiro teste, apenas no RS, o HydroFiLMMonotonicNet superou o TabFM em todas as métricas factuais:
ROC-AUC: 0.7585 vs 0.6218
PR-AUC: 0.0508 vs. 0.0170
Brier: 0.00766 vs. 0.00781

- 2ª fase:
Para a segunda fase o teste teve abrangência nacional, os modelos usados foram o HydroTFTNet e o TabFM, o HydroTFTNet foi usado por ter uma Variable Selection Network e por ser temporal, a VSN calcula os pesos para cada váriavel e tem uma célula separada para o histórico, gerada a partir do contexto geográfico.
Otimizações:
- O TabFM precisou ter o número de amostras limitado para 8000 por sua arquitetura quadrática, que se aplicada a todos os dados estouraria a VRAM instantaneamente.
- Dataset 2D indexado: o Dataset 3D precisou ser fatiado em Datasets 2D fatiados que eram buscados sobre demanda para diminuir o consumo da RAM de aproximadamente 1,7GB de RAM para 110MB.
- Diferenciação de chuvas em temporadas chuvosas e chuvas em épocas de secas.
- Os modelos tiveram um Early Stopping devido ao OVerfitting, causado pela quantidade de ruído.
O HydroTFTNet superou o TabFM:
- ROC-AUC: 0.8834 vs. 0.8762
- PR-AUC: 0.0328 vs. 0.0255
A conclusão dos resultados foi que o HydroFiLMMonotonicNet é melhor para uma quantidade menor de dados devido ao Bi-GRU, já o HydroTFTNet foi melhor no âmbito nacional, já que lida melhor com uma grande carga de dados.
- 3ª fase:
Para a terceira fase o desafio foi cruzar os dados com o número de leitos por especialidade, número de habitantes, área do território com os dados obtidos previamente, comparando o TabFM com o HydroFiLMMonotonicNet o RS e com o HydroTFTNet no Brasil. As principais mudanças foram
Normalização demográfica e territorial, para eliminar distorções entre cidades pequenas e polos regionais.
O HydroFiLMMonotonicNet venceu o TabFM no RS em 100% das métricas:
ROC-AUC: 0.7585 vs 0.6218
PR-AUC: 0.0508 vs 0.0170
Já o HydroTFTNet conseguiu prever as enchentes de 2024 com mais facilidade que o TabFM - porém em épocas sem enchente teve dificuldades pois o dataset de enchentes não registra chuvas fortes, apenas eventos climáticos com impactos grandes, então em muitos meses o nível de chuvas fortes foi 0, mesmo que não seja a exata realidade - e foi usado para prever número de leitos que seriam necessários, após o balanceamento demográfico e territorial o modelo conseguiu um resultado próximo à realidade.
=== RS 2024 MÊS A MÊS: ENCHENTE E LEITOS (PREVISTO vs. ACONTECIDO) ===
 mes  Prob_Enchente_Prev  Nivel_Alag_Prev  Nivel_Alag_Real  Leitos_Clin_Prev  Leitos_Clin_Real  Leitos_UTI_Prev  Leitos_UTI_Real  Leitos_Clin_1000hab
   1               0.006            1.845            0.489            4990.0              4997         1955.000             1955                1.108
   2               0.006            1.074            0.000            5172.0              5216         2036.000             2145                1.119
   3               0.008            2.251            0.425            5272.0              5229         2203.308             2193                1.108
   4               0.012            4.806            4.157            5229.0              5407         2193.000             2204                1.114
   5               0.021            6.578            5.470            5493.0              5550         2224.000             2387                1.159
   6               0.010            3.883            0.060            5550.0              5649         2388.528             2388                1.173
   7               0.005            0.156            0.000            5649.0              5624         2388.000             2277                1.166
   8               0.010            3.154            0.000            5624.0              5666         2277.000             2282                1.165
   9               0.013            3.739            0.691            5584.0              5593         2254.000             2268                1.189
  10               0.021            4.435            0.487            5593.0              5573         2268.000             2278                1.183
  11               0.013            2.322            0.000            5573.0              5624         2278.000             2241                1.186
  12               0.012            3.064            0.000            5624.0              5598         2241.000             2229                1.156
<img width="1289" height="440" alt="image" src="https://github.com/user-attachments/assets/f04dc47f-636d-49a3-8bd4-55a2b1c5cf1c" />
