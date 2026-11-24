# Glossário — Contexto: Hidrologia e Risco de Enchentes

### Cota Fluviométrica Observada
Nível físico da superfície da água de um rio medido em estação telemétrica ou convencional (em centímetros ou metros), comparado às cotas oficiais de atenção, alerta e inundação da estação.

### Extensão Inundada Observada (SAR)
Área territorial fisicamente coberta por água durante um evento de cheia, delimitada por radar de abertura sintética (Sentinel-1) capaz de atravessar cobertura de nuvens.

### Índice Proxy de Severidade Hidrológica
Indicador derivado de variáveis meteorológicas e de uso do solo que estima a gravidade potencial de alagamento na ausência de medição fluviométrica direta.
_Avoid:_ Nível de Alagamento Real, Nível Observado (quando derivado de chuva e decreto).

### Índice de Precipitação Antecedente (API)
Medida contínua do encharcamento prévio do solo acumulada por decaimento exponencial da precipitação diária em dias calendário consecutivos.

### Ottobacia
Unidade espacial hierárquica e topologicamente orientada da Base Hidrográfica Ottocodificada da ANA que conecta trechos de drenagem de montante para jusante.

### HAND (*Height Above Nearest Drainage*)
Diferença altimétrica entre um ponto do terreno e o curso d'água mais próximo para o qual ele drena, indicando suscetibilidade física à inundação.

### Decreto Administrativo de Desastre (S2iD / SEDEC)
Reconhecimento oficial de Situação de Emergência ou Estado de Calamidade Pública registrado pela Defesa Civil. Representa um ato administrativo com possível atraso e subnotificação, não uma medição física instantânea da enchente.
_Avoid:_ Ocorrência física de enchente (como sinônimo exato).

### Danos Humanos (Desabrigados vs. Desalojados)
Contagem oficial de pessoas afetadas registrada em Formulário de Informações do Desastre (FIDE): **desabrigados** necessitam de abrigo público; **desalojados** deixaram suas residências mas permanecem em casas de familiares ou amigos.

### Área Aterrada / Planície Artificialmente Ocupada (Pôlder)
Porção urbana construída sobre antigas várzeas, banhados ou margens fluviais/lagunares com baixo `HAND` (frequentemente protegida por diques e estações de bombeamento — EBAPs), caracterizada por falha em degrau (*threshold failure*) quando a cota supera a proteção ou falha o bombeamento, seguida de alagamento estagnado prolongado (*ponding*).

### Efeito de Remanso (*Backwater*)
Represamento da descarga fluvial em zonas estuarinas ou lagunares de baixíssima declividade (ex.: Lago Guaíba ↔ Lagoa dos Patos ↔ Canal São Gonçalo) causado por nível elevado a jusante e/ou empilhamento hídrico induzido por ventos do quadrante Sul/Sudeste (`V10M > 0`).

### Tempo de Trânsito Hidrológico ($\tau_{u \to v}$)
Intervalo físico (em dias ou semanas) decorrido entre o pico de precipitação/escoamento em um município ou sub-bacia a montante ($u$) e a chegada da onda de cheia a um município ou sub-bacia a jusante ($v$).

### Pressão Hidrológica de Área de Captação (*Catchment Flood Pressure*)
Exposição hidrológica e de áreas aterradas ponderada pela matriz empírica histórica de referência hospitalar ($W_{\text{ref}}$, `MUNIC_RES → CNES`, congelada em `<= ano - 1`), capturando o transbordamento de demanda assistencial quando cidades vizinhas alagam.
