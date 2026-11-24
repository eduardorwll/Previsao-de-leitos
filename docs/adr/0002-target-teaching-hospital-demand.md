# ADR-0002: Pivô da Variável-Alvo — De Oferta CNES Municipal para Demanda/LOS em Hospitais-Escola

- **Status:** ACCEPTED
- **Data:** 2026-11-24

## Contexto e Decisão

O protótipo legado previa a variação mensal de oferta cadastrada no CNES agregada por município (`HistGradientBoostingRegressor` com perda L1), o que colapsou para persistência ($\hat{\Delta}_t = 0$ em 7/11 meses no RS) e não mede quantos leitos são **necessários** em **hospitais-escola** durante uma enchente.

**Decisão:** Separar formalmente **Oferta Cadastrada/Funcional (CNES no grão `id_estabelecimento_cnes` + `leito` + `estabelecimento_ensino`, distinguindo SUS vs. não-SUS e UTI vs. UCI)** de **Demanda Real de Leitos**, modelada por: (1) volume de internações de urgência/emergência e agravos pós-enchente (`SIH-SUS` agregado por hospital-escola e grupo CID-10, complementado por `SINAN`/painéis GERINT quando disponíveis), e (2) **Tempo de Permanência (LOS)** sob riscos competitivos (`scikit-survival`), acoplados a uma **Macrorregião de Referência (*catchment*)** em vez de municípios isolados.

## Alternativas Rejeitadas

- **Continuar prevendo apenas delta mensal CNES municipal:** aprende restrições orçamentárias e atraso de recadastramento, não necessidade clínica; cego ao papel de retaguarda dos hospitais-escola (HCPA, HUSM, HE-UFPel, HU-FURG, Santa Casa/UFCSPA, HSL-PUCRS).
- **Usar microdados nominais de prontuário:** desnecessário para planejamento de capacidade e introduz risco elevado de LGPD.

## Consequências

Exige ingestão de `br_ms_cnes.leito`, `br_ms_cnes.estabelecimento_ensino` e `br_ms_sih` (agregado sem PII na query SQL), além de simulação de filas (`simpy`) para converter curvas de chegada + LOS em necessidade operacional de leitos de contingência.
