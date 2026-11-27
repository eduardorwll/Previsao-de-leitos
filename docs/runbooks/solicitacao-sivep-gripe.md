# Solicitação de Acesso — SIVEP-Gripe (perfil de leitor estadual)

> **Fronteira:** *Not for patient care or live clinical use* | **LGPD:** zero PII; uso agregado e anonimizado.

## Texto da justificativa (para colar no formulário)

Solicito o perfil de **leitor estadual** do SIVEP-Gripe para fins **exclusivamente acadêmicos**.
Nossa pesquisa, da **Universidade Federal de Pelotas (UFPEL)** sob orientação do **Prof. Marilton**,
investiga a **previsão da necessidade de leitos em hospitais-escola do Rio Grande do Sul durante
crises climáticas (enchentes)**. O trabalho encontra-se em **fase inicial**: ainda estamos
negociando com os hospitais-escola do RS o uso de seus dados para **treinar modelos preditivos**
de demanda e ocupação de leitos em períodos de crise como o atual.

O acesso agregado às internações por SRAG (SIVEP-Gripe) é necessário como **indicador antecipado**
da pressão respiratória sobre leitos de UTI e enfermaria, para **calibrar e validar** os modelos.
Trata-se de uso **agregado e anonimizado** (sem dados individuais de pacientes) e **sem finalidade
assistencial**.

## Dados do solicitante (a confirmar no cadastro)

- **Instituição:** Universidade Federal de Pelotas (UFPEL)
- **Orientador:** Prof. Marilton
- **Finalidade:** pesquisa acadêmica — modelos preditivos de necessidade de leitos em hospitais-escola sob risco de enchentes (RS → Brasil)

## Vínculo com o projeto

Esta solicitação corresponde à base **SIVEP-Gripe (SRAG viral)** do catálogo de dados
([`docs/data-catalog.md`](../data-catalog.md), item 5) e ao ponto pendente registrado em
[`docs/runbooks/acesso-bases-pendentes.md`](acesso-bases-pendentes.md). As colunas agregadas
desejadas são `srag_total`, `srag_uti` e `srag_viral_confirmado` por `(região de saúde, semana
epidemiológica)`, já previstas pelo agregador LGPD
`ingestion/epidemiology_sources.py::aggregate_sivep_gripe_weekly`.
