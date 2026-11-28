# Acesso às Bases de Dados — Tentativa e Status (SCPA)

> **Status:** 📋 Acesso solicitado; bases parcialmente bloqueadas | **Atualizado:** 2026-11-27
> **Fronteira:** *Not for patient care or live clinical use* | **LGPD:** zero PII; agregação na fronteira; supressão $n < 5$ (`ADR-0005`)

## 1. Contexto

O projeto depende de um subconjunto de bases assistenciais e de desastres para fechar o ciclo
científico (demanda real, Onda 2 pós-enchente, danos e deslocamento populacional). Foi feita a
**tentativa de acesso às bases via SCPA**, com pedido formal de permissão para os grupos de
informações listados abaixo. Enquanto o acesso não é concedido, o pipeline permanece operante com
as fontes já disponíveis (`SIH-SUS`, `CNES`, `MERGE-CPTEC`, `NASA POWER`, `ERA5-Land`, `MapBiomas`,
`IBGE`, `GERINT-POA`, `NOAA CPC`, `BOM`, `Zenodo IPH-UFRGS`).

## 2. Bases solicitadas e estado atual

| # | Base / Grupo | Finalidade no projeto | Estado atual | Alavanca desbloqueada |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **SIVEP-Gripe (SRAG viral)** | Indicador líder da onda respiratória epidêmica (influenza/COVID/VSR) que *lidera* o pico de demanda | ⛔ `403/404` sem API key da Base dos Dados | **Reduz o erro de fase (4,3–7,4 sem.)** — colunas `srag_viral_confirmado`/`frac_srag_viral` já previstas no `two_stage.py` |
| 2 | **SINAN (Leptospirose)** | Onda 2 pós-enchente (leptospirose grave → UTI, D+7 a D+30) | ⚠️ dengue já extraída; leptospirose aguarda | Modelagem da Onda 2 |
| 3 | **S2iD / FIDE (MIDR)** | Decretos + danos humanos (`desabrigados`, `enfermos`, danos a hospitais) | ⛔ download CSV via Playwright não dispara | Deslocamento populacional + dano à rede |
| 4 | **ANA HidroWeb API v2** | Cotas/vazões (sinal físico real de enchente) | ✅ **acesso concedido**; credenciais mantidas — se o `401` persistir, seguir com fallback Zenodo | Hidrologia observada (fallback ativo) |

## 3. Justificativa de acesso (uso pretendido)

Todo acesso solicitado se destina **exclusivamente a pesquisa aberta e avaliação metodológica**, sem
uso assistencial ou operacional ao vivo. O tratamento é estritamente agregado na origem:

- **Zero PII:** nenhum dado individualizado de paciente (CPF, CNS, nome, nascimento, prontuário)
  é lido ou persistido; apenas contagens agregadas por `(região de saúde, semana epidemiológica)`
  ou `(município, semana)`.
- **Supressão de pequenas células:** células com $n < 5$ são suprimidas automaticamente (`ADR-0005`).
- **Fronteira clínica:** *"Not for patient care or live clinical use"* — os modelos são de pesquisa
  e planejamento agregado, nunca triagem individual nem alerta operacional ao vivo.
- **Finalidade científica:** prever, com incerteza calibrada, a necessidade de leitos por
  especialidade (UTI adulto, UTI pediátrica/neonatal, clínico, cirúrgico) em hospitais-escola do
  Rio Grande do Sul frente a enchentes, com foco em horizonte $h = 2$ semanas (D+14).

## 4. Próximos passos

1. **SIVEP-Gripe:** solicitação enviada — **aguardando resposta** (perfil de leitor estadual,
   justificativa em `docs/runbooks/solicitacao-sivep-gripe.md`).
2. **SINAN:** incluir leptospirose na mesma via de acesso (Base dos Dados / OpenDataSUS).
3. **S2iD:** retomar inspecionando o *datepicker* do formulário (campo de data inicial ausente —
   ver `docs/runbooks/ingest-s2id-sinan.md`).
4. **ANA HidroWeb (email de 2026-11-27):** acesso **concedido**, mas **credenciais permanecem as
   mesmas** (`.env` inalterado). Se o `HTTP 401` persistir, **seguir com o fallback Zenodo** sem
   bloquear o trabalho — a instabilidade de banco reportada pela ANA provavelmente explica o erro.
   Reteste adiado até o banco estabilizar; sem alterar credenciais.

## 5. Rastreabilidade

- Pontos pendentes espelhados em `src/previsao_leitos/audit/pending_items.py` (fonte única usada
  pelos relatórios PDF).
- Bloqueios operacionais também registrados em `tasks/todo.md` (arquivo local de handoff).
- Justificativa SIVEP-Gripe: `docs/runbooks/solicitacao-sivep-gripe.md`.
