# Feature F18 — Série de Necessidade de Leitos (Pré-registro v3) (`Task 1.3.3`)

> **Status:** ✅ Congelada para modelagem | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, `2017–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

---

## 1. Por que trocar o alvo

Medimos que a **ocupação realizada cai** quando o choque sobe (correlação `−0,10` em UTI
adulto), porque a enchente suspende cirurgias eletivas e dificulta o acesso. Avaliar modelos
contra ocupação confunde **necessidade** com **capacidade de atendimento** — e foi por isso
que o `F04.1`, movido a choque com sinal forçado positivo, perdeu feio (`MASE 1,77`).

A série de necessidade é a variável-alvo correta para o `ADR-0001`: nela, a monotonicidade é
verdadeira **por construção**, não uma restrição imposta ao modelo.

---

## 2. Fórmula congelada

```
necessidade = max( ocupação , baseline + w · max(0, a − b) · LOS )
```

| Termo | Definição |
| :--- | :--- |
| `baseline` | mediana da mesma semana ISO nos 3 anos anteriores × fator de nível estadual (mediana robusta de 12 semanas, causal) |
| `a`, `b` | internações da causa na semana e sua mediana sazonal |
| `LOS` | média móvel de `y/a` nas 4 semanas **anteriores**, em `[1, 60]` dias |
| `w` | `prob_enchente_f03_oof` (F03.1) |

**Axiomas:** baseline/LOS estritamente passados (A6); excesso não negativo (A3); atribuição
proporcional ao choque (A4); **ocupação é piso da necessidade (A7)**.

---

## 3. Três iterações — e o que cada uma corrigiu

| Versão | O que fazia | Falha detectada | Correção |
| :--- | :--- | :--- | :--- |
| **v1** | baseline = mediana sazonal de 3 anos | necessidade `7–12%` **abaixo** da ocupação em semanas sem enchente (baseline defasado do nível de 2024) | fator de nível |
| **v2** | baseline × fator de nível (4 semanas) | fator contaminado pelo próprio evento (dupla contagem) e ruidoso em alvos pequenos (respiratório) | fator estadual, janela de 12 semanas, mediana robusta |
| **v2.1** | fator estadual | necessidade ainda `6–7%` abaixo da ocupação **durante** a enchente em UTI/clínico | **A7: ocupação é piso** |
| **v3** | `max(ocupação, baseline + excesso)` | — | **congelada** |

O critério de aceitação C4 ("correlação entre necessidade e choque") também foi corrigido:
correlação temporal é confundida pela sazonalidade e **não** mede monotonicidade. Foi
substituído por um teste **estrutural**: recalcular com `w + 0,2` e verificar que nenhuma
linha cai.

---

## 4. Resultados (2024)

| Alvo | `necessidade/ocupação` W18–30 | Semanas com lacuna | Pico | Nível pré-evento |
| :--- | ---: | :---: | :---: | ---: |
| UTI adulto | **1,038** | 13/13 | W30 | 1,027 |
| UTI total | 1,035 | 12/13 | W30 | 1,005 |
| Clínico | **1,026** | 12/13 | W30 | 1,000 |
| Respiratório | **1,324** | 13/13 | W30 | 1,000 |

O respiratório é onde a lacuna é real (`+32%`): é a causa sensível à enchente. UTI e clínico
têm lacuna pequena — a série assume o piso observado em vez de inventar necessidade.

**Critérios:** 7/7 satisfeitos nos quatro alvos (`no_holdout`, `non_negative`,
`above_baseline`, `at_least_occupancy`, `event_excess_2024 > 0`, `monotone_in_shock`,
`pre_event_level_ratio ∈ [0,90; 1,10]`).

---

## 5. Artefatos

| Arquivo | Conteúdo |
| :--- | :--- |
| `data/processed/rs_need_<alvo>_v3.parquet` | série por região × semana (12 colunas, `20.280` linhas) |
| `data/processed/rs_need_<alvo>_v3.manifest.json` | SHA-256 do painel e da série, critérios, pré-registro |
| `docs/preregistro/serie-necessidade-v{1,2,3}.md` | três versões, com o motivo de cada revisão |

---

## 6. Limitações

- É uma **proxy auditável**, não a necessidade "verdadeira": as premissas A1–A7 estão
  declaradas e a sensibilidade (`w=1`, média vs. mediana, LOS 4 vs. 8) precisa ser executada
  no backtest antes de qualquer promoção.
- `uti_total` usa internações de UTI adulto como proxy (A5).
- A série não substitui o holdout: a validação final segue no `Marco M2`.

---

## 7. Próximo passo

Rodar o **backtest (S1) sobre a série de necessidade** e comparar o ranking com o da ocupação.
Se o `F04.1` recuperar posições, isso confirma a hipótese do alvo errado; se não, o gargalo é
o lag do choque (S3).
