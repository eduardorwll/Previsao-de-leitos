# Pré-registro — Série de Necessidade de Leitos v1

> **Status:** congelado antes da avaliação em `2024` | **Versão:** `v1` | **Data:** 26/11/2026
> **Escopo:** RS, `2010–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

Este documento congela a definição da variável-alvo **necessidade** antes de qualquer
resultado ser calculado sobre `2024`. Alterações exigem nova versão (`v2`) e nova execução
completa — nunca um ajuste silencioso após ver o resultado.

---

## 1. Motivo

A ocupação realizada (SIH) é **constrangida pela oferta**: em enchente há suspensão de
cirurgias eletivas, transferências e dificuldade de acesso. Medimos que ela **cai** quando o
choque sobe (correlação `−0,10` em UTI adulto), enquanto a necessidade de cuidado **sobe**.
Avaliar modelos contra ocupação realizada confunde necessidade com capacidade de atendimento.

O `ADR-0001` (monotonicidade) continua válido — para **necessidade**, não para ocupação.

---

## 2. Definição (congelada)

Para cada região `r`, semana `t` e alvo `c`:

```
necessidade_{r,t,c} = baseline_{r,t,c} + w_{r,t} · max(0, a_{r,t,c} − b_{r,t,c}) · LOS_{r,t,c}
```

| Termo | Definição operacional | Fonte |
| :--- | :--- | :--- |
| `baseline` | mediana da ocupação na **mesma semana ISO** nos 3 anos anteriores (`t−1, t−2, t−3`) | SIH |
| `a` | internações da semana `t` na causa correspondente | SIH |
| `b` | mediana das internações na mesma semana ISO nos 3 anos anteriores | SIH |
| `LOS` | média móvel de `y/a` nas 4 semanas **anteriores** (limitada a `[1, 60]` dias) | SIH |
| `w` | peso do evento `= prob_enchente_f03_oof ∈ [0,1]` | F03.1 (OOF) |

**Unidade:** leitos-dia por semana (mesma unidade de `uti_adulto_diarias`).

### 2.1 Mapeamento por alvo

| Alvo | `y` (ocupação) | `a` (internações) | Observação |
| :--- | :--- | :--- | :--- |
| `uti_adulto` | `uti_adulto_diarias` | `uti_adulto_internacoes` | direto |
| `uti_total` | `diarias_uti_total_ativas` | `uti_adulto_internacoes` | **proxy** (A5) |
| `clinico` | `clinico_pacientes_dia` | `clinico_internacoes` | direto |
| `respiratorio` | `respiratorio_pacientes_dia` | `respiratorio_internacoes` | direto |

---

## 3. Premissas declaradas

- **A1.** LOS é estável dentro da causa na janela de 4 semanas.
- **A2.** O baseline sazonal de 3 anos captura tendência e sazonalidade sem o evento.
- **A3.** O excesso é não negativo: semanas abaixo do baseline **não** geram necessidade negativa.
- **A4.** A atribuição do excesso ao evento é proporcional ao choque (`w`), não binária.
- **A5.** Para `uti_total`, as internações de UTI adulto são proxy das internações totais de UTI.
- **A6.** Toda a informação usada em `baseline`, `b` e `LOS` é **estritamente passada**;
  apenas `a` e `w` são contemporâneos ao rótulo.

**Consequência de A3+A4:** a série é monotônica não decrescente em `w` por construção, o que
torna o `ADR-0001` verdadeiro por desenho — e não uma restrição imposta ao modelo.

---

## 4. Guardas anti-vazamento

1. `baseline`, `b` e `LOS` usam apenas `t−1 … t−4` (nunca `t`).
2. Nenhum ano `>= 2025` é lido; a série termina em `2024`.
3. O código falha se encontrar `ano_epi >= 2025` no painel.
4. O hash do painel de origem e do código é gravado no manifesto da série.

---

## 5. Análises de sensibilidade (pré-declaradas)

| Variação | Alternativa | Esperado |
| :--- | :--- | :--- |
| Peso do evento | `w = 1` (atribuição total) | necessidade maior em anos de enchente |
| Baseline | média em vez de mediana | mais sensível a outliers |
| Janela de LOS | 8 semanas | suaviza mudanças de prática |
| Proxy de UTI total | excluir `uti_total` do leaderboard | perda de um alvo |

O ranking de modelos precisa ser **estável** sob pelo menos 3 das 4 variações; caso contrário,
a série é considerada frágil e volta para revisão (`v2`).

---

## 6. Critérios de aceitação da série

1. `necessidade >= 0` em 100% das linhas.
2. `necessidade >= baseline` em 100% das linhas (por A3).
3. Em `2024` semanas `18–30`, excesso agregado `> 0` em pelo menos 2 dos 4 alvos.
4. Correlação entre `necessidade` e `w` **positiva** por região (mediana `> 0`).
5. Nenhuma linha com `ano_epi >= 2025`.

Falha em qualquer critério ⇒ a série não é publicada e a definição é revisitada.

---

## 7. O que este pré-registro **não** afirma

- Não afirma que a série é a necessidade "verdadeira": é uma **proxy auditável**, com
  premissas explícitas e sensibilidade declarada.
- Não substitui a avaliação no holdout (`2025`/`2026`), que segue no `Marco M2`.
- Não altera o `ADR-0001`; apenas define o alvo ao qual ele se aplica.
