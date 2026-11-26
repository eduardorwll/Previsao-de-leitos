# Pré-registro — Série de Necessidade de Leitos v2

> **Substitui:** `serie-necessidade-v1.md` (mantido como histórico)
> **Motivo da revisão:** a v1 falhou na validação de face — em semanas **sem** enchente a
> necessidade ficou `7–12%` abaixo da ocupação observada, porque a mediana sazonal de 3 anos
> está defasada do nível de 2024.
> **Congelado antes** de qualquer avaliação de modelo sobre a série v2.

---

## 1. O que muda

A fórmula permanece:

```
necessidade = baseline + w · max(0, a − b) · LOS
```

O **baseline** passa a ser corrigido por nível:

```
baseline_v2 = baseline_sazonal × fator_nível
fator_nível = média(y_{t−1..t−4}) / média(baseline_sazonal_{t−1..t−4})
fator_nível ∈ [0,5 ; 2,0]
```

- `baseline_sazonal`: mediana da mesma semana ISO nos 3 anos anteriores (como na v1).
- `fator_nível`: razão entre o nível recente observado e o nível sazonal esperado na mesma
  janela — captura mudanças de patamar (pandemia, abertura de leitos, envelhecimento).
- Toda a informação usada é **estritamente passada** (`t−1..t−4`).

O excesso (`w · max(0, a − b) · LOS`) **não muda**, preservando a monotonicidade em `w`.

---

## 2. Critério de face adicionado

A v1 não detectava defasagem de nível. A v2 acrescenta:

**C6 — nível pré-evento:** `mediana(necessidade / ocupação)` em `2024` semanas `1–17`
deve estar em `[0,90 ; 1,10]`.

Interpretação: antes de qualquer evento, necessidade e ocupação realizada devem praticamente
coincidir. Durante o evento, necessidade pode exceder ocupação (é a lacuna que o projeto quer
medir).

Os critérios C1–C5 da v1 permanecem.

---

## 3. Por que não corrigir a v1

Ajustar a fórmula depois de ver o resultado de 2024 seria exatamente o que o pré-registro
existe para impedir. A v1 fica registrada como tentativa auditada; a v2 é uma hipótese nova,
com o critério que faltava declarado **antes** de rodar.

---

## 4. Sensibilidade (inalterada)

`w = 1` vs. `w = prob`; mediana vs. média; LOS 4 vs. 8 semanas. O ranking de modelos precisa
ser estável em pelo menos 3 das 4 variações.
