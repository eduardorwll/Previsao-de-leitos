# Pré-registro — Série de Necessidade de Leitos v3

> **Substitui:** `v2` (que substituiu `v1`). Histórico preservado.
> **Motivo:** a v2 corrigiu o nível, mas em `W18–30` de 2024 a necessidade ainda ficava
> `6–7%` **abaixo** da ocupação realizada em UTI e clínico. Isso é logicamente impossível:
> todo leito ocupado foi necessário. Falta o **piso de realização**.

---

## 1. Fórmula (congelada)

```
necessidade = max( ocupação_realizada , baseline + w · max(0, a − b) · LOS )
```

- `baseline`, `w`, `a`, `b`, `LOS`: idênticos à v2 (baseline sazonal × fator de nível
  estadual; `LOS` e `baseline` estritamente passados).
- **A7 (novo):** a ocupação realizada é **lower bound** da necessidade. Se não conseguimos
  estimar necessidade acima do que foi efetivamente ocupado, o observado é o piso.

### 1.1 Propriedades preservadas

| Propriedade | Como |
| :--- | :--- |
| `necessidade >= 0` | ambos os termos são não negativos |
| `necessidade >= baseline` | `baseline + excesso >= baseline` e `max` preserva |
| **monotônica em `w`** | `max` de constante com função crescente é crescente |
| `necessidade >= ocupação` | por construção (A7) |

---

## 2. Por que isso não é "copiar a ocupação"

Só há ganho informacional quando `baseline + excesso > ocupação` — aí a série expressa
necessidade **acima** do realizado (o caso respiratório, onde a lacuna é real). Nos demais
casos, a série assume o piso observado: não inventamos necessidade não fundamentada.

A diferença relevante para modelagem permanece: **durante a enchente a ocupação pode cair
por suspensão de eletivas, mas o piso não cai** — porque a ocupação de 2024 foi alta. A série
deixa de punir modelos que preveem alta quando a oferta caiu.

---

## 3. Critérios (v3)

C1–C5 da v1, C6 da v2 e o novo **C7**: `necessidade >= ocupação` em 100% das linhas.

A janela de face continua `2024`, semanas `5–17`, com razão mediana em `[0,90 ; 1,10]`.

---

## 4. Status

Esta é a versão **congelada para modelagem**. Qualquer mudança posterior exige `v4` e nova
execução completa do backtest.
