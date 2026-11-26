# 1. Metodologia

## 1.1 Por que origem móvel

Julgar modelos por um único ano confunde sorte com mérito. Enchente é evento raro: um ano
favorável pode eleger o modelo errado. A régua usa **folds expansivos** e decide pela
**mediana**, não pela média (que é sensível a um único fold extremo).

## 1.2 Folds

Para cada ano `y` da lista:

| Conjunto | Filtro | Uso |
| :--- | :--- | :--- |
| Treino | `target_ano_epi <= y-2` | ajuste dos modelos |
| Calibração | `target_ano_epi == y-1` | conformal (CQR, Mondrian, AgACI, SPCI) |
| Avaliação | `target_ano_epi == y` | métricas |

Folds padrão: `2019, 2020, 2021, 2022, 2023, 2024`. Cada fold é independente: o modelo é
reajustado do zero, nunca reaproveita o fold anterior.

## 1.3 Alvo e horizonte

- Horizonte padrão `h = 2` semanas (a previsão de `t` mira `t+2`).
- Alvos de **ocupação**: `uti_adulto`, `uti_total`, `clinico`, `respiratorio`.
- Alvos de **necessidade** (`need_*`): ver [pré-registro v3](../preregistro/serie-necessidade-v3.md).
- Unidades: UTI em **leitos-dia por semana**; clínico/respiratório em **pacientes-dia**.
  Comparações de MASE só valem **dentro** do alvo.

## 1.4 Guardas de vazamento

1. Nenhum ano `>= 2025` é lido (holdout do `Marco M2`).
2. Features usam apenas informação `<= t_0`; `y_seasonal_naive` usa `t-52`.
3. A série de necessidade usa `baseline`, `LOS` e `b` estritamente passados.
4. O painel de modelagem é gravado sem `2025` (ausência física, não filtro de runtime).

## 1.5 Limitações declaradas

- `2020` e `2021` são atípicos (COVID) e entram na régua: eles dominam o pior fold. Excluí-los
  esconderia o regime que se quer prever.
- `2024` é ano de **desenvolvimento/auditoria**, não *out-of-time*.
- 30 regiões de saúde, 52 semanas/ano: amostra pequena para eventos extremos.
- O pico semanal é medido por `argmax`; em topos achatados isso é ruidoso (ver 02-metricas).

