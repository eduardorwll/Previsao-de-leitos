# Feature F19 — Nível Local no Estágio 1 (`Task 1.3.4`)

> **Status:** ✅ Concluída | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

---

## 1. Diagnóstico que motivou a mudança

O backtest (F17) e o teste do alvo de necessidade (F18) mostraram que o `F04.1` não se
recuperava com o alvo correto: seguia em `MASE 1,09–1,91`. A causa apontada foi o
**Estágio 1 sem nível**: o Ridge recebia apenas harmônicos sazonais (`target_week_sin/cos`),
então previa a *taxa média* da região e ignorava mudanças de patamar.

Sintoma que confirmava: `ridge_pooled`, que carrega `y_roll12`, era o melhor em UTI sob o
alvo de necessidade — um modelo linear simples vencia a arquitetura de dois estágios só por
ter âncora de nível.

---

## 2. Implementação

`OrthogonalTwoStageOccupancyForecaster` ganhou `use_local_level` e `level_source`:

```python
OrthogonalTwoStageOccupancyForecaster(..., use_local_level=True, level_source="y_roll12")
```

- O Estágio 1 recebe uma coluna adicional `y_roll12 / capacidade_sem` — **média móvel de
  12 semanas em taxa relativa**, não a última observação.
- A escolha é deliberada: `y_origin` (persistência) causaria atraso de fase, que era o
  motivo original do desenho ortogonal; a média suavizada fornece nível sem copiar a semana.
- Nada muda no Estágio 2 (GBDT monotônico sobre o resíduo) nem na escala heterocedástica.
- Os calibradores conformais (`f041_mondrian`, `f041_agaci`, `f041_spci`) passaram a usar o
  núcleo com nível local, e um candidato explícito (`f041_local_level`) entrou no catálogo.

---

## 3. Resultado (`bt_final_occ` e `bt_final_need`, 6 folds)

| Alvo | `f041_core` | `f041_local_level` | Δ | GBDT (referência) |
| :--- | ---: | ---: | ---: | ---: |
| UTI adulto | 1,699 | **1,135** | **−33%** | 0,943 |
| UTI total | 1,743 | **1,156** | **−34%** | 0,943 |
| Clínico | 1,591 | **1,103** | **−31%** | 0,924 |
| Respiratório | 1,088 | **0,940** | **−14%** | 0,866 |
| Necessidade — UTI adulto | 1,743 | **1,172** | **−33%** | 0,959 |
| Necessidade — UTI total | 1,911 | **1,133** | **−41%** | — |
| Necessidade — clínico | 1,563 | **1,140** | **−27%** | 0,884 |
| Necessidade — respiratório | 1,168 | **0,957** | **−18%** | 0,840 |

Cobertura também melhorou (`f041_mondrian`: `0,79–0,81` contra `0,61–0,65` do núcleo).

**Leitura honesta:** a hipótese estava certa — o nível ausente era o defeito dominante —, mas
o `F04.1` **ainda não alcança o GBDT** (`0,94–1,17` contra `0,84–0,96`). A mudança foi grande
o suficiente para justificar mantê-la como base dos conformais, não para promover o modelo.

---

## 4. O que ainda falta

1. **Fase:** o erro de pico segue em `8,4–9,2` semanas (GBDT `4,3–5,5`; persistência `2,0`).
   É o próximo alvo — lag distribuído do choque (`0..8` semanas) e "semanas desde o pico
   hidrológico".
2. **Nível mais rico:** média de 12 semanas é um nível constante; um nível local com
   tendência (state space) pode capturar deriva sem copiar observação.
3. Multi-target e ensemble continuam depois.

---

## 5. Arquivos

- `src/previsao_leitos/demand/two_stage.py` (`use_local_level`, `level_source`)
- `src/previsao_leitos/demand/registry.py` (`f041_local_level`; conformais no núcleo novo)
- `tests/unit/test_demand_registry_and_benchmark.py` (nível melhora MAE e preserva monotonicidade)
- `reports/runs/bt_final_occ/`, `reports/runs/bt_final_need/`
