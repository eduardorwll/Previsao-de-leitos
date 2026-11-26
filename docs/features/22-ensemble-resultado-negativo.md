# Feature F22 — Ensemble com Pesos Calibrados (resultado negativo) (`Task 1.3.9a`)

> **Status:** ⚠️ Implementado e **rejeitado** pela régua | **Etapa:** Etapa 3 (`Ciclo V1.3 RS`)
> **Escopo:** `RS`, folds `2019–2024` | **Holdout:** `2025`/`2026` intocados
> **Fronteira:** *Not for patient care or live clinical use.*

## 1. Hipótese

Combinar GBDT + Ridge + F04.1 (nível local) com pesos estimados no **ano de calibração**
(`y−1`, nunca no ano avaliado) reduziria o erro por diversificação — ganho típico de 2–5%
em ensembles.

## 2. Implementação

`demand/ensemble.py`: `EnsembleForecaster` com `fit_mode="train_cal"`.

- pesos ∝ `1/MAE_cal^power`, com **piso** (`min_weight`) redistribuído por projeção — nenhum
  membro é zerado nem domina;
- média ponderada dos quatro quantis, com reordenação para evitar cruzamento;
- `ensemble_top3` no catálogo: `gbdt_v1` + `ridge_pooled` + `f041_local_level`.

## 3. Resultado (`bt_ensemble_occ`, `bt_ensemble_need`)

| Alvo | `gbdt_v1` | `ensemble_top3` | Δ |
| :--- | ---: | ---: | ---: |
| UTI adulto | **0,943** | 0,950 | +0,7% (pior) |
| Clínico | **0,924** | 0,956 | +3,5% (pior) |
| Respiratório | 0,866 | **0,862** | −0,5% (melhor) |
| Necessidade — UTI adulto | **0,959** | 0,966 | +0,7% (pior) |
| Necessidade — clínico | **0,884** | 0,935 | +5,8% (pior) |

Cobertura melhora em clínico (`0,729 → 0,778`), mas o selo continua limitado pela fase.

## 4. Por que falhou

Os membros são **correlacionados e hierarquicamente inferiores**: o Ridge e o F04.1 erram nos
mesmos períodos que o GBDT, só que mais. Ensemble ajuda quando os erros são diversos; aqui a
diversificação real é pequena. O piso de peso (necessário para não zerar membros) ainda injeta
10–15% de um modelo pior.

## 5. Consequência

Terceira hipótese rejeitada (alvo, memória do choque, ensemble). O campeão segue `gbdt_v1` /
`gbdt_conformal`. Ganhos adicionais exigem **dados novos** (bloqueados por BigQuery) ou
**classe de modelo diferente** (sequencial), não recomposição dos modelos atuais.

## 6. Arquivos

- `src/previsao_leitos/demand/ensemble.py`
- `src/previsao_leitos/demand/registry.py` (`ensemble_top3`)
- `tests/unit/test_demand_ensemble.py`
- `reports/runs/bt_ensemble_occ/`, `reports/runs/bt_ensemble_need/`
