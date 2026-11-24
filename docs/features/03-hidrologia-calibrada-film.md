# Feature F03 — Hidrologia Calibrada (`HydroFiLMMonotonicNet` em PyTorch Lightning)

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída (V1 RS — Monotonicidade `autograd 0,00%` + `BSS > 0` em CPU) |
| **Etapa** | Etapa 2 — Pivô Científico RS (Hidrologia Calibrada) |
| **Especificação** | [spec/SPEC-hydrology-film-net.md](../../spec/SPEC-hydrology-film-net.md) |
| **Pacote Implementado** | `src/previsao_leitos/hydrology/` + `scripts/run_rs_f03_hydrology.py` |
| **Verificação** | `ruff` OK · `mypy --strict` OK · `check_constraints.py` OK · `75` testes (`87,2%` cov.) · `pip-audit` OK |

---

## 1. Arquitetura Entregue (`ADR-0001` e `ADR-0006`)

- **[`dataset.py`](../../src/previsao_leitos/hydrology/dataset.py) (`89%` cov.):** `IndexedHydroSequenceDataset` (indexação 2D sob demanda sem jamais cruzar lacunas ISO ou fronteiras entre regiões) com validação automática que impede qualquer vazamento de features monotônicas (`frac_urbano`, `frac_banhado_agua`, `frac_area_polder_regiao`, `indice_aterro_varzea`, `frac_floresta`) no ramo dinâmico (`x_dyn`) ou no FiLM (`x_ctx`).
- **[`model.py`](../../src/previsao_leitos/hydrology/model.py) (`95%` cov.):** `HydroFiLMMonotonicNet` (Bi-GRU 2 camadas + Multi-Head Self-Attention + FiLM climatológico sem `mun_embedding` nem `lat/lon` + ramo monotônico `F.softplus` aplicado a **ambas** as cabeças `event_logits` e `severity_pred`) encapsulada em `HydroLightningModule`.
- **[`calibration.py`](../../src/previsao_leitos/hydrology/calibration.py) (`91%` cov.):** `TemperatureScaler` (`fit_bias=False` para temperatura pura ou `fit_bias=True` para Platt Scaling, sempre com $T > 0$ preservando monotonicidade).

---

## 2. Resultados Reais RS (`Treino <= 2022 | Cal 2023 | Audit 2024 | h = 2 sem. | 2025 intocado`)

Executado via [`scripts/run_rs_f03_hydrology.py`](../../scripts/run_rs_f03_hydrology.py) em **`20,6s` na CPU** (`torch 2.15.0+cpu` + `lightning 2.6.6`):

| Experimento (`pos_weight`) | ROC-AUC (2024) | PR-AUC (2024) | ECE (2024) | BSS (2024) | Auditoria `autograd` Monotonicidade |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `pos_weight=4.0` + Platt(`2023`) | `0,6992` | **`0,3626`** | `0,1218` | `-0,0957` | **`0,00%` viol. (APROVADA)** |
| **`pos_weight=1.0` (Proper Scoring Rule — Recomendado)** | **`0,7199`** | `0,3266` | **`0,0521`** | **`+0,0756`** | **`0,00%` viol. (APROVADA)** |

> **Diagnóstico científico:** o treinamento sob *proper scoring rule* (`pos_weight=1.0`) entrega probabilidades nativamente bem calibradas (`ECE = 0,0521`, `BSS = +0,0756 > 0`), enquanto calibrar sobre um único ano anômalo El Niño (`2023`) introduz deslocamento de taxa base para `2024` — reforçando o uso de calibração *rolling* multi-ano na expansão nacional (V2).

---

## 3. Checklist da Feature F03

- [x] Redigir especificação [spec/SPEC-hydrology-film-net.md](../../spec/SPEC-hydrology-film-net.md)
- [x] Implementar `IndexedHydroSequenceDataset` (`dataset.py`) com integridade de lacunas e separação estrita `ADR-0001`
- [x] Implementar `HydroFiLMMonotonicNet` + `HydroLightningModule` (`model.py`) com monotonicidade verificada por `torch.autograd.grad` em ambas as cabeças
- [x] Implementar `TemperatureScaler` (`calibration.py`)
- [x] Validar com testes *Prove-It* e rodar treino/calibração real no RS (`2025` intocado)
- [x] **Evolução `F03.1 HydroFiLM-ENSO`:** adicionar condicionamento `FiLM` explícito por teleconexões El Niño/ENSO + Atlântico Sul (`enso_dim` em `model.py`), auditoria `audit_autograd_monotonicity` (`0,00%` violações) e gerador *out-of-fold* estritamente causal `generate_causal_hydro_oof_predictions` (`hydrology/oof.py`) para acoplamento direto à `F04.1`
