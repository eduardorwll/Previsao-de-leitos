# Especificação V1 — Módulo `hydrology`: `HydroFiLMMonotonicNet` (PyTorch Lightning) e Calibração Pós-Hoc (`spec/SPEC-hydrology-film-net.md`)

- **Status:** `approved` *(Feature F03 — último bloco arquitetural antes da Auditoria Final no holdout `2025`)*
- **Etapa / Feature:** Etapa 2–3 — Feature `F03 — Hidrologia Calibrada (HydroFiLMMonotonicNet)`
- **ADRs de Referência:** [ADR-0001](../docs/adr/0001-structural-monotonicity.md), [ADR-0003](../docs/adr/0003-causal-ex-ante-splits-and-calibration.md), [ADR-0004](../docs/adr/0004-multiscale-architecture-rs-to-brazil.md), [ADR-0006](../docs/adr/0006-landfill-polders-and-intermunicipal-routing.md)

---

## 1. Correções Estruturais frente ao Protótipo Legado (`ADR-0001`)

1. **Monotonicidade Estrutural Real (Zero Vazamento no Ramo Irrestrito):**
   - `x_dyn` contém **apenas** dinâmica meteorológica/hidrológica pura (`chuva_sem_mm`, `chuva_montante_pond`, `api_max_sem`, `api_montante_pond`, `remanso_lagunar_idx`, `umidade_solo_sem`, `temp_media_sem_c`, sazonalidade cíclica).
   - **Jamais** passa impermeabilização, `indice_aterro_varzea` ou `escoamento_efetivo_mm` no ramo dinâmico ou no FiLM.
   - Tanto `event_logits` quanto `severity_pred` recebem o ramo monotônico `F.softplus`:
     $$\frac{\partial \text{logit}}{\partial x_{\text{pos}}} > 0, \quad \frac{\partial \text{severity}}{\partial x_{\text{pos}}} > 0, \quad \frac{\partial \text{logit}}{\partial x_{\text{neg}}} < 0, \quad \frac{\partial \text{severity}}{\partial x_{\text{neg}}} < 0$$
     onde $x_{\text{pos}} = (\text{frac\_urbano}, \text{frac\_banhado\_agua}, \text{frac\_area\_polder\_regiao}, \text{indice\_aterro\_varzea})$ e $x_{\text{neg}} = (\text{frac\_floresta})$.
2. **Sem Memorização Espacial:** nenhum `mun_embedding` e nenhum `(lat, lon)` cru; o contexto FiLM recebe apenas estatísticas climatológicas de bacia ajustadas em `<= 2022`.
3. **Integridade de Grade Calendário (`IndexedHydroSequenceDataset`):** nenhuma janela cruza fronteira de região ou lacuna semanal (`gap_weeks > 1`), mantendo consumo de RAM baixo (indexação 2D sob demanda).
4. **Calibração Pós-Hoc (`TemperatureScaler` em `cal_year = 2023`):** ajusta temperatura $T > 0$ por minimização de NLL sobre logits de `2023`, auditada em `2024` via `ROC-AUC`, `PR-AUC`, `ECE` e `BSS` (`2025` permanece intocado).

---

## 2. Estrutura do Módulo `src/previsao_leitos/hydrology/`

```text
src/previsao_leitos/hydrology/
├── __init__.py          # Superfície pública
├── dataset.py           # IndexedHydroSequenceDataset (janelas 2D sem cruzar lacunas/regiões)
├── model.py             # HydroFiLMMonotonicNet (Bi-GRU + MHA + FiLM + F.softplus) + HydroLightningModule
└── calibration.py       # TemperatureScaler pós-hoc (calibrado em 2023)
```
