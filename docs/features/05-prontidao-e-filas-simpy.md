# Feature F05 — Prontidão Hospitalar, Filas (`SimPy`) e Calibração de Choque

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída (Etapa 3 Fatia 1 — Conformal 2023 + SimPy + `Fig. C5`) |
| **Etapa** | Etapa 3 — Incerteza Calibrada, Simulação de Filas e Auditoria RS |
| **Especificação** | [spec/SPEC-readiness-simpy.md](../../spec/SPEC-readiness-simpy.md) |
| **Pacotes Implementados** | `src/previsao_leitos/demand/calibration.py` · `src/previsao_leitos/readiness/` · `Fig. C5` em `audit/` |
| **Verificação** | `ruff` OK · `mypy --strict` OK · `check_constraints.py` OK · `69` testes (`89,1%` cov.) · `pip-audit` OK |

> **Fronteira Clínica:** Simulações e gatilhos exclusivamente para pesquisa e planejamento agregado (*Not for patient care or live clinical use*).

---

## 1. Arquitetura Entregue

- **[`ConformalFloodCalibrator`](../../src/previsao_leitos/demand/calibration.py) (`84%` cov.):** CQR heterocedástico calibrado estritamente sobre `cal_year = 2023` (sem tocar `2024` ou `2025`), expandindo intervalos sob estresse hidrológico (`api_max_sem`, `api_montante_pond`, `remanso_lagunar_idx`) mantendo **`0,00%` de violações monotônicas** em `P50/P90/P95`.
- **[`readiness/simulator.py`](../../src/previsao_leitos/readiness/simulator.py) (`95%` cov.):** Simulador de eventos discretos em **`SimPy`** com *Common Random Numbers* (CRN), histerese segura sobre leitos base, *lead time* de ativação de *surge* (`24–72h`), adiamento de eletivas e cenário de **perda temporária de capacidade em hospitais em áreas aterradas/pôlderes**.
- **[`readiness/triggers.py`](../../src/previsao_leitos/readiness/triggers.py):** Classificação observável (`VERDE / AMARELO / LARANJA / VERMELHO`) vinculada à previsão $P_{90}$ e ao estresse hidrológico/remanso, sempre rotulada como pesquisa.
- **[`plot_c5_simpy_contingency`](../../src/previsao_leitos/audit/plots.py) (`99%` cov.):** Integrado diretamente ao `report_builder.py`, completando o pacote de **5 gráficos contrafactuais (`Fig. C1–C5`)** definidos no [ADR-0007](../adr/0007-counterfactual-analysis-and-verification-report.md).

---

## 2. Resultados Reais RS 2024 ([`run_rs_etapa3_readiness.py`](../../scripts/run_rs_etapa3_readiness.py))

1. **Calibração Conformal Heterocedástica (`cal_year = 2023` → avaliação `2024`, `Sem. 18–28`):**
   - Cobertura `P10–P90` durante o choque extremo de **Diárias UTI Adulto** subiu de **`34,1%` (`PooledRidge`)** → **`54,5%` (`QuantileGBDT`)** → **`61,4%` (`QuantileGBDT + Conformal(2023)`)**, mantendo **`0,00%` de violação monotônica**.
2. **Simulação Contrafactual `SimPy` — Região 10 (Capital/POA — `431` leitos UTI Adulto SUS reais do CNES Abr/2024 + estresse de `15%` de perda em áreas aterradas/pôlderes + `90` leitos *surge* com *lead time* 72h):**
   - **Reativo (`h = 0`):** pico de fila **`38` pacientes** · **`30` dias** com ocupação `>= 95%`.
   - **Proativo (`h = 2` semanas, guiado pelo $P_{90}$ calibrado):** pico de fila **`24` pacientes** (**`-36,8%`**) · **`14` dias** em sobrecarga (**`16` dias evitados**), eliminando completamente a fila secundária da segunda onda!
   - **Relatório completo gerado em `3,9s`:** `reports/rs_2024_etapa3_uti/verificacao_cientifica.md` (`Fig. C1–C5` + `run_manifest.json`).

---

## 3. Checklist da Feature F05

- [x] Redigir especificação [spec/SPEC-readiness-simpy.md](../../spec/SPEC-readiness-simpy.md)
- [x] Implementar `ConformalFloodCalibrator` (`demand/calibration.py`) calibrado exclusivamente em `2023`
- [x] Implementar simulador `SimPy` (`readiness/simulator.py`) com Common Random Numbers e estresse em pôlder
- [x] Implementar gatilhos observáveis (`readiness/triggers.py`) com fronteira explícita de pesquisa
- [x] Acoplar **`Fig. C5`** em `audit/plots.py` e `audit/report_builder.py` e validar no RS `2024`
