# Especificação V1 — Calibração Conformal (`demand/calibration.py`) e Prontidão Hospitalar (`readiness` / `SimPy`) (`spec/SPEC-readiness-simpy.md`)

- **Status:** `approved` *(Etapa 3 — Fatia 1)*
- **Etapa / Feature:** Etapa 3 — Calibração Conformal de Choque (`2023`) + Feature `F05 — Prontidão, Filas Hospitalares (SimPy) e Cenários de Estresse` + `Fig. C5` em `F06`
- **ADRs de Referência:** [ADR-0002](../docs/adr/0002-target-teaching-hospital-demand.md), [ADR-0003](../docs/adr/0003-causal-ex-ante-splits-and-calibration.md), [ADR-0006](../docs/adr/0006-landfill-polders-and-intermunicipal-routing.md), [ADR-0007](../docs/adr/0007-counterfactual-analysis-and-verification-report.md)

> **Fronteira Clínica:** Simulações exclusivamente voltadas a planejamento agregado e pesquisa metodológica (*"Not for patient care or live clinical use"*).

---

## 1. Problema que Esta Especificação Resolve

1. **Subcobertura em Janelas de Choque Extremo (diagnosticada pela F06):** no RS 2024, modelos pontuais/quantílicos não calibrados tiveram cobertura `P10–P90` de `34,1%` em Diárias UTI Adulto durante o pico da enchente (`Sem. 18–28`). Precisamos de **Calibração Conformal Heterocedástica** ajustada **exclusivamente em `cal_year = 2023`** (sem tocar `2024` nem `2025`), expandindo a banda sob estresse hidrológico (`api_max_sem + api_montante_pond + remanso_lagunar_idx`) mantendo **`0,00%` de violação monotônica**.
2. **Tradução de Previsão em Decisão de Contingência (`SimPy` + `Fig. C5`):** prever chegadas ou diárias semanais não basta para dimensionar filas de UTI, adiamento de cirurgias eletivas (*lead time* `24–72h`) nem o impacto de **evacuação/perda de capacidade em hospitais em áreas aterradas/pôlderes** (ex.: Canoas, 4º Distrito em POA, Pelotas).

---

## 2. Estrutura dos Módulos

```text
src/previsao_leitos/demand/
└── calibration.py       # ConformalFloodCalibrator (CQR heterocedástico calibrado estritamente em 2023)

src/previsao_leitos/readiness/
├── __init__.py          # Superfície pública
├── simulator.py         # Simulador de eventos discretos SimPy (UTI/Enfermaria, eletivas, surge lead-time, evacuação pôlder)
└── triggers.py          # Tabela auditável de gatilhos observáveis (Verde/Amarelo/Laranja/Vermelho — pesquisa)

src/previsao_leitos/audit/
├── plots.py             # + plot_c5_simpy_contingency (Fig. C5 obrigatória do ADR-0007)
└── report_builder.py    # + integração opcional da Fig. C5 e KPIs SimPy no relatório final
```

---

## 3. Contratos e Testes *Prove-It*

- **`ConformalFloodCalibrator`:**
  - Ajustado estritamente em `cal_df` (`target_ano_epi == 2023`), com estatísticas de normalização de estresse hidrológico calculadas estritamente em `train_df` (`<= 2022`).
  - Garante $0 \le P_{10} \le P_{50} \le P_{90} \le P_{95}$ e **zero violações monotônicas** em `P50/P90/P95`.
- **`simulate_counterfactual_readiness` (`SimPy` com Common Random Numbers):**
  - Compara sob exatamente o mesmo fluxo estocástico de chegadas:
    1. `reactive_h0`: adiamento de eletivas e leitos de contingência só ativam quando ocupação real ultrapassa `95%` (com *lead time* de `48h`);
    2. `proactive_h2`: acionamento antecipado orientado pela previsão *ex-ante* $P_{90}$ (`h = 2` semanas antes), ativando redução gradual de eletivas e leitos de contingência antes do pico;
    3. `polder_evacuation_stress`: estresse composto simulando perda de `20%` da oferta em hospitais situados em áreas aterradas/pôlderes (redirecionando demanda via `W_ref`), comparando `reactive_h0` vs. `proactive_h2`.
  - **Teste *Prove-It*:** `proactive_h2` reduz estritamente `pico_fila_pacientes` e `dias_sobrecarga_gt95` frente a `reactive_h0`, e `polder_hospital_loss_frac > 0` aumenta a fila sem intervenção.
