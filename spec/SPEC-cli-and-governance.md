# Especificação V1 — Módulo `cli` (F07) e Governança `Net Benefit` / TRIPOD+AI (`spec/SPEC-cli-and-governance.md`)

- **Status:** `approved` *(Etapa 3 — Fatia 2)*
- **Etapa / Feature:** Etapa 3 — Feature `F07 — CLI Reprodutível e Logging JSON` + extensão de `F06` (`Net Benefit`, `ECE`/`BSS` e governança TRIPOD+AI / PROBAST+AI / RECORD)
- **ADRs de Referência:** [ADR-0003](../docs/adr/0003-causal-ex-ante-splits-and-calibration.md), [ADR-0005](../docs/adr/0005-data-layer-and-lgpd.md), [ADR-0007](../docs/adr/0007-counterfactual-analysis-and-verification-report.md)

---

## 1. Objetivo

1. **Composição raiz limpa (`src/previsao_leitos/cli/`):** oferecer o comando instalável `previsao-leitos` com logging JSON estruturado (`run_id`, `entry_point`, `git_sha`, `timestamp_utc`, `event`) e subcomandos determinísticos (`doctor`, `verify-rs`) protegendo o *holdout* `2025` contra acionamento acidental.
2. **Métricas de Utilidade Clínica/Decisória e Calibração (`audit/metrics.py`):** adicionar Análise de Curva de Decisão (**Net Benefit**, Vickers & Elkin 2006), **Expected Calibration Error (`ECE`)** e **Brier Skill Score (`BSS`)** para eventos de sobrecarga hospitalar (*surge*).
3. **Governança TRIPOD+AI / PROBAST+AI / RECORD (`audit/report_builder.py`):** embutir automaticamente a seção de conformidade metodológica e ética (fronteira de pesquisa, zero PII, splits congelados, controle de viés) em todo `verificacao_cientifica.md`.

---

## 2. Estrutura

```text
src/previsao_leitos/cli/
├── __init__.py          # Superfície pública
├── logging_json.py      # Emissor JSON estruturado (zero segredos)
└── main.py              # CLI argparse (`previsao-leitos doctor` | `previsao-leitos verify-rs`)
```
