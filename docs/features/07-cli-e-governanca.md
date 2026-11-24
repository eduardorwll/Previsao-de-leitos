# Feature F07 — CLI Reprodutível, Logging JSON e Governança TRIPOD+AI / RECORD

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída (CLI `previsao-leitos`, Net Benefit, `ECE`/`BSS` e TRIPOD+AI/RECORD) |
| **Etapa** | Etapa 3 — Governança, Net Benefit e Orquestração |
| **Especificação** | [spec/SPEC-cli-and-governance.md](../../spec/SPEC-cli-and-governance.md) |
| **Pacotes Implementados** | `src/previsao_leitos/cli/` · `audit/metrics.py` · `audit/report_builder.py` |
| **Verificação** | `ruff` OK · `mypy --strict` OK · `check_constraints.py` OK · `72` testes (`86,8%` cov.) · `pip-audit` OK |

---

## 1. Entregas e Evidência Real (`uv run previsao-leitos verify-rs --eval-year 2024`)

- **CLI Instalável (`pyproject.toml` `[project.scripts]`):**
  - `uv run previsao-leitos doctor`: auditoria de ambiente e prontidão de dados em JSON estruturado (`run_id`, `entry_point`, `git_sha`, `timestamp_utc`).
  - `uv run previsao-leitos verify-rs --eval-year 2024 --target uti_adulto_diarias --output-dir reports/cli_rs_2024`: executa `F04 + Conformal(2023) + SimPy F05 + Auditoria Contrafactual F06 + Net Benefit + TRIPOD+AI/RECORD` em **`3,3s`** e bloqueia `2025` com código de saída `2` se `--allow-holdout` não for informado explicitamente.
- **Calibração e Utilidade Decisória (RS 2024 — evento de sobrecarga `>= 110%` da média 12 sem.):**
  - **`ECE_surge = 0,0492`** (`< 0,05`) · **`BSS_surge = +0,0881`** (`> 0`) · **`0,00%` violação monotônica**.
  - **Net Benefit (Curva de Decisão):** supera *Tratar Todos* e *Tratar Nenhum* em todo intervalo $p_t \in [0,20; 0,50]$ (ex.: em $p_t = 0,40$, modelo **`+0,068`** vs. *Tratar Todos* `-0,040` vs. *Tratar Nenhum* `0,000`).

---

## 2. Checklist da Feature F07

- [x] Redigir especificação [spec/SPEC-cli-and-governance.md](../../spec/SPEC-cli-and-governance.md)
- [x] Implementar `decision_curve_net_benefit`, `expected_calibration_error` e `brier_skill_score` em `audit/metrics.py`
- [x] Integrar tabela de Net Benefit e seção **TRIPOD+AI / PROBAST+AI / RECORD** em `audit/report_builder.py`
- [x] Implementar `cli/logging_json.py` e `cli/main.py` (`previsao-leitos doctor` e `previsao-leitos verify-rs`) com bloqueio físico de `2025`
- [x] Registrar script `previsao-leitos` em `pyproject.toml` e validar fim a fim com testes *Prove-It*
