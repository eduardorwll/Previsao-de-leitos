# Feature F16 — Benchmark Multi-Modelo, CLI Única e Relatório Comparativo (`Task 1.3.1`)

> **Status:** ✅ Implementada (run auditado em `RS 2024`) | **Etapa:** Etapa 3 (`Ciclo V1.2 RS`)
> **Escopo de dados:** `RS`, `ano_epi <= 2024` (holdout `2025` e `2026` fisicamente fora do artefato)
> **Fronteira obrigatória:** *Not for patient care or live clinical use.* | **LGPD:** zero PII, agregação na fronteira, supressão de células com $n < 5$.

---

## 1. Objetivo

Transformar o conjunto de modelos e métricas do projeto em um **produto executável**: um comando que treina todos os candidatos no mesmo split causal, mede os mesmos indicadores, compara leitos **observados vs. preditos** e emite um relatório auditável — sem planilha manual e sem número escolhido a dedo.

Antes desta feature, o benchmark do dossiê tinha **duas linhas** (V1 e V1.2, mesma família), o alvo era fixo em UTI adulto e a CLI apontava para o painel `v2`.

---

## 2. Comando único

```bash
# prepara o painel de modelagem (rede apenas aqui; grava <= 2024)
uv run --no-sync python scripts/build_rs_modeling_panel.py

# benchmark + relatório + espelho verificado
uv run --no-sync previsao-leitos run \
  --target all --models all --eval-year 2024 \
  --run-id rs_v1_3_benchmark_2024
```

Subcomandos disponíveis: `doctor`, `verify-rs` (agora em painel `v3`), `benchmark`, `run`, `report`.

Artefatos por run:

| Arquivo | Conteúdo |
| :--- | :--- |
| `predictions.parquet` | tidy: região × semana-alvo × modelo × observado × P10/P50/P90/P95 |
| `benchmark.csv` | uma linha por modelo × alvo com todas as métricas |
| `comparison.csv` | observado vs. predito por região (viés, MAE, picos), $n \geq 5$ |
| `decisions.csv` | catraca Champion–Challenger por alvo |
| `fig_observed_vs_predicted_<alvo>.png` | séries semanais observadas vs. preditas (300 DPI, Okabe-Ito) |
| `fig_residuals_<alvo>.png` | viés médio por região de saúde |
| `fig_calibration_<alvo>.png` | cobertura P10–P90 e Pinball P95 por modelo |
| `run_manifest.json` | git SHA, seed, SHA-256 do painel, versões, tempo, notas |
| `relatorio_benchmark_leitos.pdf` | HTML5/CSS3 editorial renderizado por Chromium |

---

## 3. Guardas de confiabilidade

1. **Holdout físico:** `build_rs_modeling_panel.py` recusa `max_year >= 2025` e o painel gravado não contém `2025`; `make_modeling_splits` rejeita `eval_year >= 2025` sem `--allow-holdout`.
2. **Fail-closed por candidato:** dependência opcional ausente vira `status=unavailable`; exceção de ajuste vira `status=failed` com a mensagem. Nenhum candidato desaparece em silêncio.
3. **Determinismo:** mesma seed + mesmo painel produzem `benchmark.csv` idêntico (teste dedicado).
4. **Monotonicidade (`ADR-0001`):** cada modelo é reavaliado com o choque hidrológico majorado em `+0,10`; quedas de P50 são contadas e o gate exige `0`.
5. **Catraca:** `ChampionChallengerRatchet` compara cada challenger ao campeão registrado em MASE, Pinball P95, cobertura e *timing* de pico.
6. **LGPD:** a tabela por região suprime `n < 5` na exportação, não apenas na exibição.

---

## 4. Roster avaliado

Baselines obrigatórios: persistência (`y_origin`) e sazonal-ingênuo (`lag 52`).
Modelos: `ridge_pooled`, `negbin_hurdle`, `gbdt_v1`, `gbdt_conformal`, `f041_core`,
`f041_mondrian`, `f041_agaci`, `f041_spci`, `hierarchical_bayes`.

Campeão registrado: **`gbdt_conformal`** (V1 auditado). O benchmark revelou que o F04.1
não supera o V1 neste split — o registro foi corrigido para refletir a evidência, e não o contrário.

---

## 5. Resultados (run `rs_v1_3_benchmark_2024`)

> Fonte: `reports/runs/rs_v1_3_benchmark_2024/benchmark.csv`. `44/44` candidatos concluídos em `1.141 s` de CPU.

| Alvo | Melhor MASE | Modelo | MASE vs. persistência | Cobertura P10–P90 | Erro pontual |
| :--- | :---: | :--- | :---: | :---: | :---: |
| UTI adulto | 0,9843 | `gbdt_conformal` / `gbdt_v1` | `-1,6%` vs. persistência | `70,0%` | `3,51` leitos/dia |
| UTI total | 0,9716 | `gbdt_conformal` / `gbdt_v1` | `-2,8%` | `68,3%` | `4,33` leitos/dia |
| Clínico | 0,9831 | `gbdt_conformal` / `gbdt_v1` | `-1,7%` | `66,5%` | `127,3` pacientes-dia |
| Respiratório | 0,8837 | `ridge_pooled` | `-11,6%` | `69,6%` | `9,71` pacientes-dia |
| Respiratório (calibrado) | 0,8873 | `gbdt_conformal` | `-11,3%` | **`79,0%`** | `9,75` pacientes-dia |

### 5.1 Achados relevantes

- **F04.1 não é o melhor modelo no horizonte `h = 2`.** A arquitetura ortogonal entrega
  `MASE 1,71–1,77` (`6,29` leitos/dia em UTI adulto) contra `0,98` do GBDT monotônico,
  confirmando o `REJECT` da `Task 1.2.1`. Nenhuma promoção automática é possível.
- **O campeão V1 está subcoberto.** A cobertura empírica P10–P90 fica em `66–70%` contra
  os `80%` nominais em três dos quatro alvos — a catraca reprova o próprio campeão no
  critério de cobertura. É o principal item de recalibração para o `Marco M1`.
- **A calibração conformal de 2023 não transfere para 2024** nos modelos F04.1
  (`f041_spci` chega a `57,8%` de cobertura em UTI adulto) e é neutra no GBDT em três alvos,
  com exceção de respiratório (`76,4% → 79,0%`).
- **Bayes hierárquico** tem a melhor cobertura em UTI adulto (`92,5%`), mas pior acurácia
  (`MASE 2,23`) e custo de `~4 min/alvo`; é rejeitado por regressão em métrica secundária.
- **Baselines são competitivos em alvos difíceis:** `ridge_pooled` lidera respiratório
  (`MASE 0,884`), o que limita o ganho atribuível à modelagem não linear nesse alvo.
- **Monotonicidade:** `0` violações em todos os `44` pares modelo × alvo, incluindo o
  bayesiano após a correção de prior (`Normal → HalfNormal`).


---

## 6. Limitações

1. `2024` é ano de **desenvolvimento e auditoria**, não *out-of-time*; a afirmação de
   generalização pertence ao `Marco M2`.
2. A catraca aqui é **quantílica**. Especificidade de surto e ECE binário continuam no
   caminho dedicado da `Task 1.2.4`.
3. O custo do bayesiano (~4 min por alvo, CPU) é reportado por modelo; ele não é removido
   do relatório por ser lento.
4. Modelos ancorados em observação (baselines) não usam covariáveis de choque e, portanto,
   não exercem o teste de monotonicidade — isso é registrado no relatório.

---

## 7. Arquivos

- `src/previsao_leitos/demand/registry.py`, `baselines.py`, `modeling.py`, `hierarchical_forecaster.py`
- `src/previsao_leitos/audit/model_benchmark.py`, `comparison_plots.py`, `benchmark_report.py`
- `src/previsao_leitos/cli/main.py` (`benchmark`, `run`, `report`; `verify-rs` em `v3`)
- `scripts/build_rs_modeling_panel.py`
- `tests/unit/test_demand_registry_and_benchmark.py`, `tests/unit/test_cli_benchmark.py`
- `reports/runs/rs_v1_3_benchmark_2024/`
