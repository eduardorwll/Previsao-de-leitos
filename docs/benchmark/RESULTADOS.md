# Resultados do benchmark (snapshot)

> Run de referência: `rs_v1_3_benchmark_2024` · rubrica `v1`.
> Este arquivo é gerado por `scripts/render_benchmark_docs.py`; não editar à mão.

## 1. Benchmark anual (`--eval-year 2024`)

### Respiratório

| candidate | wmape | mase_vs_pers | peak_timing_error_weeks | cobertura_p10_p90 | erro_pct_p90 | acuracia | fase | selo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ensemble_top3 | 34.8 | 0.884 | 4.8 | 0.7 | 199.6 | D | C | D |
| ridge_pooled | 34.8 | 0.884 | 4.1 | 0.7 | 210.3 | D | C | D |
| gbdt_v1 | 35.0 | 0.887 | 4.0 | 0.8 | 204.2 | D | B | D |
| gbdt_conformal | 35.0 | 0.887 | 4.0 | 0.8 | 204.2 | D | B | D |
| gbdt_agaci | 35.0 | 0.887 | 4.0 | 0.8 | 204.2 | D | B | D |
| gbdt_spci | 35.0 | 0.887 | 4.0 | 0.8 | 204.2 | D | B | D |
| f041_local_level | 38.2 | 0.970 | 8.0 | 0.7 | 174.7 | D | D | D |
| f041_mondrian | 38.2 | 0.970 | 8.0 | 0.8 | 174.7 | D | D | D |

### Clínico

| candidate | wmape | mase_vs_pers | peak_timing_error_weeks | cobertura_p10_p90 | erro_pct_p90 | acuracia | fase | selo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gbdt_v1 | 9.7 | 0.983 | 2.4 | 0.7 | 25.6 | B | B | D |
| gbdt_conformal | 9.7 | 0.983 | 2.4 | 0.7 | 25.6 | B | B | D |
| gbdt_agaci | 9.7 | 0.983 | 2.4 | 0.8 | 25.6 | B | B | B |
| gbdt_spci | 9.7 | 0.983 | 2.4 | 0.6 | 25.6 | B | B | D |
| persistencia | 9.8 | 1.000 | 2.0 | 0.7 | 26.5 | B | A | B |
| ridge_pooled | 10.6 | 1.079 | 2.7 | 0.7 | 26.6 | B | B | D |
| ensemble_top3 | 11.1 | 1.126 | 2.6 | 0.7 | 26.9 | B | B | D |
| f041_local_level | 14.6 | 1.489 | 6.3 | 0.6 | 33.6 | C | D | D |

### UTI total

| candidate | wmape | mase_vs_pers | peak_timing_error_weeks | cobertura_p10_p90 | erro_pct_p90 | acuracia | fase | selo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gbdt_spci | 8.4 | 0.971 | 3.4 | 0.7 | 24.7 | B | B | C |
| gbdt_v1 | 8.4 | 0.972 | 3.4 | 0.7 | 24.7 | B | B | C |
| gbdt_conformal | 8.4 | 0.972 | 3.4 | 0.7 | 24.7 | B | B | C |
| gbdt_agaci | 8.4 | 0.972 | 3.4 | 0.8 | 24.7 | B | B | B |
| persistencia | 8.7 | 1.000 | 1.9 | 0.7 | 25.7 | B | A | C |
| ridge_pooled | 8.8 | 1.016 | 3.2 | 0.7 | 24.9 | B | B | D |
| ensemble_top3 | 9.4 | 1.083 | 3.9 | 0.7 | 26.2 | B | B | C |
| f041_local_level | 13.1 | 1.514 | 8.1 | 0.6 | 32.3 | C | D | D |

### UTI adulto

| candidate | wmape | mase_vs_pers | peak_timing_error_weeks | cobertura_p10_p90 | erro_pct_p90 | acuracia | fase | selo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gbdt_v1 | 9.5 | 0.984 | 3.3 | 0.7 | 28.4 | B | B | C |
| gbdt_conformal | 9.5 | 0.984 | 3.3 | 0.7 | 28.4 | B | B | C |
| gbdt_agaci | 9.5 | 0.984 | 3.3 | 0.8 | 28.4 | B | B | B |
| gbdt_spci | 9.5 | 0.984 | 3.3 | 0.7 | 28.4 | B | B | C |
| ridge_pooled | 9.6 | 0.997 | 3.3 | 0.7 | 29.1 | B | B | D |
| persistencia | 9.6 | 1.000 | 1.9 | 0.7 | 29.5 | B | A | D |
| ensemble_top3 | 10.0 | 1.037 | 3.5 | 0.7 | 28.3 | B | B | C |
| f041_local_level | 13.3 | 1.377 | 7.5 | 0.7 | 34.1 | C | D | D |

## 2. Régua de origem móvel (medianas entre folds)

### Ocupação

| target | candidate | mase_median | mase_worst | coverage_median | peak_abs_median |
| :--- | :--- | :--- | :--- | :--- | :--- |
| respiratorio | gbdt_agaci | 0.866 | 0.878 | 0.8 | 6.2 |
| respiratorio | gbdt_spci | 0.866 | 0.878 | 0.8 | 6.2 |
| respiratorio | gbdt_conformal | 0.866 | 0.878 | 0.8 | 6.2 |
| respiratorio | gbdt_v1 | 0.866 | 0.878 | 0.8 | 6.2 |
| respiratorio | ridge_pooled | 0.868 | 0.878 | 0.8 | 5.5 |
| clinico | gbdt_spci | 0.924 | 1.029 | 0.8 | 4.5 |
| clinico | gbdt_v1 | 0.924 | 1.030 | 0.7 | 4.5 |
| clinico | gbdt_agaci | 0.924 | 1.030 | 0.8 | 4.5 |

### Necessidade

| target | candidate | mase_median | mase_worst | coverage_median | peak_abs_median |
| :--- | :--- | :--- | :--- | :--- | :--- |
| need_respiratorio | gbdt_v1 | 0.840 | 0.865 | 0.8 | 7.4 |
| need_respiratorio | gbdt_spci | 0.840 | 0.865 | 0.8 | 7.4 |
| need_respiratorio | gbdt_agaci | 0.840 | 0.865 | 0.8 | 7.4 |
| need_respiratorio | gbdt_conformal | 0.840 | 0.865 | 0.8 | 7.4 |
| need_respiratorio | ridge_pooled | 0.853 | 0.894 | 0.8 | 7.5 |
| need_clinico | gbdt_spci | 0.884 | 1.036 | 0.7 | 4.3 |
| need_clinico | gbdt_agaci | 0.884 | 1.038 | 0.8 | 4.3 |
| need_clinico | gbdt_v1 | 0.884 | 1.038 | 0.7 | 4.3 |

