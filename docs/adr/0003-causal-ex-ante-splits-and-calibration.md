# ADR-0003: Contrato Causal Ex-Ante, Splits Congelados e Validação Espacial

- **Status:** ACCEPTED
- **Data:** 2026-11-24

## Contexto e Decisão

Na auditoria do protótipo legado foram detectados múltiplos vazamentos: (1) uso de chuva e decretos realizados no próprio mês-alvo para prever leitos daquele mês; (2) reutilização de **2024** (ano usado para calibração e *early stopping* nas Fases 1–2) como auditoria *out-of-time* na Fase 3; (3) `shift(-2)` antes do corte anual vazando janeiro/2024 no treino de 2023; (4) mediana global `2010–2025` na imputação populacional; e (5) janelas posicionais quebradas por dias faltantes.

**Decisão:** Estabelecer contrato estritamente causal:
1. Toda série é reindexada em **grade calendário completa** antes de lags/janelas; todo pré-processamento aprendido é ajustado apenas no treino dentro de `Pipeline`.
2. Modelos de demanda consomem exclusivamente covariáveis **ex-ante** (incluindo previsões calibradas do modelo hidrológico geradas *out-of-fold* / prospectivamente em $t_0$).
3. Protocolo temporal congelado: **Treino (`<= 2022`)**, **Calibração & Early Stopping (`2023`)**, **Auditoria de Desenvolvimento RS (`2024`, incl. catástrofe de maio/2024)** e **Holdout Intocado (`2025`)**. Na expansão nacional, adicionar **Validação Cruzada Espacial (*Leave-Basin-Out* e *Leave-UF-Out*)**.
4. Incerteza de todas as métricas calculada via **block/cluster bootstrap (por bacia e episódio)**, nunca linha-a-linha.

## Alternativas Rejeitadas

- **Reutilizar 2024 para calibração e auditoria final:** gera overfitting adaptativo e invalida conclusões *out-of-time*.
- **Avaliar modelos de leitos com chuva/decretos observados do mês-alvo:** produz estimativa oráculo inútil para prevenção antecipada.

## Consequências

Reduzirá artificialismos (como $R^2 > 0,9995$ por persistência) e entregará métricas honestas de habilidade preditiva (MASE, pinball loss, ECE, Brier Skill Score) comparadas explicitamente contra Persistência e Sazonal-Ingênuo.
