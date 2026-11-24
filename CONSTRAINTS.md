# CONSTRAINTS.md — Padrões Mecânicos de Qualidade do Projeto

Este arquivo é a barra de qualidade canônica e verificável do projeto `Previsao-de-leitos`. Nenhuma alteração pode enfraquecer uma restrição abaixo para fazer um PR passar.

- **Regime de Licença do Projeto:** Pesquisa aberta (compatível com ferramentas abertas; para modelos fundacionais usar **TimesFM 2.5 Apache-2.0** por compatibilidade com futura adoção pública SUS/EBSERH).
- **Escopo de Aplicação Imediata:** Todo código novo em `src/previsao_leitos/`, `tests/` e `scripts/`. O script legado monolítico foi arquivado em `legacy/previsao_leitos_colab.py` (evitando conflito de importação com `src/previsao_leitos/`) apenas como referência histórica até sua remoção controlada na Etapa 1 (registrado como exceção temporária abaixo).

---

## 1. Floor Inegociável (Verificação Mecânica)

| ID | Restrição | Comando de Verificação | Ciclo | Status |
| :--- | :--- | :--- | :---: | :---: |
| **C-01** | **Estilo, imports e bugs estáticos Python (`ruff`)** — zero violações em `src/`, `tests/`, `scripts/` | `uv run ruff check src tests scripts` | `BUILD` (<5s) | Enforced |
| **C-02** | **Tipagem estática estrita (`mypy`)** — zero erros de tipo em `src/` | `uv run mypy src` | `VERIFY` (<30s) | Enforced |
| **C-03** | **Testes unitários e cobertura mínima ($\ge 80\%$ em `src/`)** — suíte determinística sem rede/BigQuery/GPU | `uv run pytest -m "not slow" --cov=src/previsao_leitos --cov-fail-under=80` | `VERIFY` (<90s) | Enforced |
| **C-04** | **Floor de segurança e rigor científico em código (`scripts/check_constraints.py`)** — proíbe `warnings.filterwarnings("ignore")`, `except: pass`, `# noqa`/`# type: ignore` sem código/motivo, `allow_pickle=True`, SQL f-string não parametrizado em `src/` | `uv run python scripts/check_constraints.py src tests` | `BUILD` (<5s) | Enforced |
| **C-05** | **Varredura de segredos externa (`gitleaks --redact`)** — zero credenciais ou `.env` no working tree/commits | `gitleaks dir . --redact --no-banner` (CI: `gitleaks-action`) | `REVIEW` / `SHIP` | Enforced (CI) |
| **C-06** | **Auditoria de vulnerabilidades de dependências (`pip-audit`)** — zero CVEs críticos/altos exploráveis no lockfile | `uv run pip-audit` | `SHIP` (CI) | Enforced |
| **C-07** | **Zero side-effects em tempo de importação** — `import previsao_leitos` não requer env vars, rede nem GPU | `uv run python -c "import previsao_leitos"` | `BUILD` (<5s) | Enforced |
| **C-08** | **Zero PII / conformidade LGPD** — nenhum campo identificável de paciente em schemas, fixtures ou logs; células pequenas suprimidas | Teste unitário de contrato de schema + revisão da camada de dados | `VERIFY` / `REVIEW` | Enforced |
| **C-09** | **Zero vazamento temporal/espacial em features e splits** — perturbar `2024+` deixa artefatos de treino `<=2022` bit-idênticos | Teste *Prove-It* obrigatório a partir da Etapa 1 (`tests/unit/test_leakage.py`) | `VERIFY` | Planned (Etapa 1) |
| **C-10** | **Unidades físicas e calibração** — unidades verificadas via `pint`; classificadores reportam ECE e Brier Skill Score | Testes de unidade física + auditoria de calibração a partir da Etapa 2 | `REVIEW` | Planned (Etapa 2) |

---

## 2. Ratchets (Catracas — Não Podem Regredir)

| Métrica | Baseline Registrado | Tolerância | Regra |
| :--- | :---: | :---: | :--- |
| Cobertura de testes em `src/previsao_leitos/` | **100%** (Etapa 0 bootstrap) | `-0.5%` (piso mínimo absoluto `80%`) | Atualizar baseline ao subir; nunca reduzir abaixo do piso |
| Número de violações de `ruff` / `mypy` em `src/` | **0** | `0` | Qualquer regressão bloqueia merge |
| Cobertura de municípios do RS no painel hidrológico | **~8% (INMET legado)** → Meta Etapa 1: **100% (497/497)** | `0%` após Etapa 1 | Nunca regredir para inner-join silencioso |

---

## 3. Regras Anti-Gaming de Diff

- Proibido silenciar erros com `# noqa`, `# type: ignore`, `nosemgrep`, `gitleaks:allow` ou `warnings.filterwarnings("ignore")` sem justificativa formal na tabela de Exceções abaixo.
- Proibido pular (`@pytest.mark.skip`) ou deletar testes para fazer o CI passar sem motivo explícito no corpo do commit.
- Proibido inflar cobertura com testes sem asserções comportamentais (*Mutation Probe*: inverter uma condição crítica adicionada deve quebrar a suíte).

---

## 4. Exceções Temporárias (Obrigatório: Dono + Expiração)

| ID | Alvo | Motivo | Dono | Expira em |
| :--- | :--- | :--- | :--- | :---: |
| **EX-01** | `legacy/previsao_leitos_colab.py` | Protótipo legado Colab arquivado somente para leitura/paridade durante a migração modular da Etapa 1; excluído de `ruff`/`mypy`/`check_constraints.py` até ser removido em commit dedicado. | Equipe do projeto | `2027-02-22` (90 dias) |

---

## 5. Registro de Calibração dos Verificadores

Toda restrição marcada como `Enforced` passa por teste de calibração (violação sintética **falha**; código limpo **passa**) registrado na entrega da Etapa 0.
