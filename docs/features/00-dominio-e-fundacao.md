# Feature F00 — Domínio Base, Tipos Validados e Fundação de Qualidade

| Campo | Valor |
| :--- | :--- |
| **Status** | ✅ Concluída |
| **Etapa** | Etapa 0 — Fundação do Pacote, Segurança e Qualidade |
| **Pacote** | `src/previsao_leitos/domain/` |
| **Testes** | `tests/unit/test_domain_types.py` (`17/17` passando · `100%` cobertura linhas e branches) |

---

## 1. Por Que Esta Feature Existe

No protótipo inicial (arquivado em `legacy/previsao_leitos_colab.py`), códigos oficiais de saúde e geografia (CNES, IBGE municipal, UF) circulavam como `str` ou `int` sem validação, sujeitos a perda de zeros à esquerda e trocas acidentais entre chaves. Além disso, o script executava autenticação na nuvem e leitura obrigatória de variáveis de ambiente no momento do `import`, impedindo testes unitários isolados.

A feature **F00** estabelece a fundação limpa do pacote `previsao_leitos`:
- **Zero efeitos colaterais no `import`** (importável sem credenciais GCP, rede ou GPU).
- **Validação estrita na fronteira de entrada** para identificadores oficiais brasileiros.
- **Hierarquia única de exceções** para distinguir claramente erros de configuração, contrato de dados, vazamento temporal e indisponibilidade de fontes externas.
- **Qualidade mecânica automatizada** via `ruff`, `mypy --strict`, `pytest --cov`, `pip-audit` e verificador de padrões proibidos (`scripts/check_constraints.py`).

---

## 2. Componentes Implementados

### 2.1 Tipos Nominais e Validadores (`src/previsao_leitos/domain/types.py`)

| Símbolo | Contrato de Validação | Exemplo Válido | Exemplo Rejeitado |
| :--- | :--- | :--- | :--- |
| `CodigoCNES` / `parse_codigo_cnes` | Exatamente 7 dígitos numéricos; preserva zeros à esquerda e remove espaços laterais | `"0012345"` | `"123456"`, `"223760A"` |
| `CodigoMunicipioIBGE` / `parse_codigo_municipio_ibge` | Exatamente 7 dígitos numéricos com prefixo de UF oficial do IBGE | `"4314902"` (Porto Alegre/RS) | `"431490"`, `"9914902"` |
| `CodigoUF` / `parse_codigo_uf` | Código oficial IBGE de 2 dígitos (27 UFs) | `"43"` (RS) | `"99"`, `"RS"` |
| `CompetenciaMensal` / `parse_competencia_mensal` | Dataclass imutável e ordenável cronologicamente (`1990 <= ano <= 2100`, `1 <= mes <= 12`) com `.as_yyyymm()` | `CompetenciaMensal(2024, 5)` → `"202405"` | `(2024, 0)`, `(2024, 13)` |

### 2.2 Hierarquia Única de Exceções (`src/previsao_leitos/domain/exceptions.py`)

```text
PrevisaoLeitosError
├── ConfigurationError      # Variáveis de ambiente, caminhos ou parâmetros inválidos
├── DataContractError       # Schema, chave, cardinalidade de join ou unidade física inválida
├── LeakageViolationError   # Tentativa de uso de informação futura (ex-post) ou quebra de split
└── SourceUnavailableError  # Falha de conectividade ou acesso em fonte externa
```

### 2.3 Verificador de Regras e Pipeline CI

- **`scripts/check_constraints.py`**: audita arquivos Python contra supressões silenciosas (`warnings.filterwarnings("ignore")`, `except: pass`, `# noqa`/`# type: ignore` sem código), `allow_pickle=True` e SQL montado por f-string dentro de `src/`.
- **`.github/workflows/ci.yml`**: executa em paralelo lint (`ruff`), tipagem estrita (`mypy`), verificador de regras, import limpo, `pytest` com cobertura mínima de `80%`, `pip-audit` e varredura de segredos (`gitleaks`).

---

## 3. Como Validar Localmente

```bash
uv sync --python 3.12 --extra dev
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/check_constraints.py src tests
uv run pytest -m "not slow" --cov=src/previsao_leitos --cov-fail-under=80
uv run pip-audit
```

---

## 4. Checklist da Feature F00

- [x] Definir tipos nominais (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`) com validação na fronteira
- [x] Criar hierarquia única de exceções de domínio
- [x] Garantir `import previsao_leitos` livre de efeitos colaterais e variáveis de ambiente
- [x] Cobrir `100%` das linhas e branches com testes determinísticos (`pytest`)
- [x] Calibrar `scripts/check_constraints.py` (rejeitando violações conhecidas no protótipo legado e aprovando `src/` e `tests/`)
- [x] Configurar CI automatizado (`.github/workflows/ci.yml`) e atualizações de dependências (`.github/dependabot.yml`)
