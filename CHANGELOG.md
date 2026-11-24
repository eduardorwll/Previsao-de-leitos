# CHANGELOG.md

Histórico de mudanças orientado ao impacto para pesquisadores e mantenedores ([Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)).

## [Unreleased]

### Added
- **Etapa 0 / Feature F00 (`domain`):** Empacotamento Python 3.12 (`pyproject.toml`, `.python-version`, `uv.lock`), tipos nominais validados (`CodigoCNES`, `CodigoMunicipioIBGE`, `CodigoUF`, `CompetenciaMensal`), hierarquia única de exceções (`src/previsao_leitos/domain/`) e 17 testes unitários com `100%` de cobertura de linhas e branches (`tests/unit/test_domain_types.py`).
- **Qualidade e CI:** Verificador mecânico de regras (`scripts/check_constraints.py`), pipeline CI rápido CPU-only (`.github/workflows/ci.yml`: `ruff`, `mypy`, `pytest --cov`, `pip-audit`, `gitleaks`), Dependabot (`.github/dependabot.yml`) e `.env.example`.
- **Documentação Técnica e Científica:** Painel de progresso por etapa e feature (`docs/PROGRESSO.md`), documentação da feature F00 (`docs/features/00-dominio-e-fundacao.md`), catálogo de 14 bases de dados (`docs/data-catalog.md`), mapa de capacidades (`spec/capability-map.md`), glossário em 3 contextos (`GLOSSARY-MAP.md` + `docs/glossaries/`), 5 ADRs (`docs/adr/0001–0005`), padrões de qualidade (`CONSTRAINTS.md`), diário de desempenho (`PERF.md`) e runbook de expurgo de segredos Git (`docs/runbooks/git-history-secret-purge.md`).

### Changed
- Arquivado o protótipo monolítico Colab em `legacy/previsao_leitos_colab.py` (via `git mv`, preservando histórico) para evitar conflito de importação com `src/previsao_leitos/`.
- Atualizado `README.md` com navegação para a documentação estruturada, instruções de início rápido e síntese metodológica das lacunas identificadas no protótipo inicial.

## [0.0.1] - 2026-11-23 (Protótipo Colab Legado)
- Versão inicial monolítica `previsao_leitos.py` (`HydroTFTNet` + delta mensal CNES via `HistGradientBoostingRegressor`) e relatório exploratório `README.md`.
