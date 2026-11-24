# ADR-0005: Camada de Dados Parametrizada (BigQuery + Polars + DataLad) e Fronteira LGPD

- **Status:** ACCEPTED
- **Data:** 2026-11-24

## Contexto e Decisão

O script legado executa SQL BigQuery via f-string, escaneia tabelas amplas sem `dry_run`, usa cache Parquet sem chave de parâmetros nem escrita atômica, depende de um `flood_dataset_2010_2025.parquet` órfão fora do repositório e cobre apenas ~10% dos municípios brasileiros (estações INMET). Além disso, a entrada em dados de internação (`SIH-SUS`) exige salvaguardas estritas da LGPD.

**Decisão:**
1. **Arquitetura Medallion (Bronze/Silver/Gold)** sobre Parquet particionado lido via **`polars`** (*lazy/streaming*, preservando zeros à esquerda em códigos IBGE/CNES) e versionado em layout **YODA (`datalad`)** para reprodutibilidade (`datalad rerun`).
2. **Acesso BigQuery seguro:** 100% parametrizado (`ScalarQueryParameter`), checagem prévia de custo (`dry_run=True`), teto configurável (`PREVISAO_MAX_BQ_MB`), colunas explícitas e cache atômico indexado por SHA-256 de `(query + parâmetros + versão de schema)`.
3. **Cobertura territorial de 100%:** substituir dependência exclusiva de estações pontuais INMET por grade contínua (`MERGE-CPTEC` / `ERA5-Land`) integrada a cotas físicas (`ANA/SGB`), relevo/inundação (`Sentinel-1 SAR` + `DEM/HAND` em `SIRGAS 2000`) e malha viária (`OSMnx`).
4. **Fronteira LGPD inegociável:** microdados hospitalares (`SIH-SUS`, `SINAN`, `GERINT`) são **sempre agregados na consulta de origem** (zero colunas identificadoras/PHI no disco local), passando por verificação automática de **supressão primária e complementar de pequenas células** ($n < 5$).

## Alternativas Rejeitadas

- **Pandas eager em memória com arquivo Parquet monolítico:** propenso a OOM em máquina local (6,9 GiB RAM) e lento nos rolling windows.
- **Baixar AIHs individuais com dados demográficos finos localmente:** eleva superfície de risco LGPD sem benefício para modelagem de capacidade hospitalar agregada.

## Consequências

Exige definição explícita de contratos `pandera` e testes com adaptador local (`ParquetFixtureAdapter`) para que todo o CI rode rápido e sem credenciais GCP.
