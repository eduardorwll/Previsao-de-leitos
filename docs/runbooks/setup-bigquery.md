# Guia Rápido — Configuração Segura do Google BigQuery (Base dos Dados)

Este guia explica como obter seu `GCP_PROJECT_ID`, autenticar-se localmente **sem criar arquivos `.json` de chave privada** e testar o acesso gratuito às tabelas públicas da **Base dos Dados** (`basedosdados.br_ms_cnes`, `basedosdados.br_ms_sih`, etc.).

> **Custo:** O Google BigQuery oferece **1 TB gratuito por mês** de consultas processadas (*free tier*). As tabelas públicas da `basedosdados` não cobram armazenamento do seu projeto — você paga apenas pelos bytes processados nas suas queries (protegidos pelo teto `PREVISAO_MAX_BQ_MB` e por `dry_run=True` no pipeline).

---

## Passo 1 — Criar ou Localizar seu `GCP_PROJECT_ID`

1. Acesse [console.cloud.google.com](https://console.cloud.google.com/) com sua conta Google.
2. No seletor de projetos no topo da barra azul, clique em **Novo Projeto** (ou escolha um projeto existente de pesquisa, ex.: `previsao-leitos-pesquisa`).
3. Copie o **ID do projeto** (*Project ID* — ex.: `previsao-leitos-123456`, **não** o nome exibido nem o número do projeto).
4. No menu lateral, vá em **APIs e serviços → Biblioteca**, procure por **BigQuery API** e clique em **Ativar** (se ainda não estiver ativa).
5. *(Recomendado)* Em **Faturamento → Orçamentos e alertas**, configure um alerta preventivo caso habilite conta de faturamento (no modo *BigQuery Sandbox* gratuito nem cartão de crédito é obrigatório para até 1 TB/mês).

---

## Passo 2 — Autenticar Localmente via ADC (Sem Baixar Chave `.json`)

O método mais seguro é **Application Default Credentials (ADC)** via Google Cloud CLI (`gcloud`), evitando arquivos de chave de conta de serviço no disco.

### 2.1 Instalar a CLI `gcloud` (Linux)

- **Arch Linux / derivadas:**
  ```bash
  # Via AUR (ex.: yay ou paru) ou pacote oficial tarball abaixo
  yay -S google-cloud-cli
  ```
- **Ubuntu / Debian:**
  ```bash
  sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
  sudo apt-get update && sudo apt-get install -y google-cloud-cli
  ```
- **Qualquer Linux (sem `sudo`, direto na pasta do usuário):**
  ```bash
  curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
  tar -xf google-cloud-cli-linux-x86_64.tar.gz -C "$HOME"
  "$HOME/google-cloud-sdk/install.sh"
  ```

### 2.2 Fazer Login e Definir Projeto de Cobrança (*Quota Project*)

```bash
# 1. Abre o navegador para autenticar sua conta Google localmente
gcloud auth application-default login

# 2. Define o projeto que contabilizará a cota gratuita do BigQuery
gcloud auth application-default set-quota-project SEU_GCP_PROJECT_ID
```

As credenciais ficam armazenadas fora do repositório em `~/.config/gcloud/application_default_credentials.json`.

---

## Passo 3 — Configurar o `.env` Local (Nunca Commitado)

Na raiz do repositório:

```bash
cp .env.example .env
```

Edite `.env` com seu editor local:

```dotenv
GCP_PROJECT_ID=SEU_GCP_PROJECT_ID
PREVISAO_DATA_DIR=data
PREVISAO_ARTIFACTS_DIR=artifacts
PREVISAO_LOG_LEVEL=INFO
PREVISAO_MAX_BQ_MB=2000
```

> **Atenção:** `.env` já está listado no `.gitignore` — jamais faça `git add -f .env`.

### Alternativa no Google Colab

Se for executar em notebook do Colab:
1. Clique no ícone de **chave (Secrets)** na barra lateral esquerda.
2. Adicione o segredo `GCP_PROJECT_ID` com o ID do seu projeto e ative acesso ao notebook.
3. Autentique a sessão com `from google.colab import auth; auth.authenticate_user()`.

---

## Passo 4 — Teste Gratuito de Conectividade (`0 bytes` cobrados)

Após instalar os extras de dados (`uv sync --python 3.12 --extra dev --extra data`), valide a conexão listando apenas os metadados de schema de `br_ms_cnes.leito` (operação 100% gratuita):

```bash
uv run python -c "
import os
from google.cloud import bigquery

project_id = os.environ['GCP_PROJECT_ID']
client = bigquery.Client(project=project_id)
table = client.get_table('basedosdados.br_ms_cnes.leito')
print('Conexão OK! Colunas em br_ms_cnes.leito:', [f.name for f in table.schema])
"
```

Se preferir testar carregando o `.env` automaticamente no shell:

```bash
set -a && source .env && set +a
```
