# Runbook — Expurgo Seguro de `.env` do Histórico Git (`b3ffad6`)

> [!CAUTION]
> **Regra de Segurança:** Um valor que chegou a um repositório remoto (`origin/master`) deve ser considerado exposto. **Primeiro** audite/rotacione ou restrinja permissões IAM no projeto GCP; **somente depois** reescreva o histórico Git e faça force-push coordenado.

## Contexto

O commit `b3ffad6` (`Add GCP project ID to .env file`) adicionou `.env` contendo `GCP_PROJECT_ID`. Embora o commit `d395a41` (`Delete .env`) tenha removido o arquivo do `HEAD`, o blob permanece acessível no histórico Git (`git show b3ffad6:.env`).

## Passo 1 — Contenção no GCP (Antes de mexer no Git)

1. Verifique no console do Google Cloud se o projeto referenciado em `b3ffad6:.env` possui:
   - Restrições de IAM (nenhum acesso público/não autenticado ao billing ou BigQuery).
   - Orçamentos e alertas de faturamento ativos (*Budgets & Alerts*).
   - Se desejar isolar completamente o identificador antigo, crie um novo projeto GCP de pesquisa (ou rotacione quaisquer chaves associadas, caso existam fora deste arquivo) e atualize seu `.env` local (nunca commitado).

## Passo 2 — Backup Local de Segurança

```bash
# Crie um clone espelho fora do diretório de trabalho antes de reescrever histórico
git clone --mirror /home/eduardorwll/Projects/Previsao-de-leitos /tmp/previsao-leitos-backup.git
```

## Passo 3 — Reescrita do Histórico com `git-filter-repo` (via `uvx`)

> Execute em um clone fresco ou com working tree 100% limpo (`git status`).

```bash
# 1. Garanta que não há alterações não commitadas
git status

# 2. Remova `.env` de TODOS os commits do histórico
uvx git-filter-repo --path .env --invert-paths --force

# 3. Verifique se `.env` sumiu completamente do histórico
git log --all --full-history -- .env
# Saída esperada: vazia
```

## Passo 4 — Verificação com Scanner de Segredos (`gitleaks --redact`)

```bash
# Caso gitleaks esteja instalado localmente (ou via CI após push em branch de teste):
gitleaks git . --redact --no-banner
```

## Passo 5 — Reconexão do Remote e Force-Push Seguro

> `git-filter-repo` remove o `origin` por segurança para evitar push acidental.

```bash
# 1. Reconfigure o remote (substitua pela URL real do repositório)
git remote add origin https://github.com/GuilhermeSPedroso07/Previsao-de-leitos.git

# 2. Faça push seguro com lease após avisar todos os colaboradores
git push --force-with-lease origin master
```

## Passo 6 — Pós-Expurgo (Obrigatório)

1. Avisar todos os colaboradores para refazerem `git fetch origin && git reset --hard origin/master` (ou novo `git clone`) para não reintroduzirem o commit `b3ffad6`.
2. Se o repositório estiver no GitHub, solicitar limpeza de objetos em cache/commits órfãos caso necessário (GitHub Support) e conferir se nenhuma PR/issue exibe o valor antigo.
3. Remover o backup temporário (`/tmp/previsao-leitos-backup.git`) após confirmar a integridade do repositório remoto.
