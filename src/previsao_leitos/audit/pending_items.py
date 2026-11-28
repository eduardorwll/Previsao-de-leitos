"""Pontos pendentes e acessos a dados ainda não resolvidos (fonte única de verdade).

Cada item lista o que ficou para ser estudado/destravado depois, o estado atual e
a alavanca que ele abre. Consumido pelos relatórios PDF (Capítulo 6 do dossiê e
seção de limitações do benchmark) e pela documentação em
`docs/runbooks/acesso-bases-pendentes.md`.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingItem:
    """Um ponto pendente: base/tópico, estado atual e por que importa."""

    title: str
    status: str
    note: str


#: Catálogo canônico de pendências (dados + análises). Mantenha sincronizado com
#: `docs/runbooks/acesso-bases-pendentes.md`.
PENDING_ITEMS: tuple[PendingItem, ...] = (
    PendingItem(
        title="SIVEP-Gripe (SRAG viral)",
        status="solicitação enviada; aguardando resposta (perfil de leitor estadual)",
        note=(
            "indicador líder da onda respiratória epidêmica (influenza/COVID/VSR); as colunas "
            "srag_viral_confirmado/frac_srag_viral já são previstas no two_stage.py, mas estão "
            "ausentes do painel — é a alavanca nº 1 para reduzir o erro de fase (4,3-7,4 sem.)."
        ),
    ),
    PendingItem(
        title="SINAN (Leptospirose)",
        status="acesso pendente (dengue já extraída; leptospirose aguarda)",
        note=(
            "fonte da Onda 2 pós-enchente (leptospirose grave com insuficiência renal/UTI), "
            "que pressiona enfermarias e UTIs de D+7 a D+30."
        ),
    ),
    PendingItem(
        title="S2iD / FIDE (MIDR)",
        status="acesso pendente (download CSV via Playwright não dispara)",
        note=(
            "decretos de emergência/calamidade e danos humanos (desabrigados, enfermos, danos a "
            "hospitais) — sinal de deslocamento populacional e dano à rede assistencial."
        ),
    ),
    PendingItem(
        title="ANA HidroWeb API v2",
        status="acesso concedido; credenciais mantidas (fallback Zenodo se 401 persistir)",
        note=(
            "cotas e vazões fluviométricas (sinal físico real de enchente); o fallback Zenodo "
            "IPH-UFRGS segue como fonte enquanto a API ao vivo (instabilidade de banco reportada "
            "pela ANA) não for revalidada."
        ),
    ),
    PendingItem(
        title="Erro de fase epidêmico (4,3-7,4 semanas)",
        status="investigado; causa raiz = timing epidêmico (não sazonal fixo)",
        note=(
            "a semana de pico observada varia 13-22 semanas entre anos (UTI: 18,4 em 2021 a 40,2 "
            "em 2020); o harmônico único não acompanha o deslocamento — depende de SIVEP-Gripe ou "
            "de modelo TFT/GNN (V2) para ser atacado."
        ),
    ),
    PendingItem(
        title="Trilha ENSO E1/E2/E3 (RONI, SOI, BOM semanal)",
        status="3/3 rejeitadas na régua (redundantes com el_nino_intensity)",
        note=(
            "índices ENSO alternativos não injetam sinal preditivo novo em h=2; E4 (índice "
            "consenso) permanece na fila, com baixa expectativa de ganho."
        ),
    ),
    PendingItem(
        title="Promoção da catraca (Task 1.3.7)",
        status="reprovada no critério de fase (pico ≤ 2 sem.)",
        note=(
            "campeão gbdt_conformal em need_*: MASE 0,840-0,959, cobertura 78,3-78,9% (no piso) "
            "e erro de pico 4,3-7,4 sem.; congelamento adiado até a fase ser atacada."
        ),
    ),
)


def render_pending_items_html() -> str:
    """Renderiza os pontos pendentes como lista HTML (sem Markdown cru)."""
    items = "".join(
        f"<li><strong>{html.escape(item.title)}</strong> — {html.escape(item.status)}. "
        f"{html.escape(item.note)}</li>"
        for item in PENDING_ITEMS
    )
    return f"<ul class='notes'>{items}</ul>"
