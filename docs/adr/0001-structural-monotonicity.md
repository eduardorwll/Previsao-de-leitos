# ADR-0001: Restrição Monotônica Estrutural Real vs. Penalidade Suave (`lambda_pinn`)

- **Status:** ACCEPTED
- **Data:** 2026-11-24

## Contexto e Decisão

No protótipo inicial, penalidades suaves de perda (`lambda_pinn`) degradavam o desempenho preditivo sem eliminar inversões causais no uso do solo, enquanto `mun_embedding` permitia memorizar cidades. Adotamos **restrição monotônica estrutural via pesos estritamente positivos (`F.softplus`)** para impermeabilização (+) e cobertura vegetal (-), removendo `mun_embedding`. Porém, a auditoria revelou dois vazamentos arquiteturais no código legado (`previsao_leitos.py`): (1) `indice_impermeabilidade` também entrava em `escoamento_efetivo_mm` dentro do ramo dinâmico irrestrito (quebrando a garantia matemática de monotonicidade), e (2) `level_head` não recebia o ramo monotônico, enquanto coordenadas cruas `(lat, lon, altitude)` em um MLP de 64 dimensões reintroduziam memorização de estações.

**Decisão:** Manter o ramo monotônico estrutural `F.softplus` aplicado a **todas** as cabeças de risco/severidade, **removendo `indice_impermeabilidade` das entradas dinâmicas irrestritas** e substituindo coordenadas geográficas cruas por descritores geomorfológicos/climáticos físicos (`HAND`, declividade, área de drenagem a montante, regime pluviométrico da bacia, exposição em áreas aterradas/baixo `HAND` e impermeabilização ponderada a montante — ver [ADR-0006](0006-landfill-polders-and-intermunicipal-routing.md)), verificados por teste unitário de derivada parcial ($\partial \hat{y}/\partial x_{\text{imp/aterro/montante}} \ge 0$, $\partial \hat{y}/\partial x_{\text{veg}} \le 0$).

## Alternativas Rejeitadas

- **Penalidade suave `lambda_pinn` na loss:** já demonstrou degradação preditiva e não garante monotonicidade fora do suporte de treino.
- **Manter `escoamento_efetivo_mm` com impermeabilidade no ramo irrestrito:** anula a garantia estrutural documentada.

## Consequências

Garantia matemática real verificável via *pytest* + *Mutation Probe*; requer recalcular o termo de saturação dinâmica apenas com variáveis meteorológicas/hidrológicas puras antes da combinação no ramo monotônico.
