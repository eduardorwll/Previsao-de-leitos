# 7. Como ler um cartão de modelo

## 7.1 Anatomia

```
# <modelo>
Selo geral · rubrica · família

Como ler estas métricas     ← glossário das 5 métricas do cartão
Resultados por alvo         ← wMAPE, MASE, fase, cobertura, P90, subestimação
Categorias                  ← letra por quesito + selo por alvo
Leitura                     ← selo geral e ponto mais fraco
Reprodutibilidade           ← run, git SHA, seed, SHA-256 do painel
```

## 7.2 Ordem de leitura recomendada

1. **MASE**: bate a persistência? Se ≥ 1, o modelo não se paga.
2. **Fase**: avisa o pico? Acima de 4 semanas, não serve para planejar.
3. **Cobertura**: o intervalo é honesto? Fora de 78–82%, recalibrar.
4. **P90 e subestimação**: qual o pior caso e ele tende a subestimar?
5. **Selo**: pior dos três primeiros.

## 7.3 Erros comuns de leitura

| Leitura errada | Por que está errada |
| :--- | :--- |
| "wMAPE 8% é ótimo, promove" | pode ter fase de 8 semanas e cobertura de 60% |
| "MASE 0,95 em UTI vale para clínico" | unidades e dinâmicas diferentes; nunca comparar entre alvos |
| "cobertura 92% é melhor que 80%" | intervalo largo demais: pouco informativo |
| "2024 foi bom, então generaliza" | 2024 é ano de desenvolvimento; o holdout é o M2 |
| "ficou N/A, então é ruim" | dependência opcional ausente, não desempenho |

## 7.4 Regras de decisão

- **Promoção:** MASE mediano ≤ campeão − 3%, nenhum fold pior que +5%, cobertura 78–82%,
  fase ≤ 2 semanas, `0` violações.
- **Recalibrar:** cobertura fora da faixa, mantendo MASE.
- **Rejeitar:** fase > 6 semanas ou acurácia pior que a persistência em algum fold relevante.
- **Investigar:** selo D em um único alvo com selo A nos demais.
