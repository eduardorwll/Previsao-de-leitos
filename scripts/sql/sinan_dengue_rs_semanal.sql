-- Agregação semanal de notificações de dengue no RS (LGPD: apenas contagens).
-- `semana_sintomas` vem como "YYYY-WW"; extraímos a semana epidemiológica.
SELECT
  ano,
  SAFE_CAST(SPLIT(semana_sintomas, '-')[SAFE_OFFSET(1)] AS INT64) AS semana,
  id_municipio_residencia AS id_municipio,COUNT(*) AS casos
FROM `basedosdados.br_ms_sinan.microdados_dengue`
WHERE ano BETWEEN 2010 AND 2024 AND sigla_uf_residencia = 'RS'
GROUP BY ano, semana, id_municipio
HAVING semana BETWEEN 1 AND 53
ORDER BY ano, semana, id_municipio
