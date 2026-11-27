-- Agregação semanal de notificações de dengue no RS (LGPD: apenas contagens).
SELECT
  ano,
  semana_sintomas,
  id_municipio_residencia,
  COUNT(*) AS casos
FROM `basedosdados.br_ms_sinan.microdados_dengue`
WHERE ano BETWEEN 2010 AND 2024
  AND sigla_uf_residencia = 'RS'
  AND semana_sintomas BETWEEN 1 AND 53
GROUP BY ano, semana_sintomas, id_municipio_residencia
ORDER BY ano, semana_sintomas, id_municipio_residencia
