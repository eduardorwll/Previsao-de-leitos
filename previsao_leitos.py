# @title 1. Autenticação no Google Cloud e Imports
import os
import warnings
from typing import Dict, List, Tuple

from google.cloud import bigquery
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

try:
  from google.colab import auth

  auth.authenticate_user()
  print("[COLAB] Autenticado com sucesso no Google Cloud.")
except ImportError:
  print("[LOCAL] Usando credenciais locais do GCP.")


# ==============================================================================
# @title 2. Extração Nacional (CNES sem Indicadores + População IBGE + Área km²)
# ==============================================================================

def get_gcp_project_id() -> str:
  # 1. Tenta ler dos Secrets do Google Colab (ícone de chave na barra lateral esquerda)
  try:
    from google.colab import userdata

    project_id = userdata.get("GCP_PROJECT_ID")
    if project_id:
      return project_id
  except Exception:
    pass

  # 2. Fallback para variável de ambiente local (.env ou terminal)
  project_id = os.getenv("GCP_PROJECT_ID")
  if not project_id:
    raise ValueError(
        "Defina 'GCP_PROJECT_ID' nos Secrets do Colab ou no arquivo .env local."
    )
  return project_id


# Uso nas chamadas do pipeline:
GCP_PROJECT_ID = get_gcp_project_id()

def extract_cnes_ibge_area_br(
    project_id: str,
    ano_inicio: int = 2010,
    ano_fim: int = 2025,
    cache_path: str = "cnes_ibge_area_br_2010_2025.parquet",
    max_mb_scanned: int = 12000,
) -> Tuple[pd.DataFrame, List[str]]:
  """Extrai para TODO O BRASIL (2010–2025):

  - Leitos mensais por área hospitalar (CNES), EXCLUINDO colunas 'indicador_*'
  - População municipal anual (IBGE) com carry-forward para 2024–2025
  - Área territorial do município em km² (MapBiomas / 100)
  """
  if os.path.exists(cache_path):
    df_cached = pd.read_parquet(cache_path)
    leito_cols = [c for c in df_cached.columns if c.startswith("leitos_")]
    print(
        f"[CACHE BR] Carregado {cache_path} ({len(df_cached):,} linhas |"
        f" {len(leito_cols)} áreas hospitalares)."
    )
    return df_cached, leito_cols

  client = bigquery.Client(project=project_id)
  cnes_table = "basedosdados.br_ms_cnes.estabelecimento"

  # 1. Descoberta gratuita (0 bytes) apenas de contagens reais de leitos (sem indicadores)
  table_meta = client.get_table(cnes_table)
  raw_leito_cols = [
      f.name
      for f in table_meta.schema
      if "leito" in f.name.lower()
      and not f.name.lower().startswith("indicador")
  ]
  print(f"[SCHEMA CNES LIMPO] {len(raw_leito_cols)} áreas de leitos: {raw_leito_cols}")

  agg_exprs = []
  clean_names = []
  for col in raw_leito_cols:
    alias = col.replace("quantidade_", "").replace("leito_", "leitos_")
    if not alias.startswith("leitos_"):
      alias = f"leitos_{alias}"
    agg_exprs.append(f"SUM(IFNULL(CAST({col} AS INT64), 0)) AS {alias}")
    clean_names.append(alias)

  sql_query = f"""
    WITH cnes_mensal AS (
        SELECT
            TRIM(CAST(id_municipio AS STRING)) AS id_municipio,
            CAST(ano AS INT64) AS ano,
            CAST(mes AS INT64) AS mes,
            {", ".join(agg_exprs)}
        FROM `{cnes_table}`
        WHERE ano BETWEEN {ano_inicio} AND {ano_fim}
          AND id_municipio IS NOT NULL
        GROUP BY id_municipio, ano, mes
    ),
    pop_ibge AS (
        SELECT
            TRIM(CAST(id_municipio AS STRING)) AS id_municipio,
            CAST(ano AS INT64) AS ano,
            CAST(populacao AS FLOAT64) AS populacao
        FROM `basedosdados.br_ibge_populacao.municipio`
        WHERE ano BETWEEN {ano_inicio} AND {ano_fim}
    ),
    area_mun AS (
        SELECT
            TRIM(CAST(id_municipio AS STRING)) AS id_municipio,
            MAX(area_total_ha) / 100.0 AS area_km2
        FROM (
            SELECT
                TRIM(CAST(id_municipio AS STRING)) AS id_municipio,
                ano,
                SUM(area) AS area_total_ha
            FROM `basedosdados.br_mapbiomas_estatisticas.cobertura_municipio_classe`
            WHERE ano BETWEEN 2020 AND 2024
            GROUP BY id_municipio, ano
        )
        GROUP BY id_municipio
    )
    SELECT
        c.*,
        LAST_VALUE(p.populacao IGNORE NULLS) OVER (
            PARTITION BY c.id_municipio ORDER BY c.ano, c.mes
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS populacao,
        a.area_km2
    FROM cnes_mensal c
    LEFT JOIN pop_ibge p ON c.id_municipio = p.id_municipio AND c.ano = p.ano
    LEFT JOIN area_mun a ON c.id_municipio = a.id_municipio
    ORDER BY c.id_municipio, c.ano, c.mes
    """

  run_cfg = bigquery.QueryJobConfig(
      maximum_bytes_billed=max_mb_scanned * (1024**2), use_query_cache=True
  )
  print("[BIGQUERY BR] Extraindo CNES (limpo) + População IBGE + Área (km²)...")
  df_cnes = client.query(sql_query, job_config=run_cfg).to_dataframe()
  df_cnes["populacao"] = df_cnes["populacao"].fillna(
      df_cnes.groupby("id_municipio")["populacao"].transform("median")
  )
  df_cnes["area_km2"] = df_cnes["area_km2"].clip(lower=1.0)
  df_cnes["densidade_hab_km2"] = df_cnes["populacao"] / df_cnes["area_km2"]
  df_cnes.to_parquet(cache_path, index=False)
  print(f"[BIGQUERY BR] Concluído: {len(df_cnes):,} linhas município-mês.")
  return df_cnes, clean_names


# ==============================================================================
# @title 3. Engenharia Hidrológica + Índice de Nível de Alagamento
# ==============================================================================

BASE_DYNAMIC_COLS = [
    "precip_dia_mm",
    "precip_max_horaria_mm",
    "precip_acum_3d",
    "precip_acum_7d",
    "precip_acum_15d",
    "api_indice_saturacao",
    "escoamento_efetivo_mm",
    "temp_media_c",
    "temp_max_c",
    "umidade_rel_media",
    "pressao_media_hpa",
    "doy_sin",
    "doy_cos",
]

STATIC_SOIL_COLS = [
    "indice_impermeabilidade",
    "indice_retencao_vegetal",
    "altitude_estacao",
    "latitude",
    "longitude",
]


def prepare_national_climate_data(
    climate_parquet: str = "flood_dataset_2010_2025.parquet",
) -> pd.DataFrame:
  df = pd.read_parquet(climate_parquet)
  df["data"] = pd.to_datetime(df["data"])
  df["id_municipio"] = df["id_municipio"].astype(str).str.strip()
  df["uf_code"] = df["id_municipio"].str[:2]
  df = df.sort_values(["id_municipio", "data"]).reset_index(drop=True)

  doy = df["data"].dt.dayofyear.to_numpy(dtype=np.float32)
  df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
  df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

  # API (saturação antecedente)
  k_decay = 0.88
  p_all = df["precip_dia_mm"].to_numpy(dtype=np.float32)
  mun_codes = df["id_municipio"].to_numpy()
  api_all = np.zeros(len(df), dtype=np.float32)
  same_mun = mun_codes[1:] == mun_codes[:-1]
  for t in range(1, len(p_all)):
    if same_mun[t - 1]:
      api_all[t] = k_decay * api_all[t - 1] + p_all[t - 1]
  df["api_indice_saturacao"] = api_all

  grp = df.groupby("id_municipio")
  for w in [3, 7, 15]:
    df[f"precip_acum_{w}d"] = grp["precip_dia_mm"].transform(
        lambda s: s.rolling(w, min_periods=1).sum()
    )

  sat_norm = np.tanh(df["api_indice_saturacao"] / 120.0)
  imp = df["indice_impermeabilidade"].fillna(0.05)
  df["escoamento_efetivo_mm"] = df["precip_dia_mm"] * (
      imp + (1.0 - imp) * sat_norm
  )

  df["alvo_janela_3d"] = (
      grp["alvo_enchente"]
      .transform(lambda s: s.rolling(3, min_periods=1).max().shift(-2))
      .fillna(0)
      .astype(np.int64)
  )

  # Índice Contínuo de Nível/Intensidade de Alagamento (proxy hidrológico em eventos)
  df["nivel_alagamento_obs"] = np.log1p(
      df["alvo_janela_3d"]
      * df["escoamento_efetivo_mm"]
      * (df["precip_acum_3d"] / 10.0)
  ).astype(np.float32)

  for col in ["latitude", "longitude"]:
    if col not in df.columns:
      df[col] = 0.0

  return df.dropna(subset=BASE_DYNAMIC_COLS + STATIC_SOIL_COLS).reset_index(
      drop=True
  )


# ==============================================================================
# @title 4. Dataset 2D Indexado e HydroTFTNet (Prevê Probabilidade + Nível)
# ==============================================================================


class IndexedFloodDataset(Dataset):

  def __init__(
      self, df: pd.DataFrame, uf_to_idx: Dict[str, int], seq_len: int = 15
  ):
    self.seq_len = seq_len
    self.dyn_2d = df[BASE_DYNAMIC_COLS].to_numpy(dtype=np.float32)
    self.stat_2d = df[STATIC_SOIL_COLS].to_numpy(dtype=np.float32)
    self.uf_1d = df["uf_code"].map(uf_to_idx).fillna(0).to_numpy(dtype=np.int64)
    self.ycls_1d = df["alvo_janela_3d"].to_numpy(dtype=np.int64)
    self.yniv_1d = df["nivel_alagamento_obs"].to_numpy(dtype=np.float32)

    muns = df["id_municipio"].to_numpy()
    valid_end_idx = []
    start, n = 0, len(muns)
    while start < n:
      end = start + 1
      while end < n and muns[end] == muns[start]:
        end += 1
      if end - start >= seq_len:
        valid_end_idx.append(
            np.arange(start + seq_len - 1, end, dtype=np.int64)
        )
      start = end

    self.end_indices = np.concatenate(valid_end_idx)
    self.targets_cls = self.ycls_1d[self.end_indices]
    self.meta_df = df.iloc[self.end_indices][
        ["id_municipio", "uf_code", "data", "ano", "alvo_janela_3d", "nivel_alagamento_obs"]
    ].reset_index(drop=True)

  def __len__(self):
    return len(self.end_indices)

  def __getitem__(self, idx):
    end_i = self.end_indices[idx]
    return (
        torch.from_numpy(self.dyn_2d[end_i - self.seq_len + 1 : end_i + 1]),
        torch.from_numpy(self.stat_2d[end_i]),
        torch.tensor(self.uf_1d[end_i]),
        torch.tensor(self.ycls_1d[end_i]),
        torch.tensor(self.yniv_1d[end_i]),
    )


class GatedResidualNetwork(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim=None, context_dim=None, dropout=0.2):
    super().__init__()
    output_dim = output_dim or input_dim
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
    self.fc2 = nn.Linear(hidden_dim, output_dim * 2)
    self.dropout = nn.Dropout(dropout)
    self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    self.norm = nn.LayerNorm(output_dim)

  def forward(self, x, context=None):
    h = self.fc1(x)
    if self.context_proj is not None and context is not None:
      h = h + self.context_proj(context)
    h = self.dropout(self.fc2(F.elu(h)))
    val, gate = torch.chunk(h, 2, dim=-1)
    return self.norm(self.skip(x) + val * torch.sigmoid(gate))


class VariableSelectionNetwork(nn.Module):

  def __init__(self, num_vars, hidden_dim, context_dim=None, dropout=0.2):
    super().__init__()
    self.num_vars = num_vars
    self.var_embeds = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_vars)])
    self.var_grns = nn.ModuleList([GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_vars)])
    self.weight_grn = GatedResidualNetwork(num_vars * hidden_dim, hidden_dim, num_vars, context_dim, dropout)

  def forward(self, x, context=None):
    emb = [self.var_embeds[i](x[..., i : i + 1]) for i in range(self.num_vars)]
    weights = F.softmax(self.weight_grn(torch.cat(emb, dim=-1), context), dim=-1)
    proc = torch.stack([self.var_grns[i](emb[i]) for i in range(self.num_vars)], dim=-2)
    return torch.sum(proc * weights.unsqueeze(-1), dim=-2), weights


class HydroTFTNet(nn.Module):

  def __init__(self, num_ufs: int, hidden_dim: int = 64, dropout: float = 0.2):
    super().__init__()
    self.uf_embedding = nn.Embedding(num_ufs + 1, 4)
    self.geo_proj = GatedResidualNetwork(7, hidden_dim, hidden_dim, dropout=dropout)
    self.c_sel = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
    self.c_enr = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
    self.c_h = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
    self.c_c = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)

    self.dyn_vsn = VariableSelectionNetwork(len(BASE_DYNAMIC_COLS), hidden_dim, hidden_dim, dropout)
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    self.post_lstm = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
    self.stat_enr = GatedResidualNetwork(
        hidden_dim, hidden_dim, output_dim=hidden_dim, context_dim=hidden_dim, dropout=dropout
    )
    self.attn = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
    self.post_attn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
    self.norm = nn.LayerNorm(hidden_dim)

    self.cls_head = nn.Linear(hidden_dim, 1)
    self.level_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    self.raw_w_imp = nn.Parameter(torch.tensor(-1.5))
    self.raw_w_veg = nn.Parameter(torch.tensor(-1.5))
    self.raw_w_int = nn.Parameter(torch.tensor(-2.0))

  def forward(self, x_dyn, x_stat, uf_idx):
    _, T, _ = x_dyn.shape
    s_emb = self.geo_proj(torch.cat([self.uf_embedding(uf_idx), x_stat[:, 2:5]], dim=-1))
    v_dyn, _ = self.dyn_vsn(x_dyn, context=self.c_sel(s_emb).unsqueeze(1).expand(-1, T, -1))
    lstm_out, _ = self.lstm(v_dyn, (self.c_h(s_emb).unsqueeze(0), self.c_c(s_emb).unsqueeze(0)))
    temp_f = self.post_lstm(lstm_out + v_dyn)
    enr = self.stat_enr(temp_f, context=self.c_enr(s_emb).unsqueeze(1).expand(-1, T, -1))
    attn_out, _ = self.attn(enr, enr, enr)
    h_last = self.norm(enr + self.post_attn(attn_out))[:, -1, :]

    w_imp, w_veg, w_int = F.softplus(self.raw_w_imp), F.softplus(self.raw_w_veg), F.softplus(self.raw_w_int)
    soil_eff = (w_imp + w_int * F.softplus(x_dyn[:, -1, 2])) * x_stat[:, 0] - w_veg * x_stat[:, 1]
    return self.cls_head(h_last).squeeze(-1) + soil_eff, self.level_head(h_last).squeeze(-1)


# ==============================================================================
# @title 5. Preditor de Leitos Hospitalares (2010–2023 -> Previsão 2024)
# ==============================================================================


def train_and_predict_beds_2024(
    df_cnes: pd.DataFrame,
    df_climate_monthly: pd.DataFrame,
    target_areas: List[str],
) -> pd.DataFrame:
  """Treina um HistGradientBoostingRegressor por área hospitalar em 2010–2023

  prevendo a VARIAÇÃO MENSAL (delta = atual - lag_1m) em vez do valor absoluto,
  eliminando o viés de subestimação nos polos regionais.
  """
  df = df_cnes.merge(
      df_climate_monthly, on=["id_municipio", "ano", "mes"], how="left"
  ).sort_values(["id_municipio", "ano", "mes"]).reset_index(drop=True)

  df["chuva_mes_mm"] = df["chuva_mes_mm"].fillna(0.0)
  df["eventos_mes"] = df["eventos_mes"].fillna(0.0)

  results_2024 = df[df["ano"] == 2024][
      ["id_municipio", "ano", "mes", "populacao", "area_km2", "densidade_hab_km2"]
  ].copy()

  print("\n=== PREVISÃO DE LEITOS EM 2024 VIA DELTA MENSAL (TREINO 2010–2023 vs. REAL 2024) ===")
  for area_col in target_areas:
    if area_col not in df.columns:
      continue

    grp = df.groupby("id_municipio")[area_col]
    df["lag_1m"] = grp.shift(1)
    df["lag_2m"] = grp.shift(2)
    df["lag_3m"] = grp.shift(3)
    df["lag_12m"] = grp.shift(12)

    # Features de variação e tendência (invariantes à escala da cidade)
    df["delta_lag1m"] = df["lag_1m"] - df["lag_2m"]
    df["tendencia_3m"] = (df["lag_1m"] - df["lag_3m"]) / 3.0
    df["tendencia_12m"] = (df["lag_1m"] - df["lag_12m"]) / 12.0
    df["leitos_por_1000hab_lag1"] = (df["lag_1m"] / df["populacao"].clip(lower=1.0)) * 1000.0

    # Alvo estacionário: variação mensal real
    df["target_delta"] = df[area_col] - df["lag_1m"]

    feats = [
        "mes",
        "populacao",
        "area_km2",
        "densidade_hab_km2",
        "chuva_mes_mm",
        "eventos_mes",
        "lag_1m",
        "delta_lag1m",
        "tendencia_3m",
        "tendencia_12m",
        "leitos_por_1000hab_lag1",
    ]

    sub = df.dropna(subset=["lag_1m", "lag_12m", "populacao", area_col, "target_delta"])
    tr = sub[sub["ano"] <= 2023]
    te = sub[sub["ano"] == 2024]

    # loss='absolute_error' (L1) reduz sensibilidade a outliers pontuais de recadastramento no CNES
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=6,
        max_iter=150,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(tr[feats], tr["target_delta"])

    pred_delta_24 = model.predict(te[feats])
    preds_24 = np.maximum(0.0, te["lag_1m"].to_numpy() + pred_delta_24)
    reais_24 = te[area_col].to_numpy()

    mae = mean_absolute_error(reais_24, preds_24)
    r2 = r2_score(reais_24, preds_24)
    vies_medio = float(np.mean(reais_24 - preds_24))
    print(
        f"Área {area_col:22s} | MAE: {mae:5.2f} leitos | R²: {r2:.4f} |"
        f" Viés Médio (Real-Prev): {vies_medio:+.3f}"
    )

    pred_df = te[["id_municipio", "ano", "mes", area_col]].copy()
    pred_df = pred_df.rename(columns={area_col: f"{area_col}_real"})
    pred_df[f"{area_col}_previsto"] = preds_24
    results_2024 = results_2024.merge(
        pred_df, on=["id_municipio", "ano", "mes"], how="left"
    )

  return results_2024


# ==============================================================================
# @title 6. Execução Principal: Treino 2010–2023 -> Auditoria Completa de 2024
# ==============================================================================


def run_2024_national_audit(
    gcp_project_id: str = GCP_PROJECT_ID,
    climate_parquet: str = "flood_dataset_2010_2025.parquet",
    epochs: int = 6,
):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[DISPOSITIVO] {device.upper()} | Auditoria Nacional 2024 (Treino: 2010–2023)")

  # 1. Carrega Clima Nacional + CNES Limpo + IBGE População + Área km²
  df_clim = prepare_national_climate_data(climate_parquet)
  df_cnes, leito_cols = extract_cnes_ibge_area_br(project_id=gcp_project_id)

  # 2. Normalização estrita em 2010–2023 (Zero Leakage de 2024)
  unique_ufs = sorted(df_clim["uf_code"].unique())
  uf_to_idx = {u: i + 1 for i, u in enumerate(unique_ufs)}

  df_tr_raw = df_clim[df_clim["ano"] <= 2023]
  dyn_m, dyn_s = df_tr_raw[BASE_DYNAMIC_COLS].mean().to_numpy(), df_tr_raw[BASE_DYNAMIC_COLS].std().to_numpy() + 1e-6
  stat_m, stat_s = df_tr_raw[STATIC_SOIL_COLS].mean().to_numpy(), df_tr_raw[STATIC_SOIL_COLS].std().to_numpy() + 1e-6

  df_norm = df_clim.copy()
  df_norm[BASE_DYNAMIC_COLS] = (df_clim[BASE_DYNAMIC_COLS] - dyn_m) / dyn_s
  df_norm[STATIC_SOIL_COLS] = (df_clim[STATIC_SOIL_COLS] - stat_m) / stat_s

  train_ds = IndexedFloodDataset(df_norm[df_norm["ano"] <= 2023], uf_to_idx, seq_len=15)
  test_2024_ds = IndexedFloodDataset(
      df_norm[(df_norm["data"] >= "2023-12-17") & (df_norm["ano"] <= 2024)],
      uf_to_idx,
      seq_len=15,
  )

  # WeightedRandomSampler (~8% positivos por batch)
  y_tr = train_ds.targets_cls
  weights = np.where(y_tr == 1, 0.08 / max(1, (y_tr == 1).sum()), 0.92 / max(1, (y_tr == 0).sum()))
  sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=250_000, replacement=True)
  train_loader = DataLoader(train_ds, batch_size=1024, sampler=sampler, num_workers=2)
  test_2024_loader = DataLoader(test_2024_ds, batch_size=2048, shuffle=False, num_workers=2)

  # 3. Treino do HydroTFTNet (Probabilidade + Nível de Alagamento)
  model = HydroTFTNet(num_ufs=len(unique_ufs), hidden_dim=64).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
  huber = nn.HuberLoss()

  print("\n--- Treinando HydroTFTNet Nacional (2010–2023) ---")
  for ep in range(1, epochs + 1):
    model.train()
    ep_loss = 0.0
    for x_dyn, x_stat, uf_idx, y_cls, y_niv in train_loader:
      opt.zero_grad()
      logits, pred_niv = model(x_dyn.to(device), x_stat.to(device), uf_idx.to(device))
      bce = F.binary_cross_entropy_with_logits(logits, y_cls.to(device).float(), reduction="none")
      p_t = torch.sigmoid(logits) * y_cls.to(device) + (1 - torch.sigmoid(logits)) * (1 - y_cls.to(device))
      focal = ((0.75 * y_cls.to(device) + 0.25 * (1 - y_cls.to(device))) * ((1 - p_t) ** 2) * bce).mean()
      loss = focal + 0.30 * huber(pred_niv, y_niv.to(device))
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
      ep_loss += loss.item()
    print(f"Epoch {ep:02d}/{epochs:02d} | Loss (Focal + Nível): {ep_loss / len(train_loader):.4f}")

  # 4. Previsão Out-of-Time de 2024 (Brasil Inteiro)
  model.eval()
  p_list, niv_list = [], []
  with torch.no_grad():
    for x_dyn, x_stat, uf_idx, _, _ in test_2024_loader:
      logits, pred_niv = model(x_dyn.to(device), x_stat.to(device), uf_idx.to(device))
      # Correção Bayesiana simples do oversampling de 8% para o prior real de treino
      prior_tr = float(y_tr.mean())
      shift = np.log(prior_tr / (1 - prior_tr)) - np.log(0.08 / 0.92)
      p_list.append(torch.sigmoid(logits + shift).cpu().numpy())
      niv_list.append(pred_niv.cpu().numpy())

  df_2024_pred = test_2024_ds.meta_df.copy()
  df_2024_pred["prob_enchente_pred"] = np.concatenate(p_list)
  df_2024_pred["nivel_alagamento_pred"] = np.concatenate(niv_list)
  df_2024_pred = df_2024_pred[df_2024_pred["ano"] == 2024].copy()
  df_2024_pred["mes"] = df_2024_pred["data"].dt.month

  # Métricas Factuais 2024 (Brasil vs. RS)
  print("\n=== AUDITORIA DA ENCHENTE DE 2024 (PREVISTO vs. REAL) ===")
  for label, sub in [
      ("BRASIL INTEIRO (2024)", df_2024_pred),
      ("RIO GRANDE DO SUL (2024)", df_2024_pred[df_2024_pred["uf_code"] == "43"]),
      ("RS — MAIO DE 2024 (PICO)", df_2024_pred[(df_2024_pred["uf_code"] == "43") & (df_2024_pred["mes"] == 5)]),
  ]:
    if len(sub["alvo_janela_3d"].unique()) > 1:
      roc = roc_auc_score(sub["alvo_janela_3d"], sub["prob_enchente_pred"])
      pr = average_precision_score(sub["alvo_janela_3d"], sub["prob_enchente_pred"])
      brier = brier_score_loss(sub["alvo_janela_3d"], sub["prob_enchente_pred"])
      mae_niv = mean_absolute_error(sub["nivel_alagamento_obs"], sub["nivel_alagamento_pred"])
      print(
          f"{label:26s} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} |"
          f" Brier: {brier:.5f} | MAE Nível: {mae_niv:.4f}"
      )

  # 5. Previsão de Leitos 2024 e Comparação com o Real (CNES)
  df_clim["mes"] = df_clim["data"].dt.month
  df_clim_mes = df_clim.groupby(["id_municipio", "ano", "mes"], as_index=False).agg(
      chuva_mes_mm=("precip_dia_mm", "sum"),
      eventos_mes=("alvo_enchente", "sum"),
  )
  principais_areas = [c for c in ["leitos_clinico", "leitos_cirurgico", "leitos_complementar"] if c in leito_cols]
  df_beds_2024 = train_and_predict_beds_2024(df_cnes, df_clim_mes, principais_areas)

  # 6. Cruzamento Final Normalizado por População e Área (RS Maio 2024 vs. Brasil 2024)
  df_2024_mes_risco = df_2024_pred.groupby(["id_municipio", "uf_code", "mes"], as_index=False).agg(
      prob_media_mes=("prob_enchente_pred", "mean"),
      nivel_pred_max=("nivel_alagamento_pred", "max"),
      nivel_real_max=("nivel_alagamento_obs", "max"),
      eventos_reais=("alvo_janela_3d", "max"),
  )
  df_final_2024 = df_2024_mes_risco.merge(df_beds_2024, on=["id_municipio", "mes"], how="inner")

  for area in principais_areas:
    df_final_2024[f"{area}_real_por_1000hab"] = (df_final_2024[f"{area}_real"] / df_final_2024["populacao"]) * 1000.0
    df_final_2024[f"{area}_prev_por_1000hab"] = (df_final_2024[f"{area}_previsto"] / df_final_2024["populacao"]) * 1000.0
    df_final_2024[f"{area}_real_por_100km2"] = (df_final_2024[f"{area}_real"] / df_final_2024["area_km2"]) * 100.0

  # Comparativo Mensal no RS em 2024 (Previsto vs. Real em Maio/Junho/Julho)
  rs_2024 = df_final_2024[df_final_2024["uf_code"] == "43"]
  rs_mensal = rs_2024.groupby("mes", as_index=False).agg(
      Prob_Enchente_Prev=("prob_media_mes", "mean"),
      Nivel_Alag_Prev=("nivel_pred_max", "mean"),
      Nivel_Alag_Real=("nivel_real_max", "mean"),
      Leitos_Clin_Prev=("leitos_clinico_previsto", "sum"),
      Leitos_Clin_Real=("leitos_clinico_real", "sum"),
      Leitos_UTI_Prev=("leitos_complementar_previsto", "sum"),
      Leitos_UTI_Real=("leitos_complementar_real", "sum"),
      Leitos_Clin_1000hab=("leitos_clinico_real_por_1000hab", "mean"),
  )
  print("\n=== RS 2024 MÊS A MÊS: ENCHENTE E LEITOS (PREVISTO vs. ACONTECIDO) ===")
  print(rs_mensal.round(3).to_string(index=False))

  # Gráfico Comparativo RS 2024
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  axes[0].plot(rs_mensal["mes"], rs_mensal["Nivel_Alag_Real"], "o-", label="Nível Alagamento Real (Obs)", linewidth=2)
  axes[0].plot(rs_mensal["mes"], rs_mensal["Nivel_Alag_Prev"], "s--", label="Nível Alagamento Previsto (TFT)", linewidth=2)
  axes[0].axvline(5, color="red", linestyle=":", alpha=0.7, label="Maio/2024 (Pico RS)")
  axes[0].set_title("RS 2024: Nível de Alagamento Previsto vs. Real")
  axes[0].set_xlabel("Mês (2024)")
  axes[0].legend()
  axes[0].grid(alpha=0.3)

  axes[1].plot(rs_mensal["mes"], rs_mensal["Leitos_UTI_Real"], "o-", label="Leitos UTI/Compl. Reais (CNES)", linewidth=2)
  axes[1].plot(rs_mensal["mes"], rs_mensal["Leitos_UTI_Prev"], "s--", label="Leitos UTI/Compl. Previstos", linewidth=2)
  axes[1].axvline(5, color="red", linestyle=":", alpha=0.7, label="Maio/2024 (Pico RS)")
  axes[1].set_title("RS 2024: Leitos Complementares (UTI) Previstos vs. Reais")
  axes[1].set_xlabel("Mês (2024)")
  axes[1].legend()
  axes[1].grid(alpha=0.3)

  plt.tight_layout()
  plt.show()

  return df_final_2024, rs_mensal


if __name__ == "__main__":
  df_auditoria_2024, df_rs_mensal_2024 = run_2024_national_audit(
      gcp_project_id= GCP_PROJECT_ID,
      climate_parquet="flood_dataset_2010_2025.parquet",
      epochs=6,
  )
