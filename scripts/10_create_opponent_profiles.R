library(data.table)

# ============================================================
# 1. LEER ROLLING FEATURES
# ============================================================

dt <- fread(
  "data/processed/america_male/rolling_features.csv",
  encoding = "UTF-8"
)

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. USAR SOLO MINUTOS CON 5 MINUTOS DE HISTORIA
# ============================================================

base <- dt[
  has_full_history == TRUE &
  !is.na(opponent)
]


# ============================================================
# 3. PERFIL HISTÓRICO DEL RIVAL
# ============================================================

opponent_profiles <- base[
  ,
  .(
    matches_vs_opponent = uniqueN(match_id),

    opponent_avg_pressure_last5 =
      mean(opponent_pressures_last5, na.rm = TRUE),

    opponent_avg_xg_last5 =
      mean(opponent_xg_last5, na.rm = TRUE),

    opponent_avg_obv_last5 =
      mean(opponent_obv_last5, na.rm = TRUE),

    opponent_avg_passes_last5 =
      mean(opponent_passes_last5, na.rm = TRUE),

    opponent_avg_recoveries_last5 =
      mean(opponent_recoveries_last5, na.rm = TRUE),

    opponent_avg_carries_last5 =
      mean(opponent_carries_last5, na.rm = TRUE),

    opponent_avg_shots_last5 =
      mean(opponent_shots_last5, na.rm = TRUE),

    opponent_avg_possession =
      mean(
        1 - america_possession_share_last5,
        na.rm = TRUE
      ),

    opponent_avg_directness =
      mean(opponent_directness_last5, na.rm = TRUE)
  ),
  by = opponent
]


# ============================================================
# 4. ORDENAR
# ============================================================

setorder(
  opponent_profiles,
  -matches_vs_opponent
)


# ============================================================
# 5. GUARDAR
# ============================================================

fwrite(
  opponent_profiles,
  "data/processed/america_male/opponent_profiles.csv",
  bom = TRUE
)

cat("\nPERFILES DE RIVALES CREADOS\n")

print(opponent_profiles)