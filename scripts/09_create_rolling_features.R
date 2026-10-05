library(data.table)

# ============================================================
# 1. LEER FEATURES MINUTO A MINUTO
# ============================================================

dt <- fread(
  "data/processed/america_male/minute_features.csv",
  encoding = "UTF-8"
)

setorder(dt, match_id, minute)

cat("Filas originales:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. VARIABLES QUE VAMOS A SUMAR EN LOS ÚLTIMOS 5 MINUTOS
# ============================================================

sum_features <- c(
  "america_passes",
  "america_completed_passes",
  "america_carries",
  "america_pressures",
  "america_counterpress",
  "america_recoveries",
  "america_interceptions",
  "america_duels",
  "america_dribbles",
  "america_shots",
  "america_xg",
  "america_obv",
  "america_possessions",

  "opponent_passes",
  "opponent_completed_passes",
  "opponent_carries",
  "opponent_pressures",
  "opponent_counterpress",
  "opponent_recoveries",
  "opponent_interceptions",
  "opponent_shots",
  "opponent_xg",
  "opponent_obv",
  "opponent_possessions"
)

sum_features <- sum_features[
  sum_features %in% names(dt)
]


# ============================================================
# 3. ROLLING SUM DE 5 MINUTOS
#
# Ejemplo minuto 63:
# usa minutos 59, 60, 61, 62 y 63
# ============================================================

for (col in sum_features) {

  nueva <- paste0(
    col,
    "_last5"
  )

  dt[
    ,
    (nueva) := frollsum(
      get(col),
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
    by = match_id
  ]
}


# ============================================================
# 4. VARIABLES QUE TIENE MÁS SENTIDO PROMEDIAR
# ============================================================

mean_features <- c(
  "america_directness",
  "opponent_directness",
  "america_pass_completion",
  "opponent_pass_completion",
  "america_possession_proxy"
)

mean_features <- mean_features[
  mean_features %in% names(dt)
]

for (col in mean_features) {

  nueva <- paste0(
    col,
    "_last5"
  )

  dt[
    ,
    (nueva) := frollmean(
      get(col),
      n = 5,
      align = "right",
      fill = NA,
      na.rm = TRUE
    ),
    by = match_id
  ]
}


# ============================================================
# 5. DIFERENCIAS AMÉRICA - RIVAL
# ============================================================

dt[
  ,
  pressure_diff_last5 :=
    america_pressures_last5 -
    opponent_pressures_last5
]

dt[
  ,
  xg_diff_last5 :=
    america_xg_last5 -
    opponent_xg_last5
]

dt[
  ,
  obv_diff_last5 :=
    america_obv_last5 -
    opponent_obv_last5
]

dt[
  ,
  passes_diff_last5 :=
    america_passes_last5 -
    opponent_passes_last5
]

dt[
  ,
  recoveries_diff_last5 :=
    america_recoveries_last5 -
    opponent_recoveries_last5
]


# ============================================================
# 6. INTENSIDAD OFENSIVA
# ============================================================

dt[
  ,
  america_attack_intensity_last5 :=
    america_shots_last5 +
    america_carries_last5 +
    america_recoveries_last5
]

dt[
  ,
  opponent_attack_intensity_last5 :=
    opponent_shots_last5 +
    opponent_carries_last5 +
    opponent_recoveries_last5
]


# ============================================================
# 7. PROPORCIÓN DE PRESIÓN
# ============================================================

dt[
  ,
  total_pressures_last5 :=
    america_pressures_last5 +
    opponent_pressures_last5
]

dt[
  ,
  america_pressure_share_last5 :=
    fifelse(
      total_pressures_last5 > 0,
      america_pressures_last5 /
        total_pressures_last5,
      NA_real_
    )
]


# ============================================================
# 8. POSESIÓN RELATIVA EN ÚLTIMOS 5 MINUTOS
# ============================================================

dt[
  ,
  total_possessions_last5 :=
    america_possessions_last5 +
    opponent_possessions_last5
]

dt[
  ,
  america_possession_share_last5 :=
    fifelse(
      total_possessions_last5 > 0,
      america_possessions_last5 /
        total_possessions_last5,
      NA_real_
    )
]


# ============================================================
# 9. QUITAR LOS PRIMEROS MINUTOS SIN HISTORIA SUFICIENTE
#
# No los borramos todavía.
# Solo creamos indicador.
# ============================================================

dt[
  ,
  has_full_history :=
    minute >= 4
]


# ============================================================
# 10. GUARDAR
# ============================================================

fwrite(
  dt,
  "data/processed/america_male/rolling_features.csv",
  bom = TRUE
)

cat("\n=====================================\n")
cat("ROLLING FEATURES TERMINADO\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")



dt[
  match_id == dt$match_id[1] & minute >= 4,
  .(
    minute,
    game_state,
    america_pressures_last5,
    opponent_pressures_last5,
    pressure_diff_last5,
    america_xg_last5,
    opponent_xg_last5,
    xg_diff_last5,
    america_possession_share_last5
  )
][1:20]