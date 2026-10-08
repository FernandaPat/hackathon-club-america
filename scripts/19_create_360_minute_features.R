library(data.table)

# ============================================================
# 1. LEER 360 + EVENTS
# ============================================================

dt360 <- fread(
  "data/processed/america_male/features_360_events.csv",
  encoding = "UTF-8"
)

cat("Filas 360:", nrow(dt360), "\n")


# ============================================================
# 2. QUEDARNOS SOLO CON FRAMES QUE SÍ HICIERON MATCH
# ============================================================

dt360 <- dt360[
  !is.na(minute) &
  !is.na(team_name)
]

cat(
  "Frames 360 utilizables:",
  nrow(dt360),
  "\n"
)


# ============================================================
# 3. VARIABLES EXTRA DE PRESIÓN / DENSIDAD
# ============================================================

dt360[
  ,
  under_local_pressure_5 :=
    opponents_within_5 >= 1
]

dt360[
  ,
  under_local_pressure_10 :=
    opponents_within_10 >= 1
]

dt360[
  ,
  crowded_by_opponents_10 :=
    opponents_within_10 >= 2
]

dt360[
  ,
  supported_by_teammates_10 :=
    teammates_within_10 >= 2
]


# ============================================================
# 4. FEATURES DE AMÉRICA POR MINUTO
# ============================================================

america_360_minute <- dt360[
  team_name == "América",
  .(

    america_360_events = .N,

    # Visibilidad
    america_visible_teammates =
      mean(visible_teammates, na.rm = TRUE),

    america_visible_opponents =
      mean(visible_opponents, na.rm = TRUE),

    # Jugadores cercanos
    america_teammates_within_5 =
      mean(teammates_within_5, na.rm = TRUE),

    america_opponents_within_5 =
      mean(opponents_within_5, na.rm = TRUE),

    america_teammates_within_10 =
      mean(teammates_within_10, na.rm = TRUE),

    america_opponents_within_10 =
      mean(opponents_within_10, na.rm = TRUE),

    # Superioridad local
    america_local_superiority_5 =
      mean(local_superiority_5, na.rm = TRUE),

    america_local_superiority_10 =
      mean(local_superiority_10, na.rm = TRUE),

    # Distancias
    america_nearest_opponent_distance =
      mean(
        nearest_opponent_distance,
        na.rm = TRUE
      ),

    america_avg_opponent_distance =
      mean(
        avg_opponent_distance,
        na.rm = TRUE
      ),

    america_avg_teammate_distance =
      mean(
        avg_teammate_distance,
        na.rm = TRUE
      ),

    # Presión / densidad
    america_pressure_share_5 =
      mean(
        under_local_pressure_5,
        na.rm = TRUE
      ),

    america_pressure_share_10 =
      mean(
        under_local_pressure_10,
        na.rm = TRUE
      ),

    america_crowded_share_10 =
      mean(
        crowded_by_opponents_10,
        na.rm = TRUE
      ),

    america_support_share_10 =
      mean(
        supported_by_teammates_10,
        na.rm = TRUE
      )

  ),
  by = .(
    match_id,
    minute
  )
]


# ============================================================
# 5. FEATURES DEL RIVAL POR MINUTO
# ============================================================

opponent_360_minute <- dt360[
  team_name != "América",
  .(

    opponent_360_events = .N,

    # Visibilidad
    opponent_visible_teammates =
      mean(visible_teammates, na.rm = TRUE),

    opponent_visible_opponents =
      mean(visible_opponents, na.rm = TRUE),

    # Jugadores cercanos
    opponent_teammates_within_5 =
      mean(teammates_within_5, na.rm = TRUE),

    opponent_opponents_within_5 =
      mean(opponents_within_5, na.rm = TRUE),

    opponent_teammates_within_10 =
      mean(teammates_within_10, na.rm = TRUE),

    opponent_opponents_within_10 =
      mean(opponents_within_10, na.rm = TRUE),

    # Superioridad
    opponent_local_superiority_5 =
      mean(local_superiority_5, na.rm = TRUE),

    opponent_local_superiority_10 =
      mean(local_superiority_10, na.rm = TRUE),

    # Distancias
    opponent_nearest_opponent_distance =
      mean(
        nearest_opponent_distance,
        na.rm = TRUE
      ),

    opponent_avg_opponent_distance =
      mean(
        avg_opponent_distance,
        na.rm = TRUE
      ),

    opponent_avg_teammate_distance =
      mean(
        avg_teammate_distance,
        na.rm = TRUE
      ),

    # Presión / densidad
    opponent_pressure_share_5 =
      mean(
        under_local_pressure_5,
        na.rm = TRUE
      ),

    opponent_pressure_share_10 =
      mean(
        under_local_pressure_10,
        na.rm = TRUE
      ),

    opponent_crowded_share_10 =
      mean(
        crowded_by_opponents_10,
        na.rm = TRUE
      ),

    opponent_support_share_10 =
      mean(
        supported_by_teammates_10,
        na.rm = TRUE
      )

  ),
  by = .(
    match_id,
    minute
  )
]


# ============================================================
# 6. LIMPIAR NaN
# ============================================================

clean_nan <- function(DT) {

  numeric_cols <- names(DT)[
    sapply(DT, is.numeric)
  ]

  for (col in numeric_cols) {

    idx <- which(
      is.nan(DT[[col]])
    )

    if (length(idx) > 0) {

      set(
        DT,
        idx,
        col,
        NA_real_
      )
    }
  }
}


clean_nan(america_360_minute)
clean_nan(opponent_360_minute)


# ============================================================
# 7. UNIR AMÉRICA + RIVAL
# ============================================================

minute_360 <- merge(
  america_360_minute,
  opponent_360_minute,
  by = c(
    "match_id",
    "minute"
  ),
  all = TRUE
)


# ============================================================
# 8. CREAR DIFERENCIAS
# ============================================================

minute_360[
  ,
  local_superiority_diff_10 :=
    america_local_superiority_10 -
    opponent_local_superiority_10
]

minute_360[
  ,
  nearest_opponent_distance_diff :=
    america_nearest_opponent_distance -
    opponent_nearest_opponent_distance
]

minute_360[
  ,
  pressure_share_diff_5 :=
    america_pressure_share_5 -
    opponent_pressure_share_5
]

minute_360[
  ,
  crowded_share_diff_10 :=
    america_crowded_share_10 -
    opponent_crowded_share_10
]


# ============================================================
# 9. GUARDAR FEATURES 360 POR MINUTO
# ============================================================

fwrite(
  minute_360,
  "data/processed/america_male/minute_features_360.csv",
  bom = TRUE
)


# ============================================================
# 10. UNIR CON DATASET PRINCIPAL
# ============================================================

model <- fread(
  "data/processed/america_male/model_context_spatial_rolling.csv",
  encoding = "UTF-8"
)

model_360 <- merge(
  model,
  minute_360,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)


# ============================================================
# 11. LOS CONTEOS PUEDEN SER 0
# ============================================================

model_360[
  is.na(america_360_events),
  america_360_events := 0
]

model_360[
  is.na(opponent_360_events),
  opponent_360_events := 0
]


# ============================================================
# 12. GUARDAR DATASET COMPLETO
# ============================================================

fwrite(
  model_360,
  "data/processed/america_male/model_context_360.csv",
  bom = TRUE
)


# ============================================================
# 13. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("360 MINUTE FEATURES TERMINADAS\n")
cat("=====================================\n")

cat(
  "Minute 360 filas:",
  nrow(minute_360),
  "\n"
)

cat(
  "Dataset final filas:",
  nrow(model_360),
  "\n"
)

cat(
  "Dataset final columnas:",
  ncol(model_360),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(model_360$match_id),
  "\n"
)