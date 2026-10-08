library(data.table)

# ============================================================
# 1. LEER DATASET CON FEATURES POR MINUTO + 360
# ============================================================

dt <- fread(
  "data/processed/america_male/model_context_360.csv",
  encoding = "UTF-8"
)

setorder(
  dt,
  match_id,
  minute
)

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. FEATURES EVENT/SPATIAL A SUMAR EN 3 MINUTOS
# ============================================================

sum_features <- c(

  # EVENTS
  "america_passes",
  "opponent_passes",

  "america_pressures",
  "opponent_pressures",

  "america_recoveries",
  "opponent_recoveries",

  "america_shots",
  "opponent_shots",

  "america_xg",
  "opponent_xg",

  "america_obv",
  "opponent_obv",

  # SPATIAL
  "america_final_third_events",
  "opponent_final_third_events",

  "america_entries_final_third",
  "opponent_entries_final_third",

  "america_entries_box",
  "opponent_entries_box",

  "america_progressive_passes",
  "opponent_progressive_passes",

  "america_progressive_carries",
  "opponent_progressive_carries",

  "america_left_events",
  "america_center_events",
  "america_right_events",

  "opponent_left_events",
  "opponent_center_events",
  "opponent_right_events"
)

sum_features <- sum_features[
  sum_features %in% names(dt)
]


# ============================================================
# 3. ROLLING SUM 3 MIN
# ============================================================

for (col in sum_features) {

  new_col <- paste0(
    col,
    "_last3"
  )

  dt[
    ,
    (new_col) :=
      frollsum(
        get(col),
        n = 3,
        align = "right",
        fill = NA,
        na.rm = TRUE
      ),
    by = match_id
  ]
}


# ============================================================
# 4. VARIABLES A PROMEDIAR EN 3 MIN
# ============================================================

mean_features <- c(

  "america_recovery_height",
  "opponent_recovery_height",

  "america_avg_width",
  "opponent_avg_width",

  "america_directness",
  "opponent_directness"
)

mean_features <- mean_features[
  mean_features %in% names(dt)
]


for (col in mean_features) {

  new_col <- paste0(
    col,
    "_last3"
  )

  dt[
    ,
    (new_col) :=
      frollmean(
        get(col),
        n = 3,
        align = "right",
        fill = NA,
        na.rm = TRUE
      ),
    by = match_id
  ]
}


# ============================================================
# 5. FIELD TILT 3 MIN
# ============================================================

if (
  all(
    c(
      "america_final_third_events_last3",
      "opponent_final_third_events_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    total_final_third_events_last3 :=
      america_final_third_events_last3 +
      opponent_final_third_events_last3
  ]

  dt[
    ,
    america_field_tilt_last3 :=
      fifelse(
        total_final_third_events_last3 > 0,

        america_final_third_events_last3 /
          total_final_third_events_last3,

        NA_real_
      )
  ]
}


# ============================================================
# 6. DIFERENCIAS EVENT/SPATIAL
# ============================================================

if (
  all(
    c(
      "america_pressures_last3",
      "opponent_pressures_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    pressure_diff_last3 :=
      america_pressures_last3 -
      opponent_pressures_last3
  ]
}


if (
  all(
    c(
      "america_xg_last3",
      "opponent_xg_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    xg_diff_last3 :=
      america_xg_last3 -
      opponent_xg_last3
  ]
}


if (
  all(
    c(
      "america_obv_last3",
      "opponent_obv_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    obv_diff_last3 :=
      america_obv_last3 -
      opponent_obv_last3
  ]
}


if (
  all(
    c(
      "america_recoveries_last3",
      "opponent_recoveries_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    recoveries_diff_last3 :=
      america_recoveries_last3 -
      opponent_recoveries_last3
  ]
}


if (
  all(
    c(
      "america_progressive_passes_last3",
      "opponent_progressive_passes_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    progressive_pass_diff_last3 :=
      america_progressive_passes_last3 -
      opponent_progressive_passes_last3
  ]
}


if (
  all(
    c(
      "america_progressive_carries_last3",
      "opponent_progressive_carries_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    progressive_carry_diff_last3 :=
      america_progressive_carries_last3 -
      opponent_progressive_carries_last3
  ]
}


if (
  all(
    c(
      "america_entries_final_third_last3",
      "opponent_entries_final_third_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    entries_final_third_diff_last3 :=
      america_entries_final_third_last3 -
      opponent_entries_final_third_last3
  ]
}


if (
  all(
    c(
      "america_entries_box_last3",
      "opponent_entries_box_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    entries_box_diff_last3 :=
      america_entries_box_last3 -
      opponent_entries_box_last3
  ]
}


# ============================================================
# 7. RECOVERY HEIGHT DIFF 3 MIN
# ============================================================

if (
  all(
    c(
      "america_recovery_height_last3",
      "opponent_recovery_height_last3"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    recovery_height_diff_last3 :=
      america_recovery_height_last3 -
      opponent_recovery_height_last3
  ]
}


# ============================================================
# 8. ROLLING 360 PONDERADO DE 3 MIN
# ============================================================

weighted_roll_3 <- function(
  DT,
  variable,
  weight,
  new_name
) {

  temp_num <- paste0(
    "__num_",
    new_name
  )

  temp_den <- paste0(
    "__den_",
    new_name
  )

  DT[
    ,
    (temp_num) :=
      fifelse(
        is.na(get(variable)),
        0,
        get(variable) * get(weight)
      )
  ]

  DT[
    ,
    (temp_den) :=
      fifelse(
        is.na(get(variable)),
        0,
        get(weight)
      )
  ]

  num_roll <- paste0(
    temp_num,
    "_roll"
  )

  den_roll <- paste0(
    temp_den,
    "_roll"
  )

  DT[
    ,
    (num_roll) :=
      frollsum(
        get(temp_num),
        n = 3,
        align = "right",
        fill = NA
      ),
    by = match_id
  ]

  DT[
    ,
    (den_roll) :=
      frollsum(
        get(temp_den),
        n = 3,
        align = "right",
        fill = NA
      ),
    by = match_id
  ]

  DT[
    ,
    (new_name) :=
      fifelse(
        get(den_roll) > 0,
        get(num_roll) /
          get(den_roll),
        NA_real_
      )
  ]

  DT[
    ,
    c(
      temp_num,
      temp_den,
      num_roll,
      den_roll
    ) := NULL
  ]
}


# ============================================================
# 9. EVENTOS 360 ACUMULADOS 3 MIN
# ============================================================

dt[
  ,
  america_360_events_last3 :=
    frollsum(
      america_360_events,
      n = 3,
      align = "right",
      fill = NA
    ),
  by = match_id
]

dt[
  ,
  opponent_360_events_last3 :=
    frollsum(
      opponent_360_events,
      n = 3,
      align = "right",
      fill = NA
    ),
  by = match_id
]


# ============================================================
# 10. MÉTRICAS 360 AMÉRICA
# ============================================================

america_360_metrics <- c(

  "america_visible_teammates",
  "america_visible_opponents",

  "america_teammates_within_5",
  "america_opponents_within_5",

  "america_teammates_within_10",
  "america_opponents_within_10",

  "america_local_superiority_5",
  "america_local_superiority_10",

  "america_nearest_opponent_distance",

  "america_avg_opponent_distance",
  "america_avg_teammate_distance",

  "america_pressure_share_5",
  "america_pressure_share_10",

  "america_crowded_share_10",
  "america_support_share_10"
)


for (variable in america_360_metrics) {

  if (variable %in% names(dt)) {

    weighted_roll_3(
      DT = dt,
      variable = variable,
      weight = "america_360_events",
      new_name = paste0(
        variable,
        "_last3"
      )
    )
  }
}


# ============================================================
# 11. MÉTRICAS 360 RIVAL
# ============================================================

opponent_360_metrics <- c(

  "opponent_visible_teammates",
  "opponent_visible_opponents",

  "opponent_teammates_within_5",
  "opponent_opponents_within_5",

  "opponent_teammates_within_10",
  "opponent_opponents_within_10",

  "opponent_local_superiority_5",
  "opponent_local_superiority_10",

  "opponent_nearest_opponent_distance",

  "opponent_avg_opponent_distance",
  "opponent_avg_teammate_distance",

  "opponent_pressure_share_5",
  "opponent_pressure_share_10",

  "opponent_crowded_share_10",
  "opponent_support_share_10"
)


for (variable in opponent_360_metrics) {

  if (variable %in% names(dt)) {

    weighted_roll_3(
      DT = dt,
      variable = variable,
      weight = "opponent_360_events",
      new_name = paste0(
        variable,
        "_last3"
      )
    )
  }
}


# ============================================================
# 12. DIFERENCIAS 360 3 MIN
# ============================================================

dt[
  ,
  local_superiority_diff_10_last3 :=
    america_local_superiority_10_last3 -
    opponent_local_superiority_10_last3
]

dt[
  ,
  nearest_opponent_distance_diff_last3 :=
    america_nearest_opponent_distance_last3 -
    opponent_nearest_opponent_distance_last3
]

dt[
  ,
  pressure_share_diff_5_last3 :=
    america_pressure_share_5_last3 -
    opponent_pressure_share_5_last3
]

dt[
  ,
  pressure_share_diff_10_last3 :=
    america_pressure_share_10_last3 -
    opponent_pressure_share_10_last3
]

dt[
  ,
  crowded_share_diff_10_last3 :=
    america_crowded_share_10_last3 -
    opponent_crowded_share_10_last3
]

dt[
  ,
  support_share_diff_10_last3 :=
    america_support_share_10_last3 -
    opponent_support_share_10_last3
]


# ============================================================
# 13. HISTORIA COMPLETA
#
# En ventana de 3 min,
# desde minuto 2 ya tenemos 3 observaciones.
# ============================================================

dt[
  ,
  has_full_3min_history :=
    minute >= 2
]


# ============================================================
# 14. LIMPIAR NaN / Inf
# ============================================================

numeric_cols <- names(dt)[
  sapply(
    dt,
    is.numeric
  )
]

for (col in numeric_cols) {

  idx <- which(
    is.nan(dt[[col]]) |
    is.infinite(dt[[col]])
  )

  if (length(idx) > 0) {

    set(
      dt,
      idx,
      col,
      NA_real_
    )
  }
}


# ============================================================
# 15. GUARDAR
# ============================================================

fwrite(
  dt,
  "data/processed/america_male/model_context_360_rolling_3min.csv",
  bom = TRUE
)

cat("\n=====================================\n")
cat("3-MINUTE FEATURES TERMINADAS\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")