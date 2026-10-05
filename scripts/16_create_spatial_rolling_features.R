library(data.table)

# ============================================================
# 1. LEER DATASET ESPACIAL
# ============================================================

dt <- fread(
  "data/processed/america_male/model_context_spatial.csv",
  encoding = "UTF-8"
)

setorder(dt, match_id, minute)

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. VARIABLES ESPACIALES A SUMAR EN ÚLTIMOS 5 MIN
# ============================================================

sum_features <- c(
  "america_final_third_events",
  "america_entries_final_third",
  "america_entries_box",
  "america_progressive_passes",
  "america_progressive_carries",
  "america_left_events",
  "america_center_events",
  "america_right_events",

  "opponent_final_third_events",
  "opponent_entries_final_third",
  "opponent_entries_box",
  "opponent_progressive_passes",
  "opponent_progressive_carries",
  "opponent_left_events",
  "opponent_center_events",
  "opponent_right_events"
)

sum_features <- sum_features[
  sum_features %in% names(dt)
]

for (col in sum_features) {

  new_col <- paste0(col, "_last5")

  dt[
    ,
    (new_col) := frollsum(
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
# 3. VARIABLES ESPACIALES A PROMEDIAR EN ÚLTIMOS 5 MIN
# ============================================================

mean_features <- c(
  "america_recovery_height",
  "opponent_recovery_height",
  "america_avg_width",
  "opponent_avg_width"
)

mean_features <- mean_features[
  mean_features %in% names(dt)
]

for (col in mean_features) {

  new_col <- paste0(col, "_last5")

  dt[
    ,
    (new_col) := frollmean(
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
# 4. FIELD TILT REAL DE 5 MIN
# ============================================================

dt[
  ,
  total_final_third_events_last5 :=
    america_final_third_events_last5 +
    opponent_final_third_events_last5
]

dt[
  ,
  america_field_tilt_last5 :=
    fifelse(
      total_final_third_events_last5 > 0,
      america_final_third_events_last5 /
        total_final_third_events_last5,
      NA_real_
    )
]


# ============================================================
# 5. DIFERENCIAS ESPACIALES
# ============================================================

dt[
  ,
  recovery_height_diff_last5 :=
    america_recovery_height_last5 -
    opponent_recovery_height_last5
]

dt[
  ,
  progressive_pass_diff_last5 :=
    america_progressive_passes_last5 -
    opponent_progressive_passes_last5
]

dt[
  ,
  progressive_carry_diff_last5 :=
    america_progressive_carries_last5 -
    opponent_progressive_carries_last5
]

dt[
  ,
  entries_final_third_diff_last5 :=
    america_entries_final_third_last5 -
    opponent_entries_final_third_last5
]

dt[
  ,
  entries_box_diff_last5 :=
    america_entries_box_last5 -
    opponent_entries_box_last5
]


# ============================================================
# 6. BALANCE DE CARRILES
# ============================================================

dt[
  ,
  america_wide_events_last5 :=
    america_left_events_last5 +
    america_right_events_last5
]

dt[
  ,
  opponent_wide_events_last5 :=
    opponent_left_events_last5 +
    opponent_right_events_last5
]

dt[
  ,
  america_wide_ratio_last5 :=
    fifelse(
      (
        america_left_events_last5 +
        america_center_events_last5 +
        america_right_events_last5
      ) > 0,

      america_wide_events_last5 /
        (
          america_left_events_last5 +
          america_center_events_last5 +
          america_right_events_last5
        ),

      NA_real_
    )
]

dt[
  ,
  opponent_wide_ratio_last5 :=
    fifelse(
      (
        opponent_left_events_last5 +
        opponent_center_events_last5 +
        opponent_right_events_last5
      ) > 0,

      opponent_wide_events_last5 /
        (
          opponent_left_events_last5 +
          opponent_center_events_last5 +
          opponent_right_events_last5
        ),

      NA_real_
    )
]


# ============================================================
# 7. LIMPIAR NaN
# ============================================================

numeric_cols <- names(dt)[
  sapply(dt, is.numeric)
]

for (col in numeric_cols) {

  idx <- which(
    is.nan(dt[[col]])
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
# 8. GUARDAR
# ============================================================

fwrite(
  dt,
  "data/processed/america_male/model_context_spatial_rolling.csv",
  bom = TRUE
)

cat("\n=====================================\n")
cat("SPATIAL ROLLING FEATURES TERMINADO\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")