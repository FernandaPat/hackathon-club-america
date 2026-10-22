# ============================================================
# 54_export_female_dashboard.R
# CLUB AMÉRICA FEMENIL
#
# EXPORTA LOS ARCHIVOS MÍNIMOS PARA STREAMLIT
# ============================================================

library(data.table)


# ============================================================
# 1. RUTAS
# ============================================================

project_root <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

processed_dir <- file.path(
  project_root,
  "data",
  "processed",
  "america_female"
)

model_dir <- file.path(
  project_root,
  "outputs",
  "models",
  "female",
  "change_model"
)

dashboard_dir <- file.path(
  project_root,
  "dashboard",
  "data"
)

if (!dir.exists(dashboard_dir)) {
  dir.create(
    dashboard_dir,
    recursive = TRUE
  )
}


model_ready_file <- file.path(
  processed_dir,
  "model_ready_female_v2.csv"
)

matches_file <- file.path(
  processed_dir,
  "america_female_matches.csv"
)

metrics_file <- file.path(
  model_dir,
  "female_change_test_metrics.csv"
)

features_file <- file.path(
  model_dir,
  "female_change_features.csv"
)

selection_file <- file.path(
  model_dir,
  "female_change_selection.csv"
)

predictions_file <- file.path(
  model_dir,
  "female_change_test_predictions.csv"
)


# ============================================================
# 2. VALIDAR ARCHIVOS
# ============================================================

required_files <- c(
  model_ready_file,
  matches_file,
  metrics_file,
  features_file,
  selection_file,
  predictions_file
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {

  stop(
    paste(
      "Faltan archivos:",
      paste(
        missing_files,
        collapse = "\n"
      )
    )
  )
}


# ============================================================
# 3. CARGAR
# ============================================================

model_dt <- fread(
  model_ready_file
)

matches <- fread(
  matches_file
)

metrics <- fread(
  metrics_file
)

model_features <- fread(
  features_file
)

selection <- fread(
  selection_file
)

predictions <- fread(
  predictions_file
)


cat("\n")
cat("============================================\n")
cat("EXPORT FEMALE DASHBOARD\n")
cat("============================================\n\n")


# ============================================================
# 4. FECHAS
# ============================================================

if ("match_date" %in% names(model_dt)) {

  model_dt[
    ,
    match_date := as.IDate(
      match_date
    )
  ]
}


if ("match_date" %in% names(matches)) {

  matches[
    ,
    match_date := as.IDate(
      match_date
    )
  ]
}


if ("match_date" %in% names(predictions)) {

  predictions[
    ,
    match_date := as.IDate(
      match_date
    )
  ]
}


# ============================================================
# 5. VALIDAR MODEL READY
# ============================================================

required_model_columns <- c(
  "match_id",
  "match_date",
  "minute",
  "opponent",
  "venue",
  "current_tactical_state",
  "future_tactical_state_5min",
  "split",
  "game_state_num"
)

missing_model_columns <- setdiff(
  required_model_columns,
  names(model_dt)
)

if (length(missing_model_columns) > 0) {

  stop(
    paste(
      "Faltan columnas en model_ready_female_v2:",
      paste(
        missing_model_columns,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 6. SIMULADOR
#
# SOLO TRAIN + VALIDATION
# ============================================================

simulator_dt <- model_dt[
  split %in% c(
    "TRAIN",
    "VALIDATION"
  )
]

if (nrow(simulator_dt) == 0) {

  stop(
    "No existen filas TRAIN/VALIDATION."
  )
}


# ============================================================
# 7. FEATURES DEL MODELO
# ============================================================

if (!"feature" %in% names(model_features)) {

  stop(
    "female_change_features.csv no contiene feature."
  )
}


change_features <- unique(
  model_features$feature
)


available_model_features <- change_features[
  change_features %in% names(simulator_dt)
]


# ============================================================
# 8. FEATURES ADICIONALES DE CONTEXTO
# ============================================================

context_candidates <- c(
  "america_possession_context",
  "america_possession_context_last3",
  "america_possession_context_last5",

  "america_field_tilt",
  "america_field_tilt_last3",
  "america_field_tilt_last5",

  "entries_final_third_diff",
  "entries_final_third_diff_last3",
  "entries_final_third_diff_last5",

  "entries_box_diff",
  "entries_box_diff_last3",
  "entries_box_diff_last5",

  "progressive_passes_diff",
  "progressive_passes_diff_last3",
  "progressive_passes_diff_last5",

  "progressive_carries_diff",
  "progressive_carries_diff_last3",
  "progressive_carries_diff_last5",

  "pressures_diff",
  "pressures_diff_last3",
  "pressures_diff_last5",

  "recoveries_diff",
  "recoveries_diff_last3",
  "recoveries_diff_last5",

  "recovery_height_diff",
  "recovery_height_diff_last3",
  "recovery_height_diff_last5",

  "shots_diff",
  "shots_diff_last3",
  "shots_diff_last5"
)


context_features <- context_candidates[
  context_candidates %in% names(simulator_dt)
]


# ============================================================
# 9. COLUMNAS BASE
# ============================================================

base_columns <- c(
  "match_id",
  "match_date",
  "minute",
  "opponent",
  "venue",
  "current_tactical_state",
  "future_tactical_state_5min",
  "game_state_num",
  "split"
)


simulator_columns <- unique(
  c(
    base_columns,
    available_model_features,
    context_features
  )
)


simulator_columns <- simulator_columns[
  simulator_columns %in% names(simulator_dt)
]


female_simulator_context <- simulator_dt[
  ,
  ..simulator_columns
]


setorder(
  female_simulator_context,
  match_date,
  match_id,
  minute
)


# ============================================================
# 10. GARANTIZAR QUE TEST NO ESTÉ EN SIMULADOR
# ============================================================

if (
  "split" %in%
  names(female_simulator_context)
) {

  if (
    female_simulator_context[
      split == "TEST",
      .N
    ] > 0
  ) {

    stop(
      "TEST apareció dentro del simulador."
    )
  }
}


# ============================================================
# 11. HISTORIAL DE PARTIDOS
# ============================================================

history_candidates <- c(
  "match_id",
  "match_date",
  "home_team",
  "away_team",
  "opponent",
  "venue",
  "home_score",
  "away_score",
  "america_score",
  "opponent_score"
)


history_columns <- history_candidates[
  history_candidates %in% names(matches)
]


female_match_history <- matches[
  ,
  ..history_columns
]


if (
  all(
    c(
      "home_score",
      "away_score"
    ) %in%
    names(female_match_history)
  )
) {

  female_match_history[
    ,
    played :=
      !is.na(home_score)
      &
      !is.na(away_score)
      &
      match_date <= Sys.Date()
  ]

} else {

  female_match_history[
    ,
    played :=
      match_date <= Sys.Date()
  ]
}


setorder(
  female_match_history,
  match_date
)


# ============================================================
# 12. HISTORIAL DEL MODELO FEMENIL
#
# SOLO PREDICCIONES TEST YA GENERADAS
# ============================================================

female_change_history <- copy(
  predictions
)


required_prediction_columns <- c(
  "match_id",
  "match_date",
  "minute",
  "opponent",
  "venue",
  "current_state",
  "future_state",
  "actual_change",
  "predicted_change",
  "probability_change",
  "threshold"
)


missing_prediction_columns <- setdiff(
  required_prediction_columns,
  names(female_change_history)
)


if (length(missing_prediction_columns) > 0) {

  stop(
    paste(
      "Faltan columnas en predictions:",
      paste(
        missing_prediction_columns,
        collapse = ", "
      )
    )
  )
}


setorder(
  female_change_history,
  match_date,
  match_id,
  minute
)


# ============================================================
# 13. MÉTRICAS
# ============================================================

if (
  !all(
    c(
      "metric",
      "value"
    ) %in%
    names(metrics)
  )
) {

  stop(
    "El archivo de métricas no contiene metric/value."
  )
}


female_change_metrics <- copy(
  metrics
)


female_change_features <- copy(
  model_features
)


female_change_selection <- copy(
  selection
)


# ============================================================
# 14. SALIDAS
# ============================================================

simulator_output <- file.path(
  dashboard_dir,
  "female_simulator_context.csv"
)

history_output <- file.path(
  dashboard_dir,
  "female_match_history.csv"
)

change_history_output <- file.path(
  dashboard_dir,
  "female_change_history.csv"
)

metrics_output <- file.path(
  dashboard_dir,
  "female_change_metrics.csv"
)

features_output <- file.path(
  dashboard_dir,
  "female_change_model_features.csv"
)

selection_output <- file.path(
  dashboard_dir,
  "female_change_selection.csv"
)


# ============================================================
# 15. EXPORTAR
# ============================================================

fwrite(
  female_simulator_context,
  simulator_output
)

fwrite(
  female_match_history,
  history_output
)

fwrite(
  female_change_history,
  change_history_output
)

fwrite(
  female_change_metrics,
  metrics_output
)

fwrite(
  female_change_features,
  features_output
)

fwrite(
  female_change_selection,
  selection_output
)


# ============================================================
# 16. VALIDACIÓN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACION EXPORT FEMENIL\n")
cat("============================================\n\n")


cat(
  "SIMULADOR\n"
)

cat(
  "Filas:",
  nrow(female_simulator_context),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(
    female_simulator_context$match_id
  ),
  "\n"
)


if (
  "split" %in%
  names(female_simulator_context)
) {

  cat(
    "Splits:\n"
  )

  print(
    table(
      female_simulator_context$split
    )
  )
}


cat(
  "\nHISTORIAL PARTIDOS\n"
)

cat(
  "Total:",
  nrow(female_match_history),
  "\n"
)

cat(
  "Jugados:",
  sum(
    female_match_history$played,
    na.rm = TRUE
  ),
  "\n"
)


cat(
  "\nHISTORIAL MODELO TEST\n"
)

cat(
  "Filas:",
  nrow(female_change_history),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(
    female_change_history$match_id
  ),
  "\n"
)


cat(
  "\nMÉTRICAS:",
  nrow(female_change_metrics),
  "\n"
)


cat(
  "FEATURES:",
  nrow(female_change_features),
  "\n"
)


cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Archivos creados:\n\n"
)

cat(
  simulator_output,
  "\n"
)

cat(
  history_output,
  "\n"
)

cat(
  change_history_output,
  "\n"
)

cat(
  metrics_output,
  "\n"
)

cat(
  features_output,
  "\n"
)

cat(
  selection_output,
  "\n"
)