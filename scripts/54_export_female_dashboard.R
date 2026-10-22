# ============================================================
# 54_export_female_dashboard.R
# CLUB AMÉRICA FEMENIL
#
# EXPORTA SOLO LOS ARCHIVOS MÍNIMOS NECESARIOS
# PARA EL DASHBOARD STREAMLIT.
#
# Salidas:
# - dashboard/data/female_simulator_context.csv
# - dashboard/data/female_match_history.csv
# - dashboard/data/female_change_metrics.csv
# - dashboard/data/female_change_model_features.csv
# - dashboard/data/female_change_selection.csv
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


# ============================================================
# 2. VALIDAR ARCHIVOS
# ============================================================

required_files <- c(
  model_ready_file,
  matches_file,
  metrics_file,
  features_file,
  selection_file
)


missing_files <- required_files[
  !file.exists(
    required_files
  )
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


cat("\n")
cat("============================================\n")
cat("EXPORT FEMALE DASHBOARD\n")
cat("============================================\n\n")


cat(
  "Observaciones model_ready:",
  nrow(model_dt),
  "\n"
)


cat(
  "Partidos model_ready:",
  uniqueN(model_dt$match_id),
  "\n\n"
)


# ============================================================
# 4. VALIDAR COLUMNAS DEL MODELO
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
# 5. FECHAS
# ============================================================

model_dt[
  ,
  match_date :=
    as.IDate(
      match_date
    )
]


matches[
  ,
  match_date :=
    as.IDate(
      match_date
    )
]


# ============================================================
# 6. CONTEXTO DEL SIMULADOR
#
# IMPORTANTE:
# SOLO TRAIN + VALIDATION
# TEST NO SE UTILIZA PARA CONSTRUIR ESCENARIOS.
# ============================================================

simulator_dt <- model_dt[
  split %in% c(
    "TRAIN",
    "VALIDATION"
  )
]


if (nrow(simulator_dt) == 0) {

  stop(
    "No existen filas TRAIN/VALIDATION para simulador."
  )
}


# ============================================================
# 7. FEATURES REALES DEL MODELO
#
# El archivo female_change_features.csv contiene
# las variables utilizadas por el modelo y sus medianas.
# ============================================================

if (
  !"feature"
  %in%
  names(model_features)
) {

  stop(
    "female_change_features.csv no contiene columna feature."
  )
}


change_features <- unique(
  model_features$feature
)


# ============================================================
# 8. FEATURES QUE YA EXISTEN EN DATASET
#
# Las current_state_* se reconstruyen en Python,
# así que aquí exportamos las variables base disponibles.
# ============================================================

available_change_features <- change_features[
  change_features
  %in%
  names(simulator_dt)
]


state_feature_names <- change_features[
  grepl(
    "^current_state_",
    change_features
  )
]


# ============================================================
# 9. FEATURES DE CONTEXTO ADICIONALES PARA SIMILITUD
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
  context_candidates
  %in%
  names(simulator_dt)
]


# ============================================================
# 10. COLUMNAS BASE
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


# ============================================================
# 11. ARMAR EXPORT DE SIMULADOR
# ============================================================

simulator_columns <- unique(
  c(
    base_columns,
    available_change_features,
    context_features
  )
)


simulator_columns <- simulator_columns[
  simulator_columns
  %in%
  names(simulator_dt)
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
# 12. VALIDAR QUE TEST NO ESTÉ PRESENTE
# ============================================================

if (
  "split"
  %in%
  names(
    female_simulator_context
  )
) {

  test_rows <- female_simulator_context[
    split == "TEST",
    .N
  ]


  if (test_rows > 0) {

    stop(
      "Se detectaron filas TEST dentro del simulador."
    )
  }
}


# ============================================================
# 13. HISTORIAL FEMENIL
# ============================================================

required_match_columns <- c(
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


history_columns <- required_match_columns[
  required_match_columns
  %in%
  names(matches)
]


female_match_history <- matches[
  ,
  ..history_columns
]


# ============================================================
# 14. PARTIDOS JUGADOS VS FUTUROS
# ============================================================

if (
  "home_score"
  %in%
  names(female_match_history)
  &&
  "away_score"
  %in%
  names(female_match_history)
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
# 15. MÉTRICAS
# ============================================================

if (
  !all(
    c(
      "metric",
      "value"
    )
    %in%
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


# ============================================================
# 16. VALIDAR MÉTRICAS IMPORTANTES
# ============================================================

important_metrics <- c(
  "accuracy",
  "balanced_accuracy",
  "auc",
  "sensitivity_change",
  "specificity_no_change",
  "precision_change",
  "f1_change",
  "logloss",
  "brier_score",
  "majority_baseline_accuracy",
  "majority_baseline_balanced",
  "selected_threshold",
  "test_observations",
  "test_matches"
)


missing_metrics <- setdiff(
  important_metrics,
  female_change_metrics$metric
)


if (length(missing_metrics) > 0) {

  warning(
    paste(
      "Faltan algunas métricas:",
      paste(
        missing_metrics,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 17. SELECCIÓN DEL MODELO
# ============================================================

female_change_selection <- copy(
  selection
)


# ============================================================
# 18. FEATURES + MEDIANAS
# ============================================================

female_change_features <- copy(
  model_features
)


# ============================================================
# 19. RUTAS DE SALIDA
# ============================================================

simulator_output <- file.path(
  dashboard_dir,
  "female_simulator_context.csv"
)


history_output <- file.path(
  dashboard_dir,
  "female_match_history.csv"
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
# 20. EXPORTAR
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
# 21. VALIDACIÓN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACION EXPORT FEMENIL\n")
cat("============================================\n\n")


cat(
  "Simulador:\n"
)


cat(
  "  Filas:",
  nrow(
    female_simulator_context
  ),
  "\n"
)


cat(
  "  Partidos:",
  uniqueN(
    female_simulator_context$match_id
  ),
  "\n"
)


cat(
  "  Columnas:",
  ncol(
    female_simulator_context
  ),
  "\n\n"
)


cat(
  "Splits incluidos:\n"
)


if (
  "split"
  %in%
  names(
    female_simulator_context
  )
) {

  print(
    table(
      female_simulator_context$split
    )
  )
}


cat(
  "\nHistorial:\n"
)


cat(
  "  Partidos totales:",
  nrow(
    female_match_history
  ),
  "\n"
)


if (
  "played"
  %in%
  names(
    female_match_history
  )
) {

  cat(
    "  Partidos jugados:",
    sum(
      female_match_history$played,
      na.rm = TRUE
    ),
    "\n"
  )


  cat(
    "  Partidos futuros/no terminados:",
    sum(
      !female_match_history$played,
      na.rm = TRUE
    ),
    "\n"
  )
}


cat(
  "\nMétricas exportadas:",
  nrow(
    female_change_metrics
  ),
  "\n"
)


cat(
  "Features del modelo:",
  nrow(
    female_change_features
  ),
  "\n\n"
)


# ============================================================
# 22. MOSTRAR MÉTRICAS
# ============================================================

cat(
  "Métricas del modelo Femenil:\n\n"
)


print(
  female_change_metrics
)


# ============================================================
# 23. MOSTRAR SELECCIÓN
# ============================================================

cat(
  "\nModelo seleccionado en VALIDATION:\n\n"
)


print(
  female_change_selection
)


# ============================================================
# 24. ARCHIVOS
# ============================================================

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