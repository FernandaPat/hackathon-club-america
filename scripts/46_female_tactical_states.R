# ============================================================
# 46_female_tactical_states.R
# CLUB AMÉRICA FEMENIL
# ESTADOS TÁCTICOS DATA-DRIVEN
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

data_dir <- file.path(
  project_root,
  "data",
  "processed",
  "america_female"
)

input_file <- file.path(
  data_dir,
  "rolling_features_female.csv"
)

output_file <- file.path(
  data_dir,
  "model_with_tactical_states_female.csv"
)

threshold_file <- file.path(
  data_dir,
  "female_tactical_state_thresholds.csv"
)


# ============================================================
# 2. CARGAR
# ============================================================

if (!file.exists(input_file)) {

  stop(
    paste(
      "No existe:",
      input_file
    )
  )
}


dt <- fread(
  input_file
)


cat("\n")
cat("============================================\n")
cat("COACHLENS - FEMALE TACTICAL STATES\n")
cat("============================================\n\n")


cat(
  "Filas:",
  nrow(dt),
  "\n"
)


cat(
  "Partidos:",
  uniqueN(dt$match_id),
  "\n"
)


cat(
  "Columnas:",
  ncol(dt),
  "\n\n"
)


# ============================================================
# 3. FEATURES PARA DEFINIR ESTADOS
# ============================================================

state_features <- c(
  "america_field_tilt_last5",
  "america_possession_context_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_passes_diff_last5",
  "pressures_diff_last5",
  "recovery_height_diff_last5"
)


missing_features <- setdiff(
  state_features,
  names(dt)
)


if (length(missing_features) > 0) {

  stop(
    paste(
      "Faltan features:",
      paste(
        missing_features,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 4. NUMÉRICO
# ============================================================

for (feature in state_features) {

  dt[
    ,
    (feature) := as.numeric(
      get(feature)
    )
  ]
}


# ============================================================
# 5. FUNCIÓN CUANTIL
# ============================================================

safe_quantile <- function(
  x,
  probability
) {

  x <- x[
    is.finite(x)
  ]


  if (length(x) == 0) {

    return(
      NA_real_
    )
  }


  as.numeric(
    quantile(
      x,
      probs = probability,
      na.rm = TRUE,
      type = 7
    )
  )
}


# ============================================================
# 6. THRESHOLDS DESDE LOS DATOS
# ============================================================

threshold_table <- data.table(
  feature = state_features,
  q25 = NA_real_,
  q50 = NA_real_,
  q75 = NA_real_
)


for (i in seq_len(nrow(threshold_table))) {

  feature_name <- threshold_table$feature[i]

  values <- dt[
    ,
    get(feature_name)
  ]


  threshold_table[
    i,
    q25 := safe_quantile(
      values,
      0.25
    )
  ]


  threshold_table[
    i,
    q50 := safe_quantile(
      values,
      0.50
    )
  ]


  threshold_table[
    i,
    q75 := safe_quantile(
      values,
      0.75
    )
  ]
}


cat(
  "Thresholds calculados desde América Femenil:\n\n"
)


print(
  threshold_table
)


# ============================================================
# 7. FUNCIÓN PARA LEER THRESHOLD
# ============================================================

get_threshold <- function(
  feature_name,
  quantile_name
) {

  value <- threshold_table[
    feature == feature_name,
    get(quantile_name)
  ]


  if (length(value) == 0) {

    return(
      NA_real_
    )
  }


  as.numeric(
    value[1]
  )
}


# ============================================================
# 8. EXTRAER THRESHOLDS
# ============================================================

field_q25 <- get_threshold(
  "america_field_tilt_last5",
  "q25"
)

field_q50 <- get_threshold(
  "america_field_tilt_last5",
  "q50"
)

field_q75 <- get_threshold(
  "america_field_tilt_last5",
  "q75"
)


possession_q25 <- get_threshold(
  "america_possession_context_last5",
  "q25"
)

possession_q50 <- get_threshold(
  "america_possession_context_last5",
  "q50"
)

possession_q75 <- get_threshold(
  "america_possession_context_last5",
  "q75"
)


third_q25 <- get_threshold(
  "entries_final_third_diff_last5",
  "q25"
)

third_q50 <- get_threshold(
  "entries_final_third_diff_last5",
  "q50"
)

third_q75 <- get_threshold(
  "entries_final_third_diff_last5",
  "q75"
)


box_q25 <- get_threshold(
  "entries_box_diff_last5",
  "q25"
)

box_q50 <- get_threshold(
  "entries_box_diff_last5",
  "q50"
)

box_q75 <- get_threshold(
  "entries_box_diff_last5",
  "q75"
)


progress_q25 <- get_threshold(
  "progressive_passes_diff_last5",
  "q25"
)

progress_q50 <- get_threshold(
  "progressive_passes_diff_last5",
  "q50"
)

progress_q75 <- get_threshold(
  "progressive_passes_diff_last5",
  "q75"
)


pressure_q25 <- get_threshold(
  "pressures_diff_last5",
  "q25"
)

pressure_q50 <- get_threshold(
  "pressures_diff_last5",
  "q50"
)

pressure_q75 <- get_threshold(
  "pressures_diff_last5",
  "q75"
)


recovery_q25 <- get_threshold(
  "recovery_height_diff_last5",
  "q25"
)

recovery_q50 <- get_threshold(
  "recovery_height_diff_last5",
  "q50"
)

recovery_q75 <- get_threshold(
  "recovery_height_diff_last5",
  "q75"
)


# ============================================================
# 9. SCORE CONTROL
# ============================================================

dt[
  ,
  control_score :=
    as.integer(
      america_field_tilt_last5 >= field_q50
    )
    +
    as.integer(
      america_possession_context_last5 >= possession_q50
    )
    +
    as.integer(
      progressive_passes_diff_last5 >= progress_q50
    )
]


# ============================================================
# 10. SCORE PRESSURE
# ============================================================

dt[
  ,
  pressure_score :=
    as.integer(
      pressures_diff_last5 >= pressure_q75
    )
    +
    as.integer(
      recovery_height_diff_last5 >= recovery_q50
    )
    +
    as.integer(
      entries_final_third_diff_last5 >= third_q50
    )
]


# ============================================================
# 11. SCORE TRANSITION
# ============================================================

dt[
  ,
  transition_score :=
    as.integer(
      progressive_passes_diff_last5 >= progress_q75
    )
    +
    as.integer(
      entries_final_third_diff_last5 >= third_q75
    )
    +
    as.integer(
      entries_box_diff_last5 >= box_q75
    )
]


# ============================================================
# 12. SCORE PROTECTION
# ============================================================

dt[
  ,
  protection_score :=
    as.integer(
      america_field_tilt_last5 <= field_q25
    )
    +
    as.integer(
      america_possession_context_last5 <= possession_q25
    )
    +
    as.integer(
      entries_final_third_diff_last5 <= third_q25
    )
]


# ============================================================
# 13. SCORE URGENCY
# ============================================================

dt[
  ,
  urgency_score :=
    as.integer(
      entries_box_diff_last5 >= box_q75
    )
    +
    as.integer(
      progressive_passes_diff_last5 >= progress_q75
    )
    +
    as.integer(
      pressures_diff_last5 >= pressure_q75
    )
]


# ============================================================
# 14. CONTEXTO DEL MARCADOR
# ============================================================

if (
  "game_state_num"
  %in%
  names(dt)
) {

  dt[
    game_state_num > 0,
    protection_score :=
      protection_score + 1
  ]


  dt[
    game_state_num < 0,
    urgency_score :=
      urgency_score + 1
  ]
}


# ============================================================
# 15. MATRIZ DE SCORES
# ============================================================

score_columns <- c(
  "control_score",
  "pressure_score",
  "transition_score",
  "protection_score",
  "urgency_score"
)


score_matrix <- as.matrix(
  dt[
    ,
    ..score_columns
  ]
)


score_matrix[
  is.na(
    score_matrix
  )
] <- 0


# ============================================================
# 16. ESTADO GANADOR
# ============================================================

state_names <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)


winner_index <- max.col(
  score_matrix,
  ties.method = "first"
)


dt[
  ,
  current_tactical_state :=
    state_names[
      winner_index
    ]
]


# ============================================================
# 17. DETECTAR EMPATES
# ============================================================

row_maximum <- apply(
  score_matrix,
  1,
  max
)


tie_count <- rowSums(
  score_matrix
  ==
  row_maximum
)


dt[
  ,
  tactical_state_tie :=
    tie_count > 1
]


# ============================================================
# 18. TARGET +3 MIN
# ============================================================

dt[
  ,
  future_tactical_state_3min :=
    shift(
      current_tactical_state,
      n = 3,
      type = "lead"
    ),
  by = match_id
]


# ============================================================
# 19. TARGET +5 MIN
# ============================================================

dt[
  ,
  future_tactical_state_5min :=
    shift(
      current_tactical_state,
      n = 5,
      type = "lead"
    ),
  by = match_id
]


# ============================================================
# 20. USABLE
# ============================================================

dt[
  ,
  usable_for_model_3min :=
    !is.na(
      future_tactical_state_3min
    )
]


dt[
  ,
  usable_for_model_5min :=
    !is.na(
      future_tactical_state_5min
    )
]


# ============================================================
# 21. CAMBIO DE ESTADO
# ============================================================

dt[
  ,
  state_changes_5min :=
    current_tactical_state
    !=
    future_tactical_state_5min
]


# ============================================================
# 22. DISTRIBUCIÓN ACTUAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("DISTRIBUCION ESTADOS ACTUALES\n")
cat("============================================\n\n")


state_distribution <- dt[
  ,
  .(
    observations = .N
  ),
  by = current_tactical_state
]


state_distribution[
  ,
  share :=
    observations
    /
    sum(observations)
]


setorder(
  state_distribution,
  -observations
)


print(
  state_distribution
)


# ============================================================
# 23. DISTRIBUCIÓN TARGET +5
# ============================================================

cat("\n")
cat("============================================\n")
cat("DISTRIBUCION TARGET +5 MIN\n")
cat("============================================\n\n")


future_distribution <- dt[
  usable_for_model_5min == TRUE,
  .(
    observations = .N
  ),
  by = future_tactical_state_5min
]


future_distribution[
  ,
  share :=
    observations
    /
    sum(observations)
]


setorder(
  future_distribution,
  -observations
)


print(
  future_distribution
)


# ============================================================
# 24. EMPATES
# ============================================================

tie_rows <- dt[
  tactical_state_tie == TRUE,
  .N
]


tie_share <- mean(
  dt$tactical_state_tie,
  na.rm = TRUE
)


cat(
  "\nFilas con empate:",
  tie_rows,
  "\n"
)


cat(
  "Porcentaje de empates:",
  round(
    tie_share * 100,
    2
  ),
  "%\n"
)


# ============================================================
# 25. CAMBIO EN 5 MIN
# ============================================================

change_rate <- mean(
  dt$state_changes_5min,
  na.rm = TRUE
)


cat(
  "\nCambio de estado en +5 min:",
  round(
    change_rate * 100,
    2
  ),
  "%\n"
)


# ============================================================
# 26. VALIDACIONES
# ============================================================

current_classes <- uniqueN(
  dt$current_tactical_state
)


future_classes <- uniqueN(
  dt[
    usable_for_model_5min == TRUE,
    future_tactical_state_5min
  ]
)


cat(
  "\nClases actuales:",
  current_classes,
  "\n"
)


cat(
  "Clases target +5:",
  future_classes,
  "\n"
)


if (current_classes < 2) {

  stop(
    "Solo se generó una clase táctica."
  )
}


if (future_classes < 2) {

  stop(
    "El target +5 contiene menos de dos clases."
  )
}


# ============================================================
# 27. EXPORTAR
# ============================================================

fwrite(
  dt,
  output_file
)


fwrite(
  threshold_table,
  threshold_file
)


cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Archivo estados:\n",
  output_file,
  "\n\n"
)


cat(
  "Archivo thresholds:\n",
  threshold_file,
  "\n"
)