library(data.table)

# ============================================================
# 1. CARGAR DATASET FINAL 5 MIN
# ============================================================

dt5 <- fread(
  "data/processed/america_male/model_with_tactical_states.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

setorder(
  dt5,
  match_id,
  minute
)


# ============================================================
# 2. CARGAR DATASET 3 MIN
#
# Lo usaremos como contexto adicional.
# NO como target.
# ============================================================

dt3 <- fread(
  "data/processed/america_male/model_with_tactical_states_3min.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

setorder(
  dt3,
  match_id,
  minute
)


# ============================================================
# 3. SELECCIONAR INFORMACIÓN 3 MIN
# ============================================================

features_3min <- dt3[
  ,
  .(
    match_id,
    minute,

    tactical_state_3min,

    tactical_state_confidence_3min,

    america_field_tilt_last3,

    entries_final_third_diff_last3,

    entries_box_diff_last3,

    progressive_pass_diff_last3,

    pressure_diff_last3,

    recovery_height_diff_last3,

    america_progressive_carries_last3,

    america_progressive_passes_last3,

    america_local_superiority_10_last3,

    america_pressure_share_5_last3,

    america_nearest_opponent_distance_last3,

    america_support_share_10_last3
  )
]


# ============================================================
# 4. UNIR CONTEXTO 3 MIN CON BASE 5 MIN
# ============================================================

model_data <- merge(
  dt5,
  features_3min,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)

setorder(
  model_data,
  match_id,
  minute
)


# ============================================================
# 5. CREAR TARGET t + 5
#
# Estado táctico dentro de cinco minutos.
# ============================================================

model_data[
  ,
  future_tactical_state_5min :=
    shift(
      tactical_state,
      n = 5,
      type = "lead"
    ),
  by = match_id
]


# ============================================================
# 6. ESTADO ACTUAL
# ============================================================

model_data[
  ,
  current_tactical_state :=
    tactical_state
]


# ============================================================
# 7. ¿CAMBIARÁ EL ESTADO?
# ============================================================

model_data[
  ,
  tactical_change_next5 :=
    !is.na(current_tactical_state) &
    !is.na(future_tactical_state_5min) &
    current_tactical_state !=
      future_tactical_state_5min
]


# ============================================================
# 8. MINUTO DEL TARGET
# ============================================================

model_data[
  ,
  target_minute :=
    minute + 5
]


# ============================================================
# 9. FILTRAR FILAS MODELABLES
# ============================================================

model_ready <- model_data[
  !is.na(current_tactical_state) &
  !is.na(future_tactical_state_5min)
]


cat("\n=====================================\n")
cat("TARGET PREDICTIVO\n")
cat("=====================================\n")

cat(
  "Observaciones:",
  nrow(model_ready),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(model_ready$match_id),
  "\n"
)


# ============================================================
# 10. FECHA
# ============================================================

model_ready[
  ,
  match_date :=
    as.IDate(
      match_date
    )
]


# ============================================================
# 11. VALIDAR TEMPORALIDAD
#
# target_minute siempre debe ser mayor
# que minute.
# ============================================================

temporal_check <- model_ready[
  ,
  .(
    rows = .N,

    bad_rows =
      sum(
        target_minute <= minute
      ),

    min_horizon =
      min(
        target_minute - minute
      ),

    max_horizon =
      max(
        target_minute - minute
      )
  )
]


# ============================================================
# 12. CREAR TABLA ÚNICA DE PARTIDOS
# ============================================================

matches <- unique(
  model_ready[
    ,
    .(
      match_id,
      match_date,
      season,
      opponent,
      venue
    )
  ]
)

setorder(
  matches,
  match_date,
  match_id
)


# ============================================================
# 13. SPLIT TEMPORAL POR TEMPORADA
#
# TRAIN:
# 2021/22
# 2022/23
# 2023/24
#
# VALIDATION:
# 2024/25
#
# TEST:
# 2025/26
# 2026/27
#
# Esto es completamente temporal.
# ============================================================

matches[
  ,
  dataset_split := fifelse(

    season %in% c(
      "2021/2022",
      "2022/2023",
      "2023/2024"
    ),

    "TRAIN",

    fifelse(

      season == "2024/2025",

      "VALIDATION",

      fifelse(

        season %in% c(
          "2025/2026",
          "2026/2027"
        ),

        "TEST",

        NA_character_
      )
    )
  )
]


# ============================================================
# 14. VERIFICAR SI ALGUNA TEMPORADA QUEDÓ FUERA
# ============================================================

unknown_seasons <- matches[
  is.na(dataset_split)
]

if (nrow(unknown_seasons) > 0) {

  View(
    unknown_seasons
  )

  stop(
    "Hay temporadas sin asignar al split."
  )
}


# ============================================================
# 15. UNIR SPLIT
# ============================================================

model_ready <- merge(
  model_ready,
  matches[
    ,
    .(
      match_id,
      dataset_split
    )
  ],
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 16. VALIDAR QUE UN PARTIDO SOLO ESTÉ EN UN SPLIT
# ============================================================

split_leakage_check <- model_ready[
  ,
  .(
    number_of_splits =
      uniqueN(
        dataset_split
      )
  ),
  by = match_id
][
  number_of_splits > 1
]


if (nrow(split_leakage_check) > 0) {

  stop(
    "ERROR: hay partidos presentes en más de un split."
  )
}


# ============================================================
# 17. DISTRIBUCIÓN DEL TARGET
# ============================================================

target_distribution <- model_ready[
  ,
  .N,
  by = future_tactical_state_5min
]

target_distribution[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    )
]

setorder(
  target_distribution,
  -N
)


# ============================================================
# 18. RESUMEN DEL SPLIT
# ============================================================

split_summary <- model_ready[
  ,
  .(
    observations = .N,

    matches =
      uniqueN(
        match_id
      ),

    first_date =
      min(
        match_date
      ),

    last_date =
      max(
        match_date
      )
  ),
  by = dataset_split
]


# ============================================================
# 19. TARGET POR SPLIT
# ============================================================

target_by_split <- model_ready[
  ,
  .N,
  by = .(
    dataset_split,
    future_tactical_state_5min
  )
]

target_by_split[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    ),
  by = dataset_split
]


# ============================================================
# 20. BASELINE 1:
# PERSISTENCIA
#
# "Dentro de 5 min seguirá en el mismo estado".
# ============================================================

model_ready[
  ,
  baseline_persistence :=
    current_tactical_state
]


model_ready[
  ,
  baseline_persistence_correct :=
    baseline_persistence ==
    future_tactical_state_5min
]


# ============================================================
# 21. BASELINE 2:
# CLASE MÁS FRECUENTE EN TRAIN
#
# Importante:
# la clase se obtiene SOLO de training.
# ============================================================

majority_state_train <- model_ready[
  dataset_split == "TRAIN",
  .N,
  by = future_tactical_state_5min
][
  which.max(N),
  future_tactical_state_5min
]


model_ready[
  ,
  baseline_majority :=
    majority_state_train
]


model_ready[
  ,
  baseline_majority_correct :=
    baseline_majority ==
    future_tactical_state_5min
]


# ============================================================
# 22. MÉTRICAS DE LOS BASELINES
# ============================================================

baseline_summary <- model_ready[
  ,
  .(

    observations = .N,

    persistence_accuracy =
      round(
        mean(
          baseline_persistence_correct
        ) *
          100,
        2
      ),

    majority_accuracy =
      round(
        mean(
          baseline_majority_correct
        ) *
          100,
        2
      ),

    tactical_change_rate =
      round(
        mean(
          tactical_change_next5
        ) *
          100,
        2
      )
  ),
  by = dataset_split
]


# ============================================================
# 23. MATRIZ DE TRANSICIÓN DE ESTADOS
# ============================================================

state_transition_matrix <- model_ready[
  ,
  .N,
  by = .(
    current_tactical_state,
    future_tactical_state_5min
  )
]

state_transition_matrix[
  ,
  transition_pct :=
    round(
      N /
        sum(N) *
        100,
      2
    ),
  by = current_tactical_state
]


# ============================================================
# 24. CHEQUEO DE LEAKAGE DE COLUMNAS
#
# Estas columnas NO se deberán usar como X.
# ============================================================

forbidden_features <- c(

  "future_tactical_state_5min",

  "target_minute",

  "baseline_persistence",

  "baseline_persistence_correct",

  "baseline_majority",

  "baseline_majority_correct",

  "tactical_change_next5"
)


leakage_feature_check <- data.table(

  variable =
    forbidden_features,

  present =
    forbidden_features %in%
      names(model_ready)
)


# ============================================================
# 25. GUARDAR DATASET FINAL
# ============================================================

fwrite(
  model_ready,
  "data/processed/america_male/model_ready_predictive.csv",
  bom = TRUE,
  na = "NA"
)


fwrite(
  matches,
  "data/processed/america_male/predictive_match_split.csv",
  bom = TRUE,
  na = "NA"
)


# ============================================================
# 26. GUARDAR OUTPUTS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  target_distribution,
  "outputs/models/predictive_target_distribution.csv",
  bom = TRUE
)


fwrite(
  split_summary,
  "outputs/models/predictive_split_summary.csv",
  bom = TRUE
)


fwrite(
  baseline_summary,
  "outputs/models/predictive_baseline_summary.csv",
  bom = TRUE
)


fwrite(
  state_transition_matrix,
  "outputs/models/tactical_state_transition_matrix.csv",
  bom = TRUE
)


# ============================================================
# 27. SOLO LAS VISUALIZACIONES IMPORTANTES
# ============================================================

View(
  split_summary
)

View(
  baseline_summary
)

View(
  target_distribution
)


# ============================================================
# 28. RESUMEN TERMINAL
# ============================================================

cat("\n=====================================\n")
cat("PREDICTIVE TARGET READY\n")
cat("=====================================\n")

cat(
  "Observaciones:",
  nrow(model_ready),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(model_ready$match_id),
  "\n"
)

cat(
  "Majority baseline:",
  majority_state_train,
  "\n"
)

cat(
  "Leakage partidos:",
  nrow(split_leakage_check),
  "\n"
)

cat(
  "Horizonte:",
  temporal_check$min_horizon,
  "minutos\n"
)