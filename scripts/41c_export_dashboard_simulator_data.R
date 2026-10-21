library(data.table)

# ============================================================
# 1. CARGAR DATASET
# ============================================================

dt <- fread(
  "data/processed/america_male/model_ready_predictive.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

matches <- fread(
  "data/processed/america_male/america_matches.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

# ============================================================
# 2. ESTADOS
# ============================================================

state_levels <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

# ============================================================
# 3. ONE HOT DEL ESTADO ACTUAL
# ============================================================

for (state in state_levels) {

  new_col <- paste0(
    "current_state_",
    tolower(state)
  )

  dt[
    ,
    (new_col) :=
      as.integer(
        as.character(current_tactical_state) == state
      )
  ]
}

state_features <- paste0(
  "current_state_",
  tolower(state_levels)
)

# ============================================================
# 4. FEATURES EXACTAS DEL MODELO
# ============================================================

model_features <- c(

  "minute",
  "game_state_num",

  state_features,

  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_pass_diff_last3",
  "america_progressive_carries_last3",
  "america_progressive_passes_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3"
)

missing_features <- setdiff(
  model_features,
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
# 5. AGREGAR VENUE SI HACE FALTA
# ============================================================

if (!"venue" %in% names(dt)) {

  venue_data <- unique(
    matches[
      ,
      .(
        match_id,
        venue
      )
    ]
  )

  dt <- merge(
    dt,
    venue_data,
    by = "match_id",
    all.x = TRUE
  )
}

# ============================================================
# 6. BUSCAR COLUMNA DE POSESIÓN
# ============================================================

possession_candidates <- c(
  "america_possession_share_last5",
  "america_possession_share_5_last5",
  "america_possession_last5",
  "possession_share_last5",
  "america_possession_share",
  "america_possession"
)

possession_col <- possession_candidates[
  possession_candidates %in%
    names(dt)
]

if (length(possession_col) > 0) {

  possession_col <- possession_col[1]

  dt[
    ,
    america_possession_context :=
      as.numeric(
        get(possession_col)
      )
  ]

  cat(
    "Possession variable:",
    possession_col,
    "\n"
  )

} else {

  # No inventamos posesión.
  dt[
    ,
    america_possession_context :=
      NA_real_
  ]

  cat(
    "No explicit possession variable found.\n"
  )
}

# ============================================================
# 7. METADATA
# ============================================================

metadata_cols <- c(
  "match_id",
  "match_date",
  "season",
  "opponent",
  "venue",
  "dataset_split",
  "current_tactical_state",
  "america_possession_context"
)

metadata_cols <- metadata_cols[
  metadata_cols %in%
    names(dt)
]

# ============================================================
# 8. DATASET LIGERO
# ============================================================

keep_cols <- unique(
  c(
    metadata_cols,
    model_features
  )
)

simulator_context <- dt[
  ,
  ..keep_cols
]

# ============================================================
# 9. SOLO TRAIN + VALIDATION PARA COMPLETAR CONTEXTOS
#
# TEST queda fuera de la búsqueda de vecinos.
# ============================================================

simulator_context <- simulator_context[
  dataset_split %in%
    c(
      "TRAIN",
      "VALIDATION"
    )
]

# ============================================================
# 10. GUARDAR
# ============================================================

dir.create(
  "dashboard/data",
  recursive = TRUE,
  showWarnings = FALSE
)

fwrite(
  simulator_context,
  "dashboard/data/simulator_context.csv",
  bom = TRUE
)

# ============================================================
# 11. VALIDACIÓN
# ============================================================

simulator_summary <- data.table(
  metric = c(
    "Rows",
    "Matches",
    "Opponents",
    "Features",
    "Possession available"
  ),

  value = c(
    nrow(simulator_context),
    uniqueN(simulator_context$match_id),
    uniqueN(simulator_context$opponent),
    length(model_features),
    sum(
      !is.na(
        simulator_context$
          america_possession_context
      )
    )
  )
)

View(
  simulator_summary
)

cat(
  "\nDashboard simulator data exported.\n"
)