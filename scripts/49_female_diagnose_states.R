# ============================================================
# 49_female_diagnose_states.R
# CLUB AMÉRICA FEMENIL
# DIAGNÓSTICO DE ESTADOS TÁCTICOS
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
  "model_ready_female.csv"
)

output_dir <- file.path(
  project_root,
  "outputs",
  "models",
  "female",
  "diagnostics"
)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}


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
cat("FEMALE TACTICAL STATE DIAGNOSTICS\n")
cat("============================================\n\n")


# ============================================================
# 3. VALIDACIONES
# ============================================================

required <- c(
  "match_id",
  "match_date",
  "split",
  "minute",
  "current_tactical_state",
  "future_tactical_state_5min"
)


missing <- setdiff(
  required,
  names(dt)
)


if (length(missing) > 0) {

  stop(
    paste(
      "Faltan columnas:",
      paste(
        missing,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 4. DISTRIBUCIÓN CURRENT STATE
# ============================================================

current_distribution <- dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    split,
    current_tactical_state
  )
]


current_distribution[
  ,
  share :=
    observations /
    sum(observations),
  by = split
]


cat(
  "============================================\n"
)
cat(
  "CURRENT STATE POR SPLIT\n"
)
cat(
  "============================================\n\n"
)


print(
  current_distribution[
    order(
      split,
      -observations
    )
  ]
)


# ============================================================
# 5. DISTRIBUCIÓN TARGET
# ============================================================

future_distribution <- dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    split,
    future_tactical_state_5min
  )
]


future_distribution[
  ,
  share :=
    observations /
    sum(observations),
  by = split
]


cat("\n")
cat("============================================\n")
cat("TARGET +5 MIN POR SPLIT\n")
cat("============================================\n\n")


print(
  future_distribution[
    order(
      split,
      -observations
    )
  ]
)


# ============================================================
# 6. MATRIZ DE TRANSICIÓN
# ============================================================

transition_matrix <- dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    current_tactical_state,
    future_tactical_state_5min
  )
]


transition_matrix[
  ,
  probability :=
    observations /
    sum(observations),
  by = current_tactical_state
]


cat("\n")
cat("============================================\n")
cat("MATRIZ DE TRANSICIÓN\n")
cat("============================================\n\n")


print(
  transition_matrix[
    order(
      current_tactical_state,
      -probability
    )
  ]
)


# ============================================================
# 7. PERSISTENCIA
# ============================================================

dt[
  ,
  same_state_5min :=
    current_tactical_state ==
    future_tactical_state_5min
]


persistence_overall <- mean(
  dt$same_state_5min,
  na.rm = TRUE
)


persistence_by_state <- dt[
  ,
  .(
    observations = .N,

    persistence =
      mean(
        same_state_5min,
        na.rm = TRUE
      )
  ),
  by = current_tactical_state
]


cat("\n")
cat("============================================\n")
cat("PERSISTENCIA\n")
cat("============================================\n\n")


cat(
  "Persistencia global:",
  round(
    persistence_overall * 100,
    2
  ),
  "%\n\n"
)


print(
  persistence_by_state[
    order(
      -persistence
    )
  ]
)


# ============================================================
# 8. EMPATES EN CONSTRUCCIÓN DE ESTADOS
# ============================================================

if (
  "tactical_state_tie"
  %in%
  names(dt)
) {

  tie_rate <- mean(
    dt$tactical_state_tie,
    na.rm = TRUE
  )


  cat("\n")
  cat("============================================\n")
  cat("EMPATES EN ETIQUETADO\n")
  cat("============================================\n\n")


  cat(
    "Porcentaje de filas con empate:",
    round(
      tie_rate * 100,
      2
    ),
    "%\n"
  )
}


# ============================================================
# 9. FEATURES PARA SEPARABILIDAD
# ============================================================

candidate_features <- c(
  "america_field_tilt_last5",
  "america_possession_context_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_passes_diff_last5",
  "progressive_carries_diff_last5",
  "pressures_diff_last5",
  "recoveries_diff_last5",
  "recovery_height_diff_last5",
  "shots_diff_last5",

  "america_field_tilt_last3",
  "america_possession_context_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_passes_diff_last3",
  "progressive_carries_diff_last3",
  "pressures_diff_last3",
  "recoveries_diff_last3",
  "recovery_height_diff_last3",
  "shots_diff_last3"
)


features <- candidate_features[
  candidate_features
  %in%
  names(dt)
]


cat("\n")
cat("Features analizadas:")
cat(
  length(features),
  "\n\n"
)


# ============================================================
# 10. MEDIA POR ESTADO
# ============================================================

state_feature_means <- dt[
  ,
  lapply(
    .SD,
    function(x) {

      mean(
        as.numeric(x),
        na.rm = TRUE
      )
    }
  ),
  by = current_tactical_state,
  .SDcols = features
]


cat("\n")
cat("============================================\n")
cat("MEDIAS DE FEATURES POR ESTADO\n")
cat("============================================\n\n")


print(
  state_feature_means
)


# ============================================================
# 11. MEDIANAS POR ESTADO
# ============================================================

state_feature_medians <- dt[
  ,
  lapply(
    .SD,
    function(x) {

      median(
        as.numeric(x),
        na.rm = TRUE
      )
    }
  ),
  by = current_tactical_state,
  .SDcols = features
]


# ============================================================
# 12. VARIANZA ENTRE ESTADOS
#
# Si una variable tiene medias casi iguales entre las clases,
# aporta poca separación.
# ============================================================

separation_rows <- list()


for (feature in features) {

  feature_dt <- dt[
    ,
    .(
      feature_mean =
        mean(
          as.numeric(
            get(feature)
          ),
          na.rm = TRUE
        ),

      feature_sd =
        sd(
          as.numeric(
            get(feature)
          ),
          na.rm = TRUE
        )
    ),
    by = current_tactical_state
  ]


  between_sd <- sd(
    feature_dt$feature_mean,
    na.rm = TRUE
  )


  overall_sd <- sd(
    as.numeric(
      dt[
        ,
        get(feature)
      ]
    ),
    na.rm = TRUE
  )


  separation_ratio <- (
    if (
      is.finite(overall_sd)
      &&
      overall_sd > 0
    ) {

      between_sd /
      overall_sd

    } else {

      NA_real_
    }
  )


  separation_rows <- append(
    separation_rows,
    list(
      data.table(
        feature = feature,
        between_state_sd = between_sd,
        overall_sd = overall_sd,
        separation_ratio = separation_ratio
      )
    )
  )
}


separation_table <- rbindlist(
  separation_rows,
  fill = TRUE
)


setorder(
  separation_table,
  -separation_ratio
)


cat("\n")
cat("============================================\n")
cat("SEPARABILIDAD DE FEATURES\n")
cat("============================================\n\n")


print(
  separation_table
)


# ============================================================
# 13. DISTRIBUCIÓN DE SCORES DE ETIQUETADO
# ============================================================

score_columns <- c(
  "control_score",
  "pressure_score",
  "transition_score",
  "protection_score",
  "urgency_score"
)


score_columns <- score_columns[
  score_columns
  %in%
  names(dt)
]


if (length(score_columns) > 0) {

  cat("\n")
  cat("============================================\n")
  cat("SCORES DE ETIQUETADO\n")
  cat("============================================\n\n")


  score_summary <- dt[
    ,
    lapply(
      .SD,
      function(x) {

        c(
          min =
            min(
              x,
              na.rm = TRUE
            ),

          q25 =
            quantile(
              x,
              0.25,
              na.rm = TRUE
            ),

          median =
            median(
              x,
              na.rm = TRUE
            ),

          mean =
            mean(
              x,
              na.rm = TRUE
            ),

          q75 =
            quantile(
              x,
              0.75,
              na.rm = TRUE
            ),

          max =
            max(
              x,
              na.rm = TRUE
            )
        )
      }
    ),
    .SDcols = score_columns
  ]


  print(
    score_summary
  )
}


# ============================================================
# 14. CLASE DOMINANTE POR SPLIT
# ============================================================

dominant_class <- future_distribution[
  ,
  .SD[
    which.max(
      observations
    )
  ],
  by = split
]


cat("\n")
cat("============================================\n")
cat("CLASE DOMINANTE POR SPLIT\n")
cat("============================================\n\n")


print(
  dominant_class
)


# ============================================================
# 15. EXPORTAR
# ============================================================

fwrite(
  current_distribution,
  file.path(
    output_dir,
    "current_state_distribution.csv"
  )
)


fwrite(
  future_distribution,
  file.path(
    output_dir,
    "future_state_distribution.csv"
  )
)


fwrite(
  transition_matrix,
  file.path(
    output_dir,
    "transition_matrix.csv"
  )
)


fwrite(
  persistence_by_state,
  file.path(
    output_dir,
    "persistence_by_state.csv"
  )
)


fwrite(
  state_feature_means,
  file.path(
    output_dir,
    "state_feature_means.csv"
  )
)


fwrite(
  state_feature_medians,
  file.path(
    output_dir,
    "state_feature_medians.csv"
  )
)


fwrite(
  separation_table,
  file.path(
    output_dir,
    "feature_separation.csv"
  )
)


# ============================================================
# 16. FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Diagnóstico guardado en:\n",
  output_dir,
  "\n"
)