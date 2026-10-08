library(data.table)

# ============================================================
# 1. CARGAR DATASET
# ============================================================

dt <- fread(
  "data/processed/america_male/model_context_360_rolling.csv",
  encoding = "UTF-8"
)

setorder(
  dt,
  match_id,
  minute
)

cat("\n=====================================\n")
cat("TACTICAL STATES - VERSION LIMPIA\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. COLUMNAS QUE NECESITAMOS
# ============================================================

required_cols <- c(
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "progressive_pass_diff_last5",

  "recovery_height_diff_last5",
  "pressure_diff_last5",
  "opponent_nearest_opponent_distance_last5",

  "america_progressive_carries_last5",
  "america_progressive_passes_last5",

  "game_state",

  "america_local_superiority_10_last5",
  "america_support_share_10_last5",
  "america_pressure_share_5_last5",

  "america_entries_final_third_last5",
  "america_entries_box_last5"
)

validation <- data.table(
  variable = required_cols,
  exists = required_cols %in% names(dt)
)

print(validation)


# ============================================================
# 3. DETENER SI FALTA ALGUNA VARIABLE
# ============================================================

missing_cols <- validation[
  exists == FALSE,
  variable
]

if (length(missing_cols) > 0) {

  stop(
    paste(
      "Faltan columnas:",
      paste(
        missing_cols,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 4. Z-SCORE SEGURO
# ============================================================

zscore_safe <- function(x) {

  x <- as.numeric(x)

  media <- mean(
    x,
    na.rm = TRUE
  )

  desv <- sd(
    x,
    na.rm = TRUE
  )

  if (
    !is.finite(media) ||
    !is.finite(desv) ||
    desv == 0
  ) {

    return(
      rep(
        0,
        length(x)
      )
    )
  }

  z <- (
    x - media
  ) / desv

  z[
    !is.finite(z)
  ] <- 0

  z
}


# ============================================================
# 5. CREAR VARIABLES ESTANDARIZADAS
# ============================================================

dt[
  ,
  z_field_tilt :=
    zscore_safe(
      america_field_tilt_last5
    )
]

dt[
  ,
  z_entries_diff :=
    zscore_safe(
      entries_final_third_diff_last5
    )
]

dt[
  ,
  z_progressive_pass_diff :=
    zscore_safe(
      progressive_pass_diff_last5
    )
]


dt[
  ,
  z_recovery_height :=
    zscore_safe(
      recovery_height_diff_last5
    )
]

dt[
  ,
  z_pressure_diff :=
    zscore_safe(
      pressure_diff_last5
    )
]

dt[
  ,
  z_opponent_space :=
    zscore_safe(
      -opponent_nearest_opponent_distance_last5
    )
]


dt[
  ,
  z_progressive_carries :=
    zscore_safe(
      america_progressive_carries_last5
    )
]

dt[
  ,
  z_progressive_passes :=
    zscore_safe(
      america_progressive_passes_last5
    )
]


dt[
  ,
  z_game_state :=
    zscore_safe(
      game_state
    )
]

dt[
  ,
  z_local_superiority :=
    zscore_safe(
      america_local_superiority_10_last5
    )
]

dt[
  ,
  z_support :=
    zscore_safe(
      america_support_share_10_last5
    )
]

dt[
  ,
  z_pressure_received :=
    zscore_safe(
      america_pressure_share_5_last5
    )
]


dt[
  ,
  z_final_third_entries :=
    zscore_safe(
      america_entries_final_third_last5
    )
]

dt[
  ,
  z_box_entries :=
    zscore_safe(
      america_entries_box_last5
    )
]

dt[
  ,
  z_minute :=
    zscore_safe(
      minute
    )
]


# ============================================================
# 6. CONTROL
#
# Dominio territorial + progresión.
#
# Usamos promedio y no suma para que todos los estados
# tengan escalas comparables.
# ============================================================

dt[
  ,
  control_score :=
    (
      z_field_tilt +
      z_entries_diff +
      z_progressive_pass_diff
    ) / 3
]


# ============================================================
# 7. PRESSURE
#
# Presión activa + recuperación alta +
# poco espacio para el rival.
# ============================================================

dt[
  ,
  pressure_score :=
    (
      z_recovery_height +
      z_pressure_diff +
      z_opponent_space
    ) / 3
]


# ============================================================
# 8. TRANSITION
#
# Mucha progresión rápida con pases y conducciones.
# ============================================================

dt[
  ,
  transition_score :=
    (
      z_progressive_carries +
      z_progressive_passes
    ) / 2
]


# ============================================================
# 9. PROTECTION
#
# América tiene ventaja y estructura de apoyo.
#
# Si América no está ganando,
# penalizamos este estado.
# ============================================================

dt[
  ,
  protection_score :=
    (
      z_game_state +
      z_local_superiority +
      z_support -
      z_pressure_received
    ) / 4
]

dt[
  game_state <= 0,
  protection_score :=
    protection_score - 1
]


# ============================================================
# 10. URGENCY
#
# América pierde / partido avanzado /
# busca llegar al último tercio y área.
# ============================================================

dt[
  ,
  urgency_score :=
    (
      zscore_safe(-game_state) +
      z_minute +
      z_final_third_entries +
      z_box_entries
    ) / 4
]

# Si América va ganando, reducimos urgencia

dt[
  game_state > 0,
  urgency_score :=
    urgency_score - 1
]


# ============================================================
# 11. COMPROBAR SCORES
# ============================================================

cat("\n=====================================\n")
cat("SCORES CREADOS\n")
cat("=====================================\n")

print(
  c(
    "control_score",
    "pressure_score",
    "transition_score",
    "protection_score",
    "urgency_score"
  ) %in% names(dt)
)


# ============================================================
# 12. MATRIZ DE SCORES
# ============================================================

score_matrix <- as.matrix(
  dt[
    ,
    .(
      CONTROL =
        control_score,

      PRESSURE =
        pressure_score,

      TRANSITION =
        transition_score,

      PROTECTION =
        protection_score,

      URGENCY =
        urgency_score
    )
  ]
)


# ============================================================
# 13. LIMPIAR VALORES NO FINITOS
# ============================================================

score_matrix[
  !is.finite(score_matrix)
] <- -999


# ============================================================
# 14. ESTADO GANADOR
# ============================================================

winner <- max.col(
  score_matrix,
  ties.method = "first"
)

state_names <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

dt[
  ,
  tactical_state :=
    state_names[
      winner
    ]
]


# ============================================================
# 15. PRIMEROS 4 MINUTOS
# ============================================================

dt[
  minute < 4,
  tactical_state :=
    NA_character_
]


# ============================================================
# 16. CONFIANZA
#
# Diferencia entre el score más alto
# y el segundo más alto.
# ============================================================

dt[
  ,
  tactical_state_confidence :=
    apply(
      score_matrix,
      1,
      function(x) {

        ordered <- sort(
          x,
          decreasing = TRUE
        )

        ordered[1] -
          ordered[2]
      }
    )
]

dt[
  minute < 4,
  tactical_state_confidence :=
    NA_real_
]


# ============================================================
# 17. ESTADO ANTERIOR
# ============================================================

dt[
  ,
  previous_tactical_state :=
    shift(
      tactical_state,
      1
    ),
  by = match_id
]


# ============================================================
# 18. CAMBIO DE ESTADO
# ============================================================

dt[
  ,
  tactical_state_changed :=
    !is.na(tactical_state) &
    !is.na(previous_tactical_state) &
    tactical_state !=
      previous_tactical_state
]


# ============================================================
# 19. DISTRIBUCIÓN
# ============================================================

state_distribution <- dt[
  !is.na(tactical_state),
  .N,
  by = tactical_state
]

state_distribution[
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
  state_distribution,
  -N
)


# ============================================================
# 20. MOSTRAR DISTRIBUCIÓN
# ============================================================

cat("\n=====================================\n")
cat("DISTRIBUCIÓN TÁCTICA\n")
cat("=====================================\n")

print(
  state_distribution
)


# ============================================================
# 21. GUARDAR
# ============================================================

fwrite(
  dt,
  "data/processed/america_male/model_with_tactical_states.csv",
  bom = TRUE
)


# ============================================================
# 22. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("TACTICAL STATES TERMINADO\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(dt),
  "\n"
)

cat(
  "Columnas:",
  ncol(dt),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(dt$match_id),
  "\n"
)

cat(
  "Estados clasificados:",
  sum(
    !is.na(dt$tactical_state)
  ),
  "\n"
)

cat(
  "Cambios de estado:",
  sum(
    dt$tactical_state_changed,
    na.rm = TRUE
  ),
  "\n"
)