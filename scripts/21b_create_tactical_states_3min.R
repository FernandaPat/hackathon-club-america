library(data.table)

# ============================================================
# 1. LEER FEATURES DE 3 MINUTOS
# ============================================================

dt3 <- fread(
  "data/processed/america_male/model_context_360_rolling_3min.csv",
  encoding = "UTF-8"
)

setorder(
  dt3,
  match_id,
  minute
)

cat("Filas:", nrow(dt3), "\n")
cat("Partidos:", uniqueN(dt3$match_id), "\n")


# ============================================================
# 2. GAME STATE NUMÉRICO
# ============================================================

if (is.numeric(dt3$game_state)) {

  dt3[
    ,
    game_state_num := game_state
  ]

} else {

  dt3[
    ,
    game_state_num :=
      suppressWarnings(
        as.numeric(game_state)
      )
  ]

  dt3[
    is.na(game_state_num) &
    tolower(game_state) %in% c(
      "winning",
      "win",
      "ganando"
    ),
    game_state_num := 1
  ]

  dt3[
    is.na(game_state_num) &
    tolower(game_state) %in% c(
      "drawing",
      "draw",
      "empatando"
    ),
    game_state_num := 0
  ]

  dt3[
    is.na(game_state_num) &
    tolower(game_state) %in% c(
      "losing",
      "loss",
      "perdiendo"
    ),
    game_state_num := -1
  ]
}


# ============================================================
# 3. VALIDAR VARIABLES NECESARIAS
# ============================================================

required_cols <- c(

  "america_field_tilt_last3",

  "entries_final_third_diff_last3",
  "progressive_pass_diff_last3",

  "recovery_height_diff_last3",
  "pressure_diff_last3",

  "opponent_nearest_opponent_distance_last3",

  "america_progressive_carries_last3",
  "america_progressive_passes_last3",

  "america_local_superiority_10_last3",
  "america_support_share_10_last3",
  "america_pressure_share_5_last3",

  "america_entries_final_third_last3",
  "america_entries_box_last3"
)

validation <- data.table(
  variable = required_cols,
  exists = required_cols %in% names(dt3)
)

View(validation)

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
# 5. VARIABLES ESTANDARIZADAS
# ============================================================

dt3[
  ,
  z_field_tilt :=
    zscore_safe(
      america_field_tilt_last3
    )
]

dt3[
  ,
  z_entries_diff :=
    zscore_safe(
      entries_final_third_diff_last3
    )
]

dt3[
  ,
  z_progressive_pass_diff :=
    zscore_safe(
      progressive_pass_diff_last3
    )
]


dt3[
  ,
  z_recovery_height :=
    zscore_safe(
      recovery_height_diff_last3
    )
]

dt3[
  ,
  z_pressure_diff :=
    zscore_safe(
      pressure_diff_last3
    )
]

dt3[
  ,
  z_opponent_space :=
    zscore_safe(
      -opponent_nearest_opponent_distance_last3
    )
]


dt3[
  ,
  z_progressive_carries :=
    zscore_safe(
      america_progressive_carries_last3
    )
]

dt3[
  ,
  z_progressive_passes :=
    zscore_safe(
      america_progressive_passes_last3
    )
]


dt3[
  ,
  z_game_state :=
    zscore_safe(
      game_state_num
    )
]

dt3[
  ,
  z_local_superiority :=
    zscore_safe(
      america_local_superiority_10_last3
    )
]

dt3[
  ,
  z_support :=
    zscore_safe(
      america_support_share_10_last3
    )
]

dt3[
  ,
  z_pressure_received :=
    zscore_safe(
      america_pressure_share_5_last3
    )
]


dt3[
  ,
  z_final_third_entries :=
    zscore_safe(
      america_entries_final_third_last3
    )
]

dt3[
  ,
  z_box_entries :=
    zscore_safe(
      america_entries_box_last3
    )
]

dt3[
  ,
  z_minute :=
    zscore_safe(
      minute
    )
]


# ============================================================
# 6. CONTROL
# ============================================================

dt3[
  ,
  control_score_3min :=
    (
      z_field_tilt +
      z_entries_diff +
      z_progressive_pass_diff
    ) / 3
]


# ============================================================
# 7. PRESSURE
# ============================================================

dt3[
  ,
  pressure_score_3min :=
    (
      z_recovery_height +
      z_pressure_diff +
      z_opponent_space
    ) / 3
]


# ============================================================
# 8. TRANSITION
# ============================================================

dt3[
  ,
  transition_score_3min :=
    (
      z_progressive_carries +
      z_progressive_passes
    ) / 2
]


# ============================================================
# 9. PROTECTION
# ============================================================

dt3[
  ,
  protection_score_3min :=
    (
      z_game_state +
      z_local_superiority +
      z_support -
      z_pressure_received
    ) / 4
]

dt3[
  game_state_num <= 0,
  protection_score_3min :=
    protection_score_3min - 1
]


# ============================================================
# 10. URGENCY
# ============================================================

dt3[
  ,
  urgency_score_3min :=
    (
      zscore_safe(-game_state_num) +
      z_minute +
      z_final_third_entries +
      z_box_entries
    ) / 4
]

dt3[
  game_state_num > 0,
  urgency_score_3min :=
    urgency_score_3min - 1
]


# ============================================================
# 11. MATRIZ DE SCORES
# ============================================================

score_matrix_3 <- as.matrix(
  dt3[
    ,
    .(
      CONTROL =
        control_score_3min,

      PRESSURE =
        pressure_score_3min,

      TRANSITION =
        transition_score_3min,

      PROTECTION =
        protection_score_3min,

      URGENCY =
        urgency_score_3min
    )
  ]
)

score_matrix_3[
  !is.finite(score_matrix_3)
] <- -999


# ============================================================
# 12. ESTADO TÁCTICO 3 MIN
# ============================================================

state_names <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

winner_3 <- max.col(
  score_matrix_3,
  ties.method = "first"
)

dt3[
  ,
  tactical_state_3min :=
    state_names[
      winner_3
    ]
]


# Para 3 minutos, necesitamos minutos 0,1,2.
# Desde minuto 2 ya tenemos ventana completa.

dt3[
  minute < 2,
  tactical_state_3min :=
    NA_character_
]


# ============================================================
# 13. CONFIANZA 3 MIN
# ============================================================

dt3[
  ,
  tactical_state_confidence_3min :=
    apply(
      score_matrix_3,
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

dt3[
  minute < 2,
  tactical_state_confidence_3min :=
    NA_real_
]


# ============================================================
# 14. CAMBIOS DE ESTADO
# ============================================================

dt3[
  ,
  previous_tactical_state_3min :=
    shift(
      tactical_state_3min,
      1
    ),
  by = match_id
]

dt3[
  ,
  tactical_state_changed_3min :=
    !is.na(tactical_state_3min) &
    !is.na(previous_tactical_state_3min) &
    tactical_state_3min !=
      previous_tactical_state_3min
]


# ============================================================
# 15. DISTRIBUCIÓN 3 MIN
# ============================================================

state_distribution_3min <- dt3[
  !is.na(tactical_state_3min),
  .N,
  by = tactical_state_3min
]

state_distribution_3min[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]

setorder(
  state_distribution_3min,
  -N
)

View(
  state_distribution_3min
)


# ============================================================
# 16. GUARDAR MODELO 3 MIN
# ============================================================

fwrite(
  dt3,
  "data/processed/america_male/model_with_tactical_states_3min.csv",
  bom = TRUE
)


# ============================================================
# 17. LEER ESTADOS DE 5 MIN
# ============================================================

dt5 <- fread(
  "data/processed/america_male/model_with_tactical_states.csv",
  encoding = "UTF-8"
)

states_5 <- dt5[
  ,
  .(
    match_id,
    minute,

    tactical_state_5min =
      tactical_state,

    tactical_state_confidence_5min =
      tactical_state_confidence
  )
]


# ============================================================
# 18. UNIR 3 MIN VS 5 MIN
# ============================================================

comparison <- merge(
  dt3[
    ,
    .(
      match_id,
      minute,
      opponent,
      game_state_num,

      tactical_state_3min,
      tactical_state_confidence_3min,

      control_score_3min,
      pressure_score_3min,
      transition_score_3min,
      protection_score_3min,
      urgency_score_3min
    )
  ],
  states_5,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)


# ============================================================
# 19. ¿COINCIDEN?
# ============================================================

comparison[
  ,
  states_agree :=
    tactical_state_3min ==
    tactical_state_5min
]


# ============================================================
# 20. TABLA VISUAL DE COMPARACIÓN
# ============================================================

View(
  comparison[
    !is.na(tactical_state_3min) &
    !is.na(tactical_state_5min)
  ]
)


# ============================================================
# 21. RESUMEN DE COINCIDENCIA
# ============================================================

agreement_summary <- comparison[
  !is.na(tactical_state_3min) &
  !is.na(tactical_state_5min),
  .(
    observations = .N,

    same_state =
      sum(
        states_agree,
        na.rm = TRUE
      ),

    different_state =
      sum(
        !states_agree,
        na.rm = TRUE
      ),

    agreement_percentage =
      round(
        mean(
          states_agree,
          na.rm = TRUE
        ) * 100,
        2
      )
  )
]

View(
  agreement_summary
)


# ============================================================
# 22. MATRIZ 3 MIN VS 5 MIN
# ============================================================

state_comparison_matrix <- comparison[
  !is.na(tactical_state_3min) &
  !is.na(tactical_state_5min),
  .N,
  by = .(
    tactical_state_3min,
    tactical_state_5min
  )
]

View(
  state_comparison_matrix
)


# ============================================================
# 23. SOLO DONDE CAMBIAN
# ============================================================

different_states <- comparison[
  states_agree == FALSE &
  !is.na(states_agree)
]

View(
  different_states
)


cat("\n=====================================\n")
cat("COMPARACIÓN 3 MIN VS 5 MIN TERMINADA\n")
cat("=====================================\n")

cat("Filas 3 min:", nrow(dt3), "\n")
cat("Partidos:", uniqueN(dt3$match_id), "\n")


dt3[
  tactical_state_3min == "",
  tactical_state_3min := NA_character_
]

fwrite(
  dt3,
  "data/processed/america_male/model_with_tactical_states_3min.csv",
  bom = TRUE,
  na = "NA"
)