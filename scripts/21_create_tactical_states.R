library(data.table)

# ============================================================
# 1. CARGAR BASE ORIGINAL DE 5 MINUTOS
# ============================================================

dt <- fread(
  "data/processed/america_male/model_context_360_rolling.csv",
  encoding = "UTF-8"
)

setorder(dt, match_id, minute)

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. GAME STATE NUMÉRICO
# ============================================================

dt[, game_state_num := fifelse(
  game_state == "Winning", 1,
  fifelse(
    game_state == "Losing", -1,
    fifelse(
      game_state == "Drawing", 0,
      NA_real_
    )
  )
)]


# ============================================================
# 3. FUNCIÓN Z-SCORE SEGURA
# ============================================================

zscore_safe <- function(x) {

  x <- as.numeric(x)

  m <- mean(x, na.rm = TRUE)
  s <- sd(x, na.rm = TRUE)

  if (
    !is.finite(m) ||
    !is.finite(s) ||
    s == 0
  ) {
    return(rep(0, length(x)))
  }

  z <- (x - m) / s

  z[!is.finite(z)] <- 0

  z
}


# ============================================================
# 4. VALIDAR VARIABLES
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
  "america_local_superiority_10_last5",
  "america_support_share_10_last5",
  "america_pressure_share_5_last5",
  "america_entries_final_third_last5",
  "america_entries_box_last5"
)

validation_5min <- data.table(
  variable = required_cols,
  exists = required_cols %in% names(dt)
)

View(validation_5min)

missing_cols <- validation_5min[
  exists == FALSE,
  variable
]

if (length(missing_cols) > 0) {

  stop(
    paste(
      "Faltan columnas:",
      paste(missing_cols, collapse = ", ")
    )
  )
}


# ============================================================
# 5. VARIABLES ESTANDARIZADAS
# ============================================================

dt[, z_field_tilt :=
     zscore_safe(america_field_tilt_last5)]

dt[, z_entries_diff :=
     zscore_safe(entries_final_third_diff_last5)]

dt[, z_progressive_pass_diff :=
     zscore_safe(progressive_pass_diff_last5)]

dt[, z_recovery_height :=
     zscore_safe(recovery_height_diff_last5)]

dt[, z_pressure_diff :=
     zscore_safe(pressure_diff_last5)]

dt[, z_opponent_space :=
     zscore_safe(
       -opponent_nearest_opponent_distance_last5
     )]

dt[, z_progressive_carries :=
     zscore_safe(
       america_progressive_carries_last5
     )]

dt[, z_progressive_passes :=
     zscore_safe(
       america_progressive_passes_last5
     )]

dt[, z_game_state :=
     zscore_safe(game_state_num)]

dt[, z_local_superiority :=
     zscore_safe(
       america_local_superiority_10_last5
     )]

dt[, z_support :=
     zscore_safe(
       america_support_share_10_last5
     )]

dt[, z_pressure_received :=
     zscore_safe(
       america_pressure_share_5_last5
     )]

dt[, z_final_third_entries :=
     zscore_safe(
       america_entries_final_third_last5
     )]

dt[, z_box_entries :=
     zscore_safe(
       america_entries_box_last5
     )]

dt[, z_minute :=
     zscore_safe(minute)]


# ============================================================
# 6. CONTROL
# ============================================================

dt[, control_score :=
     (
       z_field_tilt +
       z_entries_diff +
       z_progressive_pass_diff
     ) / 3]


# ============================================================
# 7. PRESSURE
# ============================================================

dt[, pressure_score :=
     (
       z_recovery_height +
       z_pressure_diff +
       z_opponent_space
     ) / 3]


# ============================================================
# 8. TRANSITION
# ============================================================

dt[, transition_score :=
     (
       z_progressive_carries +
       z_progressive_passes
     ) / 2]


# ============================================================
# 9. PROTECTION
# ============================================================

dt[, protection_score :=
     (
       z_game_state +
       z_local_superiority +
       z_support -
       z_pressure_received
     ) / 4]

dt[
  game_state_num <= 0,
  protection_score := protection_score - 1
]


# ============================================================
# 10. URGENCY
# ============================================================

dt[, urgency_score :=
     (
       zscore_safe(-game_state_num) +
       z_minute +
       z_final_third_entries +
       z_box_entries
     ) / 4]

dt[
  game_state_num > 0,
  urgency_score := urgency_score - 1
]


# ============================================================
# 11. MATRIZ DE LOS CINCO ESTADOS
# ============================================================

score_matrix <- as.matrix(
  dt[
    ,
    .(
      CONTROL = control_score,
      PRESSURE = pressure_score,
      TRANSITION = transition_score,
      PROTECTION = protection_score,
      URGENCY = urgency_score
    )
  ]
)

score_matrix[
  !is.finite(score_matrix)
] <- -999


# ============================================================
# 12. ASIGNAR ESTADO GANADOR
# ============================================================

state_names <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

winner <- max.col(
  score_matrix,
  ties.method = "first"
)

dt[, tactical_state :=
     state_names[winner]]


# ============================================================
# 13. PRIMEROS 4 MINUTOS SIN CLASIFICAR
# ============================================================

dt[
  minute < 4,
  tactical_state := NA_character_
]


# ============================================================
# 14. CONFIANZA
# ============================================================

dt[, tactical_state_confidence :=
     apply(
       score_matrix,
       1,
       function(x) {

         x <- sort(
           x,
           decreasing = TRUE
         )

         x[1] - x[2]
       }
     )]

dt[
  minute < 4,
  tactical_state_confidence := NA_real_
]


# ============================================================
# 15. ESTADO ANTERIOR Y CAMBIO
# ============================================================

dt[, previous_tactical_state :=
     shift(
       tactical_state,
       1
     ),
   by = match_id]

dt[, tactical_state_changed :=
     !is.na(tactical_state) &
     !is.na(previous_tactical_state) &
     tactical_state != previous_tactical_state]


# ============================================================
# 16. DISTRIBUCIÓN
# ============================================================

distribution_5_final <- dt[
  !is.na(tactical_state),
  .N,
  by = tactical_state
]

distribution_5_final[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]

setorder(
  distribution_5_final,
  -N
)

View(distribution_5_final)


# ============================================================
# 17. COBERTURA
# ============================================================

coverage_5_final <- dt[
  ,
  .(
    observations = .N,

    classified =
      sum(!is.na(tactical_state)),

    missing =
      sum(is.na(tactical_state)),

    coverage_pct =
      round(
        mean(
          !is.na(tactical_state)
        ) * 100,
        2
      )
  )
]

View(coverage_5_final)


# ============================================================
# 18. GUARDAR
# ============================================================

fwrite(
  dt,
  "data/processed/america_male/model_with_tactical_states.csv",
  bom = TRUE,
  na = "NA"
)

cat("\n=====================================\n")
cat("5-MIN TACTICAL STATES RECONSTRUIDOS\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")
cat(
  "Clasificados:",
  sum(!is.na(dt$tactical_state)),
  "\n"
)