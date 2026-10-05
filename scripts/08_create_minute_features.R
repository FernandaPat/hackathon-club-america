library(data.table)

# ============================================================
# 1. LEER DATOS
# ============================================================

events <- fread(
  "data/processed/america_male/events_america_clean.csv",
  encoding = "UTF-8"
)

matches <- fread(
  "data/processed/america_male/america_matches.csv",
  encoding = "UTF-8"
)

cat("Eventos:", nrow(events), "\n")
cat("Partidos:", nrow(matches), "\n")


# ============================================================
# 2. DETECTAR COLUMNAS IMPORTANTES
# ============================================================

buscar_columna <- function(opciones, columnas) {

  encontrada <- opciones[opciones %in% columnas]

  if (length(encontrada) == 0) {
    return(NA_character_)
  }

  encontrada[1]
}


xg_col <- buscar_columna(
  c(
    "shot.statsbomb_xg",
    "shot.statsbomb_xg2",
    "shot.xg"
  ),
  names(events)
)


shot_outcome_col <- buscar_columna(
  c(
    "shot.outcome.name",
    "shot.outcome"
  ),
  names(events)
)


pass_outcome_col <- buscar_columna(
  c(
    "pass.outcome.name",
    "pass.outcome"
  ),
  names(events)
)


cat("Columna xG:", xg_col, "\n")
cat("Columna shot outcome:", shot_outcome_col, "\n")
cat("Columna pass outcome:", pass_outcome_col, "\n")


# ============================================================
# 3. LIMPIEZA BÁSICA
# ============================================================

events <- events[
  !is.na(match_id) &
  !is.na(minute)
]

events[, minute := as.integer(minute)]

# Evitar minutos absurdos
events <- events[
  minute >= 0 &
  minute <= 130
]


# ============================================================
# 4. IDENTIFICAR AMÉRICA VS RIVAL
# ============================================================

events[
  ,
  is_america := team.name == "América"
]

events[
  ,
  is_opponent := !is.na(team.name) &
                 team.name != "América"
]


# ============================================================
# 5. VARIABLES BINARIAS POR EVENTO
# ============================================================

events[
  ,
  is_pass := type.name == "Pass"
]

events[
  ,
  is_carry := type.name == "Carry"
]

events[
  ,
  is_pressure := type.name == "Pressure"
]

events[
  ,
  is_recovery := type.name == "Ball Recovery"
]

events[
  ,
  is_shot := type.name == "Shot"
]

events[
  ,
  is_interception := type.name == "Interception"
]

events[
  ,
  is_duel := type.name == "Duel"
]

events[
  ,
  is_dribble := type.name == "Dribble"
]

events[
  ,
  is_counterpress := fifelse(
    is.na(counterpress),
    FALSE,
    counterpress == TRUE
  )
]


# ============================================================
# 6. PASE COMPLETADO
#
# En StatsBomb un pass.outcome vacío normalmente representa
# un pase completado.
# ============================================================

if (!is.na(pass_outcome_col)) {

  events[
    ,
    pass_completed := is_pass &
      (
        is.na(get(pass_outcome_col)) |
        get(pass_outcome_col) == ""
      )
  ]

} else {

  events[
    ,
    pass_completed := FALSE
  ]

}


# ============================================================
# 7. xG
# ============================================================

if (!is.na(xg_col)) {

  events[
    ,
    event_xg := fifelse(
      is.na(get(xg_col)),
      0,
      as.numeric(get(xg_col))
    )
  ]

} else {

  events[, event_xg := 0]

}


# ============================================================
# 8. GOLES
# ============================================================

events[, is_goal := FALSE]

if (!is.na(shot_outcome_col)) {

  events[
    type.name == "Shot" &
    get(shot_outcome_col) == "Goal",
    is_goal := TRUE
  ]

}


# ============================================================
# 9. OBV
# ============================================================

if ("obv_total_net" %in% names(events)) {

  events[
    ,
    obv_value := fifelse(
      is.na(obv_total_net),
      0,
      as.numeric(obv_total_net)
    )
  ]

} else if ("obv_for_net" %in% names(events)) {

  events[
    ,
    obv_value := fifelse(
      is.na(obv_for_net),
      0,
      as.numeric(obv_for_net)
    )
  ]

} else {

  events[, obv_value := 0]

}


# ============================================================
# 10. DIRECTNESS
# ============================================================

if ("possession_directness" %in% names(events)) {

  events[
    ,
    directness_value := suppressWarnings(
      as.numeric(possession_directness)
    )
  ]

} else {

  events[, directness_value := NA_real_]

}


# ============================================================
# 11. FEATURES DE AMÉRICA POR MINUTO
# ============================================================

america_minute <- events[
  is_america == TRUE,
  .(
    america_events = .N,

    america_passes = sum(
      is_pass,
      na.rm = TRUE
    ),

    america_completed_passes = sum(
      pass_completed,
      na.rm = TRUE
    ),

    america_carries = sum(
      is_carry,
      na.rm = TRUE
    ),

    america_pressures = sum(
      is_pressure,
      na.rm = TRUE
    ),

    america_counterpress = sum(
      is_counterpress,
      na.rm = TRUE
    ),

    america_recoveries = sum(
      is_recovery,
      na.rm = TRUE
    ),

    america_interceptions = sum(
      is_interception,
      na.rm = TRUE
    ),

    america_duels = sum(
      is_duel,
      na.rm = TRUE
    ),

    america_dribbles = sum(
      is_dribble,
      na.rm = TRUE
    ),

    america_shots = sum(
      is_shot,
      na.rm = TRUE
    ),

    america_goals = sum(
      is_goal,
      na.rm = TRUE
    ),

    america_xg = sum(
      event_xg,
      na.rm = TRUE
    ),

    america_obv = sum(
      obv_value,
      na.rm = TRUE
    ),

    america_directness = mean(
      directness_value,
      na.rm = TRUE
    ),

    america_possessions = uniqueN(
      possession
    )
  ),
  by = .(
    match_id,
    minute
  )
]


# ============================================================
# 12. FEATURES DEL RIVAL POR MINUTO
# ============================================================

opponent_minute <- events[
  is_opponent == TRUE,
  .(
    opponent_events = .N,

    opponent_passes = sum(
      is_pass,
      na.rm = TRUE
    ),

    opponent_completed_passes = sum(
      pass_completed,
      na.rm = TRUE
    ),

    opponent_carries = sum(
      is_carry,
      na.rm = TRUE
    ),

    opponent_pressures = sum(
      is_pressure,
      na.rm = TRUE
    ),

    opponent_counterpress = sum(
      is_counterpress,
      na.rm = TRUE
    ),

    opponent_recoveries = sum(
      is_recovery,
      na.rm = TRUE
    ),

    opponent_interceptions = sum(
      is_interception,
      na.rm = TRUE
    ),

    opponent_shots = sum(
      is_shot,
      na.rm = TRUE
    ),

    opponent_goals = sum(
      is_goal,
      na.rm = TRUE
    ),

    opponent_xg = sum(
      event_xg,
      na.rm = TRUE
    ),

    opponent_obv = sum(
      obv_value,
      na.rm = TRUE
    ),

    opponent_directness = mean(
      directness_value,
      na.rm = TRUE
    ),

    opponent_possessions = uniqueN(
      possession
    )
  ),
  by = .(
    match_id,
    minute
  )
]


# ============================================================
# 13. CREAR TODOS LOS MINUTOS DE CADA PARTIDO
# ============================================================

max_minutes <- events[
  ,
  .(
    max_minute = max(
      minute,
      na.rm = TRUE
    )
  ),
  by = match_id
]


minute_grid <- max_minutes[
  ,
  .(
    minute = 0:max_minute
  ),
  by = match_id
]


# ============================================================
# 14. UNIR AMÉRICA + RIVAL
# ============================================================

minute_features <- merge(
  minute_grid,
  america_minute,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)

minute_features <- merge(
  minute_features,
  opponent_minute,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)


# ============================================================
# 15. REEMPLAZAR NA DE CONTEOS POR 0
# ============================================================

columnas_conteo <- grep(
  "events|passes|completed|carries|pressures|counterpress|recoveries|interceptions|duels|dribbles|shots|goals|xg|obv|possessions",
  names(minute_features),
  value = TRUE
)

for (col in columnas_conteo) {

  set(
    minute_features,
    which(is.na(minute_features[[col]])),
    col,
    0
  )

}


# ============================================================
# 16. MARCADOR ACUMULADO
# ============================================================

setorder(
  minute_features,
  match_id,
  minute
)

minute_features[
  ,
  america_score := cumsum(
    america_goals
  ),
  by = match_id
]

minute_features[
  ,
  opponent_score := cumsum(
    opponent_goals
  ),
  by = match_id
]

minute_features[
  ,
  score_difference :=
    america_score -
    opponent_score
]


# ============================================================
# 17. ESTADO DEL MARCADOR
# ============================================================

minute_features[
  ,
  game_state := fifelse(
    score_difference > 0,
    "Winning",
    fifelse(
      score_difference < 0,
      "Losing",
      "Drawing"
    )
  )
]


# ============================================================
# 18. PROXY DE POSESIÓN
#
# No es posesión oficial por tiempo.
# Es proporción de posesiones identificadas en ese minuto.
# ============================================================

minute_features[
  ,
  total_possessions :=
    america_possessions +
    opponent_possessions
]

minute_features[
  ,
  america_possession_proxy :=
    fifelse(
      total_possessions > 0,
      america_possessions /
        total_possessions,
      NA_real_
    )
]


# ============================================================
# 19. PASS COMPLETION %
# ============================================================

minute_features[
  ,
  america_pass_completion :=
    fifelse(
      america_passes > 0,
      america_completed_passes /
        america_passes,
      NA_real_
    )
]

minute_features[
  ,
  opponent_pass_completion :=
    fifelse(
      opponent_passes > 0,
      opponent_completed_passes /
        opponent_passes,
      NA_real_
    )
]


# ============================================================
# 20. AÑADIR INFORMACIÓN DEL PARTIDO
# ============================================================

# Evitar columnas tipo lista
columnas_match <- intersect(
  c(
    "match_id",
    "match_date",
    "season",
    "opponent",
    "venue"
  ),
  names(matches)
)

match_context <- unique(
  matches[
    ,
    ..columnas_match
  ]
)

minute_features <- merge(
  minute_features,
  match_context,
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 21. ORDENAR
# ============================================================

setorder(
  minute_features,
  match_id,
  minute
)


# ============================================================
# 22. GUARDAR TABLA MINUTO A MINUTO
# ============================================================

fwrite(
  minute_features,
  "data/processed/america_male/minute_features.csv",
  bom = TRUE
)

cat("\n=====================================\n")
cat("MINUTE FEATURES CREADO\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(minute_features),
  "\n"
)

cat(
  "Columnas:",
  ncol(minute_features),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(minute_features$match_id),
  "\n"
)