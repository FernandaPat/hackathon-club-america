# ============================================================
# 44_female_minute_features.R
# CLUB AMÉRICA FEMENIL
# FEATURES MINUTO A MINUTO
# SIN LEAKAGE DE MARCADOR
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

events_file <- file.path(
  data_dir,
  "events_america_female_clean.csv"
)

output_file <- file.path(
  data_dir,
  "minute_features_female.csv"
)


# ============================================================
# 2. CARGAR
# ============================================================

if (!file.exists(events_file)) {

  stop(
    paste(
      "No existe:",
      events_file
    )
  )
}


events <- fread(
  events_file
)


cat("\n")
cat("============================================\n")
cat("COACHLENS - FEMALE MINUTE FEATURES\n")
cat("============================================\n\n")

cat(
  "Eventos:",
  nrow(events),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(events$match_id),
  "\n\n"
)


# ============================================================
# 3. COLUMNAS NECESARIAS
# ============================================================

required_columns <- c(
  "match_id",
  "minute",
  "event_type",
  "is_america",
  "opponent",
  "venue",
  "x",
  "y",
  "pass_end_x",
  "pass_end_y",
  "carry_end_x",
  "carry_end_y"
)


missing_columns <- setdiff(
  required_columns,
  names(events)
)


if (length(missing_columns) > 0) {

  stop(
    paste(
      "Faltan columnas:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 4. TIPOS
# ============================================================

events[
  ,
  minute := as.integer(
    floor(
      as.numeric(minute)
    )
  )
]


numeric_columns <- c(
  "x",
  "y",
  "pass_end_x",
  "pass_end_y",
  "carry_end_x",
  "carry_end_y"
)


for (column in numeric_columns) {

  events[
    ,
    (column) :=
      as.numeric(
        get(column)
      )
  ]
}


if (
  "period"
  %in%
  names(events)
) {

  events[
    ,
    period :=
      as.numeric(period)
  ]

} else {

  events[
    ,
    period := NA_real_
  ]
}


# ============================================================
# 5. NORMALIZAR is_america
# ============================================================

if (is.character(events$is_america)) {

  events[
    ,
    is_america :=
      toupper(is_america) %in% c(
        "TRUE",
        "T",
        "1"
      )
  ]

} else {

  events[
    ,
    is_america :=
      as.logical(
        is_america
      )
  ]
}


# ============================================================
# 6. EVENTOS BÁSICOS
# ============================================================

events[
  ,
  is_pass :=
    event_type == "Pass"
]


events[
  ,
  is_carry :=
    event_type == "Carry"
]


events[
  ,
  is_shot :=
    event_type == "Shot"
]


events[
  ,
  is_pressure :=
    event_type == "Pressure"
]


events[
  ,
  is_recovery :=
    event_type %in% c(
      "Ball Recovery",
      "Interception"
    )
]


events[
  ,
  is_duel :=
    event_type %in% c(
      "Duel",
      "50/50",
      "Dribble"
    )
]


events[
  ,
  is_turnover :=
    event_type %in% c(
      "Dispossessed",
      "Miscontrol"
    )
]


# ============================================================
# 7. PASE COMPLETADO
# ============================================================

if (
  "pass.outcome.name"
  %in%
  names(events)
) {

  events[
    ,
    pass_completed :=
      is_pass &
      (
        is.na(
          get(
            "pass.outcome.name"
          )
        )
        |
        get(
          "pass.outcome.name"
        ) == ""
      )
  ]

} else {

  events[
    ,
    pass_completed :=
      is_pass
  ]
}


# ============================================================
# 8. TERCIO FINAL
# ============================================================

events[
  ,
  in_final_third :=
    !is.na(x)
    &
    x >= 80
]


events[
  ,
  pass_enters_final_third :=
    is_pass
    &
    !is.na(x)
    &
    !is.na(pass_end_x)
    &
    x < 80
    &
    pass_end_x >= 80
]


events[
  ,
  carry_enters_final_third :=
    is_carry
    &
    !is.na(x)
    &
    !is.na(carry_end_x)
    &
    x < 80
    &
    carry_end_x >= 80
]


# ============================================================
# 9. ENTRADAS AL ÁREA
# ============================================================

events[
  ,
  pass_enters_box :=
    is_pass
    &
    !is.na(pass_end_x)
    &
    !is.na(pass_end_y)
    &
    pass_end_x >= 102
    &
    pass_end_y >= 18
    &
    pass_end_y <= 62
]


events[
  ,
  carry_enters_box :=
    is_carry
    &
    !is.na(carry_end_x)
    &
    !is.na(carry_end_y)
    &
    carry_end_x >= 102
    &
    carry_end_y >= 18
    &
    carry_end_y <= 62
]


# ============================================================
# 10. PROGRESIONES
# ============================================================

events[
  ,
  progressive_pass :=
    is_pass
    &
    !is.na(x)
    &
    !is.na(pass_end_x)
    &
    (
      pass_end_x
      -
      x
    ) > 10
]


events[
  ,
  progressive_carry :=
    is_carry
    &
    !is.na(x)
    &
    !is.na(carry_end_x)
    &
    (
      carry_end_x
      -
      x
    ) > 10
]


# ============================================================
# 11. ALTURA DE RECUPERACIÓN
# ============================================================

events[
  ,
  recovery_x :=
    fifelse(
      is_recovery
      &
      !is.na(x),
      x,
      NA_real_
    )
]


# ============================================================
# 12. PROXY DE POSESIÓN
# ============================================================

events[
  ,
  possession_event :=
    event_type %in% c(
      "Pass",
      "Carry",
      "Dribble",
      "Shot",
      "Ball Receipt*"
    )
]


# ============================================================
# 13. IDENTIFICAR GOLES
# ============================================================

events[
  ,
  shot_goal := FALSE
]


if (
  "shot.outcome.name"
  %in%
  names(events)
) {

  events[
    ,
    shot_goal :=
      is_shot
      &
      get(
        "shot.outcome.name"
      ) == "Goal"
  ]
}


# ============================================================
# 14. EVITAR TANDAS DE PENALES
# ============================================================

events[
  ,
  valid_match_goal :=
    shot_goal
    &
    (
      is.na(period)
      |
      period <= 4
    )
]


# ============================================================
# 15. GOLES NORMALES
# ============================================================

events[
  ,
  america_goal_event :=
    valid_match_goal
    &
    is_america
]


events[
  ,
  rival_goal_event :=
    valid_match_goal
    &
    !is_america
]


# ============================================================
# 16. AUTOGOLES
# ============================================================

events[
  ,
  america_own_goal_against :=
    event_type == "Own Goal Against"
    &
    is_america
]


events[
  ,
  rival_own_goal_against :=
    event_type == "Own Goal Against"
    &
    !is_america
]


events[
  ,
  america_goal_event :=
    america_goal_event
    |
    rival_own_goal_against
]


events[
  ,
  rival_goal_event :=
    rival_goal_event
    |
    america_own_goal_against
]


# ============================================================
# 17. MARCADOR POR MINUTO
# ============================================================

score_minute <- events[
  ,
  .(
    america_goals_minute =
      sum(
        america_goal_event,
        na.rm = TRUE
      ),

    rival_goals_minute =
      sum(
        rival_goal_event,
        na.rm = TRUE
      )
  ),
  by = .(
    match_id,
    minute
  )
]


setorder(
  score_minute,
  match_id,
  minute
)


score_minute[
  ,
  america_goals_live :=
    cumsum(
      america_goals_minute
    ),
  by = match_id
]


score_minute[
  ,
  rival_goals_live :=
    cumsum(
      rival_goals_minute
    ),
  by = match_id
]


score_minute[
  ,
  game_state_num :=
    america_goals_live
    -
    rival_goals_live
]


# ============================================================
# 18. AGREGAR MÉTRICAS POR EQUIPO/MINUTO
# ============================================================

team_minute <- events[
  ,
  .(
    passes =
      sum(
        is_pass,
        na.rm = TRUE
      ),

    completed_passes =
      sum(
        pass_completed,
        na.rm = TRUE
      ),

    carries =
      sum(
        is_carry,
        na.rm = TRUE
      ),

    shots =
      sum(
        is_shot,
        na.rm = TRUE
      ),

    pressures =
      sum(
        is_pressure,
        na.rm = TRUE
      ),

    recoveries =
      sum(
        is_recovery,
        na.rm = TRUE
      ),

    duels =
      sum(
        is_duel,
        na.rm = TRUE
      ),

    turnovers =
      sum(
        is_turnover,
        na.rm = TRUE
      ),

    final_third_events =
      sum(
        in_final_third,
        na.rm = TRUE
      ),

    entries_final_third =
      sum(
        pass_enters_final_third
        |
        carry_enters_final_third,
        na.rm = TRUE
      ),

    entries_box =
      sum(
        pass_enters_box
        |
        carry_enters_box,
        na.rm = TRUE
      ),

    progressive_passes =
      sum(
        progressive_pass,
        na.rm = TRUE
      ),

    progressive_carries =
      sum(
        progressive_carry,
        na.rm = TRUE
      ),

    possession_events =
      sum(
        possession_event,
        na.rm = TRUE
      ),

    recovery_height =
      mean(
        recovery_x,
        na.rm = TRUE
      )
  ),

  by = .(
    match_id,
    minute,
    is_america
  )
]


team_minute[
  !is.finite(
    recovery_height
  ),
  recovery_height :=
    NA_real_
]


# ============================================================
# 19. CONTEXTO
# ============================================================

match_context <- unique(
  events[
    ,
    .(
      match_id,
      match_date,
      opponent,
      venue,
      home_score,
      away_score,
      america_score,
      opponent_score
    )
  ]
)


# ============================================================
# 20. AMÉRICA
# ============================================================

america_minute <- copy(
  team_minute[
    is_america == TRUE
  ]
)


america_minute[
  ,
  is_america := NULL
]


america_metrics <- setdiff(
  names(
    america_minute
  ),
  c(
    "match_id",
    "minute"
  )
)


setnames(
  america_minute,
  america_metrics,
  paste0(
    "america_",
    america_metrics
  )
)


# ============================================================
# 21. RIVAL
# ============================================================

rival_minute <- copy(
  team_minute[
    is_america == FALSE
  ]
)


rival_minute[
  ,
  is_america := NULL
]


rival_metrics <- setdiff(
  names(
    rival_minute
  ),
  c(
    "match_id",
    "minute"
  )
)


setnames(
  rival_minute,
  rival_metrics,
  paste0(
    "rival_",
    rival_metrics
  )
)


# ============================================================
# 22. UNIR AMÉRICA + RIVAL
# ============================================================

minute_features <- merge(
  america_minute,
  rival_minute,
  by = c(
    "match_id",
    "minute"
  ),
  all = TRUE
)


# ============================================================
# 23. AGREGAR CONTEXTO
# ============================================================

minute_features <- merge(
  minute_features,
  match_context,
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 24. AGREGAR MARCADOR LIVE
# ============================================================

minute_features <- merge(
  minute_features,
  score_minute,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)


# ============================================================
# 25. COMPLETAR CONTEOS
# ============================================================

count_metrics <- c(
  "passes",
  "completed_passes",
  "carries",
  "shots",
  "pressures",
  "recoveries",
  "duels",
  "turnovers",
  "final_third_events",
  "entries_final_third",
  "entries_box",
  "progressive_passes",
  "progressive_carries",
  "possession_events"
)


for (metric in count_metrics) {

  america_col <- paste0(
    "america_",
    metric
  )

  rival_col <- paste0(
    "rival_",
    metric
  )


  if (
    america_col
    %in%
    names(minute_features)
  ) {

    minute_features[
      is.na(
        get(
          america_col
        )
      ),
      (america_col) := 0
    ]
  }


  if (
    rival_col
    %in%
    names(minute_features)
  ) {

    minute_features[
      is.na(
        get(
          rival_col
        )
      ),
      (rival_col) := 0
    ]
  }
}


# ============================================================
# 26. DIFERENCIAS
# ============================================================

for (metric in count_metrics) {

  america_col <- paste0(
    "america_",
    metric
  )

  rival_col <- paste0(
    "rival_",
    metric
  )

  diff_col <- paste0(
    metric,
    "_diff"
  )


  if (
    america_col
    %in%
    names(minute_features)
    &&
    rival_col
    %in%
    names(minute_features)
  ) {

    minute_features[
      ,
      (diff_col) :=
        get(
          america_col
        )
        -
        get(
          rival_col
        )
    ]
  }
}


# ============================================================
# 27. FIELD TILT
# ============================================================

minute_features[
  ,
  total_final_third_events :=
    america_final_third_events
    +
    rival_final_third_events
]


minute_features[
  ,
  america_field_tilt :=
    fifelse(
      total_final_third_events > 0,

      america_final_third_events
      /
      total_final_third_events,

      NA_real_
    )
]


# ============================================================
# 28. POSESIÓN PROXY
# ============================================================

minute_features[
  ,
  total_possession_events :=
    america_possession_events
    +
    rival_possession_events
]


minute_features[
  ,
  america_possession_context :=
    fifelse(
      total_possession_events > 0,

      america_possession_events
      /
      total_possession_events,

      NA_real_
    )
]


# ============================================================
# 29. ALTURA RECUPERACIÓN
# ============================================================

minute_features[
  ,
  recovery_height_diff :=
    america_recovery_height
    -
    rival_recovery_height
]


# ============================================================
# 30. ORDENAR
# ============================================================

setorder(
  minute_features,
  match_id,
  minute
)


# ============================================================
# 31. VALIDAR MARCADORES
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACION MARCADOR LIVE\n")
cat("============================================\n\n")


goal_matches <- minute_features[
  america_goals_live > 0
  |
  rival_goals_live > 0
]


cat(
  "Partidos con al menos un gol detectado:",
  uniqueN(
    goal_matches$match_id
  ),
  "\n\n"
)


cat(
  "Distribución game_state_num:\n"
)


print(
  table(
    minute_features$game_state_num,
    useNA = "ifany"
  )
)


cat(
  "\nEjemplo de cambios de marcador:\n"
)


score_changes <- minute_features[
  america_goals_minute > 0
  |
  rival_goals_minute > 0,
  .(
    match_id,
    minute,
    opponent,
    america_goals_minute,
    rival_goals_minute,
    america_goals_live,
    rival_goals_live,
    game_state_num
  )
]


print(
  head(
    score_changes,
    20
  )
)


# ============================================================
# 32. VALIDACIÓN GENERAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACION MINUTE FEATURES\n")
cat("============================================\n\n")


cat(
  "Filas:",
  nrow(
    minute_features
  ),
  "\n"
)


cat(
  "Partidos:",
  uniqueN(
    minute_features$match_id
  ),
  "\n"
)


cat(
  "Columnas:",
  ncol(
    minute_features
  ),
  "\n\n"
)


cat(
  "Entradas ultimo tercio America:",
  sum(
    minute_features$america_entries_final_third,
    na.rm = TRUE
  ),
  "\n"
)


cat(
  "Entradas area America:",
  sum(
    minute_features$america_entries_box,
    na.rm = TRUE
  ),
  "\n"
)


cat(
  "Pases progresivos America:",
  sum(
    minute_features$america_progressive_passes,
    na.rm = TRUE
  ),
  "\n"
)


cat(
  "Conducciones progresivas America:",
  sum(
    minute_features$america_progressive_carries,
    na.rm = TRUE
  ),
  "\n\n"
)


# ============================================================
# 33. EXPORTAR
# ============================================================

fwrite(
  minute_features,
  output_file
)


cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Archivo creado:\n",
  output_file,
  "\n"
)