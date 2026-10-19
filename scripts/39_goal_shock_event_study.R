library(data.table)

# ============================================================
# 1. CARGAR DATOS
# ============================================================

model <- fread(
  "data/processed/america_male/model_ready_predictive.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

events <- fread(
  "data/processed/america_male/events_america_clean.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

cat("\n=====================================\n")
cat("GOAL SHOCK EVENT STUDY\n")
cat("=====================================\n")

cat("Model rows:", nrow(model), "\n")
cat("Matches:", uniqueN(model$match_id), "\n")


# ============================================================
# 2. VALIDAR COLUMNAS
# ============================================================

required_event_cols <- c(
  "match_id",
  "minute",
  "type.name",
  "team.name"
)

missing_event_cols <- setdiff(
  required_event_cols,
  names(events)
)

if (length(missing_event_cols) > 0) {

  stop(
    paste(
      "Faltan columnas en events:",
      paste(
        missing_event_cols,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 3. ENCONTRAR COLUMNA SHOT OUTCOME
# ============================================================

goal_outcome_candidates <- grep(
  "shot.*outcome.*name",
  names(events),
  value = TRUE,
  ignore.case = TRUE
)

if (length(goal_outcome_candidates) == 0) {

  stop(
    "No encontré columna shot outcome."
  )
}

goal_outcome_col <- goal_outcome_candidates[1]

cat(
  "Goal outcome column:",
  goal_outcome_col,
  "\n"
)


# ============================================================
# 4. EXTRAER GOLES
# ============================================================

goals <- events[
  type.name == "Shot" &
    get(goal_outcome_col) == "Goal",
  .(
    match_id,
    minute,
    team_name = team.name
  )
]

goals[
  ,
  goal_type :=
    fifelse(
      team_name == "América",
      "AMERICA_SCORES",
      "AMERICA_CONCEDES"
    )
]


# ============================================================
# 5. QUITAR DUPLICADOS
# ============================================================

goals <- unique(
  goals,
  by = c(
    "match_id",
    "minute",
    "team_name"
  )
)

cat(
  "Goals detected:",
  nrow(goals),
  "\n"
)

cat(
  "America scores:",
  goals[
    goal_type == "AMERICA_SCORES",
    .N
  ],
  "\n"
)

cat(
  "America concedes:",
  goals[
    goal_type == "AMERICA_CONCEDES",
    .N
  ],
  "\n"
)


# ============================================================
# 6. FEATURES
# ============================================================

features <- c(
  "america_field_tilt_last5",
  "pressure_diff_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_pass_diff_last5",
  "recovery_height_diff_last5",
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5"
)

missing_features <- setdiff(
  features,
  names(model)
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
# 7. PREPARAR DATA
# ============================================================

dt <- copy(model)

setorder(
  dt,
  match_id,
  minute
)


# ============================================================
# 8. CONSTRUIR EVENT WINDOWS
#
# -5 a -1 = BEFORE
# +1 a +5 = AFTER
# ============================================================

event_rows_list <- list()

counter <- 1


for (i in seq_len(nrow(goals))) {

  g <- goals[i]

  match_data <- dt[
    match_id == g$match_id
  ]

  if (nrow(match_data) == 0) {
    next
  }

  temp <- match_data[
    minute >= g$minute - 5 &
      minute <= g$minute + 5
  ]

  if (nrow(temp) == 0) {
    next
  }

  temp[
    ,
    event_id := i
  ]

  temp[
    ,
    goal_minute := g$minute
  ]

  temp[
    ,
    goal_type := g$goal_type
  ]

  temp[
    ,
    relative_minute :=
      minute - goal_minute
  ]

  temp <- temp[
    relative_minute != 0
  ]

  event_rows_list[[counter]] <- temp

  counter <- counter + 1
}


if (length(event_rows_list) == 0) {

  stop(
    "No se pudieron construir ventanas alrededor de goles."
  )
}


event_data <- rbindlist(
  event_rows_list,
  use.names = TRUE,
  fill = TRUE
)


# ============================================================
# 9. VALIDAR VENTANA ANTES/DESPUÉS
# ============================================================

valid_event_ids <- event_data[
  ,
  .(
    before_minutes =
      uniqueN(
        relative_minute[
          relative_minute >= -5 &
          relative_minute <= -1
        ]
      ),

    after_minutes =
      uniqueN(
        relative_minute[
          relative_minute >= 1 &
          relative_minute <= 5
        ]
      )
  ),
  by = event_id
][
  before_minutes >= 3 &
    after_minutes >= 3,
  event_id
]


event_data <- event_data[
  event_id %in%
    valid_event_ids
]


if (nrow(event_data) == 0) {

  stop(
    "No quedaron eventos con ventanas válidas."
  )
}


cat(
  "Valid goal events:",
  uniqueN(event_data$event_id),
  "\n"
)


# ============================================================
# 10. BEFORE / AFTER
# ============================================================

event_data[
  ,
  period :=
    fifelse(
      relative_minute < 0,
      "BEFORE",
      "AFTER"
    )
]


# ============================================================
# 11. PROMEDIO POR EVENTO Y FEATURE
# ============================================================

event_summary_list <- list()

counter <- 1


for (feature in features) {

  temp <- event_data[
    ,
    .(
      feature_value =
        mean(
          get(feature),
          na.rm = TRUE
        )
    ),
    by = .(
      event_id,
      match_id,
      goal_type,
      period
    )
  ]

  temp[
    ,
    feature := feature
  ]

  event_summary_list[[counter]] <- temp

  counter <- counter + 1
}


event_summary <- rbindlist(
  event_summary_list,
  use.names = TRUE,
  fill = TRUE
)


# ============================================================
# 12. PASAR A WIDE
# ============================================================

event_wide <- dcast(
  event_summary,
  event_id +
    match_id +
    goal_type +
    feature ~ period,
  value.var = "feature_value"
)


# ============================================================
# 13. VALIDAR BEFORE Y AFTER
# ============================================================

if (
  !"BEFORE" %in% names(event_wide) ||
    !"AFTER" %in% names(event_wide)
) {

  stop(
    "No se generaron correctamente las columnas BEFORE y AFTER."
  )
}


event_wide[
  ,
  delta :=
    AFTER - BEFORE
]


# ============================================================
# 14. EFECTOS PROMEDIO
# ============================================================

event_effects <- event_wide[
  is.finite(BEFORE) &
    is.finite(AFTER),
  .(
    events = .N,

    mean_before =
      mean(
        BEFORE,
        na.rm = TRUE
      ),

    mean_after =
      mean(
        AFTER,
        na.rm = TRUE
      ),

    mean_delta =
      mean(
        delta,
        na.rm = TRUE
      ),

    median_delta =
      median(
        delta,
        na.rm = TRUE
      ),

    sd_delta =
      sd(
        delta,
        na.rm = TRUE
      )
  ),
  by = .(
    goal_type,
    feature
  )
]


# ============================================================
# 15. STANDARD ERROR E IC
# ============================================================

event_effects[
  ,
  se_delta :=
    sd_delta /
    sqrt(events)
]

event_effects[
  ,
  ci_low :=
    mean_delta -
    1.96 * se_delta
]

event_effects[
  ,
  ci_high :=
    mean_delta +
    1.96 * se_delta
]

event_effects[
  ,
  ci_excludes_zero :=
    ci_low > 0 |
    ci_high < 0
]


# ============================================================
# 16. PREPARAR ESTADOS TÁCTICOS
# ============================================================

state_data <- event_data[
  ,
  .(
    event_id,
    match_id,
    goal_type,
    relative_minute,

    state =
      as.character(
        current_tactical_state
      )
  )
]


# ============================================================
# 17. OBTENER ESTADO JUSTO ANTES Y DESPUÉS
# ============================================================

state_change_list <- list()

event_ids <- unique(
  state_data$event_id
)

counter <- 1


for (eid in event_ids) {

  temp <- state_data[
    event_id == eid
  ]

  before_rows <- temp[
    relative_minute < 0
  ]

  after_rows <- temp[
    relative_minute > 0
  ]

  if (
    nrow(before_rows) == 0 ||
    nrow(after_rows) == 0
  ) {
    next
  }


  before_row <- before_rows[
    which.max(relative_minute)
  ]

  after_row <- after_rows[
    which.max(relative_minute)
  ]


  before_state <- before_row$state

  after_state <- after_row$state


  state_change_list[[counter]] <- data.table(

    event_id =
      eid,

    match_id =
      temp$match_id[1],

    goal_type =
      temp$goal_type[1],

    before_state =
      before_state,

    after_state =
      after_state,

    tactical_state_changed =
      as.integer(
        !is.na(before_state) &
        !is.na(after_state) &
        before_state != after_state
      )
  )

  counter <- counter + 1
}


if (length(state_change_list) == 0) {

  stop(
    "No se pudieron calcular cambios de estado."
  )
}


state_change_summary <- rbindlist(
  state_change_list,
  use.names = TRUE,
  fill = TRUE
)


# ============================================================
# 18. TASA DE CAMBIO TÁCTICO
# ============================================================

state_change_rates <- state_change_summary[
  ,
  .(
    events = .N,

    tactical_change_rate =
      mean(
        tactical_state_changed,
        na.rm = TRUE
      )
  ),
  by = goal_type
]


# ============================================================
# 19. TRANSICIONES
# ============================================================

state_transitions <- state_change_summary[
  !is.na(before_state) &
    !is.na(after_state),
  .N,
  by = .(
    goal_type,
    before_state,
    after_state
  )
]


state_transitions[
  ,
  percentage :=
    round(
      N /
      sum(N) *
      100,
      2
    ),
  by = goal_type
]


setorder(
  state_transitions,
  goal_type,
  -N
)


# ============================================================
# 20. TRAYECTORIA POR MINUTO RELATIVO
# ============================================================

relative_time_list <- list()

counter <- 1


for (feature in features) {

  temp <- event_data[
    ,
    .(
      mean_value =
        mean(
          get(feature),
          na.rm = TRUE
        ),

      observations =
        .N
    ),
    by = .(
      goal_type,
      relative_minute
    )
  ]

  temp[
    ,
    feature := feature
  ]

  relative_time_list[[counter]] <- temp

  counter <- counter + 1
}


relative_time_effects <- rbindlist(
  relative_time_list,
  use.names = TRUE,
  fill = TRUE
)


# ============================================================
# 21. NÚMERO DE EVENTOS
# ============================================================

n_scoring <- uniqueN(
  event_data[
    goal_type == "AMERICA_SCORES",
    event_id
  ]
)

n_conceding <- uniqueN(
  event_data[
    goal_type == "AMERICA_CONCEDES",
    event_id
  ]
)


# ============================================================
# 22. TASAS DE CAMBIO SEGURAS
# ============================================================

scoring_change_rate <- state_change_rates[
  goal_type == "AMERICA_SCORES",
  tactical_change_rate
]

conceding_change_rate <- state_change_rates[
  goal_type == "AMERICA_CONCEDES",
  tactical_change_rate
]


if (length(scoring_change_rate) == 0) {
  scoring_change_rate <- NA_real_
}

if (length(conceding_change_rate) == 0) {
  conceding_change_rate <- NA_real_
}


# ============================================================
# 23. RESUMEN
# ============================================================

goal_event_summary <- data.table(

  metric = c(
    "America scoring events",
    "America conceding events",
    "Tactical change rate after scoring",
    "Tactical change rate after conceding"
  ),

  value = c(
    n_scoring,
    n_conceding,
    scoring_change_rate,
    conceding_change_rate
  )
)


# ============================================================
# 24. TOP CAMBIOS
# ============================================================

event_effects[
  ,
  abs_delta :=
    abs(
      mean_delta
    )
]


top_goal_effects <- event_effects[
  order(
    goal_type,
    -abs_delta
  ),
  head(
    .SD,
    8
  ),
  by = goal_type
]


# ============================================================
# 25. GUARDAR
# ============================================================

dir.create(
  "outputs/models/event_study",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  goal_event_summary,
  "outputs/models/event_study/goal_event_summary.csv",
  bom = TRUE
)

fwrite(
  event_effects,
  "outputs/models/event_study/goal_feature_effects.csv",
  bom = TRUE
)

fwrite(
  state_change_rates,
  "outputs/models/event_study/goal_state_change_rates.csv",
  bom = TRUE
)

fwrite(
  state_transitions,
  "outputs/models/event_study/goal_state_transitions.csv",
  bom = TRUE
)

fwrite(
  relative_time_effects,
  "outputs/models/event_study/goal_relative_time_effects.csv",
  bom = TRUE
)


# ============================================================
# 26. SOLO 3 VISTAS
# ============================================================

View(
  goal_event_summary
)

View(
  top_goal_effects
)

View(
  state_transitions[
    1:min(
      20,
      .N
    )
  ]
)


# ============================================================
# 27. TERMINAL
# ============================================================

cat("\n=====================================\n")
cat("GOAL SHOCK EVENT STUDY TERMINADO\n")
cat("=====================================\n")

cat(
  "Valid scoring events:",
  n_scoring,
  "\n"
)

cat(
  "Valid conceding events:",
  n_conceding,
  "\n"
)

cat(
  "Scoring tactical-change rate:",
  round(
    scoring_change_rate * 100,
    2
  ),
  "%\n"
)

cat(
  "Conceding tactical-change rate:",
  round(
    conceding_change_rate * 100,
    2
  ),
  "%\n"
)