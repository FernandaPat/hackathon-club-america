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
cat("CAUSAL ANALYSIS: SUBSTITUTIONS\n")
cat("=====================================\n")

cat("Model rows:", nrow(model), "\n")
cat("Matches:", uniqueN(model$match_id), "\n")


# ============================================================
# 2. VALIDAR COLUMNAS DE EVENTS
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

if (
  length(missing_event_cols) > 0
) {

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
# 3. EXTRAER SUSTITUCIONES DE AMÉRICA
# ============================================================

america_substitutions <- unique(
  events[
    team.name == "América" &
      type.name == "Substitution",
    .(
      match_id,
      minute
    )
  ]
)

america_substitutions[
  ,
  substitution :=
    1L
]


cat(
  "Sustituciones detectadas:",
  nrow(america_substitutions),
  "\n"
)


# ============================================================
# 4. PREPARAR DATASET CAUSAL
# ============================================================

dt <- copy(model)

setorder(
  dt,
  match_id,
  minute
)


# ============================================================
# 5. TARGET:
# CAMBIO TÁCTICO EN LOS SIGUIENTES 5 MIN
# ============================================================

dt[
  ,
  tactical_change_next5 :=
    as.integer(
      as.character(
        current_tactical_state
      ) !=
        as.character(
          future_tactical_state_5min
        )
    )
]


# ============================================================
# 6. UNIR TRATAMIENTO
# ============================================================

dt <- merge(
  dt,
  america_substitutions,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)

dt[
  is.na(substitution),
  substitution :=
    0L
]


# ============================================================
# 7. EVITAR MINUTOS SIN OUTCOME
# ============================================================

dt <- dt[
  !is.na(
    tactical_change_next5
  )
]


# ============================================================
# 8. COVARIABLES PRE-TRATAMIENTO
#
# Todas deben representar información conocida
# ANTES o en el momento de la sustitución.
# ============================================================

covariates <- c(

  "minute",
  "game_state_num",

  "america_field_tilt_last5",

  "entries_final_third_diff_last5",

  "entries_box_diff_last5",

  "progressive_pass_diff_last5",

  "pressure_diff_last5",

  "recovery_height_diff_last5",

  "america_local_superiority_10_last5",

  "america_pressure_share_5_last5",

  "america_nearest_opponent_distance_last5",

  "america_support_share_10_last5"
)


missing_covariates <- setdiff(
  covariates,
  names(dt)
)

if (
  length(missing_covariates) > 0
) {

  stop(
    paste(
      "Faltan covariables:",
      paste(
        missing_covariates,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 9. ESTADO ACTUAL COMO COVARIABLE CATEGÓRICA
# ============================================================

dt[
  ,
  current_state :=
    factor(
      current_tactical_state
    )
]


# ============================================================
# 10. IMPUTACIÓN
#
# Aquí no estamos evaluando predicción.
# La mediana se calcula en el dataset causal.
# ============================================================

for (
  col in covariates
) {

  med <- median(
    dt[[col]],
    na.rm = TRUE
  )

  if (
    !is.finite(med)
  ) {
    med <- 0
  }

  dt[
    is.na(get(col)),
    (col) := med
  ]
}


# ============================================================
# 11. RESTRINGIR SOPORTE
#
# Excluimos minutos extremadamente tempranos.
# Es muy raro sustituir antes del minuto 10 y no
# tendrían controles comparables.
# ============================================================

dt <- dt[
  minute >= 10
]


# ============================================================
# 12. DISTRIBUCIÓN TRATAMIENTO
# ============================================================

treatment_summary <- dt[
  ,
  .N,
  by = substitution
]

treatment_summary[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]


# ============================================================
# 13. DIFERENCIA BRUTA
#
# ESTO ES SOLO CORRELACIÓN.
# ============================================================

raw_effect <- dt[
  ,
  .(
    observations =
      .N,

    change_rate =
      mean(
        tactical_change_next5
      )
  ),
  by = substitution
]


raw_treated_rate <- raw_effect[
  substitution == 1,
  change_rate
]

raw_control_rate <- raw_effect[
  substitution == 0,
  change_rate
]


raw_difference <-
  raw_treated_rate -
  raw_control_rate


# ============================================================
# 14. PROPENSITY SCORE
#
# Probabilidad de realizar una sustitución dado
# el contexto previo.
# ============================================================

propensity_formula <- as.formula(

  paste(

    "substitution ~",

    paste(
      c(
        covariates,
        "current_state"
      ),
      collapse = " + "
    )
  )
)


propensity_model <- glm(
  propensity_formula,
  data = dt,
  family = binomial()
)


dt[
  ,
  propensity_score :=
    predict(
      propensity_model,
      newdata = dt,
      type = "response"
    )
]


# ============================================================
# 15. REVISAR PROPENSITY SCORES
# ============================================================

propensity_summary <- dt[
  ,
  .(
    observations =
      .N,

    mean_propensity =
      mean(
        propensity_score
      ),

    min_propensity =
      min(
        propensity_score
      ),

    max_propensity =
      max(
        propensity_score
      )
  ),
  by = substitution
]


# ============================================================
# 16. COMMON SUPPORT
# ============================================================

treated <- dt[
  substitution == 1
]

controls <- dt[
  substitution == 0
]


lower_support <- max(
  min(
    treated$propensity_score
  ),
  min(
    controls$propensity_score
  )
)


upper_support <- min(
  max(
    treated$propensity_score
  ),
  max(
    controls$propensity_score
  )
)


treated <- treated[
  propensity_score >=
    lower_support &
  propensity_score <=
    upper_support
]


controls <- controls[
  propensity_score >=
    lower_support &
  propensity_score <=
    upper_support
]


cat(
  "Treated on common support:",
  nrow(treated),
  "\n"
)


# ============================================================
# 17. CALIPER
#
# 0.2 SD del logit propensity es una regla común.
# ============================================================

safe_logit <- function(p) {

  p <- pmax(
    pmin(
      p,
      0.999999
    ),
    0.000001
  )

  log(
    p /
      (
        1 - p
      )
  )
}


dt[
  ,
  propensity_logit :=
    safe_logit(
      propensity_score
    )
]


treated[
  ,
  propensity_logit :=
    safe_logit(
      propensity_score
    )
]


controls[
  ,
  propensity_logit :=
    safe_logit(
      propensity_score
    )
]


caliper <-
  0.20 *
  sd(
    dt$propensity_logit,
    na.rm = TRUE
  )


cat(
  "Matching caliper:",
  round(
    caliper,
    4
  ),
  "\n"
)


# ============================================================
# 18. NEAREST-NEIGHBOR MATCHING 1:1
#
# Sin reemplazo.
#
# Además exigimos mismo estado táctico actual
# para hacer controles más comparables.
# ============================================================

treated <- treated[
  order(
    -propensity_score
  )
]


available_controls <- copy(
  controls
)


matched_pairs_list <- list()

pair_counter <- 1


for (
  i in seq_len(
    nrow(treated)
  )
) {

  trow <- treated[i]


  candidates <- available_controls[
    current_state ==
      trow$current_state
  ]


  if (
    nrow(candidates) == 0
  ) {
    next
  }


  candidates[
    ,
    distance :=
      abs(
        propensity_logit -
        trow$propensity_logit
      )
  ]


  best <- candidates[
    which.min(distance)
  ]


  if (
    best$distance >
      caliper
  ) {
    next
  }


  matched_pairs_list[
    [
      pair_counter
    ]
  ] <- data.table(

    pair_id =
      pair_counter,

    treated_match_id =
      trow$match_id,

    treated_minute =
      trow$minute,

    treated_state =
      as.character(
        trow$current_state
      ),

    treated_propensity =
      trow$propensity_score,

    treated_outcome =
      trow$tactical_change_next5,

    control_match_id =
      best$match_id,

    control_minute =
      best$minute,

    control_state =
      as.character(
        best$current_state
      ),

    control_propensity =
      best$propensity_score,

    control_outcome =
      best$tactical_change_next5,

    propensity_distance =
      best$distance
  )


  # remover control usado
  available_controls <- available_controls[
    !(
      match_id ==
        best$match_id &
      minute ==
        best$minute
    )
  ]


  pair_counter <-
    pair_counter + 1
}


if (
  length(
    matched_pairs_list
  ) == 0
) {

  stop(
    "No se encontraron pares dentro del caliper."
  )
}


matched_pairs <- rbindlist(
  matched_pairs_list
)


# ============================================================
# 19. EFECTO PROMEDIO EN TRATADOS
#
# ATT =
# resultado tratado - resultado control comparable
# ============================================================

matched_pairs[
  ,
  pair_effect :=
    treated_outcome -
    control_outcome
]


att <- mean(
  matched_pairs$pair_effect
)


treated_matched_rate <- mean(
  matched_pairs$treated_outcome
)


control_matched_rate <- mean(
  matched_pairs$control_outcome
)


# ============================================================
# 20. ERROR ESTÁNDAR PAREADO
# ============================================================

att_se <-
  sd(
    matched_pairs$pair_effect
  ) /
  sqrt(
    nrow(
      matched_pairs
    )
  )


att_ci_low <-
  att -
  1.96 *
  att_se


att_ci_high <-
  att +
  1.96 *
  att_se


# ============================================================
# 21. BALANCE:
# STANDARDIZED MEAN DIFFERENCE
# ============================================================

smd <- function(
  x_treated,
  x_control
) {

  mean_diff <-
    mean(
      x_treated,
      na.rm = TRUE
    ) -
    mean(
      x_control,
      na.rm = TRUE
    )


  pooled_sd <- sqrt(
    (
      var(
        x_treated,
        na.rm = TRUE
      ) +
      var(
        x_control,
        na.rm = TRUE
      )
    ) /
      2
  )


  if (
    !is.finite(
      pooled_sd
    ) ||
    pooled_sd == 0
  ) {

    return(0)
  }


  mean_diff /
    pooled_sd
}


# ============================================================
# 22. CREAR DATA MATCHED PARA BALANCE
# ============================================================

treated_keys <- matched_pairs[
  ,
  .(
    match_id =
      treated_match_id,

    minute =
      treated_minute
  )
]


control_keys <- matched_pairs[
  ,
  .(
    match_id =
      control_match_id,

    minute =
      control_minute
  )
]


matched_treated <- merge(
  dt,
  treated_keys,
  by = c(
    "match_id",
    "minute"
  )
)


matched_controls <- merge(
  dt,
  control_keys,
  by = c(
    "match_id",
    "minute"
  )
)


# ============================================================
# 23. BALANCE ANTES Y DESPUÉS
# ============================================================

balance_list <- list()


for (
  i in seq_along(
    covariates
  )
) {

  col <- covariates[i]


  before_smd <- smd(
    treated[[col]],
    controls[[col]]
  )


  after_smd <- smd(
    matched_treated[[col]],
    matched_controls[[col]]
  )


  balance_list[[i]] <- data.table(

    variable =
      col,

    smd_before =
      before_smd,

    smd_after =
      after_smd,

    abs_smd_before =
      abs(
        before_smd
      ),

    abs_smd_after =
      abs(
        after_smd
      )
  )
}


balance_table <- rbindlist(
  balance_list
)


setorder(
  balance_table,
  -abs_smd_after
)


# ============================================================
# 24. RESUMEN DEL EFECTO
# ============================================================

causal_summary <- data.table(

  metric = c(

    "Raw treated change rate",

    "Raw control change rate",

    "Raw difference",

    "Matched treated change rate",

    "Matched control change rate",

    "Matched ATT",

    "ATT 95% CI low",

    "ATT 95% CI high",

    "Matched pairs"
  ),

  value = c(

    raw_treated_rate,

    raw_control_rate,

    raw_difference,

    treated_matched_rate,

    control_matched_rate,

    att,

    att_ci_low,

    att_ci_high,

    nrow(
      matched_pairs
    )
  )
)


# ============================================================
# 25. BALANCE SUMMARY
#
# |SMD| < 0.10 suele considerarse buen balance.
# ============================================================

balance_summary <- data.table(

  metric = c(
    "Mean abs SMD before",
    "Mean abs SMD after",
    "Max abs SMD before",
    "Max abs SMD after",
    "Variables after matching with |SMD| < 0.10"
  ),

  value = c(

    mean(
      balance_table$
        abs_smd_before
    ),

    mean(
      balance_table$
        abs_smd_after
    ),

    max(
      balance_table$
        abs_smd_before
    ),

    max(
      balance_table$
        abs_smd_after
    ),

    mean(
      balance_table$
        abs_smd_after <
        0.10
    ) *
      100
  )
)


# ============================================================
# 26. INTERPRETACIÓN CONSERVADORA
# ============================================================

interpretation <- data.table(

  item = c(
    "Design",
    "Treatment",
    "Outcome",
    "Identification strategy",
    "Important limitation"
  ),

  description = c(

    "Observational matched study",

    "Club América substitution at minute t",

    "Tactical-state change within the following 5 minutes",

    paste(
      "1:1 nearest-neighbor propensity-score matching",
      "on pre-treatment match, event and 360 context"
    ),

    paste(
      "Substitutions are not randomized.",
      "Matching reduces observed confounding but",
      "does not prove causality against unobserved confounding."
    )
  )
)


# ============================================================
# 27. GUARDAR
# ============================================================

dir.create(
  "outputs/models/causal",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  causal_summary,
  "outputs/models/causal/substitution_causal_summary.csv",
  bom = TRUE
)


fwrite(
  balance_table,
  "outputs/models/causal/substitution_balance.csv",
  bom = TRUE
)


fwrite(
  balance_summary,
  "outputs/models/causal/substitution_balance_summary.csv",
  bom = TRUE
)


fwrite(
  matched_pairs,
  "outputs/models/causal/substitution_matched_pairs.csv",
  bom = TRUE
)


fwrite(
  interpretation,
  "outputs/models/causal/substitution_identification_strategy.csv",
  bom = TRUE
)


# ============================================================
# 28. SOLO 3 VISTAS
# ============================================================

View(
  causal_summary
)

View(
  balance_summary
)

View(
  balance_table
)


# ============================================================
# 29. RESUMEN TERMINAL
# ============================================================

cat("\n=====================================\n")
cat("CAUSAL MATCHING TERMINADO\n")
cat("=====================================\n")

cat(
  "Matched pairs:",
  nrow(
    matched_pairs
  ),
  "\n"
)

cat(
  "Raw difference:",
  round(
    raw_difference * 100,
    2
  ),
  "pp\n"
)

cat(
  "Matched ATT:",
  round(
    att * 100,
    2
  ),
  "pp\n"
)

cat(
  "95% CI:",
  round(
    att_ci_low * 100,
    2
  ),
  "to",
  round(
    att_ci_high * 100,
    2
  ),
  "pp\n"
)

cat(
  "Mean abs SMD after:",
  round(
    mean(
      balance_table$
        abs_smd_after
    ),
    3
  ),
  "\n"
)