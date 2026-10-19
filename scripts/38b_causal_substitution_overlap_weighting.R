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
cat("CAUSAL ANALYSIS - OVERLAP WEIGHTING\n")
cat("=====================================\n")


# ============================================================
# 2. EXTRAER SUSTITUCIONES
# ============================================================

subs <- unique(
  events[
    team.name == "América" &
      type.name == "Substitution",
    .(
      match_id,
      minute
    )
  ]
)

subs[
  ,
  substitution := 1L
]


# ============================================================
# 3. PREPARAR DATA
# ============================================================

dt <- copy(model)

setorder(
  dt,
  match_id,
  minute
)


# ============================================================
# 4. OUTCOME:
# CAMBIO TÁCTICO EN PRÓXIMOS 5 MIN
# ============================================================

dt[
  ,
  tactical_change_next5 :=
    as.integer(
      as.character(current_tactical_state) !=
        as.character(future_tactical_state_5min)
    )
]


# ============================================================
# 5. UNIR TRATAMIENTO
# ============================================================

dt <- merge(
  dt,
  subs,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)

dt[
  is.na(substitution),
  substitution := 0L
]


dt <- dt[
  !is.na(tactical_change_next5)
]


# ============================================================
# 6. RESTRINGIR A MINUTOS RAZONABLES
#
# Evitamos situaciones tempranas donde casi nunca
# existe posibilidad real de sustitución.
# ============================================================

dt <- dt[
  minute >= 10 &
  minute <= 90
]


# ============================================================
# 7. COVARIABLES PRE-TRATAMIENTO
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
# 8. ESTADO TÁCTICO ACTUAL
# ============================================================

dt[
  ,
  current_state :=
    factor(
      current_tactical_state
    )
]


# ============================================================
# 9. IMPUTAR
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
# 10. RESUMEN DEL TRATAMIENTO
# ============================================================

treatment_summary <- dt[
  ,
  .(
    observations = .N,

    change_rate =
      mean(
        tactical_change_next5
      )
  ),
  by = substitution
]


# ============================================================
# 11. DIFERENCIA BRUTA
# ============================================================

raw_treated <- treatment_summary[
  substitution == 1,
  change_rate
]

raw_control <- treatment_summary[
  substitution == 0,
  change_rate
]

raw_difference <-
  raw_treated -
  raw_control


# ============================================================
# 12. PROPENSITY MODEL
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
# 13. EVITAR EXTREMOS NUMÉRICOS
# ============================================================

dt[
  ,
  propensity_score :=
    pmax(
      pmin(
        propensity_score,
        0.999
      ),
      0.001
    )
]


# ============================================================
# 14. OVERLAP WEIGHTS
#
# Tratados: 1 - e(x)
# Controles: e(x)
#
# Esto pone más peso en observaciones donde ambos
# tratamientos eran plausibles.
# ============================================================

dt[
  ,
  overlap_weight :=
    fifelse(
      substitution == 1,
      1 - propensity_score,
      propensity_score
    )
]


# ============================================================
# 15. EFFECTIVE SAMPLE SIZE
# ============================================================

effective_sample_size <- function(w) {

  (
    sum(w)^2
  ) /
    sum(
      w^2
    )
}


ess <- dt[
  ,
  .(
    raw_n =
      .N,

    effective_n =
      effective_sample_size(
        overlap_weight
      )
  ),
  by = substitution
]


# ============================================================
# 16. RESULTADOS PONDERADOS
# ============================================================

weighted_rates <- dt[
  ,
  .(
    weighted_change_rate =
      weighted.mean(
        tactical_change_next5,
        overlap_weight
      )
  ),
  by = substitution
]


weighted_treated_rate <- weighted_rates[
  substitution == 1,
  weighted_change_rate
]


weighted_control_rate <- weighted_rates[
  substitution == 0,
  weighted_change_rate
]


overlap_effect <-
  weighted_treated_rate -
  weighted_control_rate


# ============================================================
# 17. STANDARDIZED MEAN DIFFERENCE
# ============================================================

weighted_mean <- function(
  x,
  w
) {

  sum(
    x * w
  ) /
    sum(w)
}


weighted_var <- function(
  x,
  w
) {

  m <- weighted_mean(
    x,
    w
  )

  sum(
    w *
      (
        x - m
      )^2
  ) /
    sum(w)
}


smd_unweighted <- function(
  treated,
  control
) {

  denom <- sqrt(
    (
      var(
        treated,
        na.rm = TRUE
      ) +
      var(
        control,
        na.rm = TRUE
      )
    ) / 2
  )

  if (
    !is.finite(denom) ||
    denom == 0
  ) {

    return(0)
  }

  (
    mean(
      treated,
      na.rm = TRUE
    ) -
    mean(
      control,
      na.rm = TRUE
    )
  ) /
    denom
}


smd_weighted <- function(
  treated,
  treated_w,
  control,
  control_w
) {

  mt <- weighted_mean(
    treated,
    treated_w
  )

  mc <- weighted_mean(
    control,
    control_w
  )


  vt <- weighted_var(
    treated,
    treated_w
  )

  vc <- weighted_var(
    control,
    control_w
  )


  denom <- sqrt(
    (
      vt + vc
    ) / 2
  )


  if (
    !is.finite(denom) ||
    denom == 0
  ) {

    return(0)
  }


  (
    mt - mc
  ) /
    denom
}


# ============================================================
# 18. BALANCE TABLE
# ============================================================

treated <- dt[
  substitution == 1
]

control <- dt[
  substitution == 0
]


balance_list <- vector(
  "list",
  length(covariates)
)


for (
  i in seq_along(
    covariates
  )
) {

  col <- covariates[i]


  smd_before <- smd_unweighted(

    treated[[col]],

    control[[col]]
  )


  smd_after <- smd_weighted(

    treated[[col]],

    treated$overlap_weight,

    control[[col]],

    control$overlap_weight
  )


  balance_list[[i]] <- data.table(

    variable =
      col,

    smd_before =
      smd_before,

    smd_after =
      smd_after,

    abs_smd_before =
      abs(
        smd_before
      ),

    abs_smd_after =
      abs(
        smd_after
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
# 19. BALANCE SUMMARY
# ============================================================

balance_summary <- data.table(

  metric = c(

    "Mean abs SMD before",

    "Mean abs SMD after",

    "Max abs SMD before",

    "Max abs SMD after",

    "Variables with |SMD| < 0.10 after weighting"
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
# 20. BOOTSTRAP POR PARTIDO
#
# Re-muestreamos PARTIDOS, no minutos,
# para respetar dependencia temporal interna.
# ============================================================

set.seed(2026)

match_ids <- unique(
  dt$match_id
)

B <- 300

bootstrap_effects <- numeric(
  B
)


for (
  b in seq_len(B)
) {

  sampled_matches <- sample(
    match_ids,
    size = length(match_ids),
    replace = TRUE
  )


  boot_parts <- vector(
    "list",
    length(sampled_matches)
  )


  for (
    j in seq_along(
      sampled_matches
    )
  ) {

    temp <- copy(
      dt[
        match_id ==
          sampled_matches[j]
      ]
    )

    temp[
      ,
      bootstrap_match :=
        j
    ]

    boot_parts[[j]] <-
      temp
  }


  boot <- rbindlist(
    boot_parts
  )


  treated_boot <- boot[
    substitution == 1
  ]

  control_boot <- boot[
    substitution == 0
  ]


  if (
    nrow(treated_boot) == 0 ||
    nrow(control_boot) == 0
  ) {

    bootstrap_effects[b] <-
      NA_real_

    next
  }


  rate_t <- weighted.mean(
    treated_boot$
      tactical_change_next5,

    treated_boot$
      overlap_weight
  )


  rate_c <- weighted.mean(
    control_boot$
      tactical_change_next5,

    control_boot$
      overlap_weight
  )


  bootstrap_effects[b] <-
    rate_t -
    rate_c
}


bootstrap_effects <- bootstrap_effects[
  is.finite(
    bootstrap_effects
  )
]


ci_low <- quantile(
  bootstrap_effects,
  0.025,
  na.rm = TRUE
)


ci_high <- quantile(
  bootstrap_effects,
  0.975,
  na.rm = TRUE
)


# ============================================================
# 21. RESUMEN CAUSAL
# ============================================================

causal_summary <- data.table(

  metric = c(

    "Raw treated change rate",

    "Raw control change rate",

    "Raw difference",

    "Overlap-weighted treated rate",

    "Overlap-weighted control rate",

    "Overlap-weighted effect",

    "Bootstrap 95% CI low",

    "Bootstrap 95% CI high",

    "Treated observations",

    "Control observations",

    "Effective treated sample",

    "Effective control sample"
  ),

  value = c(

    raw_treated,

    raw_control,

    raw_difference,

    weighted_treated_rate,

    weighted_control_rate,

    overlap_effect,

    as.numeric(
      ci_low
    ),

    as.numeric(
      ci_high
    ),

    nrow(
      treated
    ),

    nrow(
      control
    ),

    ess[
      substitution == 1,
      effective_n
    ],

    ess[
      substitution == 0,
      effective_n
    ]
  )
)


# ============================================================
# 22. IDENTIFICATION STATEMENT
# ============================================================

identification_strategy <- data.table(

  component = c(

    "Treatment",

    "Outcome",

    "Adjustment",

    "Estimator",

    "Uncertainty",

    "Interpretation limitation"
  ),

  description = c(

    "Club América substitution at minute t",

    "Change in tactical state within the following 5 minutes",

    paste(
      "Minute, score state, current tactical state,",
      "Events and StatsBomb 360 pre-treatment context"
    ),

    "Propensity-score overlap weighting",

    "300 match-cluster bootstrap replications",

    paste(
      "Observational estimate.",
      "Balances measured confounders but cannot rule out",
      "unobserved coaching intent or other latent factors."
    )
  )
)


# ============================================================
# 23. GUARDAR
# ============================================================

dir.create(
  "outputs/models/causal",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  causal_summary,
  "outputs/models/causal/substitution_overlap_summary.csv",
  bom = TRUE
)


fwrite(
  balance_summary,
  "outputs/models/causal/substitution_overlap_balance_summary.csv",
  bom = TRUE
)


fwrite(
  balance_table,
  "outputs/models/causal/substitution_overlap_balance.csv",
  bom = TRUE
)


fwrite(
  identification_strategy,
  "outputs/models/causal/substitution_overlap_identification.csv",
  bom = TRUE
)


# ============================================================
# 24. SOLO 3 TABLAS
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
# 25. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("OVERLAP WEIGHTING TERMINADO\n")
cat("=====================================\n")

cat(
  "Raw difference:",
  round(
    raw_difference *
      100,
    2
  ),
  "pp\n"
)


cat(
  "Overlap effect:",
  round(
    overlap_effect *
      100,
    2
  ),
  "pp\n"
)


cat(
  "Bootstrap 95% CI:",
  round(
    ci_low *
      100,
    2
  ),
  "to",
  round(
    ci_high *
      100,
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
    4
  ),
  "\n"
)