library(data.table)
library(xgboost)

# ============================================================
# 1. CARGAR DATASET
# ============================================================

dt <- fread(
  "data/processed/america_male/model_ready_predictive.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

state_levels <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

cat("\n=====================================\n")
cat("COACH ENGINE\n")
cat("=====================================\n")

cat("Rows:", nrow(dt), "\n")
cat("Matches:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. ESTADO ACTUAL
# ============================================================

dt[
  ,
  current_tactical_state :=
    factor(
      current_tactical_state,
      levels = state_levels
    )
]


# ============================================================
# 3. ONE-HOT DEL ESTADO ACTUAL
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
        current_tactical_state == state
      )
  ]
}

state_features <- paste0(
  "current_state_",
  tolower(state_levels)
)


# ============================================================
# 4. FEATURES DEL MODELO
# ============================================================

features <- c(

  # CONTEXTO
  "minute",
  "game_state_num",

  # ESTADO ACTUAL
  state_features,

  # EVENTS 5 MIN
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # EVENTS 3 MIN
  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_pass_diff_last3",
  "america_progressive_carries_last3",
  "america_progressive_passes_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  # 360 5 MIN
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 360 3 MIN
  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3"
)


# ============================================================
# 5. VALIDAR FEATURES
# ============================================================

missing_features <- setdiff(
  features,
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
# 6. IMPUTACIÓN
#
# Medianas calculadas SOLO con TRAIN
# ============================================================

for (col in features) {

  med <- median(
    dt[
      dataset_split == "TRAIN",
      get(col)
    ],
    na.rm = TRUE
  )

  if (!is.finite(med)) {
    med <- 0
  }

  dt[
    is.na(get(col)),
    (col) := med
  ]
}


# ============================================================
# 7. TEST PARA EVALUAR COACH ENGINE
# ============================================================

test <- copy(
  dt[
    dataset_split == "TEST"
  ]
)

if (nrow(test) == 0) {

  stop(
    "TEST está vacío."
  )
}


# ============================================================
# 8. CARGAR MODELO FINAL ACTUAL
# ============================================================

model_path <- "outputs/models/blend_xgboost_model.json"

if (!file.exists(model_path)) {

  stop(
    paste(
      "No encontré el modelo:",
      model_path
    )
  )
}

model <- xgb.load(
  model_path
)


# ============================================================
# 9. MATRIZ
# ============================================================

x_test <- as.matrix(
  test[
    ,
    ..features
  ]
)

dtest <- xgb.DMatrix(
  x_test
)


# ============================================================
# 10. FORMATEAR PROBABILIDADES
# ============================================================

format_probs <- function(
  raw_probs,
  n_rows
) {

  if (is.matrix(raw_probs)) {

    probs <- raw_probs

  } else {

    probs <- matrix(
      raw_probs,
      nrow = n_rows,
      ncol = length(state_levels),
      byrow = TRUE
    )
  }

  if (
    nrow(probs) != n_rows ||
    ncol(probs) != length(state_levels)
  ) {

    stop(
      "Dimensiones inesperadas de probabilidad."
    )
  }

  colnames(probs) <- state_levels

  probs
}


# ============================================================
# 11. PREDICCIONES
# ============================================================

prob_test <- format_probs(
  predict(
    model,
    dtest
  ),
  nrow(test)
)


# ============================================================
# 12. TOP-1 Y TOP-2
# ============================================================

ranking_matrix <- t(
  apply(
    prob_test,
    1,
    function(x) {

      order(
        x,
        decreasing = TRUE
      )
    }
  )
)


top1_index <- ranking_matrix[, 1]
top2_index <- ranking_matrix[, 2]


predicted_state <- state_levels[
  top1_index
]

second_state <- state_levels[
  top2_index
]


confidence <- prob_test[
  cbind(
    seq_len(
      nrow(prob_test)
    ),
    top1_index
  )
]


second_probability <- prob_test[
  cbind(
    seq_len(
      nrow(prob_test)
    ),
    top2_index
  )
]


confidence_margin <-
  confidence -
  second_probability


# ============================================================
# 13. SHAP
# ============================================================

cat("Calculating local explanations...\n")

raw_shap <- tryCatch(

  predict(
    model,
    dtest,
    predcontrib = TRUE,
    strict_shape = TRUE
  ),

  error = function(e) {

    predict(
      model,
      dtest,
      predcontrib = TRUE
    )
  }
)


# ============================================================
# 14. NORMALIZAR SHAP
# ============================================================

normalize_shap <- function(
  shap_object,
  n_rows,
  n_features,
  n_classes
) {

  expected_feature_dim <-
    n_features + 1

  dims <- dim(
    shap_object
  )


  # ----------------------------------------------------------
  # ARRAY 3D
  # ----------------------------------------------------------

  if (length(dims) == 3) {

    row_dim <- which(
      dims == n_rows
    )[1]

    feature_dim <- which(
      dims == expected_feature_dim
    )[1]

    class_dim <- which(
      dims == n_classes
    )[1]


    if (
      any(
        is.na(
          c(
            row_dim,
            feature_dim,
            class_dim
          )
        )
      )
    ) {

      stop(
        paste(
          "No pude interpretar SHAP:",
          paste(
            dims,
            collapse = " x "
          )
        )
      )
    }


    result <- aperm(
      shap_object,
      c(
        row_dim,
        feature_dim,
        class_dim
      )
    )


    return(result)
  }


  # ----------------------------------------------------------
  # MATRIX
  # ----------------------------------------------------------

  if (
    is.matrix(shap_object) &&
    nrow(shap_object) == n_rows &&
    ncol(shap_object) ==
      expected_feature_dim *
      n_classes
  ) {

    result <- array(
      NA_real_,
      dim = c(
        n_rows,
        expected_feature_dim,
        n_classes
      )
    )


    for (k in seq_len(n_classes)) {

      start_col <-
        (k - 1) *
        expected_feature_dim +
        1

      end_col <-
        k *
        expected_feature_dim


      result[, , k] <-
        shap_object[
          ,
          start_col:end_col,
          drop = FALSE
        ]
    }


    return(result)
  }


  stop(
    paste(
      "Formato SHAP no reconocido:",
      paste(
        dims,
        collapse = " x "
      )
    )
  )
}


shap_array <- normalize_shap(
  raw_shap,
  n_rows = nrow(test),
  n_features = length(features),
  n_classes = length(state_levels)
)


# ============================================================
# 15. DICCIONARIO DE FEATURES
# ============================================================

feature_dictionary <- c(

  minute =
    "match minute",

  game_state_num =
    "score situation",

  current_state_control =
    "current CONTROL state",

  current_state_pressure =
    "current PRESSURE state",

  current_state_transition =
    "current TRANSITION state",

  current_state_protection =
    "current PROTECTION state",

  current_state_urgency =
    "current URGENCY state",

  america_field_tilt_last5 =
    "recent territorial dominance",

  america_field_tilt_last3 =
    "short-term territorial dominance",

  entries_final_third_diff_last5 =
    "final-third entry balance",

  entries_final_third_diff_last3 =
    "recent final-third entry balance",

  entries_box_diff_last5 =
    "box-entry balance",

  entries_box_diff_last3 =
    "recent box-entry balance",

  progressive_pass_diff_last5 =
    "progressive passing balance",

  progressive_pass_diff_last3 =
    "recent progressive passing",

  america_progressive_carries_last5 =
    "progressive carries",

  america_progressive_carries_last3 =
    "recent progressive carries",

  america_progressive_passes_last5 =
    "progressive passes",

  america_progressive_passes_last3 =
    "recent progressive passes",

  pressure_diff_last5 =
    "pressure balance",

  pressure_diff_last3 =
    "recent pressure balance",

  recovery_height_diff_last5 =
    "recovery height",

  recovery_height_diff_last3 =
    "recent recovery height",

  america_local_superiority_10_last5 =
    "local numerical superiority",

  america_local_superiority_10_last3 =
    "recent local numerical superiority",

  america_pressure_share_5_last5 =
    "nearby opponent pressure",

  america_pressure_share_5_last3 =
    "recent nearby opponent pressure",

  america_nearest_opponent_distance_last5 =
    "space from nearest opponent",

  america_nearest_opponent_distance_last3 =
    "recent opponent proximity",

  america_support_share_10_last5 =
    "nearby teammate support",

  america_support_share_10_last3 =
    "recent teammate support"
)


# ============================================================
# 16. EXTRAER TOP 3 RAZONES POSITIVAS
#
# Solo SHAP que EMPUJA hacia la predicción.
# ============================================================

top_reason_1 <- character(
  nrow(test)
)

top_reason_2 <- character(
  nrow(test)
)

top_reason_3 <- character(
  nrow(test)
)


for (
  i in seq_len(
    nrow(test)
  )
) {

  class_index <- top1_index[i]


  shap_values <- as.numeric(
    shap_array[
      i,
      seq_along(features),
      class_index
    ]
  )


  positive_idx <- which(
    shap_values > 0
  )


  if (
    length(positive_idx) == 0
  ) {

    positive_idx <- order(
      abs(shap_values),
      decreasing = TRUE
    )[1:min(
      3,
      length(shap_values)
    )]

  } else {

    positive_idx <- positive_idx[
      order(
        shap_values[
          positive_idx
        ],
        decreasing = TRUE
      )
    ]
  }


  positive_idx <- head(
    positive_idx,
    3
  )


  reason_names <- unname(
    feature_dictionary[
      features[
        positive_idx
      ]
    ]
  )


  raw_names <- features[
    positive_idx
  ]


  reason_names[
    is.na(reason_names)
  ] <- raw_names[
    is.na(reason_names)
  ]


  if (length(reason_names) >= 1) {
    top_reason_1[i] <- reason_names[1]
  }

  if (length(reason_names) >= 2) {
    top_reason_2[i] <- reason_names[2]
  }

  if (length(reason_names) >= 3) {
    top_reason_3[i] <- reason_names[3]
  }
}


# ============================================================
# 17. ALERT LEVEL
#
# No todas las predicciones deben generar la misma alarma.
# ============================================================

alert_level <- fifelse(

  confidence >= 0.60 &
    confidence_margin >= 0.15,

  "HIGH",

  fifelse(

    confidence >= 0.45 &
      confidence_margin >= 0.08,

    "MEDIUM",

    "WATCH"
  )
)


# ============================================================
# 18. ¿CAMBIO DE ESTADO?
# ============================================================

current_state_char <- as.character(
  test$current_tactical_state
)


state_change_expected <-
  predicted_state !=
  current_state_char


# ============================================================
# 19. RECOMENDACIONES BASE
# ============================================================

coach_action <- character(
  length(
    predicted_state
  )
)


coach_action[
  predicted_state ==
    "CONTROL"
] <-
  paste(
    "Maintain possession structure;",
    "circulate before forcing progression;",
    "protect central rest-defense."
  )


coach_action[
  predicted_state ==
    "PRESSURE"
] <-
  paste(
    "Prepare coordinated pressure;",
    "close central outlets;",
    "use backward pass or poor touch as pressing trigger."
  )


coach_action[
  predicted_state ==
    "TRANSITION"
] <-
  paste(
    "Prepare for transition;",
    "secure the first pass after regain;",
    "attack open lanes while preserving rest-defense."
  )


coach_action[
  predicted_state ==
    "PROTECTION"
] <-
  paste(
    "Protect central zones;",
    "compact distances between lines;",
    "reduce unnecessary high-risk turnovers."
  )


coach_action[
  predicted_state ==
    "URGENCY"
] <-
  paste(
    "Increase attacking tempo;",
    "occupy box and half-spaces earlier;",
    "counterpress immediately after loss."
  )


# ============================================================
# 20. AJUSTAR RECOMENDACIÓN POR MARCADOR
#
# game_state_num:
# >0 winning
#  0 drawing
# <0 losing
# ============================================================

for (
  i in seq_len(
    nrow(test)
  )
) {

  score_state <-
    test$game_state_num[i]


  # ----------------------------------------------------------
  # PROTECTION + GANANDO
  # ----------------------------------------------------------

  if (
    predicted_state[i] ==
      "PROTECTION" &&
    score_state > 0
  ) {

    coach_action[i] <- paste(
      "Manage the advantage:",
      "compact central spaces,",
      "keep rest-defense balanced,",
      "and avoid unnecessary central turnovers."
    )
  }


  # ----------------------------------------------------------
  # URGENCY + PERDIENDO
  # ----------------------------------------------------------

  if (
    predicted_state[i] ==
      "URGENCY" &&
    score_state < 0
  ) {

    coach_action[i] <- paste(
      "Increase offensive risk:",
      "accelerate progression,",
      "occupy the box with more players,",
      "and counterpress immediately after loss."
    )
  }


  # ----------------------------------------------------------
  # CONTROL + EMPATE
  # ----------------------------------------------------------

  if (
    predicted_state[i] ==
      "CONTROL" &&
    score_state == 0
  ) {

    coach_action[i] <- paste(
      "Control the next phase:",
      "retain possession,",
      "move the opponent before progressing,",
      "and preserve central balance."
    )
  }
}


# ============================================================
# 21. DECISIÓN OPERATIVA
# ============================================================

operational_decision <- fifelse(

  alert_level == "HIGH" &
    state_change_expected,

  "PREPARE CHANGE NOW",

  fifelse(

    alert_level == "MEDIUM" &
      state_change_expected,

    "PREPARE NEXT PHASE",

    fifelse(

      state_change_expected,

      "MONITOR POSSIBLE CHANGE",

      "MAINTAIN CURRENT PLAN"
    )
  )
)


# ============================================================
# 22. COACH ENGINE TABLE
# ============================================================

coach_engine <- data.table(

  match_id =
    test$match_id,

  match_date =
    test$match_date,

  minute =
    test$minute,

  opponent =
    test$opponent,

  score_state =
    test$game_state_num,

  current_state =
    current_state_char,

  predicted_state =
    predicted_state,

  confidence =
    round(
      confidence,
      4
    ),

  second_state =
    second_state,

  second_probability =
    round(
      second_probability,
      4
    ),

  confidence_margin =
    round(
      confidence_margin,
      4
    ),

  alert_level =
    alert_level,

  expected_change =
    state_change_expected,

  operational_decision =
    operational_decision,

  reason_1 =
    top_reason_1,

  reason_2 =
    top_reason_2,

  reason_3 =
    top_reason_3,

  recommended_action =
    coach_action,

  actual_future_state =
    as.character(
      test$
        future_tactical_state_5min
    )
)


# ============================================================
# 23. EVALUACIÓN DEL ENGINE
#
# actual_future_state se usa SOLO para evaluación histórica.
# En vivo obviamente no existiría.
# ============================================================

coach_engine[
  ,
  prediction_correct :=
    predicted_state ==
    actual_future_state
]


# ============================================================
# 24. PERFORMANCE POR NIVEL DE ALERTA
# ============================================================

alert_performance <- coach_engine[
  ,
  .(
    observations =
      .N,

    average_confidence =
      round(
        mean(
          confidence
        ),
        4
      ),

    accuracy_pct =
      round(
        mean(
          prediction_correct
        ) *
        100,
        2
      ),

    expected_changes =
      sum(
        expected_change
      ),

    change_share_pct =
      round(
        mean(
          expected_change
        ) *
        100,
        2
      )
  ),
  by = alert_level
]


alert_order <- c(
  "HIGH",
  "MEDIUM",
  "WATCH"
)

alert_performance[
  ,
  alert_level :=
    factor(
      alert_level,
      levels = alert_order
    )
]

setorder(
  alert_performance,
  alert_level
)


# ============================================================
# 25. RESUMEN DE DECISIONES
# ============================================================

decision_summary <- coach_engine[
  ,
  .(
    observations =
      .N,

    percentage =
      round(
        .N /
        nrow(coach_engine) *
        100,
        2
      )
  ),
  by = operational_decision
]

setorder(
  decision_summary,
  -observations
)


# ============================================================
# 26. ALERTAS ACCIONABLES
# ============================================================

actionable_alerts <- coach_engine[
  expected_change == TRUE &
    alert_level %in%
      c(
        "HIGH",
        "MEDIUM"
      )
]


setorder(
  actionable_alerts,
  -confidence
)


# ============================================================
# 27. EJEMPLOS PARA PRESENTACIÓN
#
# Un caso fuerte por estado.
# ============================================================

coach_engine_demo_cases <- coach_engine[
  prediction_correct == TRUE
][
  order(
    -confidence
  ),
  .SD[1],
  by = predicted_state
]


# ============================================================
# 28. VERSIÓN COMPACTA PARA STAFF
# ============================================================

coach_engine_demo_view <- coach_engine_demo_cases[
  ,
  .(
    minute,
    opponent,

    current_state,

    forecast =
      predicted_state,

    confidence_pct =
      round(
        confidence *
        100,
        1
      ),

    alternative =
      second_state,

    alert =
      alert_level,

    decision =
      operational_decision,

    why_1 =
      reason_1,

    why_2 =
      reason_2,

    action =
      recommended_action
  )
]


# ============================================================
# 29. GUARDAR
# ============================================================

dir.create(
  "outputs/coach_engine",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  coach_engine,
  "outputs/coach_engine/coach_engine_all_test.csv",
  bom = TRUE
)


fwrite(
  actionable_alerts,
  "outputs/coach_engine/coach_engine_actionable_alerts.csv",
  bom = TRUE
)


fwrite(
  coach_engine_demo_cases,
  "outputs/coach_engine/coach_engine_demo_cases.csv",
  bom = TRUE
)


fwrite(
  alert_performance,
  "outputs/coach_engine/coach_engine_alert_performance.csv",
  bom = TRUE
)


fwrite(
  decision_summary,
  "outputs/coach_engine/coach_engine_decision_summary.csv",
  bom = TRUE
)


# ============================================================
# 30. SOLO 3 VISUALIZACIONES
# ============================================================

View(
  coach_engine_demo_view
)

View(
  alert_performance
)

View(
  decision_summary
)


# ============================================================
# 31. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("COACH ENGINE TERMINADO\n")
cat("=====================================\n")

cat(
  "Total predictions:",
  nrow(
    coach_engine
  ),
  "\n"
)

cat(
  "Actionable alerts:",
  nrow(
    actionable_alerts
  ),
  "\n"
)

cat(
  "High-confidence alerts:",
  coach_engine[
    alert_level == "HIGH",
    .N
  ],
  "\n"
)

cat(
  "Predicted tactical changes:",
  coach_engine[
    expected_change == TRUE,
    .N
  ],
  "\n"
)

cat(
  "Demo states represented:",
  uniqueN(
    coach_engine_demo_cases$
      predicted_state
  ),
  "\n"
)