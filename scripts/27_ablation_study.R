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

dt[
  ,
  target := factor(
    future_tactical_state_5min,
    levels = state_levels
  )
]

dt[
  ,
  current_tactical_state := factor(
    current_tactical_state,
    levels = state_levels
  )
]


# ============================================================
# 2. ONE-HOT DEL ESTADO ACTUAL
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
# 3. GRUPO A — CONTEXTO
# ============================================================

features_context <- c(

  "minute",
  "game_state_num",

  state_features
)


# ============================================================
# 4. GRUPO B — CONTEXTO + EVENTS
# ============================================================

features_events <- c(

  features_context,

  # 5 MIN
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  "progressive_pass_diff_last5",

  "america_progressive_carries_last5",
  "america_progressive_passes_last5",

  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # 3 MIN
  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",

  "progressive_pass_diff_last3",

  "america_progressive_carries_last3",
  "america_progressive_passes_last3",

  "pressure_diff_last3",
  "recovery_height_diff_last3"
)


# ============================================================
# 5. GRUPO C — CONTEXTO + EVENTS + 360
# ============================================================

features_360 <- c(

  features_events,

  # 5 MIN 360
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 3 MIN 360
  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3"
)


# ============================================================
# 6. CONTEXTO RECIENTE DEL RIVAL
# ============================================================

opponent_recent_features <- c(

  "opponent_points_last1",
  "opponent_points_last3",
  "opponent_points_last5",
  "opponent_points_weighted"
)


# ============================================================
# 7. HISTÓRICO DEL RIVAL
# ============================================================

previous_season_features <- grep(
  "^opponent_previous_team_season_",
  names(dt),
  value = TRUE
)

if (length(previous_season_features) > 0) {

  numeric_previous <- sapply(
    dt[
      ,
      ..previous_season_features
    ],
    is.numeric
  )

  previous_season_features <-
    previous_season_features[
      numeric_previous
    ]
}


# ============================================================
# 8. FILTRAR VARIABLES HISTÓRICAS
#
# SOLO CON TRAIN
# ============================================================

historical_keep <- character()

for (col in previous_season_features) {

  values <- dt[
    dataset_split == "TRAIN",
    get(col)
  ]

  coverage <- mean(
    !is.na(values)
  )

  variance <- var(
    values,
    na.rm = TRUE
  )

  if (
    is.finite(coverage) &&
    coverage >= 0.50 &&
    is.finite(variance) &&
    variance > 0
  ) {

    historical_keep <- c(
      historical_keep,
      col
    )
  }
}


# ============================================================
# 9. GRUPO D — FULL
# ============================================================

features_full <- unique(
  c(
    features_360,
    opponent_recent_features,
    historical_keep
  )
)


# ============================================================
# 10. DEFINIR EXPERIMENTOS
# ============================================================

feature_sets <- list(

  CONTEXT_ONLY =
    features_context,

  CONTEXT_EVENTS =
    features_events,

  CONTEXT_EVENTS_360 =
    features_360,

  FULL =
    features_full
)


# ============================================================
# 11. LOG LOSS
# ============================================================

multiclass_logloss <- function(
  truth,
  probabilities
) {

  truth <- factor(
    truth,
    levels = state_levels
  )

  truth_index <- as.integer(
    truth
  )

  p <- probabilities[
    cbind(
      seq_along(truth_index),
      truth_index
    )
  ]

  p <- pmax(
    pmin(
      p,
      1 - 1e-15
    ),
    1e-15
  )

  -mean(
    log(p)
  )
}


# ============================================================
# 12. MACRO F1
# ============================================================

macro_f1 <- function(
  truth,
  prediction
) {

  values <- sapply(
    state_levels,
    function(cl) {

      tp <- sum(
        truth == cl &
        prediction == cl
      )

      fp <- sum(
        truth != cl &
        prediction == cl
      )

      fn <- sum(
        truth == cl &
        prediction != cl
      )

      precision <- ifelse(
        tp + fp == 0,
        0,
        tp / (tp + fp)
      )

      recall <- ifelse(
        tp + fn == 0,
        0,
        tp / (tp + fn)
      )

      if (
        precision + recall == 0
      ) {
        return(0)
      }

      2 *
        precision *
        recall /
        (
          precision +
            recall
        )
    }
  )

  mean(values)
}


# ============================================================
# 13. BALANCED ACCURACY
# ============================================================

balanced_accuracy <- function(
  truth,
  prediction
) {

  recalls <- sapply(
    state_levels,
    function(cl) {

      idx <- truth == cl

      if (
        sum(idx) == 0
      ) {
        return(NA_real_)
      }

      mean(
        prediction[idx] == cl
      )
    }
  )

  mean(
    recalls,
    na.rm = TRUE
  )
}


# ============================================================
# 14. FUNCIÓN PARA CONVERTIR PROBABILIDADES
# ============================================================

format_probabilities <- function(
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
      paste(
        "Dimensiones inesperadas:",
        nrow(probs),
        "x",
        ncol(probs)
      )
    )
  }

  colnames(probs) <- state_levels

  probs
}


# ============================================================
# 15. RESULTADOS
# ============================================================

ablation_results_list <- list()

experiment_number <- 1


# ============================================================
# 16. ENTRENAR CADA EXPERIMENTO
# ============================================================

for (
  experiment_name
  in names(feature_sets)
) {

  cat(
    "\n=====================================\n"
  )

  cat(
    "Entrenando:",
    experiment_name,
    "\n"
  )

  cat(
    "=====================================\n"
  )

  features <- feature_sets[[
    experiment_name
  ]]


  # ----------------------------------------------------------
  # VALIDACIÓN DE VARIABLES
  # ----------------------------------------------------------

  missing_features <- setdiff(
    features,
    names(dt)
  )

  if (
    length(missing_features) > 0
  ) {

    stop(
      paste(
        "Faltan variables en",
        experiment_name,
        ":",
        paste(
          missing_features,
          collapse = ", "
        )
      )
    )
  }


  # ----------------------------------------------------------
  # COPIA
  # ----------------------------------------------------------

  experiment_dt <- copy(
    dt
  )


  # ----------------------------------------------------------
  # IMPUTACIÓN SOLO CON TRAIN
  # ----------------------------------------------------------

  for (col in features) {

    med <- median(
      experiment_dt[
        dataset_split == "TRAIN",
        get(col)
      ],
      na.rm = TRUE
    )

    if (!is.finite(med)) {
      med <- 0
    }

    experiment_dt[
      is.na(get(col)),
      (col) := med
    ]
  }


  # ----------------------------------------------------------
  # SPLITS
  # ----------------------------------------------------------

  train <- experiment_dt[
    dataset_split == "TRAIN"
  ]

  validation <- experiment_dt[
    dataset_split == "VALIDATION"
  ]

  test <- experiment_dt[
    dataset_split == "TEST"
  ]


  # ----------------------------------------------------------
  # MATRICES
  # ----------------------------------------------------------

  x_train <- as.matrix(
    train[
      ,
      ..features
    ]
  )

  x_validation <- as.matrix(
    validation[
      ,
      ..features
    ]
  )

  x_test <- as.matrix(
    test[
      ,
      ..features
    ]
  )


  # ----------------------------------------------------------
  # TARGET
  # ----------------------------------------------------------

  y_train <- as.integer(
    train$target
  ) - 1

  y_validation <- as.integer(
    validation$target
  ) - 1

  y_test <- as.integer(
    test$target
  ) - 1


  # ----------------------------------------------------------
  # DMATRIX
  # ----------------------------------------------------------

  dtrain <- xgb.DMatrix(
    x_train,
    label = y_train
  )

  dvalidation <- xgb.DMatrix(
    x_validation,
    label = y_validation
  )

  dtest <- xgb.DMatrix(
    x_test,
    label = y_test
  )


  # ----------------------------------------------------------
  # PARÁMETROS FIJOS
  #
  # MISMA CONFIGURACIÓN PARA TODOS.
  # ----------------------------------------------------------

  params <- list(

    objective =
      "multi:softprob",

    num_class =
      length(state_levels),

    eval_metric =
      "mlogloss",

    max_depth =
      3,

    eta =
      0.03,

    min_child_weight =
      8,

    subsample =
      0.8,

    colsample_bytree =
      0.7,

    lambda =
      1,

    alpha =
      0,

    nthread =
      max(
        1,
        parallel::detectCores() - 1
      )
  )


  # ----------------------------------------------------------
  # ENTRENAR
  # ----------------------------------------------------------

  set.seed(
    3000 +
      experiment_number
  )

  model <- xgb.train(

    params = params,

    data = dtrain,

    nrounds = 1000,

    watchlist = list(
      train = dtrain,
      validation = dvalidation
    ),

    early_stopping_rounds =
      40,

    maximize =
      FALSE,

    verbose =
      0
  )


  # ----------------------------------------------------------
  # PROBABILIDADES VALIDATION
  # ----------------------------------------------------------

  raw_validation <- predict(
    model,
    dvalidation
  )

  prob_validation <- format_probabilities(
    raw_validation,
    nrow(validation)
  )


  # ----------------------------------------------------------
  # PROBABILIDADES TEST
  # ----------------------------------------------------------

  raw_test <- predict(
    model,
    dtest
  )

  prob_test <- format_probabilities(
    raw_test,
    nrow(test)
  )


  # ----------------------------------------------------------
  # VALIDAR PROBABILIDADES
  # ----------------------------------------------------------

  if (
    any(
      abs(
        rowSums(
          prob_validation
        ) - 1
      ) > 0.001
    )
  ) {

    stop(
      paste(
        "Probabilidades inválidas en",
        experiment_name,
        "VALIDATION"
      )
    )
  }


  if (
    any(
      abs(
        rowSums(
          prob_test
        ) - 1
      ) > 0.001
    )
  ) {

    stop(
      paste(
        "Probabilidades inválidas en",
        experiment_name,
        "TEST"
      )
    )
  }


  # ----------------------------------------------------------
  # CLASE PREDICHA
  # ----------------------------------------------------------

  pred_validation <- state_levels[
    max.col(
      prob_validation,
      ties.method = "first"
    )
  ]

  pred_test <- state_levels[
    max.col(
      prob_test,
      ties.method = "first"
    )
  ]


  truth_validation <- as.character(
    validation$target
  )

  truth_test <- as.character(
    test$target
  )


  # ----------------------------------------------------------
  # MÉTRICAS
  # ----------------------------------------------------------

  result <- data.table(

    experiment =
      experiment_name,

    number_features =
      length(features),

    best_iteration =
      model$best_iteration,

    validation_accuracy =
      round(
        mean(
          pred_validation ==
            truth_validation
        ) * 100,
        2
      ),

    validation_balanced_accuracy =
      round(
        balanced_accuracy(
          truth_validation,
          pred_validation
        ) * 100,
        2
      ),

    validation_macro_f1 =
      round(
        macro_f1(
          truth_validation,
          pred_validation
        ),
        4
      ),

    validation_logloss =
      round(
        multiclass_logloss(
          truth_validation,
          prob_validation
        ),
        4
      ),

    test_accuracy =
      round(
        mean(
          pred_test ==
            truth_test
        ) * 100,
        2
      ),

    test_balanced_accuracy =
      round(
        balanced_accuracy(
          truth_test,
          pred_test
        ) * 100,
        2
      ),

    test_macro_f1 =
      round(
        macro_f1(
          truth_test,
          pred_test
        ),
        4
      ),

    test_logloss =
      round(
        multiclass_logloss(
          truth_test,
          prob_test
        ),
        4
      )
  )


  ablation_results_list[[
    experiment_number
  ]] <- result


  experiment_number <-
    experiment_number + 1
}


# ============================================================
# 17. UNIR RESULTADOS
# ============================================================

ablation_results <- rbindlist(
  ablation_results_list
)


# ============================================================
# 18. COMPARAR CONTRA CONTEXT ONLY
# ============================================================

context_accuracy <- ablation_results[
  experiment == "CONTEXT_ONLY",
  test_accuracy
]

context_balanced <- ablation_results[
  experiment == "CONTEXT_ONLY",
  test_balanced_accuracy
]

context_f1 <- ablation_results[
  experiment == "CONTEXT_ONLY",
  test_macro_f1
]

context_logloss <- ablation_results[
  experiment == "CONTEXT_ONLY",
  test_logloss
]


ablation_results[
  ,
  accuracy_gain_vs_context :=
    round(
      test_accuracy -
        context_accuracy,
      2
    )
]


ablation_results[
  ,
  balanced_accuracy_gain_vs_context :=
    round(
      test_balanced_accuracy -
        context_balanced,
      2
    )
]


ablation_results[
  ,
  macro_f1_gain_vs_context :=
    round(
      test_macro_f1 -
        context_f1,
      4
    )
]


ablation_results[
  ,
  logloss_improvement_vs_context :=
    round(
      context_logloss -
        test_logloss,
      4
    )
]


# ============================================================
# 19. GANANCIA INCREMENTAL
# ============================================================

experiment_order <- c(
  "CONTEXT_ONLY",
  "CONTEXT_EVENTS",
  "CONTEXT_EVENTS_360",
  "FULL"
)

ablation_results[
  ,
  experiment := factor(
    experiment,
    levels = experiment_order
  )
]

setorder(
  ablation_results,
  experiment
)


ablation_results[
  ,
  incremental_accuracy_gain :=
    round(
      test_accuracy -
        shift(
          test_accuracy
        ),
      2
    )
]


ablation_results[
  ,
  incremental_logloss_improvement :=
    round(
      shift(
        test_logloss
      ) -
        test_logloss,
      4
    )
]


# ============================================================
# 20. GUARDAR RESULTADOS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)

fwrite(
  ablation_results,
  "outputs/models/xgboost_ablation_study.csv",
  bom = TRUE
)


# ============================================================
# 21. UNA SOLA VISUALIZACIÓN
# ============================================================

View(
  ablation_results
)


# ============================================================
# 22. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("ABLATION STUDY TERMINADO\n")
cat("=====================================\n")

cat(
  "Experimentos:",
  nrow(ablation_results),
  "\n"
)

cat(
  "Context accuracy:",
  ablation_results[
    experiment == "CONTEXT_ONLY",
    test_accuracy
  ],
  "%\n"
)

cat(
  "Events accuracy:",
  ablation_results[
    experiment == "CONTEXT_EVENTS",
    test_accuracy
  ],
  "%\n"
)

cat(
  "Events + 360 accuracy:",
  ablation_results[
    experiment == "CONTEXT_EVENTS_360",
    test_accuracy
  ],
  "%\n"
)

cat(
  "Full accuracy:",
  ablation_results[
    experiment == "FULL",
    test_accuracy
  ],
  "%\n"
)