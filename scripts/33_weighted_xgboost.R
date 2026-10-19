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
# 2. ONE-HOT ESTADO ACTUAL
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
# 3. FEATURES
# ============================================================

features <- c(

  # Contexto
  "minute",
  "game_state_num",

  state_features,

  # Events 5 min
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # Events 3 min
  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_pass_diff_last3",
  "america_progressive_carries_last3",
  "america_progressive_passes_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  # 360 5 min
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 360 3 min
  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3"
)


# ============================================================
# 4. VALIDAR FEATURES
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
# 5. IMPUTACIÓN SOLO CON TRAIN
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
# 6. SPLITS
# ============================================================

train <- dt[
  dataset_split == "TRAIN"
]

validation <- dt[
  dataset_split == "VALIDATION"
]

test <- dt[
  dataset_split == "TEST"
]


# ============================================================
# 7. MATRICES
# ============================================================

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


y_train <- as.integer(
  train$target
) - 1

y_validation <- as.integer(
  validation$target
) - 1

y_test <- as.integer(
  test$target
) - 1


# ============================================================
# 8. MÉTRICAS
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

  colnames(probs) <- state_levels

  probs
}


multiclass_logloss <- function(
  truth,
  probabilities
) {

  truth <- factor(
    truth,
    levels = state_levels
  )

  index <- as.integer(
    truth
  )

  p <- probabilities[
    cbind(
      seq_along(index),
      index
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


top2_accuracy <- function(
  truth,
  probabilities
) {

  correct <- logical(
    length(truth)
  )

  for (
    i in seq_along(truth)
  ) {

    top2 <- order(
      probabilities[i, ],
      decreasing = TRUE
    )[1:2]

    correct[i] <-
      truth[i] %in%
      state_levels[top2]
  }

  mean(correct)
}


# ============================================================
# 9. CONFIGURACIONES DE PESOS
#
# Seleccionamos SOLO usando VALIDATION.
# ============================================================

weight_schemes <- list(

  UNWEIGHTED = c(
    CONTROL = 1.00,
    PRESSURE = 1.00,
    TRANSITION = 1.00,
    PROTECTION = 1.00,
    URGENCY = 1.00
  ),

  MILD = c(
    CONTROL = 1.00,
    PRESSURE = 1.20,
    TRANSITION = 1.10,
    PROTECTION = 0.90,
    URGENCY = 0.90
  ),

  MEDIUM = c(
    CONTROL = 1.00,
    PRESSURE = 1.50,
    TRANSITION = 1.25,
    PROTECTION = 0.85,
    URGENCY = 0.85
  ),

  STRONG = c(
    CONTROL = 1.00,
    PRESSURE = 1.80,
    TRANSITION = 1.40,
    PROTECTION = 0.80,
    URGENCY = 0.80
  ),

  BALANCED = {
    counts <- table(
      train$target
    )

    inverse <- 1 / as.numeric(
      counts
    )

    inverse <-
      inverse /
      mean(inverse)

    setNames(
      inverse,
      names(counts)
    )
  }
)


# ============================================================
# 10. HIPERPARÁMETROS
# ============================================================

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


# ============================================================
# 11. ENTRENAR PESOS
# ============================================================

weight_results_list <- list()
weight_models <- list()

counter <- 1


for (
  scheme_name in names(weight_schemes)
) {

  cat(
    "\nEntrenando:",
    scheme_name,
    "\n"
  )


  # IMPORTANTE:
  # los dobles corchetes deben estar juntos
  scheme <- weight_schemes[[scheme_name]]


  train_states <- as.character(
    train$target
  )


  train_weights <- as.numeric(
    scheme[
      train_states
    ]
  )


  dtrain <- xgb.DMatrix(
    x_train,
    label = y_train,
    weight = train_weights
  )


  dvalidation <- xgb.DMatrix(
    x_validation,
    label = y_validation
  )


  set.seed(
    7000 + counter
  )


  model <- xgb.train(

    params = params,

    data = dtrain,

    nrounds = 1200,

    evals = list(
      train = dtrain,
      validation = dvalidation
    ),

    early_stopping_rounds = 50,

    maximize = FALSE,

    verbose = 0
  )


  probs_validation <- format_probs(
    predict(
      model,
      dvalidation
    ),
    nrow(validation)
  )


  pred_validation <- state_levels[
    max.col(
      probs_validation,
      ties.method = "first"
    )
  ]


  truth_validation <- as.character(
    validation$target
  )


  # ==========================================================
  # RECALL PRESSURE
  # ==========================================================

  pressure_idx <- truth_validation == "PRESSURE"

  if (
    sum(pressure_idx) > 0
  ) {

    recall_pressure <- mean(
      pred_validation[
        pressure_idx
      ] == "PRESSURE"
    )

  } else {

    recall_pressure <- NA_real_
  }


  # ==========================================================
  # RECALL TRANSITION
  # ==========================================================

  transition_idx <- truth_validation == "TRANSITION"

  if (
    sum(transition_idx) > 0
  ) {

    recall_transition <- mean(
      pred_validation[
        transition_idx
      ] == "TRANSITION"
    )

  } else {

    recall_transition <- NA_real_
  }


  # ==========================================================
  # GUARDAR RESULTADOS
  # ==========================================================

  weight_results_list[[counter]] <- data.table(

    scheme =
      scheme_name,

    accuracy =
      mean(
        pred_validation ==
          truth_validation
      ),

    balanced_accuracy =
      balanced_accuracy(
        truth_validation,
        pred_validation
      ),

    macro_f1 =
      macro_f1(
        truth_validation,
        pred_validation
      ),

    top2_accuracy =
      top2_accuracy(
        truth_validation,
        probs_validation
      ),

    log_loss =
      multiclass_logloss(
        truth_validation,
        probs_validation
      ),

    pressure_recall =
      recall_pressure,

    transition_recall =
      recall_transition
  )


  weight_models[[counter]] <-
    model


  counter <- counter + 1
}

# ============================================================
# 12. RESULTADOS VALIDATION
# ============================================================

weight_validation_results <- rbindlist(
  weight_results_list
)


# ============================================================
# 13. SCORE DE SELECCIÓN
#
# Elegimos con VALIDATION, NO con TEST.
# ============================================================

weight_validation_results[
  ,
  selection_score :=
    0.50 * accuracy +
    0.30 * balanced_accuracy +
    0.20 * macro_f1
]


setorder(
  weight_validation_results,
  -selection_score,
  log_loss
)


# ============================================================
# 14. MEJOR CONFIGURACIÓN
# ============================================================

best_scheme <- weight_validation_results[
  1,
  scheme
]


best_model_index <- match(
  best_scheme,
  names(weight_schemes)
)


if (
  is.na(best_model_index)
) {
  stop(
    "No pude identificar el índice del mejor modelo."
  )
}


best_model <- weight_models[[best_model_index]]


cat(
  "\nMejor esquema:",
  best_scheme,
  "\n"
)


# ============================================================
# 15. TEST
# ============================================================

dtest <- xgb.DMatrix(
  x_test,
  label = y_test
)


prob_test <- format_probs(
  predict(
    best_model,
    dtest
  ),
  nrow(test)
)


# ============================================================
# 16. VALIDAR PROBABILIDADES
# ============================================================

if (
  any(
    abs(
      rowSums(prob_test) - 1
    ) > 0.001
  )
) {

  stop(
    "Las probabilidades TEST no suman 1."
  )
}


# ============================================================
# 17. PREDICCIÓN TEST
# ============================================================

pred_test <- state_levels[
  max.col(
    prob_test,
    ties.method = "first"
  )
]


truth_test <- as.character(
  test$target
)


# ============================================================
# 18. MÉTRICAS TEST
# ============================================================

weighted_test_metrics <- data.table(

  model =
    paste0(
      "Weighted XGBoost - ",
      best_scheme
    ),

  accuracy =
    round(
      mean(
        pred_test ==
          truth_test
      ) * 100,
      2
    ),

  balanced_accuracy =
    round(
      balanced_accuracy(
        truth_test,
        pred_test
      ) * 100,
      2
    ),

  macro_f1 =
    round(
      macro_f1(
        truth_test,
        pred_test
      ),
      4
    ),

  top2_accuracy =
    round(
      top2_accuracy(
        truth_test,
        prob_test
      ) * 100,
      2
    ),

  log_loss =
    round(
      multiclass_logloss(
        truth_test,
        prob_test
      ),
      4
    )
)


# ============================================================
# 19. MÉTRICAS POR ESTADO
# ============================================================

weighted_by_state <- rbindlist(

  lapply(
    state_levels,
    function(cl) {

      idx <- truth_test == cl


      if (
        sum(idx) == 0
      ) {

        return(
          data.table(
            state = cl,
            observations = 0,
            top1_accuracy = NA_real_,
            top2_accuracy = NA_real_
          )
        )
      }


      probs_state <- prob_test[
        idx,
        ,
        drop = FALSE
      ]


      truth_state <- truth_test[
        idx
      ]


      top1 <- mean(
        pred_test[idx] ==
          cl
      )


      top2 <- top2_accuracy(
        truth_state,
        probs_state
      )


      data.table(

        state =
          cl,

        observations =
          sum(idx),

        top1_accuracy =
          round(
            top1 * 100,
            2
          ),

        top2_accuracy =
          round(
            top2 * 100,
            2
          )
      )
    }
  )
)


# ============================================================
# 20. COMPARACIÓN CON MODELO ACTUAL
# ============================================================

final_comparison <- data.table(

  model = c(
    "Current XGBoost",
    weighted_test_metrics$model
  ),

  top1_accuracy = c(
    40.10,
    weighted_test_metrics$accuracy
  ),

  balanced_accuracy = c(
    45.14,
    weighted_test_metrics$balanced_accuracy
  ),

  macro_f1 = c(
    0.3872,
    weighted_test_metrics$macro_f1
  ),

  top2_accuracy = c(
    64.45,
    weighted_test_metrics$top2_accuracy
  ),

  log_loss = c(
    1.3428,
    weighted_test_metrics$log_loss
  )
)


# ============================================================
# 21. GUARDAR RESULTADOS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  weight_validation_results,
  "outputs/models/weighted_xgboost_validation.csv",
  bom = TRUE
)


fwrite(
  weighted_test_metrics,
  "outputs/models/weighted_xgboost_test.csv",
  bom = TRUE
)


fwrite(
  weighted_by_state,
  "outputs/models/weighted_xgboost_by_state.csv",
  bom = TRUE
)


fwrite(
  final_comparison,
  "outputs/models/weighted_xgboost_comparison.csv",
  bom = TRUE
)


xgb.save(
  best_model,
  "outputs/models/weighted_xgboost_model.json"
)


# ============================================================
# 22. SOLO LAS 3 TABLAS IMPORTANTES
# ============================================================

View(
  weight_validation_results
)

View(
  final_comparison
)

View(
  weighted_by_state
)


# ============================================================
# 23. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("WEIGHTED XGBOOST TERMINADO\n")
cat("=====================================\n")

cat(
  "Best scheme:",
  best_scheme,
  "\n"
)

cat(
  "Test Top-1:",
  weighted_test_metrics$accuracy,
  "%\n"
)

cat(
  "Test Top-2:",
  weighted_test_metrics$top2_accuracy,
  "%\n"
)

cat(
  "Balanced accuracy:",
  weighted_test_metrics$balanced_accuracy,
  "%\n"
)

cat(
  "Macro F1:",
  weighted_test_metrics$macro_f1,
  "\n"
)

cat(
  "Log-loss:",
  weighted_test_metrics$log_loss,
  "\n"
)