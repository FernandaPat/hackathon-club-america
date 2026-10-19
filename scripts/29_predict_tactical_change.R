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

cat("\n=====================================\n")
cat("TACTICAL CHANGE MODEL\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. TARGET BINARIO
#
# 0 = mantiene estado
# 1 = cambia estado en próximos 5 min
# ============================================================

dt[
  ,
  change_target :=
    as.integer(
      current_tactical_state !=
        future_tactical_state_5min
    )
]


# ============================================================
# 3. ESTADOS
# ============================================================

state_levels <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

dt[
  ,
  current_tactical_state :=
    factor(
      current_tactical_state,
      levels = state_levels
    )
]


# ============================================================
# 4. ONE-HOT ESTADO ACTUAL
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
# 5. FEATURES
#
# Usamos CONTEXT + EVENTS + 360,
# arquitectura elegida en VALIDATION.
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
# 6. VALIDAR FEATURES
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
# 7. IMPUTACIÓN SOLO CON TRAIN
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
# 8. SPLITS
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
# 9. DISTRIBUCIÓN DEL TARGET
# ============================================================

change_distribution <- dt[
  ,
  .N,
  by = .(
    dataset_split,
    change_target
  )
]

change_distribution[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    ),
  by = dataset_split
]


# ============================================================
# 10. MATRICES
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


y_train <- train$change_target
y_validation <- validation$change_target
y_test <- test$change_target


# ============================================================
# 11. DMATRIX
# ============================================================

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


# ============================================================
# 12. GRID DE HIPERPARÁMETROS
# ============================================================

parameter_grid <- CJ(

  max_depth = c(
    3L,
    5L
  ),

  eta = c(
    0.03,
    0.07
  ),

  min_child_weight = c(
    3,
    8
  )
)


# ============================================================
# 13. TUNING
# ============================================================

tuning_results <- list()
models <- list()

for (i in seq_len(nrow(parameter_grid))) {

  p <- parameter_grid[i]

  cat(
    "Entrenando change model",
    i,
    "de",
    nrow(parameter_grid),
    "\n"
  )

  params <- list(

    objective =
      "binary:logistic",

    eval_metric =
      "logloss",

    max_depth =
      p$max_depth,

    eta =
      p$eta,

    min_child_weight =
      p$min_child_weight,

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


  set.seed(
    4000 + i
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


  tuning_results[[i]] <- data.table(

    model_id =
      i,

    max_depth =
      p$max_depth,

    eta =
      p$eta,

    min_child_weight =
      p$min_child_weight,

    best_iteration =
      model$best_iteration,

    validation_logloss =
      model$best_score
  )


  models[[i]] <- model
}


# ============================================================
# 14. MEJOR MODELO
# ============================================================

tuning_results <- rbindlist(
  tuning_results
)

setorder(
  tuning_results,
  validation_logloss
)

best_model_id <- tuning_results[
  1,
  model_id
]

best_model <- models[[
  best_model_id
]]


# ============================================================
# 15. PROBABILIDADES
# ============================================================

prob_validation <- as.numeric(
  predict(
    best_model,
    dvalidation
  )
)

prob_test <- as.numeric(
  predict(
    best_model,
    dtest
  )
)


# ============================================================
# 16. ELEGIR THRESHOLD EN VALIDATION
#
# No usamos TEST para escoger threshold.
# ============================================================

threshold_candidates <- seq(
  0.20,
  0.80,
  by = 0.01
)


threshold_results <- rbindlist(
  lapply(
    threshold_candidates,
    function(threshold) {

      pred <- as.integer(
        prob_validation >=
          threshold
      )


      tp <- sum(
        y_validation == 1 &
        pred == 1
      )

      fp <- sum(
        y_validation == 0 &
        pred == 1
      )

      fn <- sum(
        y_validation == 1 &
        pred == 0
      )

      tn <- sum(
        y_validation == 0 &
        pred == 0
      )


      sensitivity <- ifelse(
        tp + fn == 0,
        0,
        tp / (tp + fn)
      )


      specificity <- ifelse(
        tn + fp == 0,
        0,
        tn / (tn + fp)
      )


      balanced_accuracy <-
        (
          sensitivity +
          specificity
        ) / 2


      precision <- ifelse(
        tp + fp == 0,
        0,
        tp / (tp + fp)
      )


      f1 <- ifelse(
        precision + sensitivity == 0,
        0,
        2 *
          precision *
          sensitivity /
          (
            precision +
            sensitivity
          )
      )


      data.table(

        threshold =
          threshold,

        balanced_accuracy =
          balanced_accuracy,

        f1 =
          f1
      )
    }
  )
)


best_threshold <- threshold_results[
  which.max(
    balanced_accuracy
  ),
  threshold
]


# ============================================================
# 17. TEST PREDICTIONS
# ============================================================

pred_test <- as.integer(
  prob_test >=
    best_threshold
)


# ============================================================
# 18. MÉTRICAS BINARIAS
# ============================================================

tp <- sum(
  y_test == 1 &
  pred_test == 1
)

fp <- sum(
  y_test == 0 &
  pred_test == 1
)

fn <- sum(
  y_test == 1 &
  pred_test == 0
)

tn <- sum(
  y_test == 0 &
  pred_test == 0
)


test_accuracy <-
  (
    tp + tn
  ) /
  length(y_test)


test_sensitivity <- ifelse(
  tp + fn == 0,
  0,
  tp / (tp + fn)
)


test_specificity <- ifelse(
  tn + fp == 0,
  0,
  tn / (tn + fp)
)


test_balanced_accuracy <-
  (
    test_sensitivity +
    test_specificity
  ) / 2


test_precision <- ifelse(
  tp + fp == 0,
  0,
  tp / (tp + fp)
)


test_f1 <- ifelse(
  test_precision +
    test_sensitivity == 0,
  0,
  2 *
    test_precision *
    test_sensitivity /
    (
      test_precision +
      test_sensitivity
    )
)


# ============================================================
# 19. LOG LOSS
# ============================================================

binary_logloss <- function(
  truth,
  prob
) {

  prob <- pmax(
    pmin(
      prob,
      1 - 1e-15
    ),
    1e-15
  )

  -mean(
    truth *
      log(prob) +
      (
        1 - truth
      ) *
      log(
        1 - prob
      )
  )
}


test_logloss <- binary_logloss(
  y_test,
  prob_test
)


# ============================================================
# 20. BRIER SCORE
# ============================================================

test_brier <- mean(
  (
    prob_test -
      y_test
  )^2
)


# ============================================================
# 21. BASELINE
#
# Clase mayoritaria de TRAIN.
# ============================================================

majority_change <- as.integer(
  mean(
    y_train
  ) >= 0.5
)


baseline_prediction <- rep(
  majority_change,
  length(y_test)
)


baseline_accuracy <- mean(
  baseline_prediction ==
    y_test
)


# ============================================================
# 22. RESUMEN MÉTRICAS
# ============================================================

change_model_metrics <- data.table(

  metric = c(

    "Test accuracy",

    "Test balanced accuracy",

    "Test precision - change",

    "Test recall - change",

    "Test F1 - change",

    "Test log-loss",

    "Test Brier",

    "Majority baseline accuracy",

    "Selected threshold"
  ),

  value = c(

    test_accuracy,

    test_balanced_accuracy,

    test_precision,

    test_sensitivity,

    test_f1,

    test_logloss,

    test_brier,

    baseline_accuracy,

    best_threshold
  )
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

change_confusion <- data.table(

  truth = c(
    "NO_CHANGE",
    "NO_CHANGE",
    "CHANGE",
    "CHANGE"
  ),

  prediction = c(
    "NO_CHANGE",
    "CHANGE",
    "NO_CHANGE",
    "CHANGE"
  ),

  N = c(
    tn,
    fp,
    fn,
    tp
  )
)


# ============================================================
# 24. FEATURE IMPORTANCE
# ============================================================

change_importance <- xgb.importance(
  model =
    best_model
)


# ============================================================
# 25. GUARDAR PREDICCIONES
# ============================================================

change_predictions <- test[
  ,
  .(

    match_id,

    match_date,

    minute,

    opponent,

    current_tactical_state,

    future_tactical_state_5min,

    change_target
  )
]


change_predictions[
  ,
  probability_change :=
    prob_test
]


change_predictions[
  ,
  predicted_change :=
    pred_test
]


# ============================================================
# 26. GUARDAR OUTPUTS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)


xgb.save(
  best_model,
  "outputs/models/xgboost_tactical_change_model.json"
)


fwrite(
  change_model_metrics,
  "outputs/models/tactical_change_model_metrics.csv",
  bom = TRUE
)


fwrite(
  change_confusion,
  "outputs/models/tactical_change_confusion.csv",
  bom = TRUE
)


fwrite(
  change_importance,
  "outputs/models/tactical_change_feature_importance.csv",
  bom = TRUE
)


fwrite(
  change_predictions,
  "outputs/models/tactical_change_test_predictions.csv",
  bom = TRUE
)


# ============================================================
# 27. SOLO 3 VISTAS
# ============================================================

View(
  change_distribution
)

View(
  change_model_metrics
)

View(
  change_importance[
    1:min(
      15,
      .N
    )
  ]
)


# ============================================================
# 28. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("TACTICAL CHANGE MODEL TERMINADO\n")
cat("=====================================\n")

cat(
  "Threshold:",
  best_threshold,
  "\n"
)

cat(
  "Test accuracy:",
  round(
    test_accuracy * 100,
    2
  ),
  "%\n"
)

cat(
  "Balanced accuracy:",
  round(
    test_balanced_accuracy * 100,
    2
  ),
  "%\n"
)

cat(
  "Change F1:",
  round(
    test_f1,
    4
  ),
  "\n"
)

cat(
  "Baseline accuracy:",
  round(
    baseline_accuracy * 100,
    2
  ),
  "%\n"
)