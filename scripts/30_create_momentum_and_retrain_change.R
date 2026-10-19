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
  current_tactical_state :=
    factor(
      current_tactical_state,
      levels = state_levels
    )
]


# ============================================================
# 2. TARGET
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
# 3. ONE-HOT ESTADO ACTUAL
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
# 4. MOMENTUM DE VARIABLES PROMEDIO / RATIOS
#
# last3 - last5
#
# positivo = subiendo recientemente
# negativo = cayendo recientemente
# ============================================================

dt[
  ,
  field_tilt_momentum :=
    america_field_tilt_last3 -
    america_field_tilt_last5
]

dt[
  ,
  pressure_received_momentum :=
    america_pressure_share_5_last3 -
    america_pressure_share_5_last5
]

dt[
  ,
  local_superiority_momentum :=
    america_local_superiority_10_last3 -
    america_local_superiority_10_last5
]

dt[
  ,
  nearest_opponent_distance_momentum :=
    america_nearest_opponent_distance_last3 -
    america_nearest_opponent_distance_last5
]

dt[
  ,
  support_momentum :=
    america_support_share_10_last3 -
    america_support_share_10_last5
]

dt[
  ,
  recovery_height_momentum :=
    recovery_height_diff_last3 -
    recovery_height_diff_last5
]


# ============================================================
# 5. MOMENTUM DE CONTEOS
#
# No comparamos directamente last3 vs last5,
# porque las ventanas tienen diferente tamaño.
#
# Convertimos primero a RATE POR MINUTO.
# ============================================================

dt[
  ,
  final_third_entry_rate_3 :=
    entries_final_third_diff_last3 / 3
]

dt[
  ,
  final_third_entry_rate_5 :=
    entries_final_third_diff_last5 / 5
]

dt[
  ,
  final_third_momentum :=
    final_third_entry_rate_3 -
    final_third_entry_rate_5
]


dt[
  ,
  box_entry_rate_3 :=
    entries_box_diff_last3 / 3
]

dt[
  ,
  box_entry_rate_5 :=
    entries_box_diff_last5 / 5
]

dt[
  ,
  box_entry_momentum :=
    box_entry_rate_3 -
    box_entry_rate_5
]


dt[
  ,
  progressive_pass_rate_3 :=
    progressive_pass_diff_last3 / 3
]

dt[
  ,
  progressive_pass_rate_5 :=
    progressive_pass_diff_last5 / 5
]

dt[
  ,
  progressive_pass_momentum :=
    progressive_pass_rate_3 -
    progressive_pass_rate_5
]


dt[
  ,
  pressure_rate_3 :=
    pressure_diff_last3 / 3
]

dt[
  ,
  pressure_rate_5 :=
    pressure_diff_last5 / 5
]

dt[
  ,
  pressure_momentum :=
    pressure_rate_3 -
    pressure_rate_5
]


dt[
  ,
  progressive_carry_rate_3 :=
    america_progressive_carries_last3 / 3
]

dt[
  ,
  progressive_carry_rate_5 :=
    america_progressive_carries_last5 / 5
]

dt[
  ,
  progressive_carry_momentum :=
    progressive_carry_rate_3 -
    progressive_carry_rate_5
]


dt[
  ,
  america_progressive_pass_rate_3 :=
    america_progressive_passes_last3 / 3
]

dt[
  ,
  america_progressive_pass_rate_5 :=
    america_progressive_passes_last5 / 5
]

dt[
  ,
  america_progressive_pass_momentum :=
    america_progressive_pass_rate_3 -
    america_progressive_pass_rate_5
]


# ============================================================
# 6. DESACUERDO 3 MIN VS 5 MIN
#
# Si la lectura táctica rápida ya difiere de la
# estable, puede indicar transición próxima.
# ============================================================

dt[
  ,
  state_3_vs_5_disagreement :=
    as.integer(
      !is.na(tactical_state_3min) &
      as.character(tactical_state_3min) !=
        as.character(current_tactical_state)
    )
]


# ============================================================
# 7. DIFERENCIA DE CONFIANZA
# ============================================================

if (
  all(
    c(
      "tactical_state_confidence_3min",
      "tactical_state_confidence"
    ) %in% names(dt)
  )
) {

  dt[
    ,
    confidence_momentum :=
      tactical_state_confidence_3min -
      tactical_state_confidence
  ]

} else {

  dt[
    ,
    confidence_momentum :=
      0
  ]
}


# ============================================================
# 8. LISTA DE MOMENTUM FEATURES
# ============================================================

momentum_features <- c(

  "field_tilt_momentum",

  "pressure_received_momentum",

  "local_superiority_momentum",

  "nearest_opponent_distance_momentum",

  "support_momentum",

  "recovery_height_momentum",

  "final_third_momentum",

  "box_entry_momentum",

  "progressive_pass_momentum",

  "pressure_momentum",

  "progressive_carry_momentum",

  "america_progressive_pass_momentum",

  "state_3_vs_5_disagreement",

  "confidence_momentum"
)


# ============================================================
# 9. FEATURES BASE
# ============================================================

base_features <- c(

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
# 10. FEATURES FINALES
# ============================================================

features <- unique(
  c(
    base_features,
    momentum_features
  )
)


# ============================================================
# 11. VALIDAR FEATURES
# ============================================================

missing_features <- setdiff(
  features,
  names(dt)
)

if (
  length(missing_features) > 0
) {

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
# 12. IMPUTAR SOLO USANDO TRAIN
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
# 13. SPLITS
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
# 14. MATRICES
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
# 15. DMATRIX
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
# 16. GRID MÁS AMPLIO
# ============================================================

parameter_grid <- CJ(

  max_depth = c(
    2L,
    3L,
    4L,
    5L
  ),

  eta = c(
    0.02,
    0.05,
    0.08
  ),

  min_child_weight = c(
    3,
    8
  )
)


# ============================================================
# 17. ENTRENAR
# ============================================================

tuning_results <- list()

models <- list()


for (i in seq_len(nrow(parameter_grid))) {

  p <- parameter_grid[i]

  cat(
    "Momentum model",
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
      0.80,

    colsample_bytree =
      0.75,

    lambda =
      2,

    alpha =
      0.1,

    nthread =
      max(
        1,
        parallel::detectCores() - 1
      )
  )


  set.seed(
    5000 + i
  )


  model <- xgb.train(

    params = params,

    data = dtrain,

    nrounds = 1500,

    watchlist = list(
      train = dtrain,
      validation = dvalidation
    ),

    early_stopping_rounds =
      60,

    maximize =
      FALSE,

    verbose =
      0
  )


  tuning_results[[i]] <- data.table(

    model_id = i,

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
# 18. MEJOR MODELO
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
# 19. PROBABILIDADES
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
# 20. BUSCAR THRESHOLD SOLO EN VALIDATION
# ============================================================

threshold_candidates <- seq(
  0.20,
  0.85,
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


      accuracy <-
        (
          tp +
          tn
        ) /
        length(
          y_validation
        )


      data.table(

        threshold =
          threshold,

        balanced_accuracy =
          balanced_accuracy,

        accuracy =
          accuracy
      )
    }
  )
)


# ============================================================
# 21. THRESHOLD POR BALANCED ACCURACY
# ============================================================

best_threshold <- threshold_results[
  which.max(
    balanced_accuracy
  ),
  threshold
]


# ============================================================
# 22. PREDICCIÓN TEST
# ============================================================

pred_test <- as.integer(
  prob_test >=
    best_threshold
)


# ============================================================
# 23. CONFUSION
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


# ============================================================
# 24. MÉTRICAS
# ============================================================

accuracy <-
  (
    tp + tn
  ) /
  length(y_test)


recall_change <- ifelse(
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
    recall_change +
    specificity
  ) / 2


precision_change <- ifelse(
  tp + fp == 0,
  0,
  tp / (tp + fp)
)


f1_change <- ifelse(

  precision_change +
    recall_change == 0,

  0,

  2 *
    precision_change *
    recall_change /
    (
      precision_change +
        recall_change
    )
)


# ============================================================
# 25. MAJORITY BASELINE
# ============================================================

majority_prediction <-
  as.integer(
    mean(y_train) >=
      0.50
  )


baseline_accuracy <- mean(
  y_test ==
    majority_prediction
)


# ============================================================
# 26. RESULTADOS ANTERIORES
# ============================================================

previous_metrics <- data.table(

  model =
    "Previous change model",

  accuracy =
    0.6584075,

  balanced_accuracy =
    0.5732758,

  precision_change =
    0.7151326,

  recall_change =
    0.8188639,

  f1_change =
    0.7634910
)


new_metrics <- data.table(

  model =
    "Momentum change model",

  accuracy =
    accuracy,

  balanced_accuracy =
    balanced_accuracy,

  precision_change =
    precision_change,

  recall_change =
    recall_change,

  f1_change =
    f1_change
)


model_comparison <- rbind(

  previous_metrics,

  new_metrics
)


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

feature_importance <- xgb.importance(
  model =
    best_model
)


# ============================================================
# 28. GUARDAR DATASET CON MOMENTUM
# ============================================================

fwrite(

  dt,

  "data/processed/america_male/model_ready_with_momentum.csv",

  bom = TRUE,

  na = "NA"
)


# ============================================================
# 29. GUARDAR OUTPUTS
# ============================================================

dir.create(

  "outputs/models",

  recursive = TRUE,

  showWarnings = FALSE
)


xgb.save(

  best_model,

  "outputs/models/xgboost_momentum_change_model.json"
)


fwrite(

  model_comparison,

  "outputs/models/momentum_change_model_comparison.csv",

  bom = TRUE
)


fwrite(

  feature_importance,

  "outputs/models/momentum_change_feature_importance.csv",

  bom = TRUE
)


fwrite(

  tuning_results,

  "outputs/models/momentum_change_tuning.csv",

  bom = TRUE
)


# ============================================================
# 30. SOLO 2 VISUALIZACIONES
# ============================================================

View(
  model_comparison
)

View(
  feature_importance[
    1:min(
      20,
      .N
    )
  ]
)


# ============================================================
# 31. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("MOMENTUM CHANGE MODEL TERMINADO\n")
cat("=====================================\n")

cat(
  "Threshold:",
  best_threshold,
  "\n"
)

cat(
  "Test accuracy:",
  round(
    accuracy * 100,
    2
  ),
  "%\n"
)

cat(
  "Balanced accuracy:",
  round(
    balanced_accuracy * 100,
    2
  ),
  "%\n"
)

cat(
  "F1 change:",
  round(
    f1_change,
    4
  ),
  "\n"
)

cat(
  "Majority baseline:",
  round(
    baseline_accuracy * 100,
    2
  ),
  "%\n"
)