library(data.table)
library(nnet)
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
# 2. FEATURES MULTINOMIAL
# ============================================================

multinom_features <- c(

  "minute",
  "game_state_num",
  "current_tactical_state",

  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  "progressive_pass_diff_last5",

  "america_progressive_carries_last5",
  "america_progressive_passes_last5",

  "pressure_diff_last5",
  "recovery_height_diff_last5",

  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",

  "progressive_pass_diff_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3",

  "opponent_points_last1",
  "opponent_points_last3",
  "opponent_points_last5",
  "opponent_points_weighted"
)


# ============================================================
# 3. ONE-HOT PARA XGBOOST
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
        current_tactical_state ==
          state
      )
  ]
}

state_features <- paste0(
  "current_state_",
  tolower(state_levels)
)


# ============================================================
# 4. FEATURES XGBOOST EVENTS + 360
# ============================================================

xgb_features <- c(

  "minute",
  "game_state_num",

  state_features,

  # Events 5
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # Events 3
  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_pass_diff_last3",
  "america_progressive_carries_last3",
  "america_progressive_passes_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  # 360 5
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 360 3
  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3"
)


# ============================================================
# 5. VALIDAR FEATURES
# ============================================================

all_features <- unique(
  c(
    multinom_features[
      multinom_features !=
        "current_tactical_state"
    ],
    xgb_features
  )
)

missing_features <- setdiff(
  all_features,
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
# 6. IMPUTACIÓN SOLO CON TRAIN
# ============================================================

numeric_imputation_features <- unique(
  c(
    multinom_features[
      multinom_features !=
        "current_tactical_state"
    ],
    xgb_features
  )
)

for (col in numeric_imputation_features) {

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
# 7. SPLITS
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
# 8. ENTRENAR MULTINOMIAL
# ============================================================

formula_multinom <- as.formula(
  paste(
    "target ~",
    paste(
      multinom_features,
      collapse = " + "
    )
  )
)

set.seed(2026)

multinom_model <- multinom(
  formula_multinom,
  data = train,
  trace = FALSE,
  MaxNWts = 10000,
  maxit = 500
)


# ============================================================
# 9. PROBABILIDADES MULTINOMIAL
# ============================================================

format_multinom_probs <- function(
  probs
) {

  probs <- as.matrix(
    probs
  )

  missing_classes <- setdiff(
    state_levels,
    colnames(probs)
  )

  if (
    length(
      missing_classes
    ) > 0
  ) {

    for (
      cl in missing_classes
    ) {

      probs <- cbind(
        probs,
        setNames(
          rep(
            0,
            nrow(probs)
          ),
          cl
        )
      )
    }
  }

  probs <- probs[
    ,
    state_levels,
    drop = FALSE
  ]

  probs
}


prob_multinom_validation <- format_multinom_probs(
  predict(
    multinom_model,
    newdata = validation,
    type = "probs"
  )
)

prob_multinom_test <- format_multinom_probs(
  predict(
    multinom_model,
    newdata = test,
    type = "probs"
  )
)


# ============================================================
# 10. XGBOOST MATRICES
# ============================================================

x_train <- as.matrix(
  train[
    ,
    ..xgb_features
  ]
)

x_validation <- as.matrix(
  validation[
    ,
    ..xgb_features
  ]
)

x_test <- as.matrix(
  test[
    ,
    ..xgb_features
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
# 11. ENTRENAR XGBOOST EVENTS + 360
# ============================================================

xgb_params <- list(

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


set.seed(2026)

xgb_model <- xgb.train(

  params =
    xgb_params,

  data =
    dtrain,

  nrounds =
    1000,

  evals =
    list(
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


# ============================================================
# 12. FORMAT XGBOOST PROBS
# ============================================================

format_xgb_probs <- function(
  raw_probs,
  n_rows
) {

  if (
    is.matrix(
      raw_probs
    )
  ) {

    probs <- raw_probs

  } else {

    probs <- matrix(
      raw_probs,
      nrow = n_rows,
      ncol = length(state_levels),
      byrow = TRUE
    )
  }

  colnames(probs) <-
    state_levels

  probs
}


prob_xgb_validation <- format_xgb_probs(
  predict(
    xgb_model,
    dvalidation
  ),
  nrow(validation)
)

prob_xgb_test <- format_xgb_probs(
  predict(
    xgb_model,
    dtest
  ),
  nrow(test)
)


# ============================================================
# 13. LOG LOSS
# ============================================================

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


# ============================================================
# 14. MACRO F1
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
# 15. BALANCED ACCURACY
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
        return(
          NA_real_
        )
      }

      mean(
        prediction[idx] ==
          cl
      )
    }
  )

  mean(
    recalls,
    na.rm = TRUE
  )
}


# ============================================================
# 16. TOP-2
# ============================================================

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
      probabilities[
        i,
      ],
      decreasing = TRUE
    )[1:2]

    correct[i] <-
      truth[i] %in%
      state_levels[top2]
  }

  mean(
    correct
  )
}


# ============================================================
# 17. BUSCAR PESO DEL ENSEMBLE SOLO EN VALIDATION
#
# alpha = peso XGBoost
# 1-alpha = peso multinomial
# ============================================================

alpha_grid <- seq(
  0,
  1,
  by = 0.05
)

validation_truth <- as.character(
  validation$target
)


blend_search <- rbindlist(
  lapply(
    alpha_grid,
    function(alpha) {

      blended_probs <-
        alpha *
        prob_xgb_validation +
        (
          1 - alpha
        ) *
        prob_multinom_validation


      prediction <- state_levels[
        max.col(
          blended_probs,
          ties.method =
            "first"
        )
      ]


      data.table(

        alpha_xgboost =
          alpha,

        alpha_multinomial =
          1 - alpha,

        accuracy =
          mean(
            prediction ==
            validation_truth
          ),

        balanced_accuracy =
          balanced_accuracy(
            validation_truth,
            prediction
          ),

        macro_f1 =
          macro_f1(
            validation_truth,
            prediction
          ),

        top2_accuracy =
          top2_accuracy(
            validation_truth,
            blended_probs
          ),

        log_loss =
          multiclass_logloss(
            validation_truth,
            blended_probs
          )
      )
    }
  )
)


# ============================================================
# 18. ELEGIR ALPHA
#
# Principal: menor log-loss.
# Desempate: accuracy más alta.
# ============================================================

setorder(
  blend_search,
  log_loss,
  -accuracy
)


best_alpha <- blend_search[
  1,
  alpha_xgboost
]


cat(
  "Best XGBoost weight:",
  best_alpha,
  "\n"
)


# ============================================================
# 19. ENSEMBLE TEST
#
# Solo ahora aplicamos el peso elegido a TEST.
# ============================================================

prob_blend_test <-
  best_alpha *
  prob_xgb_test +
  (
    1 - best_alpha
  ) *
  prob_multinom_test


test_truth <- as.character(
  test$target
)


pred_blend_test <- state_levels[
  max.col(
    prob_blend_test,
    ties.method = "first"
  )
]


# ============================================================
# 20. RESULTADOS TEST
# ============================================================

ensemble_test_metrics <- data.table(

  model =
    "Blended ensemble",

  xgboost_weight =
    best_alpha,

  multinomial_weight =
    1 - best_alpha,

  accuracy =
    round(
      mean(
        pred_blend_test ==
        test_truth
      ) * 100,
      2
    ),

  balanced_accuracy =
    round(
      balanced_accuracy(
        test_truth,
        pred_blend_test
      ) * 100,
      2
    ),

  macro_f1 =
    round(
      macro_f1(
        test_truth,
        pred_blend_test
      ),
      4
    ),

  top2_accuracy =
    round(
      top2_accuracy(
        test_truth,
        prob_blend_test
      ) * 100,
      2
    ),

  log_loss =
    round(
      multiclass_logloss(
        test_truth,
        prob_blend_test
      ),
      4
    )
)


# ============================================================
# 21. BENCHMARKS
# ============================================================

benchmark_comparison <- data.table(

  model = c(
    "Persistence baseline",
    "Multinomial",
    "XGBoost Events+360",
    "Blended ensemble"
  ),

  test_accuracy = c(
    32.67,
    36.42,
    39.91,
    ensemble_test_metrics$
      accuracy
  ),

  test_balanced_accuracy = c(
    NA_real_,
    42.93,
    44.88,
    ensemble_test_metrics$
      balanced_accuracy
  ),

  test_macro_f1 = c(
    NA_real_,
    0.3569,
    0.3861,
    ensemble_test_metrics$
      macro_f1
  ),

  test_top2_accuracy = c(
    NA_real_,
    NA_real_,
    NA_real_,
    ensemble_test_metrics$
      top2_accuracy
  ),

  test_log_loss = c(
    NA_real_,
    1.3913,
    1.3430,
    ensemble_test_metrics$
      log_loss
  )
)


# ============================================================
# 22. GUARDAR PREDICCIONES
# ============================================================

ensemble_predictions <- test[
  ,
  .(
    match_id,
    match_date,
    minute,
    opponent,

    current_tactical_state,

    future_tactical_state_5min
  )
]


ensemble_predictions[
  ,
  predicted_state :=
    pred_blend_test
]


for (
  i in seq_along(
    state_levels
  )
) {

  cl <- state_levels[i]

  ensemble_predictions[
    ,
    paste0(
      "prob_",
      tolower(cl)
    ) :=
      prob_blend_test[
        ,
        i
      ]
  ]
}


# ============================================================
# 23. GUARDAR
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  blend_search,
  "outputs/models/blend_validation_search.csv",
  bom = TRUE
)


fwrite(
  ensemble_test_metrics,
  "outputs/models/blend_test_metrics.csv",
  bom = TRUE
)


fwrite(
  benchmark_comparison,
  "outputs/models/final_predictive_model_comparison.csv",
  bom = TRUE
)


fwrite(
  ensemble_predictions,
  "outputs/models/blended_test_predictions.csv",
  bom = TRUE
)


saveRDS(
  multinom_model,
  "outputs/models/blend_multinomial_model.rds"
)


xgb.save(
  xgb_model,
  "outputs/models/blend_xgboost_model.json"
)


# ============================================================
# 24. SOLO 2 VISUALIZACIONES
# ============================================================

View(
  benchmark_comparison
)

View(
  blend_search[
    1:min(
      10,
      .N
    )
  ]
)


# ============================================================
# 25. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("ENSEMBLE TERMINADO\n")
cat("=====================================\n")


cat(
  "XGBoost weight:",
  best_alpha,
  "\n"
)


cat(
  "Multinomial weight:",
  1 - best_alpha,
  "\n"
)


cat(
  "Test accuracy:",
  ensemble_test_metrics$
    accuracy,
  "%\n"
)


cat(
  "Test balanced accuracy:",
  ensemble_test_metrics$
    balanced_accuracy,
  "%\n"
)


cat(
  "Test Top-2 accuracy:",
  ensemble_test_metrics$
    top2_accuracy,
  "%\n"
)


cat(
  "Test log-loss:",
  ensemble_test_metrics$
    log_loss,
  "\n"
)