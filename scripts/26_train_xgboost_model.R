library(data.table)

# ============================================================
# 0. CARGAR XGBOOST
# ============================================================

if (!requireNamespace("xgboost", quietly = TRUE)) {

  install.packages(
    "xgboost",
    repos = "https://cloud.r-project.org"
  )
}

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
cat("XGBOOST MULTICLASS\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. DEFINIR CLASES
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
  target :=
    factor(
      future_tactical_state_5min,
      levels = state_levels
    )
]

dt[
  ,
  current_tactical_state :=
    factor(
      current_tactical_state,
      levels = state_levels
    )
]


# ============================================================
# 3. FEATURES CORE
# ============================================================

core_features <- c(

  # Contexto partido
  "minute",
  "game_state_num",

  # 5 MIN: TERRITORIO
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  # 5 MIN: PROGRESIÓN
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",

  # 5 MIN: PRESIÓN
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # 5 MIN: 360
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 3 MIN: REACCIÓN RÁPIDA
  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_pass_diff_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  "america_progressive_carries_last3",
  "america_progressive_passes_last3",

  # 3 MIN: 360
  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3",

  # Forma reciente rival
  "opponent_points_last1",
  "opponent_points_last3",
  "opponent_points_last5",
  "opponent_points_weighted"
)


# ============================================================
# 4. FEATURES HISTÓRICAS DEL RIVAL
#
# SOLO temporada anterior:
# opponent_previous_team_season_*
# ============================================================

previous_season_features <- grep(
  "^opponent_previous_team_season_",
  names(dt),
  value = TRUE
)

if (length(previous_season_features) > 0) {

  numeric_check <- sapply(
    dt[
      ,
      ..previous_season_features
    ],
    is.numeric
  )

  previous_season_features <-
    previous_season_features[
      numeric_check
    ]
}

cat(
  "Features históricas encontradas:",
  length(previous_season_features),
  "\n"
)


# ============================================================
# 5. FILTRAR HISTÓRICAS USANDO SOLO TRAIN
#
# Requisitos:
# cobertura >= 50%
# varianza > 0
# ============================================================

historical_keep <- character()

for (col in previous_season_features) {

  train_values <- dt[
    dataset_split == "TRAIN",
    get(col)
  ]

  coverage <- mean(
    !is.na(train_values)
  )

  variance <- var(
    train_values,
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

cat(
  "Features históricas conservadas:",
  length(historical_keep),
  "\n"
)


# ============================================================
# 6. FEATURES NUMÉRICAS FINALES
# ============================================================

numeric_features <- unique(
  c(
    core_features,
    historical_keep
  )
)


# ============================================================
# 7. VALIDAR FEATURES
# ============================================================

feature_validation <- data.table(
  variable = numeric_features,
  exists = numeric_features %in% names(dt)
)

missing_features <- feature_validation[
  exists == FALSE,
  variable
]

if (length(missing_features) > 0) {

  View(
    feature_validation[
      exists == FALSE
    ]
  )

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
# 8. IMPUTACIÓN POR MEDIANA
#
# IMPORTANTÍSIMO:
# medianas calculadas SOLO con TRAIN.
# ============================================================

imputation_table <- data.table(
  variable = numeric_features,
  train_median = NA_real_
)

for (i in seq_along(numeric_features)) {

  col <- numeric_features[i]

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

  imputation_table[
    i,
    train_median := med
  ]

  dt[
    is.na(get(col)),
    (col) := med
  ]
}


# ============================================================
# 9. ONE-HOT DEL ESTADO ACTUAL
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

state_dummy_features <- paste0(
  "current_state_",
  tolower(state_levels)
)

final_features <- c(
  numeric_features,
  state_dummy_features
)

cat(
  "Total features XGBoost:",
  length(final_features),
  "\n"
)


# ============================================================
# 10. SPLITS
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
# 11. MATRICES X
# ============================================================

x_train <- as.matrix(
  train[
    ,
    ..final_features
  ]
)

x_validation <- as.matrix(
  validation[
    ,
    ..final_features
  ]
)

x_test <- as.matrix(
  test[
    ,
    ..final_features
  ]
)


# ============================================================
# 12. TARGET XGBOOST
#
# XGBoost espera clases 0,1,2,3,4
# ============================================================

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
# 13. DMATRIX
# ============================================================

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)

dvalidation <- xgb.DMatrix(
  data = x_validation,
  label = y_validation
)

dtest <- xgb.DMatrix(
  data = x_test,
  label = y_test
)


# ============================================================
# 14. GRID DE HIPERPARÁMETROS
#
# Validation decide.
# TEST NO se usa aquí.
# ============================================================

parameter_grid <- CJ(

  max_depth = c(
    3L,
    5L,
    7L
  ),

  eta = c(
    0.03,
    0.07
  ),

  min_child_weight = c(
    3,
    8
  ),

  subsample = 0.8,

  colsample_bytree = 0.7
)


# ============================================================
# 15. TUNING
# ============================================================

tuning_results <- list()
models <- list()

for (i in seq_len(nrow(parameter_grid))) {

  p <- parameter_grid[i]

  cat(
    "Entrenando modelo",
    i,
    "de",
    nrow(parameter_grid),
    "\n"
  )

  params <- list(

    objective = "multi:softprob",

    num_class =
      length(state_levels),

    eval_metric =
      "mlogloss",

    max_depth =
      p$max_depth,

    eta =
      p$eta,

    min_child_weight =
      p$min_child_weight,

    subsample =
      p$subsample,

    colsample_bytree =
      p$colsample_bytree,

    lambda = 1,

    alpha = 0,

    nthread =
      max(
        1,
        parallel::detectCores() - 1
      )
  )

  set.seed(
    2026 + i
  )

  model <- xgb.train(

    params = params,

    data = dtrain,

    nrounds = 1000,

    watchlist = list(
      train = dtrain,
      validation = dvalidation
    ),

    early_stopping_rounds = 40,

    maximize = FALSE,

    verbose = 0
  )

  tuning_results[[i]] <- data.table(

    model_id = i,

    max_depth =
      p$max_depth,

    eta =
      p$eta,

    min_child_weight =
      p$min_child_weight,

    subsample =
      p$subsample,

    colsample_bytree =
      p$colsample_bytree,

    best_iteration =
      model$best_iteration,

    validation_logloss =
      model$best_score
  )

  models[[i]] <- model
}


# ============================================================
# 16. TABLA DE TUNING
# ============================================================

tuning_results <- rbindlist(
  tuning_results
)

setorder(
  tuning_results,
  validation_logloss
)


# ============================================================
# 17. MEJOR MODELO
# ============================================================

best_model_id <- tuning_results[
  1,
  model_id
]

best_model <- models[[
  best_model_id
]]

cat(
  "\nMejor modelo ID:",
  best_model_id,
  "\n"
)

cat(
  "Validation log-loss:",
  tuning_results[
    1,
    validation_logloss
  ],
  "\n"
)


# ============================================================
# 18. FUNCIÓN ROBUSTA DE PROBABILIDADES
#
# FIX:
# XGBoost moderno puede devolver directamente
# una matriz N x clases.
# ============================================================

predict_multiclass <- function(
  model,
  dmatrix,
  n_rows
) {

  raw_probs <- predict(
    model,
    dmatrix
  )

  # ----------------------------------------------------------
  # CASO 1:
  # XGBoost moderno devuelve matriz
  # ----------------------------------------------------------

  if (is.matrix(raw_probs)) {

    probs <- raw_probs

  } else {

    # --------------------------------------------------------
    # CASO 2:
    # versiones que devuelven vector
    # --------------------------------------------------------

    probs <- matrix(
      raw_probs,
      nrow = n_rows,
      ncol = length(state_levels),
      byrow = TRUE
    )
  }

  # ----------------------------------------------------------
  # VALIDAR DIMENSIONES
  # ----------------------------------------------------------

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
# 19. PROBABILIDADES
# ============================================================

prob_train <- predict_multiclass(
  best_model,
  dtrain,
  nrow(train)
)

prob_validation <- predict_multiclass(
  best_model,
  dvalidation,
  nrow(validation)
)

prob_test <- predict_multiclass(
  best_model,
  dtest,
  nrow(test)
)


# ============================================================
# 20. VALIDAR QUE CADA FILA SUME ~1
# ============================================================

probability_check <- data.table(

  dataset = c(
    "TRAIN",
    "VALIDATION",
    "TEST"
  ),

  min_sum = c(
    min(rowSums(prob_train)),
    min(rowSums(prob_validation)),
    min(rowSums(prob_test))
  ),

  max_sum = c(
    max(rowSums(prob_train)),
    max(rowSums(prob_validation)),
    max(rowSums(prob_test))
  ),

  mean_sum = c(
    mean(rowSums(prob_train)),
    mean(rowSums(prob_validation)),
    mean(rowSums(prob_test))
  )
)


# ============================================================
# 21. CLASE PREDICHA
# ============================================================

pred_train <- state_levels[
  max.col(
    prob_train,
    ties.method = "first"
  )
]

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


# ============================================================
# 22. LOG LOSS
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
# 23. MACRO F1
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
# 24. BALANCED ACCURACY
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
# 25. FUNCIÓN DE EVALUACIÓN
# ============================================================

evaluate_model <- function(
  dataset_name,
  data,
  probs,
  preds
) {

  truth <- as.character(
    data$target
  )

  data.table(

    dataset =
      dataset_name,

    observations =
      length(truth),

    accuracy =
      round(
        mean(
          preds ==
            truth
        ) * 100,
        2
      ),

    balanced_accuracy =
      round(
        balanced_accuracy(
          truth,
          preds
        ) * 100,
        2
      ),

    macro_f1 =
      round(
        macro_f1(
          truth,
          preds
        ),
        4
      ),

    log_loss =
      round(
        multiclass_logloss(
          truth,
          probs
        ),
        4
      )
  )
}


# ============================================================
# 26. MÉTRICAS
# ============================================================

metrics_train <- evaluate_model(
  "TRAIN",
  train,
  prob_train,
  pred_train
)

metrics_validation <- evaluate_model(
  "VALIDATION",
  validation,
  prob_validation,
  pred_validation
)

metrics_test <- evaluate_model(
  "TEST",
  test,
  prob_test,
  pred_test
)

xgb_metrics <- rbind(
  metrics_train,
  metrics_validation,
  metrics_test
)


# ============================================================
# 27. MÉTRICAS POR CLASE EN TEST
# ============================================================

truth_test <- as.character(
  test$target
)

xgb_class_metrics <- rbindlist(
  lapply(
    state_levels,
    function(cl) {

      tp <- sum(
        truth_test == cl &
        pred_test == cl
      )

      fp <- sum(
        truth_test != cl &
        pred_test == cl
      )

      fn <- sum(
        truth_test == cl &
        pred_test != cl
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

      f1 <- ifelse(
        precision + recall == 0,
        0,
        2 *
          precision *
          recall /
          (
            precision +
              recall
          )
      )

      data.table(

        tactical_state =
          cl,

        precision =
          round(
            precision,
            4
          ),

        recall =
          round(
            recall,
            4
          ),

        f1 =
          round(
            f1,
            4
          )
      )
    }
  )
)


# ============================================================
# 28. CONFUSION MATRIX TEST
# ============================================================

confusion_test <- data.table(

  truth =
    truth_test,

  prediction =
    pred_test

)[
  ,
  .N,
  by = .(
    truth,
    prediction
  )
]


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

feature_importance <- xgb.importance(
  model = best_model
)


# ============================================================
# 30. PREDICCIONES TEST
# ============================================================

test_predictions <- test[
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

test_predictions[
  ,
  predicted_state :=
    pred_test
]


for (i in seq_along(state_levels)) {

  cl <- state_levels[i]

  test_predictions[
    ,
    paste0(
      "prob_",
      tolower(cl)
    ) :=
      prob_test[
        ,
        i
      ]
  ]
}


# ============================================================
# 31. COMPARAR CONTRA MULTINOMIAL
# ============================================================

multinom_metrics <- fread(
  "outputs/models/multinomial_model_metrics.csv",
  encoding = "UTF-8"
)

multinom_metrics[
  ,
  model := "Multinomial"
]

xgb_metrics[
  ,
  model := "XGBoost"
]

model_comparison <- rbind(
  multinom_metrics,
  xgb_metrics,
  fill = TRUE
)

setcolorder(
  model_comparison,
  c(
    "model",
    "dataset",
    "observations",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "log_loss"
  )
)


# ============================================================
# 32. GUARDAR MODELO Y RESULTADOS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)

xgb.save(
  best_model,
  "outputs/models/xgboost_tactical_state_model.json"
)

fwrite(
  tuning_results,
  "outputs/models/xgboost_tuning_results.csv",
  bom = TRUE
)

fwrite(
  xgb_metrics,
  "outputs/models/xgboost_model_metrics.csv",
  bom = TRUE
)

fwrite(
  xgb_class_metrics,
  "outputs/models/xgboost_test_class_metrics.csv",
  bom = TRUE
)

fwrite(
  confusion_test,
  "outputs/models/xgboost_test_confusion.csv",
  bom = TRUE
)

fwrite(
  feature_importance,
  "outputs/models/xgboost_feature_importance.csv",
  bom = TRUE
)

fwrite(
  test_predictions,
  "outputs/models/xgboost_test_predictions.csv",
  bom = TRUE
)

fwrite(
  model_comparison,
  "outputs/models/predictive_model_comparison.csv",
  bom = TRUE
)

fwrite(
  imputation_table,
  "outputs/models/xgboost_imputation_values.csv",
  bom = TRUE
)

fwrite(
  probability_check,
  "outputs/models/xgboost_probability_check.csv",
  bom = TRUE
)


# ============================================================
# 33. SOLO 3 VISUALIZACIONES
# ============================================================

View(
  model_comparison
)

View(
  probability_check
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
# 34. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("XGBOOST CORREGIDO TERMINADO\n")
cat("=====================================\n")

cat(
  "Best model ID:",
  best_model_id,
  "\n"
)

cat(
  "Test accuracy:",
  metrics_test$accuracy,
  "%\n"
)

cat(
  "Test balanced accuracy:",
  metrics_test$balanced_accuracy,
  "%\n"
)

cat(
  "Test macro F1:",
  metrics_test$macro_f1,
  "\n"
)

cat(
  "Test log-loss:",
  metrics_test$log_loss,
  "\n"
)