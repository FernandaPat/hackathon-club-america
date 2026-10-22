# ============================================================
# 53_female_change_model.R
# CLUB AMÉRICA FEMENIL
#
# MODELO BINARIO DE CAMBIO TÁCTICO
#
# Objetivo:
#   1 = el estado táctico cambia dentro de 5 minutos
#   0 = el estado táctico permanece igual
#
# Metodología:
# - TRAIN: entrenamiento
# - VALIDATION: selección de modelo + threshold
# - TEST: evaluación final una sola vez
# ============================================================

library(data.table)
library(xgboost)


# ============================================================
# 1. RUTAS
# ============================================================

project_root <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)


data_dir <- file.path(
  project_root,
  "data",
  "processed",
  "america_female"
)


input_file <- file.path(
  data_dir,
  "model_ready_female_v2.csv"
)


output_dir <- file.path(
  project_root,
  "outputs",
  "models",
  "female",
  "change_model"
)


if (!dir.exists(output_dir)) {

  dir.create(
    output_dir,
    recursive = TRUE
  )
}


model_file <- file.path(
  output_dir,
  "female_change_model.json"
)


metrics_file <- file.path(
  output_dir,
  "female_change_test_metrics.csv"
)


validation_file <- file.path(
  output_dir,
  "female_change_validation_candidates.csv"
)


predictions_file <- file.path(
  output_dir,
  "female_change_test_predictions.csv"
)


features_file <- file.path(
  output_dir,
  "female_change_features.csv"
)


selection_file <- file.path(
  output_dir,
  "female_change_selection.csv"
)


# ============================================================
# 2. CARGAR DATOS
# ============================================================

if (!file.exists(input_file)) {

  stop(
    paste(
      "No existe:",
      input_file
    )
  )
}


dt <- fread(
  input_file
)


cat("\n")
cat("============================================\n")
cat("FEMALE TACTICAL CHANGE MODEL\n")
cat("============================================\n\n")


cat(
  "Observaciones cargadas:",
  nrow(dt),
  "\n"
)


cat(
  "Partidos:",
  uniqueN(dt$match_id),
  "\n\n"
)


# ============================================================
# 3. VALIDACIONES
# ============================================================

required_columns <- c(
  "match_id",
  "match_date",
  "minute",
  "split",
  "current_tactical_state",
  "future_tactical_state_5min",
  "game_state_num"
)


missing_columns <- setdiff(
  required_columns,
  names(dt)
)


if (length(missing_columns) > 0) {

  stop(
    paste(
      "Faltan columnas:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 4. TARGET BINARIO
#
# 1 = CAMBIA
# 0 = NO CAMBIA
# ============================================================

dt[
  ,
  tactical_change :=
    as.integer(
      current_tactical_state
      !=
      future_tactical_state_5min
    )
]


if (any(is.na(dt$tactical_change))) {

  stop(
    "El target tactical_change contiene NA."
  )
}


# ============================================================
# 5. DISTRIBUCIÓN GLOBAL
# ============================================================

global_distribution <- dt[
  ,
  .(
    observations = .N
  ),
  by = tactical_change
]


global_distribution[
  ,
  label :=
    fifelse(
      tactical_change == 1,
      "CAMBIA",
      "NO_CAMBIA"
    )
]


global_distribution[
  ,
  share :=
    observations /
    sum(observations)
]


cat(
  "Distribución global del target:\n\n"
)


print(
  global_distribution
)


# ============================================================
# 6. SPLITS
# ============================================================

train_dt <- copy(
  dt[
    split == "TRAIN"
  ]
)


validation_dt <- copy(
  dt[
    split == "VALIDATION"
  ]
)


test_dt <- copy(
  dt[
    split == "TEST"
  ]
)


if (
  nrow(train_dt) == 0
  ||
  nrow(validation_dt) == 0
  ||
  nrow(test_dt) == 0
) {

  stop(
    "Uno de los splits está vacío."
  )
}


cat("\n")
cat(
  "TRAIN:",
  nrow(train_dt),
  "observaciones /",
  uniqueN(train_dt$match_id),
  "partidos\n"
)


cat(
  "VALIDATION:",
  nrow(validation_dt),
  "observaciones /",
  uniqueN(validation_dt$match_id),
  "partidos\n"
)


cat(
  "TEST:",
  nrow(test_dt),
  "observaciones /",
  uniqueN(test_dt$match_id),
  "partidos\n\n"
)


# ============================================================
# 7. DISTRIBUCIÓN TARGET POR SPLIT
# ============================================================

split_distribution <- dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    split,
    tactical_change
  )
]


split_distribution[
  ,
  share :=
    observations /
    sum(observations),
  by = split
]


cat(
  "Distribución target por split:\n\n"
)


print(
  split_distribution
)


# ============================================================
# 8. ONE-HOT DEL ESTADO ACTUAL
#
# Solo usamos estados observados en TRAIN
# ============================================================

current_states <- sort(
  unique(
    train_dt$current_tactical_state
  )
)


state_features <- character()


for (state in current_states) {

  clean_state <- tolower(
    gsub(
      "[^A-Za-z0-9]+",
      "_",
      state
    )
  )


  feature_name <- paste0(
    "current_state_",
    clean_state
  )


  state_features <- c(
    state_features,
    feature_name
  )


  train_dt[
    ,
    (feature_name) :=
      as.integer(
        current_tactical_state == state
      )
  ]


  validation_dt[
    ,
    (feature_name) :=
      as.integer(
        current_tactical_state == state
      )
  ]


  test_dt[
    ,
    (feature_name) :=
      as.integer(
        current_tactical_state == state
      )
  ]
}


# ============================================================
# 9. FEATURES
# ============================================================

rolling_features <- grep(
  "_last3$|_last5$|_trend$",
  names(train_dt),
  value = TRUE
)


if (length(rolling_features) > 0) {

  numeric_mask <- vapply(
    train_dt[
      ,
      ..rolling_features
    ],
    is.numeric,
    logical(1)
  )


  rolling_features <- rolling_features[
    numeric_mask
  ]
}


features <- unique(
  c(
    "minute",
    "game_state_num",
    state_features,
    rolling_features
  )
)


# ============================================================
# 10. BLOQUEAR LEAKAGE
# ============================================================

forbidden_patterns <- c(
  "future_",
  "target",
  "usable_",
  "cluster_id",
  "same_state",
  "tactical_change",
  "tactical_state_tie",
  "_score"
)


for (pattern in forbidden_patterns) {

  features <- features[
    !grepl(
      pattern,
      features,
      fixed = TRUE
    )
  ]
}


features <- features[
  features %in%
    names(train_dt)
]


if (length(features) < 5) {

  stop(
    "No existen suficientes features."
  )
}


cat(
  "Features utilizadas:",
  length(features),
  "\n\n"
)


print(
  features
)


# ============================================================
# 11. CONVERTIR A NUMÉRICO
# ============================================================

for (feature in features) {

  train_dt[
    ,
    (feature) :=
      as.numeric(
        get(feature)
      )
  ]


  validation_dt[
    ,
    (feature) :=
      as.numeric(
        get(feature)
      )
  ]


  test_dt[
    ,
    (feature) :=
      as.numeric(
        get(feature)
      )
  ]
}


# ============================================================
# 12. IMPUTACIÓN SOLO DESDE TRAIN
# ============================================================

imputation_table <- data.table(
  feature = features,
  train_median = NA_real_
)


for (i in seq_along(features)) {

  feature <- features[i]


  median_value <- median(
    train_dt[
      ,
      get(feature)
    ],
    na.rm = TRUE
  )


  if (!is.finite(median_value)) {

    median_value <- 0
  }


  imputation_table[
    i,
    train_median :=
      median_value
  ]


  train_dt[
    is.na(
      get(feature)
    ),
    (feature) :=
      median_value
  ]


  validation_dt[
    is.na(
      get(feature)
    ),
    (feature) :=
      median_value
  ]


  test_dt[
    is.na(
      get(feature)
    ),
    (feature) :=
      median_value
  ]
}


# ============================================================
# 13. MATRICES
# ============================================================

x_train <- as.matrix(
  train_dt[
    ,
    ..features
  ]
)


x_validation <- as.matrix(
  validation_dt[
    ,
    ..features
  ]
)


x_test <- as.matrix(
  test_dt[
    ,
    ..features
  ]
)


y_train <- train_dt$tactical_change
y_validation <- validation_dt$tactical_change
y_test <- test_dt$tactical_change


# ============================================================
# 14. PESOS DE CLASE DERIVADOS DE TRAIN
# ============================================================

class_counts <- table(
  y_train
)


class_frequency <- (
  class_counts /
  sum(class_counts)
)


class_weights <- (
  1 /
  sqrt(
    class_frequency
  )
)


class_weights <- (
  class_weights /
  mean(
    class_weights
  )
)


train_weights <- as.numeric(
  class_weights[
    as.character(
      y_train
    )
  ]
)


cat(
  "\nPesos derivados de TRAIN:\n"
)


print(
  class_weights
)


# ============================================================
# 15. DMATRIX
# ============================================================

dtrain_plain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)


dtrain_weighted <- xgb.DMatrix(
  data = x_train,
  label = y_train,
  weight = train_weights
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
# 16. AUC MANUAL
# ============================================================

calculate_auc <- function(
  actual,
  probability
) {

  positive <- sum(
    actual == 1
  )


  negative <- sum(
    actual == 0
  )


  if (
    positive == 0
    ||
    negative == 0
  ) {

    return(
      NA_real_
    )
  }


  ranks <- rank(
    probability,
    ties.method = "average"
  )


  positive_rank_sum <- sum(
    ranks[
      actual == 1
    ]
  )


  auc <- (
    positive_rank_sum
    -
    positive *
    (
      positive + 1
    ) / 2
  ) /
    (
      positive *
      negative
    )


  as.numeric(
    auc
  )
}


# ============================================================
# 17. MÉTRICAS BINARIAS
# ============================================================

evaluate_binary <- function(
  actual,
  probability,
  threshold
) {

  prediction <- as.integer(
    probability >= threshold
  )


  tp <- sum(
    actual == 1
    &
    prediction == 1
  )


  tn <- sum(
    actual == 0
    &
    prediction == 0
  )


  fp <- sum(
    actual == 0
    &
    prediction == 1
  )


  fn <- sum(
    actual == 1
    &
    prediction == 0
  )


  accuracy <- mean(
    prediction ==
    actual
  )


  sensitivity <- if (
    tp + fn > 0
  ) {

    tp /
    (
      tp + fn
    )

  } else {

    NA_real_
  }


  specificity <- if (
    tn + fp > 0
  ) {

    tn /
    (
      tn + fp
    )

  } else {

    NA_real_
  }


  balanced_accuracy <- mean(
    c(
      sensitivity,
      specificity
    ),
    na.rm = TRUE
  )


  precision <- if (
    tp + fp > 0
  ) {

    tp /
    (
      tp + fp
    )

  } else {

    NA_real_
  }


  f1 <- if (
    is.finite(precision)
    &&
    is.finite(sensitivity)
    &&
    (
      precision +
      sensitivity
    ) > 0
  ) {

    2 *
    precision *
    sensitivity /
    (
      precision +
      sensitivity
    )

  } else {

    NA_real_
  }


  clipped_probability <- pmin(
    pmax(
      probability,
      .Machine$double.eps
    ),
    1 -
      .Machine$double.eps
  )


  logloss <- -mean(
    actual *
    log(
      clipped_probability
    )
    +
    (
      1 -
      actual
    ) *
    log(
      1 -
      clipped_probability
    )
  )


  brier <- mean(
    (
      probability -
      actual
    )^2
  )


  auc <- calculate_auc(
    actual,
    probability
  )


  list(

    accuracy =
      accuracy,

    balanced_accuracy =
      balanced_accuracy,

    sensitivity =
      sensitivity,

    specificity =
      specificity,

    precision =
      precision,

    f1 =
      f1,

    auc =
      auc,

    logloss =
      logloss,

    brier =
      brier,

    tp =
      tp,

    tn =
      tn,

    fp =
      fp,

    fn =
      fn,

    prediction =
      prediction
  )
}


# ============================================================
# 18. CONFIGURACIONES CANDIDATAS
# ============================================================

candidate_configs <- data.table(

  candidate_id =
    seq_len(8),

  max_depth =
    c(
      3,
      4,
      5,
      6,
      3,
      4,
      5,
      6
    ),

  eta =
    c(
      0.03,
      0.03,
      0.03,
      0.03,
      0.05,
      0.05,
      0.05,
      0.05
    ),

  min_child_weight =
    c(
      5,
      5,
      5,
      5,
      10,
      10,
      10,
      10
    )
)


# ============================================================
# 19. THRESHOLDS A PROBAR
#
# Se seleccionan SOLO usando VALIDATION
# ============================================================

threshold_candidates <- seq(
  0.20,
  0.80,
  by = 0.02
)


# ============================================================
# 20. ENTRENAMIENTO DE CANDIDATOS
# ============================================================

validation_results_list <- list()

model_list <- list()


for (
  config_index
  in
  seq_len(
    nrow(
      candidate_configs
    )
  )
) {

  config <- candidate_configs[
    config_index
  ]


  for (
    weighted
    in
    c(
      FALSE,
      TRUE
    )
  ) {

    candidate_name <- paste0(
      "C",
      config$candidate_id,
      ifelse(
        weighted,
        "_W",
        "_U"
      )
    )


    cat(
      "\nEntrenando:",
      candidate_name,
      "\n"
    )


    params <- list(

      objective =
        "binary:logistic",

      eval_metric =
        "logloss",

      max_depth =
        config$max_depth,

      eta =
        config$eta,

      min_child_weight =
        config$min_child_weight,

      subsample =
        0.85,

      colsample_bytree =
        0.85,

      lambda =
        1
    )


    training_matrix <- if (
      weighted
    ) {

      dtrain_weighted

    } else {

      dtrain_plain
    }


    model <- xgb.train(

      params =
        params,

      data =
        training_matrix,

      nrounds =
        1500,

      evals = list(
        train =
          training_matrix,

        validation =
          dvalidation
      ),

      early_stopping_rounds =
        75,

      verbose =
        0
    )


    validation_probability <- predict(
      model,
      dvalidation
    )


    # ========================================================
    # BUSCAR MEJOR THRESHOLD EN VALIDATION
    # ========================================================

    threshold_results <- list()


    for (
      threshold
      in
      threshold_candidates
    ) {

      metrics <- evaluate_binary(
        y_validation,
        validation_probability,
        threshold
      )


      threshold_results <- append(
        threshold_results,
        list(
          data.table(

            threshold =
              threshold,

            accuracy =
              metrics$accuracy,

            balanced_accuracy =
              metrics$balanced_accuracy,

            sensitivity =
              metrics$sensitivity,

            specificity =
              metrics$specificity,

            precision =
              metrics$precision,

            f1 =
              metrics$f1,

            auc =
              metrics$auc,

            logloss =
              metrics$logloss,

            brier =
              metrics$brier
          )
        )
      )
    }


    threshold_table <- rbindlist(
      threshold_results
    )


    setorder(
      threshold_table,
      -balanced_accuracy,
      -f1,
      -accuracy,
      logloss
    )


    best_threshold_row <- (
      threshold_table[1]
    )


    result_row <- data.table(

      candidate =
        candidate_name,

      weighted =
        weighted,

      max_depth =
        config$max_depth,

      eta =
        config$eta,

      min_child_weight =
        config$min_child_weight,

      best_iteration =
        model$best_iteration,

      threshold =
        best_threshold_row$threshold,

      validation_accuracy =
        best_threshold_row$accuracy,

      validation_balanced =
        best_threshold_row$balanced_accuracy,

      validation_sensitivity =
        best_threshold_row$sensitivity,

      validation_specificity =
        best_threshold_row$specificity,

      validation_precision =
        best_threshold_row$precision,

      validation_f1 =
        best_threshold_row$f1,

      validation_auc =
        best_threshold_row$auc,

      validation_logloss =
        best_threshold_row$logloss,

      validation_brier =
        best_threshold_row$brier
    )


    validation_results_list <- append(
      validation_results_list,
      list(
        result_row
      )
    )


    model_list[
      candidate_name
    ] <- list(
      model
    )
  }
}


# ============================================================
# 21. RESULTADOS VALIDATION
# ============================================================

validation_results <- rbindlist(
  validation_results_list
)


# ============================================================
# 22. SELECCIONAR MODELO
#
# Prioridad:
# 1. balanced accuracy
# 2. AUC
# 3. F1
# 4. logloss
#
# TODO SIN TOCAR TEST
# ============================================================

setorder(
  validation_results,
  -validation_balanced,
  -validation_auc,
  -validation_f1,
  validation_logloss
)


cat("\n")
cat("============================================\n")
cat("RESULTADOS VALIDATION\n")
cat("============================================\n\n")


print(
  validation_results
)


best_candidate <- (
  validation_results$candidate[1]
)


best_threshold <- (
  validation_results$threshold[1]
)


best_model <- (
  model_list[
    best_candidate
  ][[1]]
)


cat("\n")
cat("============================================\n")
cat("MODELO SELECCIONADO\n")
cat("============================================\n\n")


cat(
  "Candidato:",
  best_candidate,
  "\n"
)


cat(
  "Threshold:",
  best_threshold,
  "\n\n"
)


# ============================================================
# 23. TEST UNA SOLA VEZ
# ============================================================

test_probability <- predict(
  best_model,
  dtest
)


test_metrics <- evaluate_binary(
  y_test,
  test_probability,
  best_threshold
)


# ============================================================
# 24. BASELINE MAYORITARIO
#
# Se deriva EXCLUSIVAMENTE de TRAIN
# ============================================================

train_majority_class <- as.integer(
  mean(
    y_train
  ) >= 0.5
)


majority_prediction <- rep(
  train_majority_class,
  length(
    y_test
  )
)


majority_accuracy <- mean(
  majority_prediction ==
  y_test
)


# ============================================================
# 25. BASELINE BALANCEADO
# ============================================================

majority_tp <- sum(
  y_test == 1
  &
  majority_prediction == 1
)


majority_tn <- sum(
  y_test == 0
  &
  majority_prediction == 0
)


majority_fp <- sum(
  y_test == 0
  &
  majority_prediction == 1
)


majority_fn <- sum(
  y_test == 1
  &
  majority_prediction == 0
)


majority_sensitivity <- if (
  majority_tp +
  majority_fn > 0
) {

  majority_tp /
  (
    majority_tp +
    majority_fn
  )

} else {

  NA_real_
}


majority_specificity <- if (
  majority_tn +
  majority_fp > 0
) {

  majority_tn /
  (
    majority_tn +
    majority_fp
  )

} else {

  NA_real_
}


majority_balanced <- mean(
  c(
    majority_sensitivity,
    majority_specificity
  ),
  na.rm = TRUE
)


# ============================================================
# 26. MÉTRICAS FINALES TEST
# ============================================================

final_metrics <- data.table(

  metric = c(
    "accuracy",
    "balanced_accuracy",
    "auc",
    "sensitivity_change",
    "specificity_no_change",
    "precision_change",
    "f1_change",
    "logloss",
    "brier_score",
    "majority_baseline_accuracy",
    "majority_baseline_balanced",
    "selected_threshold",
    "test_observations",
    "test_matches"
  ),

  value = c(
    test_metrics$accuracy,
    test_metrics$balanced_accuracy,
    test_metrics$auc,
    test_metrics$sensitivity,
    test_metrics$specificity,
    test_metrics$precision,
    test_metrics$f1,
    test_metrics$logloss,
    test_metrics$brier,
    majority_accuracy,
    majority_balanced,
    best_threshold,
    nrow(test_dt),
    uniqueN(
      test_dt$match_id
    )
  )
)


cat("\n")
cat("============================================\n")
cat("RESULTADOS TEST - CAMBIO TACTICO\n")
cat("============================================\n\n")


print(
  final_metrics
)


# ============================================================
# 27. MATRIZ DE CONFUSIÓN
# ============================================================

confusion_matrix <- table(

  REAL =
    ifelse(
      y_test == 1,
      "CAMBIA",
      "NO_CAMBIA"
    ),

  PREDICTED =
    ifelse(
      test_metrics$prediction == 1,
      "CAMBIA",
      "NO_CAMBIA"
    )
)


cat(
  "\nMatriz de confusión:\n\n"
)


print(
  confusion_matrix
)


# ============================================================
# 28. PREDICCIONES TEST
# ============================================================

test_prediction_label <- ifelse(
  test_metrics$prediction == 1,
  "CAMBIA",
  "NO_CAMBIA"
)


actual_label <- ifelse(
  y_test == 1,
  "CAMBIA",
  "NO_CAMBIA"
)


predictions <- data.table(

  match_id =
    test_dt$match_id,

  match_date =
    test_dt$match_date,

  minute =
    test_dt$minute,

  opponent =
    test_dt$opponent,

  venue =
    test_dt$venue,

  current_state =
    test_dt$current_tactical_state,

  future_state =
    test_dt$future_tactical_state_5min,

  actual_change =
    actual_label,

  predicted_change =
    test_prediction_label,

  probability_change =
    test_probability,

  threshold =
    best_threshold
)


# ============================================================
# 29. GUARDAR MODELO
# ============================================================

xgb.save(
  best_model,
  model_file
)


# ============================================================
# 30. GUARDAR RESULTADOS
# ============================================================

fwrite(
  final_metrics,
  metrics_file
)


fwrite(
  validation_results,
  validation_file
)


fwrite(
  predictions,
  predictions_file
)


feature_export <- merge(
  data.table(
    feature =
      features
  ),
  imputation_table,
  by = "feature",
  all.x = TRUE
)


fwrite(
  feature_export,
  features_file
)


selection_summary <- data.table(

  selected_candidate =
    best_candidate,

  selected_threshold =
    best_threshold,

  validation_accuracy =
    validation_results$validation_accuracy[1],

  validation_balanced =
    validation_results$validation_balanced[1],

  validation_auc =
    validation_results$validation_auc[1],

  validation_f1 =
    validation_results$validation_f1[1],

  validation_logloss =
    validation_results$validation_logloss[1]
)


fwrite(
  selection_summary,
  selection_file
)


# ============================================================
# 31. FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Modelo seleccionado:",
  best_candidate,
  "\n"
)


cat(
  "Threshold seleccionado:",
  best_threshold,
  "\n\n"
)


cat(
  "Modelo:\n",
  model_file,
  "\n\n"
)


cat(
  "Métricas:\n",
  metrics_file,
  "\n\n"
)


cat(
  "Predicciones:\n",
  predictions_file,
  "\n"
)