# ============================================================
# 52_female_horizon_selection.R
# CLUB AMÉRICA FEMENIL
#
# SELECCIÓN EMPÍRICA DEL HORIZONTE
#
# Compara:
# +1 minuto
# +3 minutos
# +5 minutos
#
# IMPORTANTE:
# - Los targets se construyen por minuto exacto.
# - TRAIN entrena.
# - VALIDATION selecciona horizonte/modelo.
# - TEST se utiliza UNA SOLA VEZ al final.
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
  "horizon_selection"
)


if (!dir.exists(output_dir)) {

  dir.create(
    output_dir,
    recursive = TRUE
  )
}


validation_file <- file.path(
  output_dir,
  "horizon_validation_results.csv"
)


test_metrics_file <- file.path(
  output_dir,
  "selected_horizon_test_metrics.csv"
)


predictions_file <- file.path(
  output_dir,
  "selected_horizon_test_predictions.csv"
)


model_file <- file.path(
  output_dir,
  "female_selected_horizon_model.json"
)


features_file <- file.path(
  output_dir,
  "selected_horizon_features.csv"
)


classes_file <- file.path(
  output_dir,
  "selected_horizon_classes.csv"
)


selection_file <- file.path(
  output_dir,
  "selected_horizon_summary.csv"
)


# ============================================================
# 2. CARGAR
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
cat("FEMALE HORIZON SELECTION\n")
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


dt[
  ,
  match_date :=
    as.IDate(
      match_date
    )
]


dt[
  ,
  minute :=
    as.integer(
      minute
    )
]


setorder(
  dt,
  match_date,
  match_id,
  minute
)


# ============================================================
# 4. HORIZONTES A COMPARAR
# ============================================================

horizons <- c(
  1L,
  3L,
  5L
)


cat(
  "Horizontes:",
  paste(
    horizons,
    collapse = ", "
  ),
  "minutos\n\n"
)


# ============================================================
# 5. FUNCIÓN: CREAR TARGET EXACTO A +N MINUTOS
# ============================================================

build_horizon_dataset <- function(
  data,
  horizon
) {

  base <- copy(
    data
  )


  future_lookup <- base[
    ,
    .(
      match_id,
      target_minute =
        minute,

      future_state =
        current_tactical_state
    )
  ]


  base[
    ,
    target_minute :=
      minute + horizon
  ]


  result <- merge(
    base,
    future_lookup,
    by = c(
      "match_id",
      "target_minute"
    ),
    all.x = TRUE,
    sort = FALSE
  )


  result[
    ,
    target_minute := NULL
  ]


  result <- result[
    !is.na(
      future_state
    )
  ]


  setorder(
    result,
    match_date,
    match_id,
    minute
  )


  result
}


# ============================================================
# 6. CREAR ONE-HOT DEL ESTADO ACTUAL
# ============================================================

add_state_features <- function(
  train_data,
  validation_data,
  test_data
) {

  states <- sort(
    unique(
      train_data$current_tactical_state
    )
  )


  feature_names <- character()


  for (state in states) {

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


    feature_names <- c(
      feature_names,
      feature_name
    )


    train_data[
      ,
      (feature_name) :=
        as.integer(
          current_tactical_state ==
            state
        )
    ]


    validation_data[
      ,
      (feature_name) :=
        as.integer(
          current_tactical_state ==
            state
        )
    ]


    test_data[
      ,
      (feature_name) :=
        as.integer(
          current_tactical_state ==
            state
        )
    ]
  }


  list(
    train =
      train_data,

    validation =
      validation_data,

    test =
      test_data,

    state_features =
      feature_names
  )
}


# ============================================================
# 7. EVALUAR PREDICCIONES
# ============================================================

evaluate_predictions <- function(
  probabilities,
  y_true,
  class_levels
) {

  n_classes <- length(
    class_levels
  )


  if (
    ncol(
      probabilities
    ) !=
    n_classes
  ) {

    stop(
      "Número incorrecto de columnas de probabilidad."
    )
  }


  top1_index <- max.col(
    probabilities,
    ties.method = "first"
  )


  actual_state <- class_levels[
    y_true + 1
  ]


  predicted_state <- class_levels[
    top1_index
  ]


  second_index <- apply(
    probabilities,
    1,
    function(values) {

      order(
        values,
        decreasing = TRUE
      )[2]
    }
  )


  second_state <- class_levels[
    second_index
  ]


  top1_accuracy <- mean(
    predicted_state ==
      actual_state
  )


  top2_accuracy <- mean(
    (
      predicted_state ==
        actual_state
    )
    |
    (
      second_state ==
        actual_state
    )
  )


  recall_values <- numeric(
    length(
      class_levels
    )
  )


  for (
    i
    in
    seq_along(
      class_levels
    )
  ) {

    state <- class_levels[i]


    mask <- (
      actual_state ==
        state
    )


    if (sum(mask) == 0) {

      recall_values[i] <- NA_real_

    } else {

      recall_values[i] <- mean(
        predicted_state[
          mask
        ]
        ==
        state
      )
    }
  }


  balanced_accuracy <- mean(
    recall_values,
    na.rm = TRUE
  )


  row_ids <- seq_len(
    nrow(
      probabilities
    )
  )


  true_probability <- probabilities[
    cbind(
      row_ids,
      y_true + 1
    )
  ]


  true_probability <- pmax(
    true_probability,
    .Machine$double.eps
  )


  logloss <- -mean(
    log(
      true_probability
    )
  )


  list(

    top1 =
      top1_accuracy,

    top2 =
      top2_accuracy,

    balanced =
      balanced_accuracy,

    logloss =
      logloss,

    actual =
      actual_state,

    predicted =
      predicted_state,

    second =
      second_state,

    top1_index =
      top1_index,

    second_index =
      second_index,

    recall =
      recall_values
  )
}


# ============================================================
# 8. CONFIGURACIONES XGBOOST
#
# Se seleccionan SOLO mediante VALIDATION.
# ============================================================

candidate_configs <- data.table(

  candidate_id =
    seq_len(6),

  max_depth =
    c(
      3,
      4,
      5,
      3,
      4,
      5
    ),

  eta =
    c(
      0.03,
      0.03,
      0.03,
      0.05,
      0.05,
      0.05
    ),

  min_child_weight =
    c(
      5,
      5,
      5,
      10,
      10,
      10
    ),

  subsample =
    c(
      0.85,
      0.85,
      0.85,
      0.85,
      0.85,
      0.85
    ),

  colsample_bytree =
    c(
      0.85,
      0.85,
      0.85,
      0.85,
      0.85,
      0.85
    )
)


# ============================================================
# 9. CONTENEDORES
# ============================================================

all_validation_results <- list()

trained_models <- list()

model_metadata <- list()


# ============================================================
# 10. LOOP POR HORIZONTE
# ============================================================

for (horizon in horizons) {

  cat("\n")
  cat("============================================\n")
  cat(
    "HORIZONTE +",
    horizon,
    " MIN\n",
    sep = ""
  )
  cat("============================================\n\n")


  horizon_dt <- build_horizon_dataset(
    dt,
    horizon
  )


  train_dt <- copy(
    horizon_dt[
      split == "TRAIN"
    ]
  )


  validation_dt <- copy(
    horizon_dt[
      split == "VALIDATION"
    ]
  )


  test_dt <- copy(
    horizon_dt[
      split == "TEST"
    ]
  )


  cat(
    "TRAIN:",
    nrow(train_dt),
    "\n"
  )


  cat(
    "VALIDATION:",
    nrow(validation_dt),
    "\n"
  )


  cat(
    "TEST:",
    nrow(test_dt),
    "\n\n"
  )


  if (
    nrow(train_dt) == 0
    ||
    nrow(validation_dt) == 0
    ||
    nrow(test_dt) == 0
  ) {

    stop(
      paste(
        "Split vacío para horizonte",
        horizon
      )
    )
  }


  # ==========================================================
  # 10.1 CLASES DESDE TRAIN
  # ==========================================================

  class_levels <- sort(
    unique(
      train_dt$future_state
    )
  )


  n_classes <- length(
    class_levels
  )


  if (n_classes < 2) {

    stop(
      paste(
        "Muy pocas clases en horizonte",
        horizon
      )
    )
  }


  validation_unknown <- setdiff(
    unique(
      validation_dt$future_state
    ),
    class_levels
  )


  test_unknown <- setdiff(
    unique(
      test_dt$future_state
    ),
    class_levels
  )


  if (length(validation_unknown) > 0) {

    stop(
      paste(
        "VALIDATION contiene clases nuevas en horizonte",
        horizon
      )
    )
  }


  if (length(test_unknown) > 0) {

    stop(
      paste(
        "TEST contiene clases nuevas en horizonte",
        horizon
      )
    )
  }


  # ==========================================================
  # 10.2 TARGET ID
  # ==========================================================

  train_dt[
    ,
    target_id :=
      match(
        future_state,
        class_levels
      ) - 1
  ]


  validation_dt[
    ,
    target_id :=
      match(
        future_state,
        class_levels
      ) - 1
  ]


  test_dt[
    ,
    target_id :=
      match(
        future_state,
        class_levels
      ) - 1
  ]


  # ==========================================================
  # 10.3 ESTADO ACTUAL ONE-HOT
  # ==========================================================

  state_result <- add_state_features(
    train_dt,
    validation_dt,
    test_dt
  )


  train_dt <- state_result$train

  validation_dt <- state_result$validation

  test_dt <- state_result$test

  state_features <- (
    state_result$state_features
  )


  # ==========================================================
  # 10.4 FEATURES
  # ==========================================================

  rolling_features <- grep(
    "_last3$|_last5$|_trend$",
    names(train_dt),
    value = TRUE
  )


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


  features <- unique(
    c(
      "minute",
      "game_state_num",
      state_features,
      rolling_features
    )
  )


  # ==========================================================
  # 10.5 BLOQUEAR LEAKAGE
  # ==========================================================

  forbidden_patterns <- c(
    "future_",
    "target",
    "usable_",
    "cluster_id",
    "same_state",
    "_score",
    "tactical_state_tie"
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
    features
    %in%
    names(train_dt)
  ]


  # ==========================================================
  # 10.6 NUMÉRICO
  # ==========================================================

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


  # ==========================================================
  # 10.7 IMPUTACIÓN DESDE TRAIN
  # ==========================================================

  imputation_table <- data.table(
    feature =
      features,

    train_median =
      NA_real_
  )


  for (
    i
    in
    seq_along(
      features
    )
  ) {

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


  # ==========================================================
  # 10.8 MATRICES
  # ==========================================================

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


  y_train <- train_dt$target_id

  y_validation <- validation_dt$target_id

  y_test <- test_dt$target_id


  # ==========================================================
  # 10.9 PESOS DE CLASE DESDE TRAIN
  # ==========================================================

  class_counts <- table(
    train_dt$future_state
  )


  class_frequency <- (
    class_counts /
    sum(
      class_counts
    )
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
      train_dt$future_state
    ]
  )


  # ==========================================================
  # 10.10 DMATRIX
  # ==========================================================

  dtrain_plain <- xgb.DMatrix(
    data =
      x_train,

    label =
      y_train
  )


  dtrain_weighted <- xgb.DMatrix(
    data =
      x_train,

    label =
      y_train,

    weight =
      train_weights
  )


  dvalidation <- xgb.DMatrix(
    data =
      x_validation,

    label =
      y_validation
  )


  dtest <- xgb.DMatrix(
    data =
      x_test,

    label =
      y_test
  )


  # ==========================================================
  # 10.11 CANDIDATOS
  # ==========================================================

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
        "H",
        horizon,
        "_C",
        config$candidate_id,
        ifelse(
          weighted,
          "_W",
          "_U"
        )
      )


      cat(
        "Entrenando:",
        candidate_name,
        "\n"
      )


      params <- list(

        objective =
          "multi:softprob",

        num_class =
          n_classes,

        eval_metric =
          "mlogloss",

        max_depth =
          config$max_depth,

        eta =
          config$eta,

        min_child_weight =
          config$min_child_weight,

        subsample =
          config$subsample,

        colsample_bytree =
          config$colsample_bytree,

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


      validation_raw <- predict(
        model,
        dvalidation
      )


      validation_probabilities <- matrix(
        validation_raw,
        ncol =
          n_classes,
        byrow =
          TRUE
      )


      validation_metrics <- evaluate_predictions(
        validation_probabilities,
        y_validation,
        class_levels
      )


      result_row <- data.table(

        candidate =
          candidate_name,

        horizon =
          horizon,

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

        validation_top1 =
          validation_metrics$top1,

        validation_top2 =
          validation_metrics$top2,

        validation_balanced =
          validation_metrics$balanced,

        validation_logloss =
          validation_metrics$logloss
      )


      all_validation_results <- append(
        all_validation_results,
        list(
          result_row
        )
      )


      trained_models[
        candidate_name
      ] <- list(
        model
      )


      model_metadata[
        candidate_name
      ] <- list(
        list(

          horizon =
            horizon,

          class_levels =
            class_levels,

          features =
            features,

          imputation =
            imputation_table,

          test_dt =
            test_dt,

          y_test =
            y_test,

          dtest =
            dtest,

          class_weights =
            class_weights
        )
      )
    }
  }
}


# ============================================================
# 11. UNIR RESULTADOS VALIDATION
# ============================================================

validation_results <- rbindlist(
  all_validation_results,
  fill = TRUE
)


# ============================================================
# 12. SELECCIÓN FINAL
#
# PRIORIDAD:
# 1. BALANCED ACCURACY
# 2. TOP-1
# 3. TOP-2
# 4. LOGLOSS
#
# TODO usando únicamente VALIDATION.
# ============================================================

setorder(
  validation_results,
  -validation_balanced,
  -validation_top1,
  -validation_top2,
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


selected_horizon <- (
  validation_results$horizon[1]
)


cat("\n")
cat("============================================\n")
cat("SELECCIÓN\n")
cat("============================================\n\n")


cat(
  "Mejor candidato:",
  best_candidate,
  "\n"
)


cat(
  "Horizonte seleccionado:",
  selected_horizon,
  "minutos\n\n"
)


# ============================================================
# 13. RECUPERAR MODELO SELECCIONADO
# ============================================================

best_model <- (
  trained_models[
    best_candidate
  ][[1]]
)


metadata <- (
  model_metadata[
    best_candidate
  ][[1]]
)


class_levels <- metadata$class_levels

features <- metadata$features

test_dt <- metadata$test_dt

y_test <- metadata$y_test

dtest <- metadata$dtest

imputation_table <- metadata$imputation


n_classes <- length(
  class_levels
)


# ============================================================
# 14. EVALUAR TEST UNA SOLA VEZ
# ============================================================

test_raw <- predict(
  best_model,
  dtest
)


test_probabilities <- matrix(
  test_raw,
  ncol =
    n_classes,
  byrow =
    TRUE
)


test_metrics <- evaluate_predictions(
  test_probabilities,
  y_test,
  class_levels
)


# ============================================================
# 15. BASELINE MAYORITARIO
#
# Calculado usando TRAIN del horizonte seleccionado.
# ============================================================

selected_dataset <- build_horizon_dataset(
  dt,
  selected_horizon
)


selected_train <- selected_dataset[
  split == "TRAIN"
]


majority_state <- names(
  sort(
    table(
      selected_train$future_state
    ),
    decreasing = TRUE
  )
)[1]


majority_baseline <- mean(
  test_metrics$actual ==
    majority_state
)


# ============================================================
# 16. BASELINE PERSISTENCIA
# ============================================================

persistence_baseline <- mean(
  test_dt$current_tactical_state ==
    test_metrics$actual
)


# ============================================================
# 17. BASELINE TRANSICIÓN
# ============================================================

transition_table <- selected_train[
  ,
  .(
    observations = .N
  ),
  by = .(
    current_tactical_state,
    future_state
  )
]


setorder(
  transition_table,
  current_tactical_state,
  -observations
)


transition_rule <- transition_table[
  ,
  .SD[1],
  by = current_tactical_state
]


transition_prediction <- character(
  nrow(
    test_dt
  )
)


for (
  i
  in
  seq_len(
    nrow(
      test_dt
    )
  )
) {

  state <- (
    test_dt$current_tactical_state[i]
  )


  predicted_rule <- transition_rule[
    current_tactical_state ==
      state,
    future_state
  ]


  if (length(predicted_rule) == 0) {

    transition_prediction[i] <- (
      majority_state
    )

  } else {

    transition_prediction[i] <- (
      predicted_rule[1]
    )
  }
}


transition_baseline <- mean(
  transition_prediction ==
    test_metrics$actual
)


# ============================================================
# 18. RESULTADOS TEST
# ============================================================

final_metrics <- data.table(

  metric = c(
    "selected_horizon_minutes",
    "top1_accuracy",
    "top2_accuracy",
    "majority_baseline",
    "persistence_baseline",
    "transition_baseline",
    "balanced_accuracy",
    "logloss",
    "test_observations",
    "test_matches"
  ),

  value = c(
    selected_horizon,
    test_metrics$top1,
    test_metrics$top2,
    majority_baseline,
    persistence_baseline,
    transition_baseline,
    test_metrics$balanced,
    test_metrics$logloss,
    nrow(test_dt),
    uniqueN(
      test_dt$match_id
    )
  )
)


cat("\n")
cat("============================================\n")
cat("RESULTADOS TEST - HORIZONTE SELECCIONADO\n")
cat("============================================\n\n")


print(
  final_metrics
)


# ============================================================
# 19. RECALL POR ESTADO
# ============================================================

recall_table <- data.table(

  tactical_state =
    class_levels,

  recall =
    test_metrics$recall
)


cat(
  "\nRecall por estado:\n\n"
)


print(
  recall_table
)


# ============================================================
# 20. MATRIZ DE CONFUSIÓN
# ============================================================

confusion_matrix <- table(

  REAL =
    test_metrics$actual,

  PREDICTED =
    test_metrics$predicted
)


cat(
  "\nMatriz de confusión:\n\n"
)


print(
  confusion_matrix
)


# ============================================================
# 21. PREDICCIONES TEST
# ============================================================

row_ids <- seq_len(
  nrow(
    test_probabilities
  )
)


top1_probability <- test_probabilities[
  cbind(
    row_ids,
    test_metrics$top1_index
  )
]


second_probability <- test_probabilities[
  cbind(
    row_ids,
    test_metrics$second_index
  )
]


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

  horizon_minutes =
    selected_horizon,

  current_state =
    test_dt$current_tactical_state,

  actual_future_state =
    test_metrics$actual,

  predicted_state =
    test_metrics$predicted,

  second_state =
    test_metrics$second,

  confidence =
    top1_probability,

  second_probability =
    second_probability
)


# ============================================================
# 22. GUARDAR
# ============================================================

xgb.save(
  best_model,
  model_file
)


fwrite(
  validation_results,
  validation_file
)


fwrite(
  final_metrics,
  test_metrics_file
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


fwrite(
  data.table(

    class_id =
      seq_along(
        class_levels
      ) - 1,

    tactical_state =
      class_levels
  ),
  classes_file
)


selection_summary <- data.table(

  selected_candidate =
    best_candidate,

  selected_horizon_minutes =
    selected_horizon,

  validation_top1 =
    validation_results$validation_top1[1],

  validation_top2 =
    validation_results$validation_top2[1],

  validation_balanced =
    validation_results$validation_balanced[1],

  validation_logloss =
    validation_results$validation_logloss[1]
)


fwrite(
  selection_summary,
  selection_file
)


# ============================================================
# 23. FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Horizonte elegido:",
  selected_horizon,
  "minutos\n"
)


cat(
  "Modelo elegido:",
  best_candidate,
  "\n\n"
)


cat(
  "Modelo guardado:\n",
  model_file,
  "\n\n"
)


cat(
  "Métricas TEST:\n",
  test_metrics_file,
  "\n"
)