# ============================================================
# 51_female_xgboost_v2.R
# CLUB AMÉRICA FEMENIL
#
# XGBOOST V2
# - Usa estados construidos por clustering
# - Selección usando VALIDATION
# - TEST permanece aislado
# - Compara modelos normales y balanceados
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
  "v2"
)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}


model_file <- file.path(
  output_dir,
  "female_xgboost_v2.json"
)

metrics_file <- file.path(
  output_dir,
  "female_v2_metrics.csv"
)

validation_file <- file.path(
  output_dir,
  "female_v2_validation_candidates.csv"
)

predictions_file <- file.path(
  output_dir,
  "female_v2_test_predictions.csv"
)

features_file <- file.path(
  output_dir,
  "female_v2_features.csv"
)

classes_file <- file.path(
  output_dir,
  "female_v2_classes.csv"
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
cat("FEMALE XGBOOST V2\n")
cat("============================================\n\n")


cat(
  "Observaciones:",
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
# 4. SPLITS
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
    "Alguno de los splits está vacío."
  )
}


cat(
  "TRAIN:",
  nrow(train_dt),
  "/",
  uniqueN(train_dt$match_id),
  "partidos\n"
)

cat(
  "VALIDATION:",
  nrow(validation_dt),
  "/",
  uniqueN(validation_dt$match_id),
  "partidos\n"
)

cat(
  "TEST:",
  nrow(test_dt),
  "/",
  uniqueN(test_dt$match_id),
  "partidos\n\n"
)


# ============================================================
# 5. CLASES DEFINIDAS DESDE TRAIN
# ============================================================

class_levels <- sort(
  unique(
    train_dt$future_tactical_state_5min
  )
)


n_classes <- length(
  class_levels
)


cat(
  "Clases:",
  n_classes,
  "\n"
)

print(
  class_levels
)


if (n_classes < 2) {

  stop(
    "No existen suficientes clases."
  )
}


# ============================================================
# 6. VALIDAR CLASES DE VALIDATION / TEST
# ============================================================

validation_unknown <- setdiff(
  unique(
    validation_dt$future_tactical_state_5min
  ),
  class_levels
)


test_unknown <- setdiff(
  unique(
    test_dt$future_tactical_state_5min
  ),
  class_levels
)


if (length(validation_unknown) > 0) {

  stop(
    paste(
      "VALIDATION contiene clases nuevas:",
      paste(
        validation_unknown,
        collapse = ", "
      )
    )
  )
}


if (length(test_unknown) > 0) {

  stop(
    paste(
      "TEST contiene clases nuevas:",
      paste(
        test_unknown,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 7. TARGET NUMÉRICO
# ============================================================

train_dt[
  ,
  target_id :=
    match(
      future_tactical_state_5min,
      class_levels
    ) - 1
]


validation_dt[
  ,
  target_id :=
    match(
      future_tactical_state_5min,
      class_levels
    ) - 1
]


test_dt[
  ,
  target_id :=
    match(
      future_tactical_state_5min,
      class_levels
    ) - 1
]


# ============================================================
# 8. ONE-HOT ESTADO ACTUAL
# ============================================================

current_states <- sort(
  unique(
    train_dt$current_tactical_state
  )
)


state_features <- character()


for (state in current_states) {

  clean_name <- tolower(
    gsub(
      "[^A-Za-z0-9]+",
      "_",
      state
    )
  )


  feature_name <- paste0(
    "current_state_",
    clean_name
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


rolling_features <- rolling_features[
  vapply(
    train_dt[
      ,
      ..rolling_features
    ],
    is.numeric,
    logical(1)
  )
]


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


cat(
  "\nFeatures utilizadas:",
  length(features),
  "\n\n"
)


print(
  features
)


# ============================================================
# 11. NUMÉRICO
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
# 12. IMPUTACIÓN DESDE TRAIN
# ============================================================

imputation_dt <- data.table(
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


  imputation_dt[
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


y_train <- train_dt$target_id
y_validation <- validation_dt$target_id
y_test <- test_dt$target_id


# ============================================================
# 14. PESOS DERIVADOS DE TRAIN
# ============================================================

class_counts <- table(
  train_dt$future_tactical_state_5min
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
    train_dt$future_tactical_state_5min
  ]
)


cat(
  "\nPesos de clase derivados de TRAIN:\n"
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
# 16. FUNCIONES DE EVALUACIÓN
# ============================================================

probability_matrix <- function(
  model,
  dmatrix,
  classes
) {

  raw <- predict(
    model,
    dmatrix
  )


  matrix(
    raw,
    ncol = classes,
    byrow = TRUE
  )
}


evaluate_predictions <- function(
  probabilities,
  y_true,
  levels
) {

  top1_index <- max.col(
    probabilities,
    ties.method = "first"
  )


  top1_state <- levels[
    top1_index
  ]


  actual_state <- levels[
    y_true + 1
  ]


  second_index <- apply(
    probabilities,
    1,
    function(x) {

      order(
        x,
        decreasing = TRUE
      )[2]
    }
  )


  second_state <- levels[
    second_index
  ]


  top1 <- mean(
    top1_state ==
      actual_state
  )


  top2 <- mean(
    (
      top1_state ==
      actual_state
    )
    |
    (
      second_state ==
      actual_state
    )
  )


  recalls <- numeric(
    length(levels)
  )


  for (i in seq_along(levels)) {

    state <- levels[i]

    mask <- (
      actual_state ==
      state
    )


    if (sum(mask) == 0) {

      recalls[i] <- NA_real_

    } else {

      recalls[i] <- mean(
        top1_state[
          mask
        ] == state
      )
    }
  }


  balanced <- mean(
    recalls,
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
    top1 = top1,
    top2 = top2,
    balanced = balanced,
    logloss = logloss,
    predicted = top1_state,
    second = second_state,
    actual = actual_state,
    top1_index = top1_index,
    second_index = second_index
  )
}


# ============================================================
# 17. CONFIGURACIONES CANDIDATAS
# ============================================================

candidate_configs <- data.table(

  candidate_id = seq_len(8),

  max_depth = c(
    3,
    4,
    5,
    6,
    3,
    4,
    5,
    6
  ),

  eta = c(
    0.03,
    0.03,
    0.03,
    0.03,
    0.05,
    0.05,
    0.05,
    0.05
  ),

  min_child_weight = c(
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
# 18. ENTRENAR CANDIDATOS
# ============================================================

results_list <- list()

models_list <- list()


for (i in seq_len(nrow(candidate_configs))) {

  config <- candidate_configs[i]


  for (weighted in c(FALSE, TRUE)) {

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
      "\nEntrenando",
      candidate_name,
      "...\n"
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
        0.85,

      colsample_bytree =
        0.85,

      lambda =
        1
    )


    train_matrix <- if (
      weighted
    ) {
      dtrain_weighted
    } else {
      dtrain_plain
    }


    model <- xgb.train(

      params = params,

      data =
        train_matrix,

      nrounds =
        1500,

      evals = list(
        train =
          train_matrix,
        validation =
          dvalidation
      ),

      early_stopping_rounds =
        75,

      verbose =
        0
    )


    validation_probabilities <- (
      probability_matrix(
        model,
        dvalidation,
        n_classes
      )
    )


    validation_metrics <- (
      evaluate_predictions(
        validation_probabilities,
        y_validation,
        class_levels
      )
    )


    results_list <- append(
      results_list,
      list(
        data.table(

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

          validation_top1 =
            validation_metrics$top1,

          validation_top2 =
            validation_metrics$top2,

          validation_balanced =
            validation_metrics$balanced,

          validation_logloss =
            validation_metrics$logloss
        )
      )
    )


    models_list[
      candidate_name
    ] <- list(
      model
    )
  }
}


candidate_results <- rbindlist(
  results_list
)


# ============================================================
# 19. SELECCIONAR MODELO SOLO CON VALIDATION
#
# Prioridad:
# 1. balanced accuracy
# 2. Top-1
# 3. logloss
# ============================================================

setorder(
  candidate_results,
  -validation_balanced,
  -validation_top1,
  validation_logloss
)


cat("\n")
cat("============================================\n")
cat("RESULTADOS VALIDATION\n")
cat("============================================\n\n")


print(
  candidate_results
)


best_candidate <- (
  candidate_results$candidate[1]
)


best_model <- (
  models_list[
    best_candidate
  ][[1]]
)


cat(
  "\nModelo seleccionado:",
  best_candidate,
  "\n"
)


# ============================================================
# 20. EVALUAR UNA SOLA VEZ EN TEST
# ============================================================

test_probabilities <- probability_matrix(
  best_model,
  dtest,
  n_classes
)


test_metrics <- evaluate_predictions(
  test_probabilities,
  y_test,
  class_levels
)


# ============================================================
# 21. BASELINE MAYORITARIO
# ============================================================

majority_state <- names(
  sort(
    table(
      train_dt$future_tactical_state_5min
    ),
    decreasing = TRUE
  )
)[1]


majority_baseline <- mean(
  test_metrics$actual ==
    majority_state
)


# ============================================================
# 22. BASELINE PERSISTENCIA
# ============================================================

persistence_baseline <- mean(
  test_dt$current_tactical_state ==
    test_metrics$actual
)


# ============================================================
# 23. BASELINE DE TRANSICIÓN
# ============================================================

transition_table <- train_dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    current_tactical_state,
    future_tactical_state_5min
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
  by =
    current_tactical_state
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

  current_state <- (
    test_dt$current_tactical_state[i]
  )


  rule <- transition_rule[
    current_tactical_state ==
      current_state,
    future_tactical_state_5min
  ]


  if (length(rule) == 0) {

    transition_prediction[i] <- (
      majority_state
    )

  } else {

    transition_prediction[i] <- (
      rule[1]
    )
  }
}


transition_baseline <- mean(
  transition_prediction ==
    test_metrics$actual
)


# ============================================================
# 24. MÉTRICAS TEST
# ============================================================

final_metrics <- data.table(

  metric = c(
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
cat("RESULTADOS TEST FEMENIL V2\n")
cat("============================================\n\n")


print(
  final_metrics
)


# ============================================================
# 25. RECALL POR ESTADO
# ============================================================

recall_table <- data.table(
  tactical_state =
    class_levels,
  recall =
    NA_real_
)


for (i in seq_along(class_levels)) {

  state <- class_levels[i]

  mask <- (
    test_metrics$actual ==
      state
  )


  if (sum(mask) > 0) {

    recall_table[
      i,
      recall :=
        mean(
          test_metrics$predicted[
            mask
          ] == state
        )
    ]
  }
}


cat(
  "\nRecall por estado:\n\n"
)


print(
  recall_table
)


# ============================================================
# 26. PREDICCIONES TEST
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
# 27. EXPORTAR
# ============================================================

xgb.save(
  best_model,
  model_file
)


fwrite(
  final_metrics,
  metrics_file
)


fwrite(
  candidate_results,
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
  imputation_dt,
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


# ============================================================
# 28. FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Mejor candidato:",
  best_candidate,
  "\n"
)


cat(
  "Modelo:\n",
  model_file,
  "\n\n"
)


cat(
  "Métricas:\n",
  metrics_file,
  "\n"
)