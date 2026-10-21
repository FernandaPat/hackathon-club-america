# ============================================================
# 48_female_xgboost.R
# CLUB AMÉRICA FEMENIL
# XGBOOST MULTICLASE - VERSIÓN MEJORADA
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
  "model_ready_female.csv"
)

model_dir <- file.path(
  project_root,
  "outputs",
  "models",
  "female"
)

if (!dir.exists(model_dir)) {

  dir.create(
    model_dir,
    recursive = TRUE
  )
}


model_file <- file.path(
  model_dir,
  "female_xgboost_model.json"
)

metrics_file <- file.path(
  model_dir,
  "female_model_metrics.csv"
)

predictions_file <- file.path(
  model_dir,
  "female_test_predictions.csv"
)

features_file <- file.path(
  model_dir,
  "female_model_features.csv"
)

classes_file <- file.path(
  model_dir,
  "female_model_classes.csv"
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
cat("FEMALE XGBOOST - IMPROVED\n")
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
# 5. CLASES
#
# Se derivan únicamente de TRAIN.
# ============================================================

state_levels <- sort(
  unique(
    train_dt$current_tactical_state
  )
)


target_levels <- sort(
  unique(
    train_dt$future_tactical_state_5min
  )
)


cat(
  "Estados actuales TRAIN:\n"
)

print(
  state_levels
)


cat(
  "\nTarget TRAIN:\n"
)

print(
  target_levels
)


# ============================================================
# 6. VALIDAR QUE VALIDATION / TEST NO TENGAN CLASES NUEVAS
# ============================================================

validation_unknown <- setdiff(
  unique(
    validation_dt$future_tactical_state_5min
  ),
  target_levels
)


test_unknown <- setdiff(
  unique(
    test_dt$future_tactical_state_5min
  ),
  target_levels
)


if (length(validation_unknown) > 0) {

  stop(
    paste(
      "VALIDATION contiene clases no observadas en TRAIN:",
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
      "TEST contiene clases no observadas en TRAIN:",
      paste(
        test_unknown,
        collapse = ", "
      )
    )
  )
}


n_classes <- length(
  target_levels
)


# ============================================================
# 7. TARGET NUMÉRICO
# ============================================================

train_dt[
  ,
  target_id :=
    match(
      future_tactical_state_5min,
      target_levels
    )
    -
    1
]


validation_dt[
  ,
  target_id :=
    match(
      future_tactical_state_5min,
      target_levels
    )
    -
    1
]


test_dt[
  ,
  target_id :=
    match(
      future_tactical_state_5min,
      target_levels
    )
    -
    1
]


# ============================================================
# 8. ONE-HOT DEL ESTADO ACTUAL
#
# Las columnas se generan desde los estados encontrados
# realmente en TRAIN.
# ============================================================

state_feature_names <- character()


for (state in state_levels) {

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


  state_feature_names <- c(
    state_feature_names,
    feature_name
  )


  train_dt[
    ,
    (feature_name) :=
      as.integer(
        current_tactical_state
        ==
        state
      )
  ]


  validation_dt[
    ,
    (feature_name) :=
      as.integer(
        current_tactical_state
        ==
        state
      )
  ]


  test_dt[
    ,
    (feature_name) :=
      as.integer(
        current_tactical_state
        ==
        state
      )
  ]
}


cat(
  "\nFeatures de estado actual:\n"
)


print(
  state_feature_names
)


# ============================================================
# 9. SELECCIONAR FEATURES DINÁMICAMENTE
#
# Se utilizan:
# - minuto
# - marcador LIVE
# - estado táctico actual
# - rolling last3
# - rolling last5
# - trends
#
# NO usamos scores finales del partido.
# NO usamos target.
# NO usamos tactical scores utilizados para crear etiquetas.
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
    state_feature_names,
    rolling_features
  )
)


# ============================================================
# 10. EXCLUSIONES DE SEGURIDAD
# ============================================================

forbidden_patterns <- c(
  "future_",
  "target",
  "usable_",
  "state_changes",
  "tactical_state_tie",
  "control_score",
  "pressure_score",
  "transition_score",
  "protection_score",
  "urgency_score"
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


cat(
  "\nFeatures finales:",
  length(features),
  "\n\n"
)


print(
  features
)


if (length(features) < 5) {

  stop(
    "No existen suficientes features."
  )
}


# ============================================================
# 11. CONVERTIR NUMÉRICO
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
# 12. IMPUTACIÓN
#
# Median calculada EXCLUSIVAMENTE con TRAIN.
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


y_train <- train_dt$target_id
y_validation <- validation_dt$target_id
y_test <- test_dt$target_id


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
# 14. MODELO
# ============================================================

params <- list(
  objective = "multi:softprob",
  num_class = n_classes,
  eval_metric = "mlogloss",
  max_depth = 5,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)


model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1500,

  evals = list(
    train = dtrain,
    validation = dvalidation
  ),

  early_stopping_rounds = 75,

  verbose = 1
)


# ============================================================
# 15. PREDICCIÓN
# ============================================================

pred_raw <- predict(
  model,
  dtest
)


pred_matrix <- matrix(
  pred_raw,
  ncol = n_classes,
  byrow = TRUE
)


# ============================================================
# 16. TOP-1
# ============================================================

top1_index <- max.col(
  pred_matrix,
  ties.method = "first"
)


predicted_state <- target_levels[
  top1_index
]


actual_state <- target_levels[
  y_test + 1
]


# ============================================================
# 17. TOP-2
# ============================================================

second_index <- apply(
  pred_matrix,
  1,
  function(row_values) {

    order(
      row_values,
      decreasing = TRUE
    )[2]
  }
)


second_state <- target_levels[
  second_index
]


# ============================================================
# 18. PROBABILIDADES
# ============================================================

row_ids <- seq_len(
  nrow(
    pred_matrix
  )
)


top1_probability <- pred_matrix[
  cbind(
    row_ids,
    top1_index
  )
]


second_probability <- pred_matrix[
  cbind(
    row_ids,
    second_index
  )
]


# ============================================================
# 19. TOP-1 ACCURACY
# ============================================================

top1_accuracy <- mean(
  predicted_state
  ==
  actual_state
)


# ============================================================
# 20. TOP-2 ACCURACY
# ============================================================

top2_accuracy <- mean(
  (
    predicted_state
    ==
    actual_state
  )
  |
  (
    second_state
    ==
    actual_state
  )
)


# ============================================================
# 21. MAJORITY BASELINE
#
# La clase mayoritaria se calcula SOLO desde TRAIN.
# ============================================================

majority_table <- sort(
  table(
    train_dt$future_tactical_state_5min
  ),
  decreasing = TRUE
)


majority_state <- names(
  majority_table
)[1]


majority_baseline <- mean(
  actual_state
  ==
  majority_state
)


# ============================================================
# 22. PERSISTENCE BASELINE
#
# Predice que el estado futuro será igual al actual.
# ============================================================

persistence_prediction <- (
  test_dt$current_tactical_state
)


persistence_baseline <- mean(
  persistence_prediction
  ==
  actual_state
)


# ============================================================
# 23. TRANSITION BASELINE
#
# Aprende de TRAIN cuál es el estado futuro más común
# condicionado al estado táctico actual.
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
  by = current_tactical_state
]


setnames(
  transition_rule,
  "future_tactical_state_5min",
  "transition_prediction"
)


test_transition <- merge(
  test_dt[
    ,
    .(
      row_id = .I,
      current_tactical_state
    )
  ],

  transition_rule[
    ,
    .(
      current_tactical_state,
      transition_prediction
    )
  ],

  by = "current_tactical_state",
  all.x = TRUE,
  sort = FALSE
)


setorder(
  test_transition,
  row_id
)


test_transition[
  is.na(
    transition_prediction
  ),
  transition_prediction :=
    majority_state
]


transition_baseline <- mean(
  test_transition$transition_prediction
  ==
  actual_state
)


# ============================================================
# 24. BALANCED ACCURACY
# ============================================================

recall_table <- data.table(
  tactical_state =
    target_levels,

  recall =
    NA_real_
)


for (i in seq_along(target_levels)) {

  state <- target_levels[i]


  mask <- (
    actual_state
    ==
    state
  )


  if (sum(mask) > 0) {

    recall_table[
      i,
      recall :=
        mean(
          predicted_state[
            mask
          ]
          ==
          state
        )
    ]
  }
}


balanced_accuracy <- mean(
  recall_table$recall,
  na.rm = TRUE
)


# ============================================================
# 25. LOG LOSS
# ============================================================

actual_probability <- pred_matrix[
  cbind(
    row_ids,
    y_test + 1
  )
]


actual_probability <- pmax(
  actual_probability,
  .Machine$double.eps
)


logloss <- -mean(
  log(
    actual_probability
  )
)


# ============================================================
# 26. MÉTRICAS
# ============================================================

metrics <- data.table(

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
    top1_accuracy,
    top2_accuracy,
    majority_baseline,
    persistence_baseline,
    transition_baseline,
    balanced_accuracy,
    logloss,
    nrow(
      test_dt
    ),
    uniqueN(
      test_dt$match_id
    )
  )
)


cat("\n")
cat("============================================\n")
cat("RESULTADOS FEMENIL - IMPROVED\n")
cat("============================================\n\n")


print(
  metrics
)


cat(
  "\nRecall por estado:\n\n"
)


print(
  recall_table
)


# ============================================================
# 27. MATRIZ DE CONFUSIÓN
# ============================================================

confusion <- table(
  REAL =
    actual_state,

  PREDICTED =
    predicted_state
)


cat(
  "\nMatriz de confusión:\n\n"
)


print(
  confusion
)


# ============================================================
# 28. PREDICCIONES
# ============================================================

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
    actual_state,

  predicted_state =
    predicted_state,

  second_state =
    second_state,

  confidence =
    top1_probability,

  second_probability =
    second_probability
)


# ============================================================
# 29. GUARDAR MODELO
# ============================================================

xgb.save(
  model,
  model_file
)


# ============================================================
# 30. GUARDAR MÉTRICAS
# ============================================================

fwrite(
  metrics,
  metrics_file
)


fwrite(
  predictions,
  predictions_file
)


# ============================================================
# 31. FEATURES
# ============================================================

feature_export <- data.table(
  feature =
    features
)


feature_export <- merge(
  feature_export,
  imputation_table,
  by = "feature",
  all.x = TRUE
)


fwrite(
  feature_export,
  features_file
)


# ============================================================
# 32. CLASES
# ============================================================

class_export <- data.table(

  class_id =
    seq_along(
      target_levels
    )
    -
    1,

  tactical_state =
    target_levels
)


fwrite(
  class_export,
  classes_file
)


# ============================================================
# 33. FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Best iteration:",
  model$best_iteration,
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