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
cat("STATE-CONDITIONAL EXPERTS\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. CLASES
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
# 3. FEATURES
# ============================================================

features <- c(

  # Contexto
  "minute",
  "game_state_num",

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

  # 360 - 5 min
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 360 - 3 min
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
# 5. SOLO TRAIN + VALIDATION
#
# TEST NO SE TOCA.
# ============================================================

development_dt <- dt[
  dataset_split %in%
    c(
      "TRAIN",
      "VALIDATION"
    )
]


# ============================================================
# 6. GRID
# ============================================================

parameter_grid <- CJ(

  max_depth = c(
    2L,
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
# 7. FORMATEAR PROBABILIDADES
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


  if (
    nrow(probs) != n_rows ||
    ncol(probs) != length(state_levels)
  ) {

    stop(
      paste(
        "Dimensiones incorrectas:",
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
# 8. MULTICLASS LOG-LOSS
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
# 9. MACRO F1
# ============================================================

macro_f1 <- function(
  truth,
  prediction
) {

  f1_values <- sapply(
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


  mean(
    f1_values
  )
}


# ============================================================
# 10. BALANCED ACCURACY
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
# 11. TOP-2 ACCURACY
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
      probabilities[i, ],
      decreasing = TRUE
    )[1:2]


    top2_states <- state_levels[
      top2
    ]


    correct[i] <-
      truth[i] %in%
      top2_states
  }


  mean(
    correct
  )
}


# ============================================================
# 12. CONTENEDORES
# ============================================================

expert_predictions_list <- vector(
  "list",
  length(state_levels)
)

names(
  expert_predictions_list
) <- state_levels


expert_metrics_list <- vector(
  "list",
  length(state_levels)
)

names(
  expert_metrics_list
) <- state_levels


expert_tuning_list <- vector(
  "list",
  length(state_levels)
)

names(
  expert_tuning_list
) <- state_levels


expert_models <- vector(
  "list",
  length(state_levels)
)

names(
  expert_models
) <- state_levels


# ============================================================
# 13. LOOP POR ESTADO ACTUAL
# ============================================================

for (
  state_index in seq_along(
    state_levels
  )
) {

  current_state <- state_levels[
    state_index
  ]


  cat(
    "\n=====================================\n"
  )

  cat(
    "ENTRENANDO EXPERTO:",
    current_state,
    "\n"
  )

  cat(
    "=====================================\n"
  )


  # ----------------------------------------------------------
  # TRAIN
  # ----------------------------------------------------------

  expert_train <- copy(
    development_dt[
      dataset_split == "TRAIN" &
      as.character(
        current_tactical_state
      ) == current_state
    ]
  )


  # ----------------------------------------------------------
  # VALIDATION
  # ----------------------------------------------------------

  expert_validation <- copy(
    development_dt[
      dataset_split == "VALIDATION" &
      as.character(
        current_tactical_state
      ) == current_state
    ]
  )


  cat(
    "Train rows:",
    nrow(expert_train),
    "\n"
  )

  cat(
    "Validation rows:",
    nrow(expert_validation),
    "\n"
  )


  if (nrow(expert_train) == 0) {

    stop(
      paste(
        "TRAIN vacío para",
        current_state
      )
    )
  }


  if (nrow(expert_validation) == 0) {

    stop(
      paste(
        "VALIDATION vacío para",
        current_state
      )
    )
  }


  # ==========================================================
  # 14. VALIDAR TARGET
  # ==========================================================

  train_target_distribution <- table(
    expert_train$target
  )

  validation_target_distribution <- table(
    expert_validation$target
  )


  cat(
    "Target classes TRAIN:",
    sum(
      train_target_distribution > 0
    ),
    "\n"
  )

  cat(
    "Target classes VALIDATION:",
    sum(
      validation_target_distribution > 0
    ),
    "\n"
  )


  # ==========================================================
  # 15. IMPUTACIÓN
  #
  # Cada experto usa medianas de su propio TRAIN.
  # ==========================================================

  for (
    col in features
  ) {

    med <- median(
      expert_train[[col]],
      na.rm = TRUE
    )


    if (!is.finite(med)) {

      med <- 0
    }


    train_missing <- which(
      is.na(
        expert_train[[col]]
      )
    )


    if (
      length(train_missing) > 0
    ) {

      set(
        expert_train,
        train_missing,
        col,
        med
      )
    }


    validation_missing <- which(
      is.na(
        expert_validation[[col]]
      )
    )


    if (
      length(validation_missing) > 0
    ) {

      set(
        expert_validation,
        validation_missing,
        col,
        med
      )
    }
  }


  # ==========================================================
  # 16. MATRICES
  # ==========================================================

  x_train <- as.matrix(
    expert_train[
      ,
      ..features
    ]
  )


  x_validation <- as.matrix(
    expert_validation[
      ,
      ..features
    ]
  )


  y_train <- as.integer(
    expert_train$target
  ) - 1


  y_validation <- as.integer(
    expert_validation$target
  ) - 1


  dtrain <- xgb.DMatrix(
    x_train,
    label = y_train
  )


  dvalidation <- xgb.DMatrix(
    x_validation,
    label = y_validation
  )


  # ==========================================================
  # 17. TUNING DEL EXPERTO
  # ==========================================================

  tuning_state_list <- vector(
    "list",
    nrow(parameter_grid)
  )


  models_state <- vector(
    "list",
    nrow(parameter_grid)
  )


  for (
    i in seq_len(
      nrow(parameter_grid)
    )
  ) {

    p <- parameter_grid[i]


    params <- list(

      objective =
        "multi:softprob",

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
        0.80,

      colsample_bytree =
        0.75,

      lambda =
        2,

      alpha =
        0.10,

      nthread =
        max(
          1,
          parallel::detectCores() - 1
        )
    )


    set.seed(
      6000 +
      state_index * 100 +
      i
    )


    # ========================================================
    # IMPORTANTE:
    # usamos evals, NO watchlist
    # ========================================================

    model <- xgb.train(

      params = params,

      data = dtrain,

      nrounds = 1200,

      evals = list(
        train = dtrain,
        validation = dvalidation
      ),

      early_stopping_rounds =
        50,

      maximize =
        FALSE,

      verbose =
        0
    )


    # ========================================================
    # 18. VALIDATION LOGLOSS CALCULADO MANUALMENTE
    #
    # Ya NO dependemos de:
    # model$best_score
    # model$evaluation_log
    # ========================================================

    raw_validation_probs <- predict(
      model,
      dvalidation
    )


    validation_probs <- format_probs(
      raw_validation_probs,
      nrow(
        expert_validation
      )
    )


    validation_truth <- as.character(
      expert_validation$target
    )


    validation_logloss_value <-
      multiclass_logloss(
        validation_truth,
        validation_probs
      )


    # ========================================================
    # 19. BEST ITERATION
    # ========================================================

    best_iteration_value <- NA_integer_


    if (
      !is.null(
        model$best_iteration
      )
    ) {

      best_iteration_value <-
        as.integer(
          model$best_iteration
        )
    }


    # ========================================================
    # 20. GUARDAR TUNING
    # ========================================================

    tuning_state_list[[i]] <- data.table(

      current_state =
        current_state,

      model_id =
        i,

      max_depth =
        as.integer(
          p$max_depth
        ),

      eta =
        as.numeric(
          p$eta
        ),

      min_child_weight =
        as.numeric(
          p$min_child_weight
        ),

      best_iteration =
        best_iteration_value,

      validation_logloss =
        validation_logloss_value
    )


    models_state[[i]] <- model
  }


  # ==========================================================
  # 21. UNIR RESULTADOS DEL TUNING
  # ==========================================================

  tuning_state <- rbindlist(
    tuning_state_list
  )


  if (
    !"validation_logloss" %in%
    names(tuning_state)
  ) {

    stop(
      paste(
        "No se creó validation_logloss para",
        current_state
      )
    )
  }


  # ==========================================================
  # 22. ELEGIR MEJOR MODELO
  # ==========================================================

  setorder(
    tuning_state,
    validation_logloss
  )


  best_model_id <- tuning_state[
    1,
    model_id
  ]


  best_model <- models_state[[
    best_model_id
  ]]


  cat(
    "Best model:",
    best_model_id,
    "\n"
  )


  cat(
    "Best validation logloss:",
    round(
      tuning_state[
        1,
        validation_logloss
      ],
      4
    ),
    "\n"
  )


  expert_models[[state_index]] <-
    best_model


  expert_tuning_list[[state_index]] <-
    tuning_state


  # ==========================================================
  # 23. PROBABILIDADES DEL MEJOR MODELO
  # ==========================================================

  raw_probs <- predict(
    best_model,
    dvalidation
  )


  probs <- format_probs(
    raw_probs,
    nrow(
      expert_validation
    )
  )


  # ==========================================================
  # 24. VALIDAR SUMA DE PROBABILIDADES
  # ==========================================================

  if (
    any(
      abs(
        rowSums(probs) - 1
      ) > 0.001
    )
  ) {

    stop(
      paste(
        "Probabilidades inválidas para",
        current_state
      )
    )
  }


  # ==========================================================
  # 25. PREDICCIÓN
  # ==========================================================

  prediction <- state_levels[
    max.col(
      probs,
      ties.method = "first"
    )
  ]


  truth <- as.character(
    expert_validation$target
  )


  # ==========================================================
  # 26. MÉTRICAS DEL EXPERTO
  # ==========================================================

  expert_metrics <- data.table(

    current_state =
      current_state,

    observations =
      length(truth),

    accuracy =
      round(
        mean(
          prediction ==
          truth
        ) * 100,
        2
      ),

    balanced_accuracy =
      round(
        balanced_accuracy(
          truth,
          prediction
        ) * 100,
        2
      ),

    macro_f1 =
      round(
        macro_f1(
          truth,
          prediction
        ),
        4
      ),

    top2_accuracy =
      round(
        top2_accuracy(
          truth,
          probs
        ) * 100,
        2
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


  expert_metrics_list[[state_index]] <-
    expert_metrics


  # ==========================================================
  # 27. PREDICCIONES
  # ==========================================================

  pred_dt <- expert_validation[
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


  pred_dt[
    ,
    predicted_state :=
      prediction
  ]


  for (
    j in seq_along(
      state_levels
    )
  ) {

    cl <- state_levels[j]


    new_col <- paste0(
      "prob_",
      tolower(cl)
    )


    pred_dt[
      ,
      (new_col) :=
        probs[
          ,
          j
        ]
    ]
  }


  if (
    nrow(pred_dt) == 0
  ) {

    stop(
      paste(
        "Predicciones vacías para",
        current_state
      )
    )
  }


  expert_predictions_list[[state_index]] <-
    pred_dt


  cat(
    "Experto",
    current_state,
    "terminado correctamente.",
    "\n"
  )
}


# ============================================================
# 28. VALIDAR LOS CINCO EXPERTOS
# ============================================================

expert_status <- data.table(

  state =
    state_levels,

  prediction_rows =
    sapply(
      expert_predictions_list,
      nrow
    ),

  metrics_rows =
    sapply(
      expert_metrics_list,
      nrow
    ),

  tuning_rows =
    sapply(
      expert_tuning_list,
      nrow
    )
)


if (
  any(
    expert_status$
    prediction_rows == 0
  )
) {

  View(
    expert_status
  )

  stop(
    "Uno o más expertos quedaron vacíos."
  )
}


# ============================================================
# 29. UNIR PREDICCIONES
# ============================================================

validation_predictions <- rbindlist(

  expert_predictions_list,

  use.names = TRUE,

  fill = TRUE
)


if (
  nrow(
    validation_predictions
  ) == 0
) {

  stop(
    "validation_predictions quedó vacío."
  )
}


# ============================================================
# 30. COLUMNAS DE PROBABILIDAD
# ============================================================

probability_columns <- paste0(
  "prob_",
  tolower(
    state_levels
  )
)


missing_probability_columns <- setdiff(
  probability_columns,
  names(
    validation_predictions
  )
)


if (
  length(
    missing_probability_columns
  ) > 0
) {

  stop(
    paste(
      "Faltan probabilidades:",
      paste(
        missing_probability_columns,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 31. MATRIZ GLOBAL DE PROBABILIDAD
# ============================================================

global_probs <- as.matrix(
  validation_predictions[
    ,
    ..probability_columns
  ]
)


colnames(
  global_probs
) <- state_levels


# ============================================================
# 32. TARGET GLOBAL
# ============================================================

global_truth <- as.character(
  validation_predictions$
  future_tactical_state_5min
)


global_prediction <-
  validation_predictions$
  predicted_state


# ============================================================
# 33. RESUMEN GLOBAL
# ============================================================

expert_validation_summary <- data.table(

  model =
    "State-conditional experts",

  observations =
    length(
      global_truth
    ),

  accuracy =
    round(
      mean(
        global_prediction ==
        global_truth
      ) * 100,
      2
    ),

  balanced_accuracy =
    round(
      balanced_accuracy(
        global_truth,
        global_prediction
      ) * 100,
      2
    ),

  macro_f1 =
    round(
      macro_f1(
        global_truth,
        global_prediction
      ),
      4
    ),

  top2_accuracy =
    round(
      top2_accuracy(
        global_truth,
        global_probs
      ) * 100,
      2
    ),

  log_loss =
    round(
      multiclass_logloss(
        global_truth,
        global_probs
      ),
      4
    )
)


# ============================================================
# 34. BENCHMARK GLOBAL
# ============================================================

global_benchmark <- data.table(

  model =
    "Global Events+360",

  observations =
    nrow(
      validation_predictions
    ),

  accuracy =
    40.06,

  balanced_accuracy =
    NA_real_,

  macro_f1 =
    NA_real_,

  top2_accuracy =
    NA_real_,

  log_loss =
    1.3303
)


# ============================================================
# 35. COMPARACIÓN
# ============================================================

validation_model_comparison <- rbind(

  global_benchmark,

  expert_validation_summary,

  fill = TRUE
)


# ============================================================
# 36. MÉTRICAS POR EXPERTO
# ============================================================

expert_by_state <- rbindlist(
  expert_metrics_list
)


# ============================================================
# 37. TUNING COMPLETO
# ============================================================

expert_tuning <- rbindlist(
  expert_tuning_list
)


# ============================================================
# 38. GUARDAR MODELOS
# ============================================================

dir.create(
  "outputs/models/state_experts",
  recursive = TRUE,
  showWarnings = FALSE
)


for (
  i in seq_along(
    state_levels
  )
) {

  safe_name <- tolower(
    state_levels[i]
  )


  xgb.save(
    expert_models[[i]],
    paste0(
      "outputs/models/state_experts/",
      safe_name,
      "_expert.json"
    )
  )
}


# ============================================================
# 39. GUARDAR RESULTADOS
# ============================================================

fwrite(
  validation_model_comparison,
  "outputs/models/state_experts_validation_comparison.csv",
  bom = TRUE
)


fwrite(
  expert_by_state,
  "outputs/models/state_experts_by_state.csv",
  bom = TRUE
)


fwrite(
  expert_tuning,
  "outputs/models/state_experts_tuning.csv",
  bom = TRUE
)


fwrite(
  validation_predictions,
  "outputs/models/state_experts_validation_predictions.csv",
  bom = TRUE
)


# ============================================================
# 40. SOLO DOS VISUALIZACIONES
# ============================================================

View(
  validation_model_comparison
)


View(
  expert_by_state
)


# ============================================================
# 41. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("STATE-CONDITIONAL EXPERTS TERMINADO\n")
cat("=====================================\n")


cat(
  "Validation accuracy:",
  expert_validation_summary$
  accuracy,
  "%\n"
)


cat(
  "Validation balanced accuracy:",
  expert_validation_summary$
  balanced_accuracy,
  "%\n"
)


cat(
  "Validation Top-2 accuracy:",
  expert_validation_summary$
  top2_accuracy,
  "%\n"
)


cat(
  "Validation log-loss:",
  expert_validation_summary$
  log_loss,
  "\n"
)