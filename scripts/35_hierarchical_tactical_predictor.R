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
cat("HIERARCHICAL TACTICAL PREDICTOR\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. ESTADOS
# ============================================================

state_levels <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION",
  "PROTECTION",
  "URGENCY"
)

dynamic_states <- c(
  "CONTROL",
  "PRESSURE",
  "TRANSITION"
)

situational_states <- c(
  "PROTECTION",
  "URGENCY"
)


# ============================================================
# 3. TARGET 5 MIN
# ============================================================

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
# 4. TARGET MACRO
# ============================================================

dt[
  ,
  future_macro :=
    fifelse(
      as.character(target) %in%
        dynamic_states,
      "DYNAMIC",
      "SITUATIONAL"
    )
]

dt[
  ,
  future_macro :=
    factor(
      future_macro,
      levels = c(
        "DYNAMIC",
        "SITUATIONAL"
      )
    )
]


# ============================================================
# 5. ONE-HOT DEL ESTADO ACTUAL
# ============================================================

for (
  state in state_levels
) {

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
# 6. FEATURES
# ============================================================

features <- c(

  # CONTEXTO
  "minute",
  "game_state_num",

  # ESTADO ACTUAL
  state_features,

  # EVENTS 5 MIN
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # EVENTS 3 MIN
  "america_field_tilt_last3",
  "entries_final_third_diff_last3",
  "entries_box_diff_last3",
  "progressive_pass_diff_last3",
  "america_progressive_carries_last3",
  "america_progressive_passes_last3",
  "pressure_diff_last3",
  "recovery_height_diff_last3",

  # 360 5 MIN
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 360 3 MIN
  "america_local_superiority_10_last3",
  "america_pressure_share_5_last3",
  "america_nearest_opponent_distance_last3",
  "america_support_share_10_last3"
)


# ============================================================
# 7. VALIDAR FEATURES
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
# 8. IMPUTACIÓN
#
# SOLO TRAIN
# ============================================================

for (
  col in features
) {

  med <- median(
    dt[
      dataset_split == "TRAIN",
      get(col)
    ],
    na.rm = TRUE
  )

  if (
    !is.finite(med)
  ) {
    med <- 0
  }

  dt[
    is.na(get(col)),
    (col) := med
  ]
}


# ============================================================
# 9. SPLITS
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
# 10. FUNCIONES
# ============================================================

format_probs <- function(
  raw_probs,
  n_rows,
  class_levels
) {

  if (
    is.matrix(raw_probs)
  ) {

    probs <- raw_probs

  } else {

    probs <- matrix(
      raw_probs,
      nrow = n_rows,
      ncol = length(class_levels),
      byrow = TRUE
    )
  }


  if (
    nrow(probs) != n_rows ||
    ncol(probs) != length(class_levels)
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


  colnames(probs) <- class_levels

  probs
}


multiclass_logloss <- function(
  truth,
  probabilities,
  class_levels
) {

  truth <- factor(
    truth,
    levels = class_levels
  )

  idx <- as.integer(
    truth
  )

  p <- probabilities[
    cbind(
      seq_along(idx),
      idx
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


balanced_accuracy <- function(
  truth,
  prediction,
  class_levels
) {

  recalls <- sapply(
    class_levels,
    function(cl) {

      idx <- truth == cl

      if (
        sum(idx) == 0
      ) {
        return(NA_real_)
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


macro_f1 <- function(
  truth,
  prediction,
  class_levels
) {

  values <- sapply(
    class_levels,
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


top2_accuracy <- function(
  truth,
  probabilities,
  class_levels
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
      class_levels[top2]
  }

  mean(correct)
}


# ============================================================
# 11. FUNCIÓN GENERAL PARA ENTRENAR/TUNEAR
#
# SOLO selecciona usando VALIDATION.
# ============================================================

train_tuned_model <- function(
  train_data,
  validation_data,
  target_column,
  class_levels,
  seed_base
) {

  x_train <- as.matrix(
    train_data[
      ,
      ..features
    ]
  )

  x_validation <- as.matrix(
    validation_data[
      ,
      ..features
    ]
  )


  train_target <- factor(
    train_data[[target_column]],
    levels = class_levels
  )

  validation_target <- factor(
    validation_data[[target_column]],
    levels = class_levels
  )


  y_train <- as.integer(
    train_target
  ) - 1

  y_validation <- as.integer(
    validation_target
  ) - 1


  dtrain <- xgb.DMatrix(
    x_train,
    label = y_train
  )

  dvalidation <- xgb.DMatrix(
    x_validation,
    label = y_validation
  )


  grid <- CJ(

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


  tuning_list <- vector(
    "list",
    nrow(grid)
  )

  models <- vector(
    "list",
    nrow(grid)
  )


  for (
    i in seq_len(
      nrow(grid)
    )
  ) {

    p <- grid[i]


    params <- list(

      objective =
        "multi:softprob",

      num_class =
        length(class_levels),

      eval_metric =
        "mlogloss",

      max_depth =
        p$max_depth,

      eta =
        p$eta,

      min_child_weight =
        p$min_child_weight,

      subsample =
        0.8,

      colsample_bytree =
        0.75,

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
      seed_base + i
    )


    model <- xgb.train(

      params =
        params,

      data =
        dtrain,

      nrounds =
        1200,

      evals =
        list(
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


    probs <- format_probs(
      predict(
        model,
        dvalidation
      ),
      nrow(validation_data),
      class_levels
    )


    truth <- as.character(
      validation_target
    )


    prediction <- class_levels[
      max.col(
        probs,
        ties.method = "first"
      )
    ]


    tuning_list[[i]] <- data.table(

      model_id =
        i,

      max_depth =
        p$max_depth,

      eta =
        p$eta,

      min_child_weight =
        p$min_child_weight,

      validation_accuracy =
        mean(
          prediction ==
          truth
        ),

      validation_balanced_accuracy =
        balanced_accuracy(
          truth,
          prediction,
          class_levels
        ),

      validation_macro_f1 =
        macro_f1(
          truth,
          prediction,
          class_levels
        ),

      validation_logloss =
        multiclass_logloss(
          truth,
          probs,
          class_levels
        )
    )


    models[[i]] <- model
  }


  tuning <- rbindlist(
    tuning_list
  )


  tuning[
    ,
    selection_score :=
      0.50 *
        validation_accuracy +
      0.30 *
        validation_balanced_accuracy +
      0.20 *
        validation_macro_f1
  ]


  setorder(
    tuning,
    -selection_score,
    validation_logloss
  )


  best_id <- tuning[
    1,
    model_id
  ]


  list(

    model =
      models[[best_id]],

    tuning =
      tuning,

    best_id =
      best_id
  )
}


# ============================================================
# 12. NIVEL 1:
# MACRO-RÉGIMEN
# ============================================================

cat("\nEntrenando NIVEL 1: MACRO\n")

macro_result <- train_tuned_model(

  train_data =
    train,

  validation_data =
    validation,

  target_column =
    "future_macro",

  class_levels =
    c(
      "DYNAMIC",
      "SITUATIONAL"
    ),

  seed_base =
    9000
)


macro_model <- macro_result$model


# ============================================================
# 13. NIVEL 2A:
# DYNAMIC
# ============================================================

train_dynamic <- train[
  as.character(target) %in%
    dynamic_states
]

validation_dynamic <- validation[
  as.character(target) %in%
    dynamic_states
]


cat(
  "\nDynamic TRAIN:",
  nrow(train_dynamic),
  "\n"
)

cat(
  "Dynamic VALIDATION:",
  nrow(validation_dynamic),
  "\n"
)


dynamic_result <- train_tuned_model(

  train_data =
    train_dynamic,

  validation_data =
    validation_dynamic,

  target_column =
    "future_tactical_state_5min",

  class_levels =
    dynamic_states,

  seed_base =
    10000
)


dynamic_model <- dynamic_result$model


# ============================================================
# 14. NIVEL 2B:
# SITUATIONAL
# ============================================================

train_situational <- train[
  as.character(target) %in%
    situational_states
]

validation_situational <- validation[
  as.character(target) %in%
    situational_states
]


cat(
  "\nSituational TRAIN:",
  nrow(train_situational),
  "\n"
)

cat(
  "Situational VALIDATION:",
  nrow(validation_situational),
  "\n"
)


situational_result <- train_tuned_model(

  train_data =
    train_situational,

  validation_data =
    validation_situational,

  target_column =
    "future_tactical_state_5min",

  class_levels =
    situational_states,

  seed_base =
    11000
)


situational_model <-
  situational_result$model


# ============================================================
# 15. FUNCIÓN PARA PREDICCIÓN JERÁRQUICA
# ============================================================

predict_hierarchical <- function(
  data
) {

  x <- as.matrix(
    data[
      ,
      ..features
    ]
  )


  # ----------------------------------------------------------
  # MACRO
  # ----------------------------------------------------------

  dmacro <- xgb.DMatrix(
    x
  )


  macro_probs <- format_probs(
    predict(
      macro_model,
      dmacro
    ),
    nrow(data),
    c(
      "DYNAMIC",
      "SITUATIONAL"
    )
  )


  # ----------------------------------------------------------
  # DYNAMIC
  # ----------------------------------------------------------

  ddynamic <- xgb.DMatrix(
    x
  )


  dynamic_probs <- format_probs(
    predict(
      dynamic_model,
      ddynamic
    ),
    nrow(data),
    dynamic_states
  )


  # ----------------------------------------------------------
  # SITUATIONAL
  # ----------------------------------------------------------

  dsituational <- xgb.DMatrix(
    x
  )


  situational_probs <- format_probs(
    predict(
      situational_model,
      dsituational
    ),
    nrow(data),
    situational_states
  )


  # ----------------------------------------------------------
  # COMBINACIÓN FINAL
  # ----------------------------------------------------------

  final_probs <- matrix(
    0,
    nrow = nrow(data),
    ncol = length(state_levels)
  )


  colnames(
    final_probs
  ) <- state_levels


  # DYNAMIC
  final_probs[
    ,
    "CONTROL"
  ] <-
    macro_probs[
      ,
      "DYNAMIC"
    ] *
    dynamic_probs[
      ,
      "CONTROL"
    ]


  final_probs[
    ,
    "PRESSURE"
  ] <-
    macro_probs[
      ,
      "DYNAMIC"
    ] *
    dynamic_probs[
      ,
      "PRESSURE"
    ]


  final_probs[
    ,
    "TRANSITION"
  ] <-
    macro_probs[
      ,
      "DYNAMIC"
    ] *
    dynamic_probs[
      ,
      "TRANSITION"
    ]


  # SITUATIONAL
  final_probs[
    ,
    "PROTECTION"
  ] <-
    macro_probs[
      ,
      "SITUATIONAL"
    ] *
    situational_probs[
      ,
      "PROTECTION"
    ]


  final_probs[
    ,
    "URGENCY"
  ] <-
    macro_probs[
      ,
      "SITUATIONAL"
    ] *
    situational_probs[
      ,
      "URGENCY"
    ]


  # ----------------------------------------------------------
  # VALIDAR SUMAS
  # ----------------------------------------------------------

  if (
    any(
      abs(
        rowSums(final_probs) -
        1
      ) > 0.001
    )
  ) {

    stop(
      "Probabilidades jerárquicas no suman 1."
    )
  }


  list(

    final_probs =
      final_probs,

    macro_probs =
      macro_probs,

    dynamic_probs =
      dynamic_probs,

    situational_probs =
      situational_probs
  )
}


# ============================================================
# 16. EVALUACIÓN VALIDATION
#
# DECISIÓN DEL MODELO SE HACE AQUÍ.
# ============================================================

validation_prediction <-
  predict_hierarchical(
    validation
  )


validation_probs <-
  validation_prediction$
    final_probs


validation_truth <-
  as.character(
    validation$target
  )


validation_predicted <-
  state_levels[
    max.col(
      validation_probs,
      ties.method =
        "first"
    )
  ]


hierarchical_validation_metrics <-
  data.table(

    model =
      "Hierarchical",

    accuracy =
      round(
        mean(
          validation_predicted ==
          validation_truth
        ) * 100,
        2
      ),

    balanced_accuracy =
      round(
        balanced_accuracy(
          validation_truth,
          validation_predicted,
          state_levels
        ) * 100,
        2
      ),

    macro_f1 =
      round(
        macro_f1(
          validation_truth,
          validation_predicted,
          state_levels
        ),
        4
      ),

    top2_accuracy =
      round(
        top2_accuracy(
          validation_truth,
          validation_probs,
          state_levels
        ) * 100,
        2
      ),

    log_loss =
      round(
        multiclass_logloss(
          validation_truth,
          validation_probs,
          state_levels
        ),
        4
      )
  )


# ============================================================
# 17. COMPARAR VALIDATION CONTRA GLOBAL
# ============================================================

validation_comparison <- data.table(

  model = c(
    "Global Events+360",
    "Hierarchical"
  ),

  accuracy = c(
    40.06,
    hierarchical_validation_metrics$
      accuracy
  ),

  balanced_accuracy = c(
    NA_real_,
    hierarchical_validation_metrics$
      balanced_accuracy
  ),

  macro_f1 = c(
    NA_real_,
    hierarchical_validation_metrics$
      macro_f1
  ),

  top2_accuracy = c(
    NA_real_,
    hierarchical_validation_metrics$
      top2_accuracy
  ),

  log_loss = c(
    1.3303,
    hierarchical_validation_metrics$
      log_loss
  )
)


# ============================================================
# 18. DECISIÓN AUTOMÁTICA
#
# Solo pasa a TEST si en validation tiene una señal razonable.
# ============================================================

use_hierarchical <-
  hierarchical_validation_metrics$
    accuracy >=
    40.06 ||
  hierarchical_validation_metrics$
    log_loss <
    1.3303


cat(
  "\n¿Hierarchical pasa a TEST?:",
  use_hierarchical,
  "\n"
)


# ============================================================
# 19. SI GANA / COMPITE EN VALIDATION, EVALUAMOS TEST
# ============================================================

hierarchical_test_metrics <-
  data.table()


hierarchical_by_state <-
  data.table()


if (
  use_hierarchical
) {

  cat(
    "\nEvaluando TEST una sola vez...\n"
  )


  test_prediction <-
    predict_hierarchical(
      test
    )


  test_probs <-
    test_prediction$
      final_probs


  test_truth <-
    as.character(
      test$target
    )


  test_predicted <-
    state_levels[
      max.col(
        test_probs,
        ties.method =
          "first"
      )
    ]


  hierarchical_test_metrics <-
    data.table(

      model =
        "Hierarchical",

      accuracy =
        round(
          mean(
            test_predicted ==
            test_truth
          ) * 100,
          2
        ),

      balanced_accuracy =
        round(
          balanced_accuracy(
            test_truth,
            test_predicted,
            state_levels
          ) * 100,
          2
        ),

      macro_f1 =
        round(
          macro_f1(
            test_truth,
            test_predicted,
            state_levels
          ),
          4
        ),

      top2_accuracy =
        round(
          top2_accuracy(
            test_truth,
            test_probs,
            state_levels
          ) * 100,
          2
        ),

      log_loss =
        round(
          multiclass_logloss(
            test_truth,
            test_probs,
            state_levels
          ),
          4
        )
    )


  # ==========================================================
  # 20. MÉTRICAS POR CLASE
  # ==========================================================

  hierarchical_by_state <-
    rbindlist(

      lapply(
        state_levels,
        function(cl) {

          idx <-
            test_truth ==
            cl


          probs_state <-
            test_probs[
              idx,
              ,
              drop = FALSE
            ]


          truth_state <-
            test_truth[
              idx
            ]


          data.table(

            state =
              cl,

            observations =
              sum(idx),

            top1_accuracy =
              round(
                mean(
                  test_predicted[idx] ==
                  cl
                ) * 100,
                2
              ),

            top2_accuracy =
              round(
                top2_accuracy(
                  truth_state,
                  probs_state,
                  state_levels
                ) * 100,
                2
              )
          )
        }
      )
    )
}


# ============================================================
# 21. COMPARACIÓN FINAL
# ============================================================

if (
  use_hierarchical
) {

  final_comparison <-
    data.table(

      model = c(
        "Current XGBoost",
        "Hierarchical"
      ),

      top1_accuracy = c(
        40.10,
        hierarchical_test_metrics$
          accuracy
      ),

      balanced_accuracy = c(
        45.14,
        hierarchical_test_metrics$
          balanced_accuracy
      ),

      macro_f1 = c(
        0.3872,
        hierarchical_test_metrics$
          macro_f1
      ),

      top2_accuracy = c(
        64.45,
        hierarchical_test_metrics$
          top2_accuracy
      ),

      log_loss = c(
        1.3428,
        hierarchical_test_metrics$
          log_loss
      )
    )

} else {

  final_comparison <-
    data.table(

      message =
        "Hierarchical model did not beat validation benchmark; TEST not evaluated."
    )
}


# ============================================================
# 22. GUARDAR
# ============================================================

dir.create(
  "outputs/models/hierarchical",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  validation_comparison,
  "outputs/models/hierarchical/validation_comparison.csv",
  bom = TRUE
)


fwrite(
  macro_result$tuning,
  "outputs/models/hierarchical/macro_tuning.csv",
  bom = TRUE
)


fwrite(
  dynamic_result$tuning,
  "outputs/models/hierarchical/dynamic_tuning.csv",
  bom = TRUE
)


fwrite(
  situational_result$tuning,
  "outputs/models/hierarchical/situational_tuning.csv",
  bom = TRUE
)


xgb.save(
  macro_model,
  "outputs/models/hierarchical/macro_model.json"
)


xgb.save(
  dynamic_model,
  "outputs/models/hierarchical/dynamic_model.json"
)


xgb.save(
  situational_model,
  "outputs/models/hierarchical/situational_model.json"
)


if (
  use_hierarchical
) {

  fwrite(
    hierarchical_test_metrics,
    "outputs/models/hierarchical/test_metrics.csv",
    bom = TRUE
  )


  fwrite(
    hierarchical_by_state,
    "outputs/models/hierarchical/test_by_state.csv",
    bom = TRUE
  )


  fwrite(
    final_comparison,
    "outputs/models/hierarchical/final_comparison.csv",
    bom = TRUE
  )
}


# ============================================================
# 23. SOLO LAS TABLAS IMPORTANTES
# ============================================================

View(
  validation_comparison
)


if (
  use_hierarchical
) {

  View(
    final_comparison
  )

  View(
    hierarchical_by_state
  )
}


# ============================================================
# 24. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("HIERARCHICAL MODEL TERMINADO\n")
cat("=====================================\n")


cat(
  "Validation Top-1:",
  hierarchical_validation_metrics$
    accuracy,
  "%\n"
)


cat(
  "Validation Top-2:",
  hierarchical_validation_metrics$
    top2_accuracy,
  "%\n"
)


cat(
  "Validation log-loss:",
  hierarchical_validation_metrics$
    log_loss,
  "\n"
)


if (
  use_hierarchical
) {

  cat(
    "TEST Top-1:",
    hierarchical_test_metrics$
      accuracy,
    "%\n"
  )

  cat(
    "TEST Top-2:",
    hierarchical_test_metrics$
      top2_accuracy,
    "%\n"
  )
}