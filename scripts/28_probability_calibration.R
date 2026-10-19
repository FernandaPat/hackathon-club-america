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
# 2. ONE-HOT ESTADO ACTUAL
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
# 3. MODELO SELECCIONADO
#
# Seleccionamos CONTEXT + EVENTS + 360
# porque fue el mejor en VALIDATION.
#
# IMPORTANTE:
# No elegimos usando TEST.
# ============================================================

selected_features <- c(

  # ------------------------------------
  # CONTEXTO
  # ------------------------------------

  "minute",
  "game_state_num",

  state_features,

  # ------------------------------------
  # EVENTS 5 MIN
  # ------------------------------------

  "america_field_tilt_last5",

  "entries_final_third_diff_last5",

  "entries_box_diff_last5",

  "progressive_pass_diff_last5",

  "america_progressive_carries_last5",

  "america_progressive_passes_last5",

  "pressure_diff_last5",

  "recovery_height_diff_last5",

  # ------------------------------------
  # EVENTS 3 MIN
  # ------------------------------------

  "america_field_tilt_last3",

  "entries_final_third_diff_last3",

  "entries_box_diff_last3",

  "progressive_pass_diff_last3",

  "america_progressive_carries_last3",

  "america_progressive_passes_last3",

  "pressure_diff_last3",

  "recovery_height_diff_last3",

  # ------------------------------------
  # 360 5 MIN
  # ------------------------------------

  "america_local_superiority_10_last5",

  "america_pressure_share_5_last5",

  "america_nearest_opponent_distance_last5",

  "america_support_share_10_last5",

  # ------------------------------------
  # 360 3 MIN
  # ------------------------------------

  "america_local_superiority_10_last3",

  "america_pressure_share_5_last3",

  "america_nearest_opponent_distance_last3",

  "america_support_share_10_last3"
)


# ============================================================
# 4. VALIDAR FEATURES
# ============================================================

missing_features <- setdiff(
  selected_features,
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
# 5. IMPUTACIÓN
#
# Medianas calculadas SOLO CON TRAIN.
# ============================================================

imputation_table <- data.table(
  variable = selected_features,
  train_median = NA_real_
)

for (i in seq_along(selected_features)) {

  col <- selected_features[i]

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
# 6. SPLITS
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
# 7. MATRICES
# ============================================================

x_train <- as.matrix(
  train[
    ,
    ..selected_features
  ]
)

x_validation <- as.matrix(
  validation[
    ,
    ..selected_features
  ]
)

x_test <- as.matrix(
  test[
    ,
    ..selected_features
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


# ============================================================
# 8. DMATRIX
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
# 9. PARÁMETROS
#
# Mismos parámetros usados en el ablation study.
# ============================================================

params <- list(

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


# ============================================================
# 10. ENTRENAR MODELO
# ============================================================

set.seed(2026)

selected_model <- xgb.train(

  params = params,

  data = dtrain,

  nrounds = 1000,

  watchlist = list(

    train =
      dtrain,

    validation =
      dvalidation
  ),

  early_stopping_rounds =
    40,

  maximize =
    FALSE,

  verbose =
    0
)


# ============================================================
# 11. FUNCIÓN PARA PROBABILIDADES
# ============================================================

format_probs <- function(
  raw_probs,
  n_rows
) {

  if (
    is.matrix(raw_probs)
  ) {

    probs <- raw_probs

  } else {

    probs <- matrix(

      raw_probs,

      nrow =
        n_rows,

      ncol =
        length(state_levels),

      byrow =
        TRUE
    )
  }


  if (
    nrow(probs) != n_rows ||
    ncol(probs) != length(state_levels)
  ) {

    stop(
      "Las dimensiones de las probabilidades son incorrectas."
    )
  }


  colnames(
    probs
  ) <- state_levels


  probs
}


# ============================================================
# 12. PROBABILIDADES
# ============================================================

prob_validation <- format_probs(

  predict(
    selected_model,
    dvalidation
  ),

  nrow(validation)
)


prob_test <- format_probs(

  predict(
    selected_model,
    dtest
  ),

  nrow(test)
)


# ============================================================
# 13. VALIDAR PROBABILIDADES
# ============================================================

if (
  any(
    abs(
      rowSums(
        prob_validation
      ) - 1
    ) > 0.001
  )
) {

  stop(
    "Probabilidades validation no suman 1."
  )
}


if (
  any(
    abs(
      rowSums(
        prob_test
      ) - 1
    ) > 0.001
  )
) {

  stop(
    "Probabilidades test no suman 1."
  )
}


# ============================================================
# 14. LOG LOSS
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
# 15. TEMPERATURE SCALING
#
# T > 1:
# reduce confianza
#
# T < 1:
# aumenta confianza
# ============================================================

temperature_scale <- function(
  probabilities,
  temperature
) {

  probabilities <- pmax(
    probabilities,
    1e-15
  )

  log_probs <- log(
    probabilities
  )

  scaled <- log_probs /
    temperature


  # estabilidad numérica
  row_max <- apply(
    scaled,
    1,
    max
  )

  scaled <- scaled -
    row_max


  exp_scaled <- exp(
    scaled
  )


  calibrated <-
    exp_scaled /
    rowSums(
      exp_scaled
    )


  colnames(
    calibrated
  ) <- state_levels


  calibrated
}


# ============================================================
# 16. TARGET VALIDATION
# ============================================================

validation_truth <- as.character(
  validation$target
)


# ============================================================
# 17. FUNCIÓN OBJETIVO DE TEMPERATURA
#
# SOLO VALIDATION.
# ============================================================

temperature_objective <- function(
  log_temperature
) {

  temperature <- exp(
    log_temperature
  )

  calibrated <- temperature_scale(

    prob_validation,

    temperature
  )


  multiclass_logloss(

    validation_truth,

    calibrated
  )
}


# ============================================================
# 18. OPTIMIZAR TEMPERATURA
# ============================================================

temperature_fit <- optimize(

  f =
    temperature_objective,

  interval = c(
    log(0.20),
    log(5.00)
  )
)


best_temperature <- exp(
  temperature_fit$minimum
)


# ============================================================
# 19. CALIBRAR
# ============================================================

prob_validation_calibrated <-
  temperature_scale(

    prob_validation,

    best_temperature
  )


prob_test_calibrated <-
  temperature_scale(

    prob_test,

    best_temperature
  )


# ============================================================
# 20. TARGET TEST
# ============================================================

test_truth <- as.character(
  test$target
)


# ============================================================
# 21. LOG LOSS ANTES / DESPUÉS
# ============================================================

validation_logloss_before <-
  multiclass_logloss(

    validation_truth,

    prob_validation
  )


validation_logloss_after <-
  multiclass_logloss(

    validation_truth,

    prob_validation_calibrated
  )


test_logloss_before <-
  multiclass_logloss(

    test_truth,

    prob_test
  )


test_logloss_after <-
  multiclass_logloss(

    test_truth,

    prob_test_calibrated
  )


# ============================================================
# 22. MULTICLASS BRIER SCORE
#
# Menor = mejor.
# ============================================================

multiclass_brier <- function(
  truth,
  probabilities
) {

  truth_factor <- factor(

    truth,

    levels =
      state_levels
  )


  y <- matrix(

    0,

    nrow =
      length(
        truth_factor
      ),

    ncol =
      length(
        state_levels
      )
  )


  y[
    cbind(
      seq_along(
        truth_factor
      ),
      as.integer(
        truth_factor
      )
    )
  ] <- 1


  mean(
    rowSums(
      (
        probabilities -
          y
      )^2
    )
  )
}


test_brier_before <-
  multiclass_brier(

    test_truth,

    prob_test
  )


test_brier_after <-
  multiclass_brier(

    test_truth,

    prob_test_calibrated
  )


# ============================================================
# 23. EXPECTED CALIBRATION ERROR
#
# Menor = mejor.
# ============================================================

multiclass_ece <- function(
  truth,
  probabilities,
  bins = 10
) {

  predicted_index <-
    max.col(
      probabilities,
      ties.method =
        "first"
    )


  predicted_class <-
    state_levels[
      predicted_index
    ]


  confidence <- apply(
    probabilities,
    1,
    max
  )


  correct <-
    predicted_class ==
    truth


  breaks <- seq(
    0,
    1,
    length.out =
      bins + 1
  )


  bin_id <- cut(

    confidence,

    breaks =
      breaks,

    include.lowest =
      TRUE,

    labels =
      FALSE
  )


  ece <- 0


  for (
    b in seq_len(
      bins
    )
  ) {

    idx <- which(
      bin_id == b
    )


    if (
      length(idx) == 0
    ) {

      next
    }


    accuracy_bin <- mean(
      correct[idx]
    )


    confidence_bin <- mean(
      confidence[idx]
    )


    weight <-
      length(idx) /
      length(truth)


    ece <-
      ece +
      weight *
      abs(
        accuracy_bin -
        confidence_bin
      )
  }


  ece
}


test_ece_before <-
  multiclass_ece(

    test_truth,

    prob_test
  )


test_ece_after <-
  multiclass_ece(

    test_truth,

    prob_test_calibrated
  )


# ============================================================
# 24. RESUMEN CALIBRACIÓN
# ============================================================

calibration_summary <- data.table(

  metric = c(

    "Validation log-loss",

    "Test log-loss",

    "Test Brier score",

    "Test ECE"
  ),

  before = c(

    validation_logloss_before,

    test_logloss_before,

    test_brier_before,

    test_ece_before
  ),

  after = c(

    validation_logloss_after,

    test_logloss_after,

    test_brier_after,

    test_ece_after
  )
)


calibration_summary[
  ,
  improvement :=
    before -
    after
]


# ============================================================
# 25. RELIABILITY TABLE
# ============================================================

predicted_index_test <- max.col(

  prob_test_calibrated,

  ties.method =
    "first"
)


predicted_class_test <-
  state_levels[
    predicted_index_test
  ]


confidence_test <- apply(

  prob_test_calibrated,

  1,

  max
)


correct_test <-
  predicted_class_test ==
  test_truth


reliability_table <- data.table(

  confidence =
    confidence_test,

  correct =
    correct_test
)


reliability_table[
  ,
  confidence_bin :=
    cut(

      confidence,

      breaks =
        seq(
          0,
          1,
          by =
            0.10
        ),

      include.lowest =
        TRUE
    )
]


reliability_summary <- reliability_table[
  ,
  .(

    observations =
      .N,

    mean_confidence =
      round(
        mean(
          confidence
        ),
        4
      ),

    empirical_accuracy =
      round(
        mean(
          correct
        ),
        4
      )
  ),

  by =
    confidence_bin
]


# ============================================================
# 26. PREDICCIONES CALIBRADAS
# ============================================================

calibrated_predictions <- test[
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


calibrated_predictions[
  ,
  predicted_state :=
    predicted_class_test
]


calibrated_predictions[
  ,
  confidence :=
    confidence_test
]


for (
  i in seq_along(
    state_levels
  )
) {

  cl <- state_levels[i]

  new_col <- paste0(
    "prob_",
    tolower(cl)
  )

  calibrated_predictions[
    ,
    (new_col) :=
      prob_test_calibrated[
        ,
        i
      ]
  ]
}


# ============================================================
# 27. GUARDAR
# ============================================================

dir.create(

  "outputs/models",

  recursive =
    TRUE,

  showWarnings =
    FALSE
)


xgb.save(

  selected_model,

  "outputs/models/xgboost_selected_events360.json"
)


saveRDS(

  best_temperature,

  "outputs/models/calibration_temperature.rds"
)


fwrite(

  calibration_summary,

  "outputs/models/calibration_summary.csv",

  bom =
    TRUE
)


fwrite(

  reliability_summary,

  "outputs/models/calibration_reliability.csv",

  bom =
    TRUE
)


fwrite(

  calibrated_predictions,

  "outputs/models/calibrated_test_predictions.csv",

  bom =
    TRUE
)


# ============================================================
# 28. SOLO 2 VISUALIZACIONES
# ============================================================

View(
  calibration_summary
)

View(
  reliability_summary
)


# ============================================================
# 29. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("CALIBRATION TERMINADA\n")
cat("=====================================\n")

cat(
  "Temperature:",
  round(
    best_temperature,
    4
  ),
  "\n"
)

cat(
  "Validation log-loss before:",
  round(
    validation_logloss_before,
    4
  ),
  "\n"
)

cat(
  "Validation log-loss after:",
  round(
    validation_logloss_after,
    4
  ),
  "\n"
)

cat(
  "Test log-loss before:",
  round(
    test_logloss_before,
    4
  ),
  "\n"
)

cat(
  "Test log-loss after:",
  round(
    test_logloss_after,
    4
  ),
  "\n"
)

cat(
  "Test Brier before:",
  round(
    test_brier_before,
    4
  ),
  "\n"
)

cat(
  "Test Brier after:",
  round(
    test_brier_after,
    4
  ),
  "\n"
)

cat(
  "Test ECE before:",
  round(
    test_ece_before,
    4
  ),
  "\n"
)

cat(
  "Test ECE after:",
  round(
    test_ece_after,
    4
  ),
  "\n"
)