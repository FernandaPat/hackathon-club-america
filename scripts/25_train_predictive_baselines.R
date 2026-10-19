library(data.table)
library(nnet)

# ============================================================
# 1. CARGAR DATASET PREDICTIVO
# ============================================================

dt <- fread(
  "data/processed/america_male/model_ready_predictive.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. DEFINIR ORDEN DE CLASES
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
  future_tactical_state_5min :=
    factor(
      future_tactical_state_5min,
      levels = state_levels
    )
]


# ============================================================
# 3. FEATURES DEL MODELO
#
# Usamos contexto actual:
# - estado actual
# - marcador
# - minuto
# - ventana 3 min
# - ventana 5 min
# - contexto rival
# ============================================================

features <- c(

  # Contexto partido
  "minute",
  "game_state_num",

  # Estado actual
  "current_tactical_state",

  # 5 min - territorio
  "america_field_tilt_last5",
  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  # 5 min - progresión
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",
  "america_progressive_passes_last5",

  # 5 min - presión
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # 5 min - 360
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # 3 min - reacción rápida
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

  # Contexto rival reciente
  "opponent_points_last1",
  "opponent_points_last3",
  "opponent_points_last5",
  "opponent_points_weighted"
)


# ============================================================
# 4. VALIDAR FEATURES
# ============================================================

feature_validation <- data.table(
  variable = features,
  exists = features %in% names(dt)
)

missing_features <- feature_validation[
  exists == FALSE,
  variable
]

if (length(missing_features) > 0) {

  View(feature_validation)

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
# 5. PREPARAR FACTOR DEL ESTADO ACTUAL
# ============================================================

dt[
  ,
  current_tactical_state :=
    factor(
      current_tactical_state,
      levels = state_levels
    )
]


# ============================================================
# 6. IMPUTACIÓN POR MEDIANA
#
# SOLO calculamos medianas usando TRAIN.
# Esto evita leakage.
# ============================================================

numeric_features <- features[
  features != "current_tactical_state"
]


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
# 7. CREAR TRAIN / VALIDATION / TEST
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
# 8. FÓRMULA
# ============================================================

formula_multinom <- as.formula(
  paste(
    "future_tactical_state_5min ~",
    paste(
      features,
      collapse = " + "
    )
  )
)


# ============================================================
# 9. ENTRENAR MULTINOMIAL LOGISTIC
# ============================================================

set.seed(2026)

multinom_model <- multinom(
  formula_multinom,
  data = train,
  trace = FALSE,
  MaxNWts = 10000,
  maxit = 500
)


# ============================================================
# 10. FUNCIÓN PARA PROBABILIDADES SEGURAS
# ============================================================

predict_probabilities <- function(model, newdata) {

  probs <- predict(
    model,
    newdata = newdata,
    type = "probs"
  )

  probs <- as.matrix(probs)

  # Asegurar todas las clases
  missing_classes <- setdiff(
    state_levels,
    colnames(probs)
  )

  if (length(missing_classes) > 0) {

    for (cl in missing_classes) {
      probs <- cbind(
        probs,
        setNames(
          rep(0, nrow(probs)),
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


# ============================================================
# 11. PREDICCIONES
# ============================================================

prob_train <- predict_probabilities(
  multinom_model,
  train
)

prob_validation <- predict_probabilities(
  multinom_model,
  validation
)

prob_test <- predict_probabilities(
  multinom_model,
  test
)


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
# 12. FUNCIÓN LOG LOSS
# ============================================================

multiclass_logloss <- function(
  truth,
  probabilities
) {

  truth <- factor(
    truth,
    levels = state_levels
  )

  truth_index <- as.integer(
    truth
  )

  p_true <- probabilities[
    cbind(
      seq_along(truth_index),
      truth_index
    )
  ]

  p_true <- pmax(
    pmin(
      p_true,
      1 - 1e-15
    ),
    1e-15
  )

  -mean(
    log(
      p_true
    )
  )
}


# ============================================================
# 13. MACRO F1
# ============================================================

macro_f1 <- function(
  truth,
  prediction
) {

  truth <- factor(
    truth,
    levels = state_levels
  )

  prediction <- factor(
    prediction,
    levels = state_levels
  )

  f1_scores <- numeric(
    length(
      state_levels
    )
  )


  for (i in seq_along(state_levels)) {

    cl <- state_levels[i]

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
      tp / (
        tp + fp
      )
    )

    recall <- ifelse(
      tp + fn == 0,
      0,
      tp / (
        tp + fn
      )
    )

    f1_scores[i] <- ifelse(
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
  }

  mean(
    f1_scores
  )
}


# ============================================================
# 14. BALANCED ACCURACY
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


# ============================================================
# 15. EVALUACIÓN GENERAL
# ============================================================

evaluate_model <- function(
  dataset_name,
  data,
  probs,
  preds
) {

  truth <- as.character(
    data$
      future_tactical_state_5min
  )

  data.table(

    dataset =
      dataset_name,

    observations =
      length(
        truth
      ),

    accuracy =
      round(
        mean(
          preds ==
            truth
        ) *
          100,
        2
      ),

    balanced_accuracy =
      round(
        balanced_accuracy(
          truth,
          preds
        ) *
          100,
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


model_metrics <- rbind(
  metrics_train,
  metrics_validation,
  metrics_test
)


# ============================================================
# 16. CONFUSION MATRIX TEST
# ============================================================

confusion_test <- data.table(
  truth = as.character(
    test$
      future_tactical_state_5min
  ),
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
# 17. MÉTRICA POR CLASE
# ============================================================

class_metrics <- rbindlist(
  lapply(
    state_levels,
    function(cl) {

      truth <- as.character(
        test$
          future_tactical_state_5min
      )

      pred <- pred_test

      tp <- sum(
        truth == cl &
        pred == cl
      )

      fp <- sum(
        truth != cl &
        pred == cl
      )

      fn <- sum(
        truth == cl &
        pred != cl
      )

      precision <- ifelse(
        tp + fp == 0,
        0,
        tp /
          (
            tp + fp
          )
      )

      recall <- ifelse(
        tp + fn == 0,
        0,
        tp /
          (
            tp + fn
          )
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
# 18. GUARDAR PREDICCIONES TEST
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
# 19. GUARDAR OUTPUTS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)


saveRDS(
  multinom_model,
  "outputs/models/multinomial_baseline_model.rds"
)


fwrite(
  model_metrics,
  "outputs/models/multinomial_model_metrics.csv",
  bom = TRUE
)


fwrite(
  confusion_test,
  "outputs/models/multinomial_test_confusion.csv",
  bom = TRUE
)


fwrite(
  class_metrics,
  "outputs/models/multinomial_test_class_metrics.csv",
  bom = TRUE
)


fwrite(
  test_predictions,
  "outputs/models/multinomial_test_predictions.csv",
  bom = TRUE
)


fwrite(
  imputation_table,
  "outputs/models/multinomial_imputation_values.csv",
  bom = TRUE
)


# ============================================================
# 20. SOLO 3 VISUALIZACIONES
# ============================================================

View(
  model_metrics
)

View(
  class_metrics
)

View(
  confusion_test
)


# ============================================================
# 21. RESUMEN TERMINAL
# ============================================================

cat("\n=====================================\n")
cat("MULTINOMIAL BASELINE TERMINADO\n")
cat("=====================================\n")

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