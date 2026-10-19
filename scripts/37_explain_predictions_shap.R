library(data.table)
library(xgboost)

# ============================================================
# 1. CARGAR DATOS
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
# 3. FEATURES DEL MODELO FINAL
# ============================================================

features <- c(

  # CONTEXTO
  "minute",
  "game_state_num",

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
# 4. VALIDAR FEATURES
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
# 5. IMPUTACIÓN
#
# Medianas calculadas SOLO con TRAIN.
# ============================================================

for (col in features) {

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
# 6. TEST
# ============================================================

test <- copy(
  dt[
    dataset_split == "TEST"
  ]
)

x_test <- as.matrix(
  test[
    ,
    ..features
  ]
)

dtest <- xgb.DMatrix(
  x_test
)


# ============================================================
# 7. CARGAR MODELO XGBOOST ACTUAL
#
# Paso 32:
# alpha XGBoost = 1.00
# por lo tanto éste es nuestro predictor actual.
# ============================================================

model_path <- "outputs/models/blend_xgboost_model.json"

if (
  !file.exists(model_path)
) {

  stop(
    paste(
      "No encontré:",
      model_path
    )
  )
}

model <- xgb.load(
  model_path
)


# ============================================================
# 8. PROBABILIDADES
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
      "Dimensiones inesperadas de probabilidades."
    )
  }

  colnames(
    probs
  ) <- state_levels

  probs
}


prob_test <- format_probs(
  predict(
    model,
    dtest
  ),
  nrow(test)
)


predicted_index <- max.col(
  prob_test,
  ties.method = "first"
)

predicted_state <- state_levels[
  predicted_index
]

confidence <- apply(
  prob_test,
  1,
  max
)


# ============================================================
# 9. SHAP NATIVO DE XGBOOST
# ============================================================

cat("\nCalculando SHAP...\n")

raw_shap <- tryCatch(

  predict(
    model,
    dtest,
    predcontrib = TRUE,
    strict_shape = TRUE
  ),

  error = function(e) {

    predict(
      model,
      dtest,
      predcontrib = TRUE
    )
  }
)


# ============================================================
# 10. NORMALIZAR FORMATO SHAP
#
# Queremos:
#
# filas x (features + bias) x clases
# ============================================================

normalize_shap <- function(
  shap_object,
  n_rows,
  n_features,
  n_classes
) {

  expected_feature_dim <-
    n_features + 1

  dims <- dim(
    shap_object
  )


  # ----------------------------------------------------------
  # CASO 3D
  # ----------------------------------------------------------

  if (
    length(dims) == 3
  ) {

    row_dim <- which(
      dims == n_rows
    )[1]

    feature_dim <- which(
      dims == expected_feature_dim
    )[1]

    class_dim <- which(
      dims == n_classes
    )[1]


    if (
      any(
        is.na(
          c(
            row_dim,
            feature_dim,
            class_dim
          )
        )
      )
    ) {

      stop(
        paste(
          "No pude interpretar dimensiones SHAP:",
          paste(
            dims,
            collapse = " x "
          )
        )
      )
    }


    result <- aperm(
      shap_object,
      c(
        row_dim,
        feature_dim,
        class_dim
      )
    )


    return(
      result
    )
  }


  # ----------------------------------------------------------
  # CASO MATRIX
  # n x ((p+1) * classes)
  # ----------------------------------------------------------

  if (
    is.matrix(
      shap_object
    ) &&
    nrow(shap_object) == n_rows &&
    ncol(shap_object) ==
      expected_feature_dim *
      n_classes
  ) {

    result <- array(
      NA_real_,
      dim = c(
        n_rows,
        expected_feature_dim,
        n_classes
      )
    )


    for (
      k in seq_len(
        n_classes
      )
    ) {

      start_col <-
        (
          k - 1
        ) *
        expected_feature_dim +
        1

      end_col <-
        k *
        expected_feature_dim


      result[
        ,
        ,
        k
      ] <- shap_object[
        ,
        start_col:end_col,
        drop = FALSE
      ]
    }


    return(
      result
    )
  }


  stop(
    paste(
      "Formato SHAP no reconocido. Dimensiones:",
      paste(
        dims,
        collapse = " x "
      )
    )
  )
}


shap_array <- normalize_shap(
  raw_shap,
  n_rows = nrow(test),
  n_features = length(features),
  n_classes = length(state_levels)
)


# ============================================================
# 11. VALIDAR DIMENSIONES
# ============================================================

expected_dims <- c(
  nrow(test),
  length(features) + 1,
  length(state_levels)
)

if (
  !all(
    dim(shap_array) ==
      expected_dims
  )
) {

  stop(
    paste(
      "SHAP quedó con dimensiones:",
      paste(
        dim(shap_array),
        collapse = " x "
      )
    )
  )
}


# ============================================================
# 12. NOMBRES
# ============================================================

dimnames(
  shap_array
)[[2]] <- c(
  features,
  "BIAS"
)

dimnames(
  shap_array
)[[3]] <-
  state_levels


# ============================================================
# 13. IMPORTANCIA GLOBAL SHAP
#
# Mean absolute SHAP
# promediado en observaciones y clases.
# ============================================================

global_importance_values <- sapply(
  seq_along(
    features
  ),
  function(j) {

    mean(
      abs(
        shap_array[
          ,
          j,
        ]
      ),
      na.rm = TRUE
    )
  }
)


shap_global_importance <- data.table(

  feature =
    features,

  mean_abs_shap =
    global_importance_values
)


setorder(
  shap_global_importance,
  -mean_abs_shap
)


shap_global_importance[
  ,
  rank :=
    .I
]


# ============================================================
# 14. CLASIFICAR ORIGEN DE FEATURE
# ============================================================

shap_global_importance[
  ,
  source :=
    fifelse(

      feature %in%
        c(
          "minute",
          "game_state_num",
          state_features
        ),

      "CONTEXT",

      fifelse(

        grepl(
          "local_superiority|pressure_share|nearest_opponent|support_share",
          feature
        ),

        "360",

        "EVENTS"
      )
    )
]


# ============================================================
# 15. IMPORTANCIA POR FUENTE
# ============================================================

importance_by_source <- shap_global_importance[
  ,
  .(
    total_mean_abs_shap =
      sum(
        mean_abs_shap,
        na.rm = TRUE
      )
  ),
  by = source
]


importance_by_source[
  ,
  share_pct :=
    round(
      total_mean_abs_shap /
        sum(
          total_mean_abs_shap
        ) *
        100,
      2
    )
]


setorder(
  importance_by_source,
  -share_pct
)


# ============================================================
# 16. IMPORTANCIA SHAP POR CLASE
# ============================================================

class_importance_list <- list()

counter <- 1


for (
  k in seq_along(
    state_levels
  )
) {

  cl <- state_levels[k]


  class_values <- sapply(
    seq_along(
      features
    ),
    function(j) {

      mean(
        abs(
          shap_array[
            ,
            j,
            k
          ]
        ),
        na.rm = TRUE
      )
    }
  )


  tmp <- data.table(

    tactical_state =
      cl,

    feature =
      features,

    mean_abs_shap =
      class_values
  )


  setorder(
    tmp,
    -mean_abs_shap
  )


  tmp[
    ,
    rank :=
      .I
  ]


  class_importance_list[
    [
      counter
    ]
  ] <- tmp


  counter <- counter + 1
}


shap_by_class <- rbindlist(
  class_importance_list
)


shap_by_class_top10 <- shap_by_class[
  rank <= 10
]


# ============================================================
# 17. SELECCIONAR UN EJEMPLO REPRESENTATIVO POR ESTADO
#
# Predicción correcta con mayor confianza.
# ============================================================

test_explain <- test[
  ,
  .(
    match_id,
    match_date,
    minute,
    opponent,
    future_tactical_state_5min
  )
]


test_explain[
  ,
  predicted_state :=
    predicted_state
]


test_explain[
  ,
  confidence :=
    confidence
]


test_explain[
  ,
  row_id :=
    .I
]


correct_predictions <- test_explain[
  predicted_state ==
    future_tactical_state_5min
]


representative_examples <- correct_predictions[
  order(
    -confidence
  ),
  head(
    .SD,
    1
  ),
  by = predicted_state
]


# ============================================================
# 18. EXPLICACIONES LOCALES
#
# Top 5 variables por ejemplo.
# ============================================================

coach_explanations_list <- list()

counter <- 1


for (
  i in seq_len(
    nrow(
      representative_examples
    )
  )
) {

  example <- representative_examples[
    i
  ]


  row_id <- example$row_id


  state <- example$predicted_state


  class_index <- match(
    state,
    state_levels
  )


  shap_values <- shap_array[
    row_id,
    seq_along(
      features
    ),
    class_index
  ]


  feature_values <- as.numeric(
    test[
      row_id,
      ..features
    ]
  )


  local <- data.table(

    match_id =
      example$match_id,

    minute =
      example$minute,

    opponent =
      example$opponent,

    predicted_state =
      state,

    confidence =
      round(
        example$confidence,
        4
      ),

    feature =
      features,

    feature_value =
      feature_values,

    shap_value =
      shap_values
  )


  local[
    ,
    abs_shap :=
      abs(
        shap_value
      )
  ]


  local[
    ,
    effect :=
      fifelse(
        shap_value > 0,
        paste(
          "Pushes toward",
          state
        ),
        paste(
          "Pushes away from",
          state
        )
      )
  ]


  setorder(
    local,
    -abs_shap
  )


  coach_explanations_list[
    [
      counter
    ]
  ] <- local[
    1:min(
      5,
      .N
    )
  ]


  counter <- counter + 1
}


coach_explanations <- rbindlist(
  coach_explanations_list
)


# ============================================================
# 19. NOMBRES MÁS ENTENDIBLES
# ============================================================

feature_dictionary <- c(

  minute =
    "Match minute",

  game_state_num =
    "Score state",

  current_state_control =
    "Currently in CONTROL",

  current_state_pressure =
    "Currently in PRESSURE",

  current_state_transition =
    "Currently in TRANSITION",

  current_state_protection =
    "Currently in PROTECTION",

  current_state_urgency =
    "Currently in URGENCY",

  america_field_tilt_last5 =
    "Field tilt - last 5 min",

  america_field_tilt_last3 =
    "Field tilt - last 3 min",

  entries_final_third_diff_last5 =
    "Final-third entry advantage - 5 min",

  entries_final_third_diff_last3 =
    "Final-third entry advantage - 3 min",

  entries_box_diff_last5 =
    "Box-entry advantage - 5 min",

  entries_box_diff_last3 =
    "Box-entry advantage - 3 min",

  progressive_pass_diff_last5 =
    "Progressive-pass advantage - 5 min",

  progressive_pass_diff_last3 =
    "Progressive-pass advantage - 3 min",

  america_progressive_carries_last5 =
    "América progressive carries - 5 min",

  america_progressive_carries_last3 =
    "América progressive carries - 3 min",

  america_progressive_passes_last5 =
    "América progressive passes - 5 min",

  america_progressive_passes_last3 =
    "América progressive passes - 3 min",

  pressure_diff_last5 =
    "Pressure advantage - 5 min",

  pressure_diff_last3 =
    "Pressure advantage - 3 min",

  recovery_height_diff_last5 =
    "Recovery height advantage - 5 min",

  recovery_height_diff_last3 =
    "Recovery height advantage - 3 min",

  america_local_superiority_10_last5 =
    "Local superiority within 10m - 5 min",

  america_local_superiority_10_last3 =
    "Local superiority within 10m - 3 min",

  america_pressure_share_5_last5 =
    "Opponent pressure nearby - 5 min",

  america_pressure_share_5_last3 =
    "Opponent pressure nearby - 3 min",

  america_nearest_opponent_distance_last5 =
    "Nearest opponent distance - 5 min",

  america_nearest_opponent_distance_last3 =
    "Nearest opponent distance - 3 min",

  america_support_share_10_last5 =
    "Nearby teammate support - 5 min",

  america_support_share_10_last3 =
    "Nearby teammate support - 3 min"
)


shap_global_importance[
  ,
  display_name :=
    feature_dictionary[
      feature
    ]
]


shap_by_class_top10[
  ,
  display_name :=
    feature_dictionary[
      feature
    ]
]


coach_explanations[
  ,
  display_name :=
    feature_dictionary[
      feature
    ]
]


# ============================================================
# 20. TOP 20 GLOBAL
# ============================================================

shap_global_top20 <- shap_global_importance[
  1:min(
    20,
    .N
  )
]


# ============================================================
# 21. GUARDAR RESULTADOS
# ============================================================

dir.create(
  "outputs/models/interpretability",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  shap_global_importance,
  "outputs/models/interpretability/shap_global_importance.csv",
  bom = TRUE
)


fwrite(
  shap_by_class,
  "outputs/models/interpretability/shap_by_class.csv",
  bom = TRUE
)


fwrite(
  importance_by_source,
  "outputs/models/interpretability/shap_importance_by_source.csv",
  bom = TRUE
)


fwrite(
  coach_explanations,
  "outputs/models/interpretability/coach_explanations.csv",
  bom = TRUE
)


# ============================================================
# 22. SOLO 3 VISTAS
# ============================================================

View(
  shap_global_top20
)


View(
  importance_by_source
)


View(
  coach_explanations
)


# ============================================================
# 23. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("SHAP INTERPRETABILITY TERMINADO\n")
cat("=====================================\n")

cat(
  "Test observations explained:",
  nrow(test),
  "\n"
)

cat(
  "Features:",
  length(features),
  "\n"
)

cat(
  "Representative coach examples:",
  uniqueN(
    coach_explanations$
      predicted_state
  ),
  "\n"
)

cat(
  "Top global feature:",
  shap_global_importance[
    1,
    feature
  ],
  "\n"
)