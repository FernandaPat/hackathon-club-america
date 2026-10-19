library(data.table)

# ============================================================
# 1. CARGAR DATASET ORIGINAL MODELABLE
# ============================================================

dt <- fread(
  "data/processed/america_male/model_ready_predictive.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

# Predicciones del modelo actual.
# En el Paso 32 alpha_xgboost = 1,
# por lo que estas predicciones representan al XGBoost actual.
pred <- fread(
  "outputs/models/blended_test_predictions.csv",
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


# ============================================================
# 2. VALIDAR PREDICCIONES
# ============================================================

required_pred_cols <- c(
  "match_id",
  "minute",
  "future_tactical_state_5min",
  "predicted_state",
  "prob_control",
  "prob_pressure",
  "prob_transition",
  "prob_protection",
  "prob_urgency"
)

missing_pred_cols <- setdiff(
  required_pred_cols,
  names(pred)
)

if (
  length(missing_pred_cols) > 0
) {

  stop(
    paste(
      "Faltan columnas en blended_test_predictions.csv:",
      paste(
        missing_pred_cols,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 3. CONFUSION MATRIX - CONTEOS
# ============================================================

confusion_counts <- pred[
  ,
  .N,
  by = .(
    truth = future_tactical_state_5min,
    prediction = predicted_state
  )
]


# ============================================================
# 4. CONFUSION MATRIX NORMALIZADA POR ESTADO REAL
#
# Ejemplo:
# PRESSURE -> CONTROL = 55%
#
# significaría que 55% de los PRESSURE reales
# fueron clasificados como CONTROL.
# ============================================================

confusion_normalized <- copy(
  confusion_counts
)

confusion_normalized[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    ),
  by = truth
]

setorder(
  confusion_normalized,
  truth,
  -percentage
)


# ============================================================
# 5. SOLO PRESSURE Y TRANSITION
# ============================================================

pressure_transition_confusion <- confusion_normalized[
  truth %in%
    c(
      "PRESSURE",
      "TRANSITION"
    )
]


# ============================================================
# 6. PROBABILIDAD PROMEDIO SEGÚN ESTADO REAL
#
# Nos ayuda a saber si el modelo está "cerca"
# aunque no ponga la clase correcta como Top-1.
# ============================================================

probability_by_truth <- pred[
  ,
  .(
    observations = .N,

    avg_prob_control =
      round(
        mean(
          prob_control,
          na.rm = TRUE
        ),
        4
      ),

    avg_prob_pressure =
      round(
        mean(
          prob_pressure,
          na.rm = TRUE
        ),
        4
      ),

    avg_prob_transition =
      round(
        mean(
          prob_transition,
          na.rm = TRUE
        ),
        4
      ),

    avg_prob_protection =
      round(
        mean(
          prob_protection,
          na.rm = TRUE
        ),
        4
      ),

    avg_prob_urgency =
      round(
        mean(
          prob_urgency,
          na.rm = TRUE
        ),
        4
      )
  ),
  by = future_tactical_state_5min
]


# ============================================================
# 7. FEATURES A ANALIZAR
#
# Usamos TRAIN + VALIDATION para estudiar separabilidad.
# NO usamos TEST para diseñar nuevas reglas.
# ============================================================

features <- c(

  # CONTEXTO
  "minute",
  "game_state_num",

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
# 8. DATA DE DESARROLLO
# ============================================================

dev <- copy(
  dt[
    dataset_split %in%
      c(
        "TRAIN",
        "VALIDATION"
      )
  ]
)

dev[
  ,
  target :=
    as.character(
      future_tactical_state_5min
    )
]


# ============================================================
# 9. IMPUTACIÓN SOLO CON TRAIN
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

  dev[
    is.na(get(col)),
    (col) := med
  ]
}


# ============================================================
# 10. FUNCIÓN DE SEPARABILIDAD
#
# Cohen-like standardized difference.
#
# ~0     = casi idénticas
# ~0.2   = diferencia pequeña
# ~0.5   = diferencia moderada
# >=0.8  = separación fuerte
# ============================================================

effect_size <- function(
  x1,
  x2
) {

  x1 <- x1[
    is.finite(x1)
  ]

  x2 <- x2[
    is.finite(x2)
  ]


  if (
    length(x1) < 2 ||
    length(x2) < 2
  ) {

    return(
      NA_real_
    )
  }


  sd1 <- sd(x1)

  sd2 <- sd(x2)


  pooled_sd <- sqrt(
    (
      sd1^2 +
      sd2^2
    ) / 2
  )


  if (
    !is.finite(pooled_sd) ||
    pooled_sd == 0
  ) {

    return(
      0
    )
  }


  abs(
    mean(x1) -
    mean(x2)
  ) /
    pooled_sd
}


# ============================================================
# 11. FUNCIÓN PARA COMPARAR DOS ESTADOS
# ============================================================

compare_states <- function(
  state_a,
  state_b
) {

  result_list <- vector(
    "list",
    length(features)
  )


  for (
    i in seq_along(
      features
    )
  ) {

    feature <- features[i]


    a <- dev[
      target == state_a,
      get(feature)
    ]


    b <- dev[
      target == state_b,
      get(feature)
    ]


    result_list[[i]] <- data.table(

      comparison =
        paste0(
          state_a,
          "_vs_",
          state_b
        ),

      feature =
        feature,

      mean_state_a =
        mean(
          a,
          na.rm = TRUE
        ),

      mean_state_b =
        mean(
          b,
          na.rm = TRUE
        ),

      effect_size =
        effect_size(
          a,
          b
        )
    )
  }


  result <- rbindlist(
    result_list
  )


  result[
    ,
    mean_state_a :=
      round(
        mean_state_a,
        4
      )
  ]

  result[
    ,
    mean_state_b :=
      round(
        mean_state_b,
        4
      )
  ]

  result[
    ,
    effect_size :=
      round(
        effect_size,
        4
      )
  ]


  setorder(
    result,
    -effect_size
  )


  result
}


# ============================================================
# 12. COMPARACIONES IMPORTANTES
# ============================================================

pressure_vs_control <- compare_states(
  "PRESSURE",
  "CONTROL"
)

pressure_vs_transition <- compare_states(
  "PRESSURE",
  "TRANSITION"
)

transition_vs_control <- compare_states(
  "TRANSITION",
  "CONTROL"
)


# ============================================================
# 13. UNIR SEPARABILIDAD
# ============================================================

pairwise_separability <- rbindlist(
  list(
    pressure_vs_control,
    pressure_vs_transition,
    transition_vs_control
  )
)


# ============================================================
# 14. TOP FEATURES DE CADA COMPARACIÓN
# ============================================================

pairwise_separability_top <- pairwise_separability[
  order(
    comparison,
    -effect_size
  ),
  head(
    .SD,
    10
  ),
  by = comparison
]


# ============================================================
# 15. RESUMEN DE SEPARABILIDAD
#
# Promedio de las 5 mejores variables por pareja.
# ============================================================

separability_summary <- pairwise_separability[
  order(
    -effect_size
  ),
  head(
    .SD,
    5
  ),
  by = comparison
][
  ,
  .(
    mean_top5_effect =
      round(
        mean(
          effect_size,
          na.rm = TRUE
        ),
        4
      ),

    max_effect =
      round(
        max(
          effect_size,
          na.rm = TRUE
        ),
        4
      )
  ),
  by = comparison
]


# ============================================================
# 16. ¿A DÓNDE SE ESTÁ YENDO PRESSURE?
# ============================================================

pressure_error_summary <- pred[
  future_tactical_state_5min ==
    "PRESSURE",
  .N,
  by = predicted_state
]

pressure_error_summary[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]

setorder(
  pressure_error_summary,
  -percentage
)


# ============================================================
# 17. ¿A DÓNDE SE ESTÁ YENDO TRANSITION?
# ============================================================

transition_error_summary <- pred[
  future_tactical_state_5min ==
    "TRANSITION",
  .N,
  by = predicted_state
]

transition_error_summary[
  ,
  percentage :=
    round(
      N / sum(N) * 100,
      2
    )
]

setorder(
  transition_error_summary,
  -percentage
)


# ============================================================
# 18. GUARDAR RESULTADOS
# ============================================================

dir.create(
  "outputs/models/diagnostics",
  recursive = TRUE,
  showWarnings = FALSE
)


fwrite(
  confusion_counts,
  "outputs/models/diagnostics/confusion_counts.csv",
  bom = TRUE
)


fwrite(
  confusion_normalized,
  "outputs/models/diagnostics/confusion_normalized.csv",
  bom = TRUE
)


fwrite(
  probability_by_truth,
  "outputs/models/diagnostics/probability_by_truth.csv",
  bom = TRUE
)


fwrite(
  pairwise_separability,
  "outputs/models/diagnostics/pairwise_separability.csv",
  bom = TRUE
)


fwrite(
  separability_summary,
  "outputs/models/diagnostics/separability_summary.csv",
  bom = TRUE
)


fwrite(
  pressure_error_summary,
  "outputs/models/diagnostics/pressure_errors.csv",
  bom = TRUE
)


fwrite(
  transition_error_summary,
  "outputs/models/diagnostics/transition_errors.csv",
  bom = TRUE
)


# ============================================================
# 19. SOLO 3 VISTAS
# ============================================================

View(
  pressure_transition_confusion
)

View(
  separability_summary
)

View(
  pairwise_separability_top
)


# ============================================================
# 20. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("STATE SEPARABILITY DIAGNOSTIC DONE\n")
cat("=====================================\n")

cat(
  "PRESSURE observations:",
  pred[
    future_tactical_state_5min ==
      "PRESSURE",
    .N
  ],
  "\n"
)

cat(
  "TRANSITION observations:",
  pred[
    future_tactical_state_5min ==
      "TRANSITION",
    .N
  ],
  "\n"
)

cat(
  "Comparaciones analizadas:",
  uniqueN(
    pairwise_separability$
      comparison
  ),
  "\n"
)