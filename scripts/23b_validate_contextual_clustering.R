library(data.table)
library(cluster)

# ============================================================
# 1. CARGAR DATASET FINAL DE 5 MINUTOS
# ============================================================

dt <- fread(
  "data/processed/america_male/model_with_tactical_states.csv",
  encoding = "UTF-8",
  na.strings = "NA"
)

setorder(
  dt,
  match_id,
  minute
)

cat("\n=====================================\n")
cat("CONTEXTUAL CLUSTER VALIDATION\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. SOLO OBSERVACIONES CON ESTADO TÁCTICO
# ============================================================

analysis_dt <- dt[
  !is.na(tactical_state)
]

cat(
  "Observaciones clasificadas:",
  nrow(analysis_dt),
  "\n"
)


# ============================================================
# 3. ASEGURAR GAME STATE NUMÉRICO
# ============================================================

if (!"game_state_num" %in% names(analysis_dt)) {

  analysis_dt[
    ,
    game_state_num := fifelse(
      game_state == "Winning",
      1,
      fifelse(
        game_state == "Losing",
        -1,
        fifelse(
          game_state == "Drawing",
          0,
          NA_real_
        )
      )
    )
  ]
}


# ============================================================
# 4. FASE DEL PARTIDO
#
# Además del minuto continuo, añadimos una variable
# categórica ordinal para contexto operacional.
#
# 1 = inicio
# 2 = desarrollo
# 3 = segunda mitad
# 4 = tramo final
# ============================================================

analysis_dt[
  ,
  match_phase := fifelse(
    minute < 30,
    1,
    fifelse(
      minute < 60,
      2,
      fifelse(
        minute < 75,
        3,
        4
      )
    )
  )
]


# ============================================================
# 5. FEATURES DEL CLUSTERING CONTEXTUAL
#
# Ahora sí incluimos:
# - comportamiento táctico
# - 360
# - marcador
# - tiempo
# ============================================================

cluster_features <- c(

  # Territorio
  "america_field_tilt_last5",

  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  # Progresión
  "progressive_pass_diff_last5",
  "america_progressive_carries_last5",

  # Presión
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # StatsBomb 360
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",
  "america_support_share_10_last5",

  # Contexto
  "game_state_num",
  "minute",
  "match_phase"
)


# ============================================================
# 6. VALIDAR VARIABLES
# ============================================================

feature_validation_contextual <- data.table(

  variable = cluster_features,

  exists =
    cluster_features %in%
    names(analysis_dt)
)

View(
  feature_validation_contextual
)


missing_features <- feature_validation_contextual[
  exists == FALSE,
  variable
]


if (length(missing_features) > 0) {

  stop(
    paste(
      "Faltan variables:",
      paste(
        missing_features,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 7. CONSTRUIR MATRIZ
# ============================================================

cluster_data <- copy(
  analysis_dt[
    ,
    ..cluster_features
  ]
)


# ============================================================
# 8. MISSING VALUES
#
# Imputación por mediana.
# ============================================================

imputation_results <- list()


for (col in cluster_features) {

  missing_before <- sum(
    is.na(
      cluster_data[[col]]
    )
  )

  mediana <- median(
    cluster_data[[col]],
    na.rm = TRUE
  )

  if (!is.finite(mediana)) {
    mediana <- 0
  }

  cluster_data[
    is.na(get(col)),
    (col) := mediana
  ]

  imputation_results[[col]] <- data.table(

    variable = col,

    missing_before =
      missing_before,

    imputation_value =
      mediana
  )
}


imputation_summary_contextual <- rbindlist(
  imputation_results
)

View(
  imputation_summary_contextual
)


# ============================================================
# 9. ESTANDARIZAR
# ============================================================

scaled_matrix <- scale(
  cluster_data
)


scaled_matrix[
  !is.finite(scaled_matrix)
] <- 0


# ============================================================
# 10. MUESTRA PARA SILHOUETTE
#
# Evitamos calcular distancias completas sobre
# las ~20 mil observaciones.
# ============================================================

set.seed(2026)

sample_size <- min(
  3000,
  nrow(scaled_matrix)
)


sample_idx <- sample(
  seq_len(
    nrow(scaled_matrix)
  ),
  size = sample_size,
  replace = FALSE
)


silhouette_matrix <- scaled_matrix[
  sample_idx,
  ,
  drop = FALSE
]


# ============================================================
# 11. PROBAR K = 2 ... 8
# ============================================================

k_results <- list()


for (k in 2:8) {

  cat(
    "Probando clustering contextual k =",
    k,
    "\n"
  )

  set.seed(
    2026 + k
  )

  model_k <- kmeans(

    silhouette_matrix,

    centers = k,

    nstart = 40,

    iter.max = 150
  )


  sil <- silhouette(

    model_k$cluster,

    dist(
      silhouette_matrix
    )
  )


  k_results[[as.character(k)]] <- data.table(

    k = k,

    avg_silhouette =
      mean(
        sil[
          ,
          "sil_width"
        ]
      ),

    total_withinss =
      model_k$tot.withinss
  )
}


k_evaluation_contextual <- rbindlist(
  k_results
)


k_evaluation_contextual[
  ,
  avg_silhouette :=
    round(
      avg_silhouette,
      4
    )
]


View(
  k_evaluation_contextual
)


# ============================================================
# 12. MEJOR K
# ============================================================

best_k <- k_evaluation_contextual[
  which.max(
    avg_silhouette
  ),
  k
]


cat(
  "\nMejor K contextual:",
  best_k,
  "\n"
)


# ============================================================
# 13. MODELO K = 5
#
# Lo mantenemos para compararlo con nuestros
# cinco estados tácticos.
# ============================================================

set.seed(2026)


kmeans_5 <- kmeans(

  scaled_matrix,

  centers = 5,

  nstart = 60,

  iter.max = 250
)


analysis_dt[
  ,
  contextual_cluster_5 :=
    factor(
      kmeans_5$cluster
    )
]


# ============================================================
# 14. TAMAÑO DE CLUSTERS
# ============================================================

cluster_sizes_contextual <- analysis_dt[
  ,
  .N,
  by = contextual_cluster_5
]


cluster_sizes_contextual[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    )
]


setorder(
  cluster_sizes_contextual,
  contextual_cluster_5
)


View(
  cluster_sizes_contextual
)


# ============================================================
# 15. PERFIL PROMEDIO DE LOS CLUSTERS
# ============================================================

cluster_profiles_contextual <- analysis_dt[
  ,
  lapply(
    .SD,
    mean,
    na.rm = TRUE
  ),
  by = contextual_cluster_5,
  .SDcols = cluster_features
]


View(
  cluster_profiles_contextual
)


# ============================================================
# 16. CLUSTER VS ESTADO
# ============================================================

cluster_state_profile_contextual <- analysis_dt[
  ,
  .N,
  by = .(
    contextual_cluster_5,
    tactical_state
  )
]


cluster_state_profile_contextual[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    ),
  by = contextual_cluster_5
]


setorder(
  cluster_state_profile_contextual,
  contextual_cluster_5,
  -percentage
)


View(
  cluster_state_profile_contextual
)


# ============================================================
# 17. ESTADO DOMINANTE DE CADA CLUSTER
# ============================================================

dominant_state_contextual <- cluster_state_profile_contextual[
  ,
  .SD[
    which.max(
      percentage
    )
  ],
  by = contextual_cluster_5
]


View(
  dominant_state_contextual
)


# ============================================================
# 18. ASIGNAR LABEL DOMINANTE
# ============================================================

cluster_labels <- dominant_state_contextual[
  ,
  .(
    contextual_cluster_5,

    contextual_cluster_label =
      tactical_state,

    cluster_dominance_pct =
      percentage
  )
]


analysis_dt <- merge(

  analysis_dt,

  cluster_labels,

  by = "contextual_cluster_5",

  all.x = TRUE
)


# ============================================================
# 19. ACUERDO SIMPLE
# ============================================================

analysis_dt[
  ,
  contextual_cluster_agree :=
    tactical_state ==
    contextual_cluster_label
]


agreement_summary_contextual <- analysis_dt[
  ,
  .(

    observations = .N,

    agreements =
      sum(
        contextual_cluster_agree,
        na.rm = TRUE
      ),

    disagreements =
      sum(
        !contextual_cluster_agree,
        na.rm = TRUE
      ),

    agreement_pct =
      round(
        mean(
          contextual_cluster_agree,
          na.rm = TRUE
        ) *
          100,
        2
      )
  )
]


View(
  agreement_summary_contextual
)


# ============================================================
# 20. CONFUSION TABLE
#
# La necesitaremos para métricas de asociación.
# ============================================================

confusion_table <- table(

  analysis_dt$tactical_state,

  analysis_dt$contextual_cluster_5
)


# ============================================================
# 21. ADJUSTED RAND INDEX
#
# Implementación manual para no depender de
# paquetes adicionales.
#
# ARI:
# 1   = acuerdo perfecto
# ~0  = parecido a asignación aleatoria
# <0  = peor que aleatorio
# ============================================================

adjusted_rand_index <- function(labels_true, labels_pred) {

  tab <- table(
    labels_true,
    labels_pred
  )

  n <- sum(tab)


  choose2 <- function(x) {

    x * (
      x - 1
    ) / 2
  }


  sum_comb <- sum(
    choose2(tab)
  )


  row_comb <- sum(
    choose2(
      rowSums(tab)
    )
  )


  col_comb <- sum(
    choose2(
      colSums(tab)
    )
  )


  total_comb <- choose2(n)


  expected_index <-
    (
      row_comb *
        col_comb
    ) /
    total_comb


  max_index <-
    (
      row_comb +
        col_comb
    ) /
    2


  denominator <-
    max_index -
    expected_index


  if (
    denominator == 0
  ) {
    return(NA_real_)
  }


  (
    sum_comb -
      expected_index
  ) /
    denominator
}


ari <- adjusted_rand_index(

  analysis_dt$tactical_state,

  analysis_dt$contextual_cluster_5
)


# ============================================================
# 22. CRAMER'S V
#
# Asociación entre estado categórico y cluster.
#
# 0 = no asociación
# 1 = asociación perfecta
# ============================================================

chi <- suppressWarnings(
  chisq.test(
    confusion_table,
    correct = FALSE
  )
)


n_total <- sum(
  confusion_table
)


min_dimension <- min(
  nrow(
    confusion_table
  ) - 1,

  ncol(
    confusion_table
  ) - 1
)


cramers_v <- sqrt(

  as.numeric(
    chi$statistic
  ) /

    (
      n_total *
        min_dimension
    )
)


# ============================================================
# 23. MÉTRICAS DE VALIDACIÓN
# ============================================================

validation_metrics <- data.table(

  metric = c(

    "Best silhouette K",

    "Silhouette K=5",

    "Simple agreement (%)",

    "Adjusted Rand Index",

    "Cramer's V"
  ),

  value = c(

    best_k,

    k_evaluation_contextual[
      k == 5,
      avg_silhouette
    ],

    agreement_summary_contextual$
      agreement_pct,

    round(
      ari,
      4
    ),

    round(
      cramers_v,
      4
    )
  )
)


View(
  validation_metrics
)


# ============================================================
# 24. VALIDAR LOS DOS MACRO-REGÍMENES
#
# Como el análisis previo favoreció k=2,
# también modelamos explícitamente esos dos
# macro-regímenes.
# ============================================================

set.seed(2026)


kmeans_2 <- kmeans(

  scaled_matrix,

  centers = 2,

  nstart = 60,

  iter.max = 250
)


analysis_dt[
  ,
  macro_regime :=
    factor(
      kmeans_2$cluster
    )
]


# ============================================================
# 25. PERFIL DE MACRO-REGÍMENES
# ============================================================

macro_profiles <- analysis_dt[
  ,
  lapply(
    .SD,
    mean,
    na.rm = TRUE
  ),
  by = macro_regime,
  .SDcols = cluster_features
]


View(
  macro_profiles
)


# ============================================================
# 26. ESTADOS DENTRO DE CADA MACRO-RÉGIMEN
# ============================================================

macro_state_profile <- analysis_dt[
  ,
  .N,
  by = .(
    macro_regime,
    tactical_state
  )
]


macro_state_profile[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    ),
  by = macro_regime
]


setorder(
  macro_state_profile,
  macro_regime,
  -percentage
)


View(
  macro_state_profile
)


# ============================================================
# 27. DESACUERDOS PARA INSPECCIÓN
# ============================================================

contextual_disagreements <- analysis_dt[
  contextual_cluster_agree == FALSE,
  .(

    match_id,
    minute,
    opponent,

    game_state,
    game_state_num,

    tactical_state,

    contextual_cluster_5,

    contextual_cluster_label,

    cluster_dominance_pct,

    america_field_tilt_last5,

    progressive_pass_diff_last5,

    pressure_diff_last5,

    entries_final_third_diff_last5,

    entries_box_diff_last5,

    recovery_height_diff_last5,

    america_local_superiority_10_last5,

    america_pressure_share_5_last5
  )
]


View(
  contextual_disagreements
)


# ============================================================
# 28. GUARDAR OUTPUTS
# ============================================================

dir.create(

  "outputs/models",

  recursive = TRUE,

  showWarnings = FALSE
)


fwrite(

  k_evaluation_contextual,

  "outputs/models/contextual_clustering_k_evaluation.csv",

  bom = TRUE
)


fwrite(

  cluster_profiles_contextual,

  "outputs/models/contextual_clustering_profiles.csv",

  bom = TRUE
)


fwrite(

  dominant_state_contextual,

  "outputs/models/contextual_clustering_dominant_states.csv",

  bom = TRUE
)


fwrite(

  validation_metrics,

  "outputs/models/contextual_clustering_validation_metrics.csv",

  bom = TRUE
)


fwrite(

  macro_profiles,

  "outputs/models/tactical_macro_regime_profiles.csv",

  bom = TRUE
)


fwrite(

  macro_state_profile,

  "outputs/models/tactical_macro_regime_states.csv",

  bom = TRUE
)


fwrite(

  analysis_dt,

  "data/processed/america_male/tactical_states_contextual_clustering.csv",

  bom = TRUE,

  na = "NA"
)


# ============================================================
# 29. RESUMEN FINAL
# ============================================================

cat("\n=====================================\n")
cat("CONTEXTUAL CLUSTERING TERMINADO\n")
cat("=====================================\n")

cat(
  "Observaciones:",
  nrow(analysis_dt),
  "\n"
)

cat(
  "Best K:",
  best_k,
  "\n"
)

cat(
  "Agreement K5:",
  agreement_summary_contextual$
    agreement_pct,
  "%\n"
)

cat(
  "ARI:",
  round(
    ari,
    4
  ),
  "\n"
)

cat(
  "Cramer's V:",
  round(
    cramers_v,
    4
  ),
  "\n"
)