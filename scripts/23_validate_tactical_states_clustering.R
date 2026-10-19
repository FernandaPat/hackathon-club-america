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
cat("TACTICAL STATE CLUSTER VALIDATION\n")
cat("=====================================\n")

cat("Filas:", nrow(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. QUEDARNOS SOLO CON MINUTOS CLASIFICADOS
# ============================================================

analysis_dt <- dt[
  !is.na(tactical_state)
]

cat(
  "Minutos utilizables:",
  nrow(analysis_dt),
  "\n"
)


# ============================================================
# 3. VARIABLES TÁCTICAS PARA CLUSTERING
#
# IMPORTANTE:
# NO usamos tactical_state como input.
#
# Queremos que el clustering vea únicamente
# comportamiento táctico observado.
# ============================================================

cluster_features <- c(

  # Territorio
  "america_field_tilt_last5",

  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  # Progresión
  "progressive_pass_diff_last5",

  "america_progressive_carries_last5",

  # Presión / recuperación
  "pressure_diff_last5",
  "recovery_height_diff_last5",

  # 360
  "america_local_superiority_10_last5",
  "america_pressure_share_5_last5",
  "america_nearest_opponent_distance_last5",

  # Apoyo estructural
  "america_support_share_10_last5"
)


# ============================================================
# 4. VALIDAR QUE EXISTAN
# ============================================================

feature_validation <- data.table(

  variable = cluster_features,

  exists =
    cluster_features %in%
    names(analysis_dt)
)

View(feature_validation)


missing_features <- feature_validation[
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
# 5. CREAR MATRIZ DE FEATURES
# ============================================================

cluster_data <- copy(
  analysis_dt[
    ,
    ..cluster_features
  ]
)


# ============================================================
# 6. IMPUTACIÓN POR MEDIANA
#
# No ponemos cero arbitrariamente.
#
# Cada NA se reemplaza por la mediana
# de esa variable.
# ============================================================

imputation_summary <- list()

for (col in cluster_features) {

  missing_before <- sum(
    is.na(cluster_data[[col]])
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

  imputation_summary[[col]] <- data.table(

    variable = col,

    missing_before =
      missing_before,

    imputation_value =
      mediana
  )
}


imputation_summary <- rbindlist(
  imputation_summary
)

View(imputation_summary)


# ============================================================
# 7. ESTANDARIZACIÓN
#
# Clustering necesita variables comparables.
# ============================================================

scaled_matrix <- scale(
  cluster_data
)

scaled_matrix[
  !is.finite(scaled_matrix)
] <- 0


# ============================================================
# 8. DATASET PARA SILHOUETTE
#
# Silhouette con 20k observaciones completas puede ser pesado.
# Usamos una muestra reproducible solo para seleccionar k.
# ============================================================

set.seed(2026)

silhouette_sample_size <- min(
  3000,
  nrow(scaled_matrix)
)

sample_idx <- sample(
  seq_len(nrow(scaled_matrix)),
  size = silhouette_sample_size,
  replace = FALSE
)

silhouette_matrix <- scaled_matrix[
  sample_idx,
  ,
  drop = FALSE
]


# ============================================================
# 9. PROBAR K = 2 ... 8
#
# Esto evita decidir arbitrariamente que k=5 es óptimo.
# ============================================================

k_results <- list()

for (k in 2:8) {

  cat(
    "Probando k =",
    k,
    "\n"
  )

  set.seed(
    2026 + k
  )

  km_test <- kmeans(
    silhouette_matrix,
    centers = k,
    nstart = 30,
    iter.max = 100
  )

  sil <- silhouette(
    km_test$cluster,
    dist(
      silhouette_matrix
    )
  )

  avg_silhouette <- mean(
    sil[
      ,
      "sil_width"
    ]
  )

  k_results[[as.character(k)]] <- data.table(

    k = k,

    avg_silhouette =
      avg_silhouette,

    total_withinss =
      km_test$tot.withinss
  )
}


k_evaluation <- rbindlist(
  k_results
)

k_evaluation[
  ,
  avg_silhouette :=
    round(
      avg_silhouette,
      4
    )
]

View(k_evaluation)


# ============================================================
# 10. MEJOR K SEGÚN SILHOUETTE
# ============================================================

best_k <- k_evaluation[
  which.max(
    avg_silhouette
  ),
  k
]

cat(
  "\nMejor k según silhouette:",
  best_k,
  "\n"
)


# ============================================================
# 11. CLUSTERING DE 5 GRUPOS
#
# Aunque best_k sea distinto,
# también hacemos k=5 porque queremos comparar
# contra nuestros cinco estados tácticos.
# ============================================================

set.seed(2026)

kmeans_5 <- kmeans(

  scaled_matrix,

  centers = 5,

  nstart = 50,

  iter.max = 200
)


analysis_dt[
  ,
  cluster_5 :=
    factor(
      kmeans_5$cluster
    )
]


# ============================================================
# 12. TAMAÑO DE CADA CLUSTER
# ============================================================

cluster_sizes <- analysis_dt[
  ,
  .N,
  by = cluster_5
]

cluster_sizes[
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
  cluster_sizes,
  cluster_5
)

View(cluster_sizes)


# ============================================================
# 13. PERFIL PROMEDIO DE CADA CLUSTER
# ============================================================

cluster_profiles <- analysis_dt[
  ,
  lapply(
    .SD,
    mean,
    na.rm = TRUE
  ),
  by = cluster_5,
  .SDcols = cluster_features
]

View(cluster_profiles)


# ============================================================
# 14. ESTADO TÁCTICO VS CLUSTER
# ============================================================

state_cluster_table <- analysis_dt[
  ,
  .N,
  by = .(
    tactical_state,
    cluster_5
  )
]

View(state_cluster_table)


# ============================================================
# 15. PORCENTAJE DE CLUSTER DENTRO DE CADA ESTADO
# ============================================================

state_cluster_pct <- analysis_dt[
  ,
  .N,
  by = .(
    tactical_state,
    cluster_5
  )
]

state_cluster_pct[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    ),
  by = tactical_state
]

setorder(
  state_cluster_pct,
  tactical_state,
  -percentage
)

View(state_cluster_pct)


# ============================================================
# 16. ESTADO DOMINANTE DE CADA CLUSTER
#
# Nos sirve para interpretar qué representa
# cada grupo data-driven.
# ============================================================

cluster_state_profile <- analysis_dt[
  ,
  .N,
  by = .(
    cluster_5,
    tactical_state
  )
]

cluster_state_profile[
  ,
  percentage :=
    round(
      N /
        sum(N) *
        100,
      2
    ),
  by = cluster_5
]

setorder(
  cluster_state_profile,
  cluster_5,
  -percentage
)

View(cluster_state_profile)


# ============================================================
# 17. ESTADO DOMINANTE POR CLUSTER
# ============================================================

dominant_state_by_cluster <- cluster_state_profile[
  ,
  .SD[
    which.max(
      percentage
    )
  ],
  by = cluster_5
]

View(dominant_state_by_cluster)


# ============================================================
# 18. ASIGNAR NOMBRE DOMINANTE DEL CLUSTER
# ============================================================

cluster_labels <- dominant_state_by_cluster[
  ,
  .(
    cluster_5,

    cluster_label =
      tactical_state,

    dominance_pct =
      percentage
  )
]


analysis_dt <- merge(

  analysis_dt,

  cluster_labels,

  by = "cluster_5",

  all.x = TRUE
)


# ============================================================
# 19. ¿COINCIDE ESTADO REGLADO VS CLUSTER?
# ============================================================

analysis_dt[
  ,
  cluster_state_agree :=
    tactical_state ==
    cluster_label
]


agreement_summary <- analysis_dt[
  ,
  .(

    observations = .N,

    agreements =
      sum(
        cluster_state_agree,
        na.rm = TRUE
      ),

    disagreements =
      sum(
        !cluster_state_agree,
        na.rm = TRUE
      ),

    agreement_pct =
      round(
        mean(
          cluster_state_agree,
          na.rm = TRUE
        ) *
          100,
        2
      )
  )
]

View(agreement_summary)


# ============================================================
# 20. TABLA PARA INSPECCIONAR DESACUERDOS
# ============================================================

cluster_disagreements <- analysis_dt[
  cluster_state_agree == FALSE,
  .(

    match_id,
    minute,
    opponent,

    tactical_state,

    cluster_5,

    cluster_label,

    dominance_pct,

    america_field_tilt_last5,

    pressure_diff_last5,

    progressive_pass_diff_last5,

    entries_final_third_diff_last5,

    entries_box_diff_last5,

    america_local_superiority_10_last5,

    america_pressure_share_5_last5
  )
]

View(cluster_disagreements)


# ============================================================
# 21. CENTROIDES EN ESCALA ESTANDARIZADA
# ============================================================

centroids_scaled <- as.data.table(
  kmeans_5$centers
)

centroids_scaled[
  ,
  cluster_5 :=
    factor(
      1:5
    )
]

setcolorder(
  centroids_scaled,
  c(
    "cluster_5",
    cluster_features
  )
)

View(centroids_scaled)


# ============================================================
# 22. GUARDAR RESULTADOS
# ============================================================

dir.create(
  "outputs/models",
  recursive = TRUE,
  showWarnings = FALSE
)

fwrite(
  k_evaluation,
  "outputs/models/clustering_k_evaluation.csv",
  bom = TRUE
)

fwrite(
  cluster_profiles,
  "outputs/models/clustering_cluster_profiles.csv",
  bom = TRUE
)

fwrite(
  state_cluster_pct,
  "outputs/models/clustering_state_comparison.csv",
  bom = TRUE
)

fwrite(
  dominant_state_by_cluster,
  "outputs/models/clustering_dominant_states.csv",
  bom = TRUE
)

fwrite(
  cluster_disagreements,
  "outputs/models/clustering_disagreements.csv",
  bom = TRUE
)


# ============================================================
# 23. GUARDAR DATASET CON CLUSTERS
# ============================================================

fwrite(
  analysis_dt,
  "data/processed/america_male/tactical_states_clustering_validation.csv",
  bom = TRUE,
  na = "NA"
)


# ============================================================
# 24. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("CLUSTER VALIDATION TERMINADA\n")
cat("=====================================\n")

cat(
  "Observaciones:",
  nrow(analysis_dt),
  "\n"
)

cat(
  "Mejor k silhouette:",
  best_k,
  "\n"
)

cat(
  "K usado para comparar estados:",
  5,
  "\n"
)

cat(
  "Agreement estados vs clustering:",
  agreement_summary$agreement_pct,
  "%\n"
)