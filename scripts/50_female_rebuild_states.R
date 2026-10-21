# ============================================================
# 50_female_rebuild_states.R
# CLUB AMÉRICA FEMENIL
#
# RECONSTRUCCIÓN DATA-DRIVEN DE ESTADOS TÁCTICOS
# K-MEANS ENTRENADO EXCLUSIVAMENTE EN TRAIN
# ============================================================

library(data.table)


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
  "rolling_features_female.csv"
)

output_file <- file.path(
  data_dir,
  "model_ready_female_v2.csv"
)

centroids_file <- file.path(
  data_dir,
  "female_state_centroids_v2.csv"
)

mapping_file <- file.path(
  data_dir,
  "female_state_mapping_v2.csv"
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
cat("FEMALE STATES V2 - DATA DRIVEN\n")
cat("============================================\n\n")


cat(
  "Filas:",
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


dt[
  ,
  match_date := as.IDate(
    match_date
  )
]


dt[
  ,
  minute := as.integer(
    minute
  )
]


setorder(
  dt,
  match_date,
  match_id,
  minute
)


# ============================================================
# 4. PARTIDOS Y SPLIT TEMPORAL
# ============================================================

matches <- unique(
  dt[
    !is.na(match_date),
    .(
      match_id,
      match_date
    )
  ]
)


setorder(
  matches,
  match_date,
  match_id
)


n_matches <- nrow(
  matches
)


if (n_matches < 10) {

  stop(
    "Muy pocos partidos."
  )
}


train_end <- floor(
  n_matches * 0.60
)


validation_end <- floor(
  n_matches * 0.80
)


matches[
  ,
  split := "TEST"
]


matches[
  seq_len(
    train_end
  ),
  split := "TRAIN"
]


validation_rows <- seq.int(
  train_end + 1,
  validation_end
)


matches[
  validation_rows,
  split := "VALIDATION"
]


dt <- merge(
  dt,
  matches[
    ,
    .(
      match_id,
      split
    )
  ],
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 5. FEATURES PARA DEFINIR EL ESTADO
#
# NO usamos el marcador para crear clusters.
# El estado describe comportamiento táctico.
# ============================================================

candidate_features <- c(
  "america_field_tilt_last5",
  "america_possession_context_last5",

  "entries_final_third_diff_last5",
  "entries_box_diff_last5",

  "progressive_passes_diff_last5",
  "progressive_carries_diff_last5",

  "pressures_diff_last5",
  "recoveries_diff_last5",

  "recovery_height_diff_last5",
  "shots_diff_last5"
)


state_features <- candidate_features[
  candidate_features
  %in%
  names(dt)
]


cat(
  "Features para clustering:",
  length(state_features),
  "\n\n"
)


print(
  state_features
)


if (length(state_features) < 5) {

  stop(
    "No hay suficientes features para clustering."
  )
}


# ============================================================
# 6. NUMÉRICO
# ============================================================

for (feature in state_features) {

  dt[
    ,
    (feature) :=
      as.numeric(
        get(feature)
      )
  ]
}


# ============================================================
# 7. IMPUTACIÓN CALCULADA SOLO CON TRAIN
# ============================================================

train_rows <- which(
  dt$split == "TRAIN"
)


if (length(train_rows) == 0) {

  stop(
    "TRAIN está vacío."
  )
}


train_medians <- setNames(
  numeric(
    length(state_features)
  ),
  state_features
)


for (feature in state_features) {

  median_value <- median(
    dt[
      split == "TRAIN",
      get(feature)
    ],
    na.rm = TRUE
  )


  if (!is.finite(median_value)) {

    median_value <- 0
  }


  train_medians[
    feature
  ] <- median_value


  dt[
    is.na(
      get(feature)
    ),
    (feature) :=
      median_value
  ]
}


# ============================================================
# 8. MEDIA Y DESVIACIÓN SOLO TRAIN
# ============================================================

train_means <- setNames(
  numeric(
    length(state_features)
  ),
  state_features
)


train_sds <- setNames(
  numeric(
    length(state_features)
  ),
  state_features
)


for (feature in state_features) {

  feature_mean <- mean(
    dt[
      split == "TRAIN",
      get(feature)
    ],
    na.rm = TRUE
  )


  feature_sd <- sd(
    dt[
      split == "TRAIN",
      get(feature)
    ],
    na.rm = TRUE
  )


  if (!is.finite(feature_mean)) {

    feature_mean <- 0
  }


  if (
    !is.finite(feature_sd)
    ||
    feature_sd == 0
  ) {

    feature_sd <- 1
  }


  train_means[
    feature
  ] <- feature_mean


  train_sds[
    feature
  ] <- feature_sd
}


# ============================================================
# 9. MATRIZ ESTANDARIZADA
# ============================================================

x_all <- as.matrix(
  dt[
    ,
    ..state_features
  ]
)


x_scaled <- sweep(
  x_all,
  2,
  train_means,
  "-"
)


x_scaled <- sweep(
  x_scaled,
  2,
  train_sds,
  "/"
)


# ============================================================
# 10. ENTRENAR K-MEANS SOLO TRAIN
#
# 5 clusters porque CoachLens trabaja con
# 5 estados tácticos.
# ============================================================

set.seed(
  2026
)


kmeans_model <- kmeans(
  x_scaled[
    train_rows,
    ,
    drop = FALSE
  ],

  centers = 5,

  nstart = 100,

  iter.max = 200
)


cat(
  "\nK-means entrenado.\n"
)


cat(
  "Between SS / Total SS:",
  round(
    kmeans_model$betweenss /
      kmeans_model$totss,
    4
  ),
  "\n\n"
)


# ============================================================
# 11. ASIGNAR CADA FILA AL CENTRO MÁS CERCANO
# ============================================================

centers <- kmeans_model$centers


assign_cluster <- function(
  matrix_data,
  center_matrix
) {

  result <- integer(
    nrow(
      matrix_data
    )
  )


  for (
    row_index
    in
    seq_len(
      nrow(
        matrix_data
      )
    )
  ) {

    row_values <- matrix_data[
      row_index,
      ,
      drop = FALSE
    ]


    distances <- rowSums(
      (
        center_matrix
        -
        matrix(
          row_values,
          nrow =
            nrow(
              center_matrix
            ),
          ncol =
            ncol(
              center_matrix
            ),
          byrow = TRUE
        )
      )^2
    )


    result[
      row_index
    ] <- which.min(
      distances
    )
  }


  result
}


dt[
  ,
  cluster_id :=
    assign_cluster(
      x_scaled,
      centers
    )
]


# ============================================================
# 12. CENTROIDES
# ============================================================

centroid_dt <- as.data.table(
  centers
)


centroid_dt[
  ,
  cluster_id :=
    seq_len(
      .N
    )
]


setcolorder(
  centroid_dt,
  c(
    "cluster_id",
    state_features
  )
)


cat(
  "\n============================================\n"
)

cat(
  "CENTROIDES ESTANDARIZADOS\n"
)

cat(
  "============================================\n\n"
)


print(
  centroid_dt
)


# ============================================================
# 13. NOMBRAR LOS CLUSTERS
#
# Las etiquetas se asignan a partir de los propios
# centroides, no usando thresholds manuales.
# ============================================================

cluster_ids <- centroid_dt$cluster_id


cluster_labels <- setNames(
  rep(
    NA_character_,
    length(
      cluster_ids
    )
  ),
  as.character(
    cluster_ids
  )
)


# ------------------------------------------------------------
# CONTROL
# mayor combinación de posesión + field tilt
# ------------------------------------------------------------

control_score <- (
  centroid_dt$america_field_tilt_last5
  +
  centroid_dt$america_possession_context_last5
)


control_cluster <- centroid_dt$cluster_id[
  which.max(
    control_score
  )
]


cluster_labels[
  as.character(
    control_cluster
  )
] <- "CONTROL"


# ------------------------------------------------------------
# PROTECTION
# menor dominio territorial/ofensivo
# ------------------------------------------------------------

protection_score <- (
  centroid_dt$america_field_tilt_last5
  +
  centroid_dt$america_possession_context_last5
  +
  centroid_dt$entries_final_third_diff_last5
  +
  centroid_dt$progressive_passes_diff_last5
)


available <- is.na(
  cluster_labels[
    as.character(
      centroid_dt$cluster_id
    )
  ]
)


protection_candidates <- centroid_dt[
  available
]


protection_candidate_score <- (
  protection_candidates$america_field_tilt_last5
  +
  protection_candidates$america_possession_context_last5
  +
  protection_candidates$entries_final_third_diff_last5
  +
  protection_candidates$progressive_passes_diff_last5
)


protection_cluster <- (
  protection_candidates$cluster_id[
    which.min(
      protection_candidate_score
    )
  ]
)


cluster_labels[
  as.character(
    protection_cluster
  )
] <- "PROTECTION"


# ------------------------------------------------------------
# PRESSURE
# mayor presión + recuperación alta
# ------------------------------------------------------------

available <- is.na(
  cluster_labels[
    as.character(
      centroid_dt$cluster_id
    )
  ]
)


pressure_candidates <- centroid_dt[
  available
]


pressure_candidate_score <- (
  pressure_candidates$pressures_diff_last5
  +
  pressure_candidates$recovery_height_diff_last5
)


pressure_cluster <- (
  pressure_candidates$cluster_id[
    which.max(
      pressure_candidate_score
    )
  ]
)


cluster_labels[
  as.character(
    pressure_cluster
  )
] <- "PRESSURE"


# ------------------------------------------------------------
# TRANSITION
# mayor progresión y avance territorial
# ------------------------------------------------------------

available <- is.na(
  cluster_labels[
    as.character(
      centroid_dt$cluster_id
    )
  ]
)


transition_candidates <- centroid_dt[
  available
]


transition_candidate_score <- (
  transition_candidates$progressive_passes_diff_last5
  +
  transition_candidates$progressive_carries_diff_last5
  +
  transition_candidates$entries_final_third_diff_last5
)


transition_cluster <- (
  transition_candidates$cluster_id[
    which.max(
      transition_candidate_score
    )
  ]
)


cluster_labels[
  as.character(
    transition_cluster
  )
] <- "TRANSITION"


# ------------------------------------------------------------
# URGENCY
# último cluster disponible
# ------------------------------------------------------------

remaining_cluster <- as.integer(
  names(
    cluster_labels
  )[
    is.na(
      cluster_labels
    )
  ]
)


if (
  length(
    remaining_cluster
  ) != 1
) {

  stop(
    "No fue posible asignar exactamente un cluster a URGENCY."
  )
}


cluster_labels[
  as.character(
    remaining_cluster
  )
] <- "URGENCY"


# ============================================================
# 14. MAPA
# ============================================================

mapping_dt <- data.table(

  cluster_id =
    as.integer(
      names(
        cluster_labels
      )
    ),

  tactical_state =
    as.character(
      cluster_labels
    )
)


setorder(
  mapping_dt,
  cluster_id
)


cat(
  "\n============================================\n"
)

cat(
  "MAPEO CLUSTERS -> ESTADOS\n"
)

cat(
  "============================================\n\n"
)


print(
  mapping_dt
)


# ============================================================
# 15. ASIGNAR ESTADO
# ============================================================

dt <- merge(
  dt,
  mapping_dt,
  by = "cluster_id",
  all.x = TRUE
)


setnames(
  dt,
  "tactical_state",
  "current_tactical_state"
)


# ============================================================
# 16. TARGET EXACTO +5 MINUTOS
#
# NO usamos shift(n=5).
# Buscamos exactamente minute + 5.
# ============================================================

future_lookup <- dt[
  ,
  .(
    match_id,
    target_minute =
      minute,

    future_tactical_state_5min =
      current_tactical_state
  )
]


dt[
  ,
  target_minute :=
    minute + 5L
]


dt <- merge(
  dt,
  future_lookup,
  by = c(
    "match_id",
    "target_minute"
  ),
  all.x = TRUE
)


dt[
  ,
  target_minute := NULL
]


# ============================================================
# 17. FILAS MODELABLES
# ============================================================

dt[
  ,
  usable_for_model_5min :=
    !is.na(
      future_tactical_state_5min
    )
]


model_dt <- dt[
  usable_for_model_5min == TRUE
]


# ============================================================
# 18. DISTRIBUCIÓN
# ============================================================

distribution <- model_dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    split,
    future_tactical_state_5min
  )
]


distribution[
  ,
  share :=
    observations /
    sum(observations),
  by = split
]


cat(
  "\n============================================\n"
)

cat(
  "TARGET V2 POR SPLIT\n"
)

cat(
  "============================================\n\n"
)


print(
  distribution[
    order(
      split,
      -observations
    )
  ]
)


# ============================================================
# 19. PERSISTENCIA
# ============================================================

model_dt[
  ,
  same_state :=
    current_tactical_state
    ==
    future_tactical_state_5min
]


persistence <- model_dt[
  ,
  .(
    observations = .N,

    persistence =
      mean(
        same_state
      )
  ),
  by = current_tactical_state
]


cat(
  "\n============================================\n"
)

cat(
  "PERSISTENCIA V2\n"
)

cat(
  "============================================\n\n"
)


print(
  persistence[
    order(
      -persistence
    )
  ]
)


cat(
  "\nPersistencia global:",
  round(
    mean(
      model_dt$same_state
    ) * 100,
    2
  ),
  "%\n"
)


# ============================================================
# 20. BALANCE GLOBAL
# ============================================================

global_distribution <- model_dt[
  ,
  .(
    observations = .N
  ),
  by = future_tactical_state_5min
]


global_distribution[
  ,
  share :=
    observations /
    sum(observations)
]


cat(
  "\n============================================\n"
)

cat(
  "BALANCE GLOBAL V2\n"
)

cat(
  "============================================\n\n"
)


print(
  global_distribution[
    order(
      -observations
    )
  ]
)


# ============================================================
# 21. EXPORTAR
# ============================================================

setorder(
  model_dt,
  match_date,
  match_id,
  minute
)


fwrite(
  model_dt,
  output_file
)


fwrite(
  centroid_dt,
  centroids_file
)


fwrite(
  mapping_dt,
  mapping_file
)


# ============================================================
# 22. FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Observaciones modelables:",
  nrow(
    model_dt
  ),
  "\n"
)


cat(
  "Partidos:",
  uniqueN(
    model_dt$match_id
  ),
  "\n\n"
)


cat(
  "Dataset V2:\n",
  output_file,
  "\n\n"
)


cat(
  "Centroides:\n",
  centroids_file,
  "\n\n"
)


cat(
  "Mapeo:\n",
  mapping_file,
  "\n"
)