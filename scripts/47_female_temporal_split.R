# ============================================================
# 47_female_temporal_split.R
# CLUB AMÉRICA FEMENIL
# SPLIT TEMPORAL TRAIN / VALIDATION / TEST
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
  "model_with_tactical_states_female.csv"
)

output_file <- file.path(
  data_dir,
  "model_ready_female.csv"
)


# ============================================================
# 2. CARGAR DATOS
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
cat("FEMALE TEMPORAL SPLIT\n")
cat("============================================\n\n")


cat(
  "Filas cargadas:",
  nrow(dt),
  "\n"
)


cat(
  "Partidos cargados:",
  uniqueN(dt$match_id),
  "\n\n"
)


# ============================================================
# 3. VALIDAR COLUMNAS
# ============================================================

required_columns <- c(
  "match_id",
  "match_date",
  "minute",
  "current_tactical_state",
  "future_tactical_state_5min",
  "usable_for_model_5min"
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
# 4. FECHA
# ============================================================

dt[
  ,
  match_date :=
    as.IDate(
      match_date
    )
]


if (
  any(
    is.na(
      dt$match_date
    )
  )
) {

  warning(
    "Existen fechas NA."
  )
}


# ============================================================
# 5. SOLO OBSERVACIONES MODELABLES
# ============================================================

model_dt <- dt[
  usable_for_model_5min == TRUE
  &
  !is.na(
    future_tactical_state_5min
  )
  &
  !is.na(
    match_date
  )
]


if (nrow(model_dt) == 0) {

  stop(
    "No existen observaciones utilizables para el modelo."
  )
}


# ============================================================
# 6. PARTIDOS ÚNICOS CRONOLÓGICOS
# ============================================================

matches <- unique(
  model_dt[
    ,
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


cat(
  "Partidos modelables:",
  n_matches,
  "\n\n"
)


if (n_matches < 10) {

  stop(
    "Muy pocos partidos para realizar un split temporal."
  )
}


# ============================================================
# 7. CORTES TEMPORALES
#
# Los cortes se calculan desde el número real de partidos.
# No se usan fechas escritas manualmente.
# ============================================================

train_end <- floor(
  n_matches * 0.60
)


validation_end <- floor(
  n_matches * 0.80
)


if (train_end < 1) {

  stop(
    "TRAIN quedó vacío."
  )
}


if (validation_end <= train_end) {

  stop(
    "VALIDATION quedó vacío."
  )
}


if (validation_end >= n_matches) {

  stop(
    "TEST quedó vacío."
  )
}


# ============================================================
# 8. ASIGNAR SPLIT
# ============================================================

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
  from = train_end + 1,
  to = validation_end
)


matches[
  validation_rows,
  split := "VALIDATION"
]


# ============================================================
# 9. ORDEN AUXILIAR DEL SPLIT
# ============================================================

matches[
  ,
  split_order :=
    fifelse(
      split == "TRAIN",
      1L,
      fifelse(
        split == "VALIDATION",
        2L,
        3L
      )
    )
]


# ============================================================
# 10. AGREGAR SPLIT AL DATASET
# ============================================================

model_dt <- merge(
  model_dt,
  matches[
    ,
    .(
      match_id,
      split,
      split_order
    )
  ],
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 11. VALIDAR QUE TODO TENGA SPLIT
# ============================================================

if (
  any(
    is.na(
      model_dt$split
    )
  )
) {

  stop(
    "Existen observaciones sin split asignado."
  )
}


# ============================================================
# 12. RESUMEN TEMPORAL
# ============================================================

split_summary <- model_dt[
  ,
  .(
    observations = .N,

    matches =
      uniqueN(
        match_id
      ),

    date_min =
      min(
        match_date,
        na.rm = TRUE
      ),

    date_max =
      max(
        match_date,
        na.rm = TRUE
      ),

    split_order =
      unique(
        split_order
      )[1]
  ),
  by = split
]


setorder(
  split_summary,
  split_order
)


split_summary[
  ,
  split_order := NULL
]


cat(
  "Resumen temporal:\n\n"
)


print(
  split_summary
)


# ============================================================
# 13. VALIDAR ORDEN CRONOLÓGICO
# ============================================================

train_max_date <- model_dt[
  split == "TRAIN",
  max(
    match_date,
    na.rm = TRUE
  )
]


validation_min_date <- model_dt[
  split == "VALIDATION",
  min(
    match_date,
    na.rm = TRUE
  )
]


validation_max_date <- model_dt[
  split == "VALIDATION",
  max(
    match_date,
    na.rm = TRUE
  )
]


test_min_date <- model_dt[
  split == "TEST",
  min(
    match_date,
    na.rm = TRUE
  )
]


cat(
  "\nÚltima fecha TRAIN:",
  as.character(
    train_max_date
  ),
  "\n"
)


cat(
  "Primera fecha VALIDATION:",
  as.character(
    validation_min_date
  ),
  "\n"
)


cat(
  "Última fecha VALIDATION:",
  as.character(
    validation_max_date
  ),
  "\n"
)


cat(
  "Primera fecha TEST:",
  as.character(
    test_min_date
  ),
  "\n\n"
)


if (
  train_max_date >
  validation_min_date
) {

  stop(
    "TRAIN contiene fechas posteriores a VALIDATION."
  )
}


if (
  validation_max_date >
  test_min_date
) {

  stop(
    "VALIDATION contiene fechas posteriores a TEST."
  )
}


# ============================================================
# 14. COMPROBAR LEAKAGE DE MATCH_ID
# ============================================================

train_matches <- unique(
  model_dt[
    split == "TRAIN",
    match_id
  ]
)


validation_matches <- unique(
  model_dt[
    split == "VALIDATION",
    match_id
  ]
)


test_matches <- unique(
  model_dt[
    split == "TEST",
    match_id
  ]
)


leak_train_validation <- length(
  intersect(
    train_matches,
    validation_matches
  )
)


leak_train_test <- length(
  intersect(
    train_matches,
    test_matches
  )
)


leak_validation_test <- length(
  intersect(
    validation_matches,
    test_matches
  )
)


cat(
  "Leak TRAIN / VALIDATION:",
  leak_train_validation,
  "\n"
)


cat(
  "Leak TRAIN / TEST:",
  leak_train_test,
  "\n"
)


cat(
  "Leak VALIDATION / TEST:",
  leak_validation_test,
  "\n\n"
)


if (
  leak_train_validation > 0
  ||
  leak_train_test > 0
  ||
  leak_validation_test > 0
) {

  stop(
    "Se detectó leakage entre los splits."
  )
}


# ============================================================
# 15. DISTRIBUCIÓN DEL TARGET
# ============================================================

target_distribution <- model_dt[
  ,
  .(
    observations = .N
  ),
  by = .(
    split,
    future_tactical_state_5min
  )
]


target_distribution[
  ,
  total_split :=
    sum(
      observations
    ),
  by = split
]


target_distribution[
  ,
  share :=
    observations
    /
    total_split
]


target_distribution[
  ,
  split_order :=
    fifelse(
      split == "TRAIN",
      1L,
      fifelse(
        split == "VALIDATION",
        2L,
        3L
      )
    )
]


setorder(
  target_distribution,
  split_order,
  -observations
)


target_distribution[
  ,
  c(
    "split_order",
    "total_split"
  ) := NULL
]


cat(
  "Distribución del target por split:\n\n"
)


print(
  target_distribution
)


# ============================================================
# 16. CLASES PRESENTES POR SPLIT
# ============================================================

class_summary <- model_dt[
  ,
  .(
    classes =
      uniqueN(
        future_tactical_state_5min
      )
  ),
  by = split
]


class_summary[
  ,
  split_order :=
    fifelse(
      split == "TRAIN",
      1L,
      fifelse(
        split == "VALIDATION",
        2L,
        3L
      )
    )
]


setorder(
  class_summary,
  split_order
)


class_summary[
  ,
  split_order := NULL
]


cat(
  "\nClases presentes por split:\n\n"
)


print(
  class_summary
)


# ============================================================
# 17. VALIDAR CLASES
# ============================================================

train_classes <- uniqueN(
  model_dt[
    split == "TRAIN",
    future_tactical_state_5min
  ]
)


validation_classes <- uniqueN(
  model_dt[
    split == "VALIDATION",
    future_tactical_state_5min
  ]
)


test_classes <- uniqueN(
  model_dt[
    split == "TEST",
    future_tactical_state_5min
  ]
)


if (train_classes < 2) {

  stop(
    "TRAIN contiene menos de dos clases."
  )
}


if (validation_classes < 2) {

  stop(
    "VALIDATION contiene menos de dos clases."
  )
}


if (test_classes < 2) {

  stop(
    "TEST contiene menos de dos clases."
  )
}


# ============================================================
# 18. LIMPIAR COLUMNA AUXILIAR
# ============================================================

model_dt[
  ,
  split_order := NULL
]


# ============================================================
# 19. ORDEN FINAL
# ============================================================

setorder(
  model_dt,
  match_date,
  match_id,
  minute
)


# ============================================================
# 20. EXPORTAR
# ============================================================

fwrite(
  model_dt,
  output_file
)


cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Observaciones:",
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
  "Archivo creado:\n",
  output_file,
  "\n"
)