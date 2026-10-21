# ============================================================
# 45_female_rolling_features.R
# CLUB AMÉRICA FEMENIL
# ROLLING FEATURES 3 Y 5 MINUTOS
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
  "minute_features_female.csv"
)

output_file <- file.path(
  data_dir,
  "rolling_features_female.csv"
)


# ============================================================
# 2. VALIDAR ARCHIVO
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
cat("COACHLENS - FEMALE ROLLING FEATURES\n")
cat("============================================\n\n")

cat(
  "Filas cargadas:",
  nrow(dt),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(dt$match_id),
  "\n"
)

cat(
  "Columnas:",
  ncol(dt),
  "\n\n"
)


# ============================================================
# 3. ORDEN
# ============================================================

setorder(
  dt,
  match_id,
  minute
)


# ============================================================
# 4. IDENTIFICAR FEATURES NUMÉRICAS ÚTILES
# ============================================================

base_metrics <- c(
  "america_field_tilt",
  "america_possession_context",

  "entries_final_third_diff",
  "entries_box_diff",
  "progressive_passes_diff",
  "progressive_carries_diff",
  "pressures_diff",
  "recoveries_diff",
  "shots_diff",
  "passes_diff",
  "completed_passes_diff",
  "carries_diff",
  "duels_diff",
  "turnovers_diff",
  "possession_events_diff",

  "america_entries_final_third",
  "america_entries_box",
  "america_progressive_passes",
  "america_progressive_carries",
  "america_pressures",
  "america_recoveries",
  "america_shots",
  "america_passes",
  "america_completed_passes",
  "america_carries",
  "america_duels",
  "america_turnovers",

  "rival_entries_final_third",
  "rival_entries_box",
  "rival_progressive_passes",
  "rival_progressive_carries",
  "rival_pressures",
  "rival_recoveries",
  "rival_shots",
  "rival_passes",
  "rival_completed_passes",
  "rival_carries",
  "rival_duels",
  "rival_turnovers",

  "america_recovery_height",
  "rival_recovery_height",
  "recovery_height_diff"
)


base_metrics <- base_metrics[
  base_metrics
  %in%
  names(dt)
]


cat(
  "Features base disponibles:",
  length(base_metrics),
  "\n\n"
)


print(
  base_metrics
)


# ============================================================
# 5. FUNCIÓN ROLLING
# ============================================================

rolling_sum <- function(
  x,
  n
) {

  frollsum(
    x,
    n = n,
    align = "right",
    fill = NA_real_,
    na.rm = TRUE
  )
}


rolling_mean <- function(
  x,
  n
) {

  frollmean(
    x,
    n = n,
    align = "right",
    fill = NA_real_,
    na.rm = TRUE
  )
}


# ============================================================
# 6. CLASIFICAR MÉTRICAS
#
# Algunas métricas son proporciones/promedios:
# field tilt, possession, recovery height
#
# El resto son conteos y diferencias acumulables.
# ============================================================

mean_metrics <- c(
  "america_field_tilt",
  "america_possession_context",
  "america_recovery_height",
  "rival_recovery_height",
  "recovery_height_diff"
)


mean_metrics <- mean_metrics[
  mean_metrics
  %in%
  base_metrics
]


sum_metrics <- setdiff(
  base_metrics,
  mean_metrics
)


# ============================================================
# 7. ROLLING 3 MINUTOS
# ============================================================

cat(
  "\nConstruyendo rolling 3 minutos...\n"
)


for (metric in sum_metrics) {

  new_col <- paste0(
    metric,
    "_last3"
  )


  dt[
    ,
    (new_col) :=
      rolling_sum(
        get(metric),
        3
      ),
    by = match_id
  ]
}


for (metric in mean_metrics) {

  new_col <- paste0(
    metric,
    "_last3"
  )


  dt[
    ,
    (new_col) :=
      rolling_mean(
        get(metric),
        3
      ),
    by = match_id
  ]
}


# ============================================================
# 8. ROLLING 5 MINUTOS
# ============================================================

cat(
  "Construyendo rolling 5 minutos...\n"
)


for (metric in sum_metrics) {

  new_col <- paste0(
    metric,
    "_last5"
  )


  dt[
    ,
    (new_col) :=
      rolling_sum(
        get(metric),
        5
      ),
    by = match_id
  ]
}


for (metric in mean_metrics) {

  new_col <- paste0(
    metric,
    "_last5"
  )


  dt[
    ,
    (new_col) :=
      rolling_mean(
        get(metric),
        5
      ),
    by = match_id
  ]
}


# ============================================================
# 9. FEATURES DERIVADAS
# ============================================================

if (
  "america_pressures_last5"
  %in%
  names(dt)
  &&
  "rival_pressures_last5"
  %in%
  names(dt)
) {

  dt[
    ,
    pressure_share_america_last5 :=
      fifelse(
        (
          america_pressures_last5
          +
          rival_pressures_last5
        ) > 0,

        america_pressures_last5
        /
        (
          america_pressures_last5
          +
          rival_pressures_last5
        ),

        NA_real_
      )
  ]
}


if (
  "america_pressures_last3"
  %in%
  names(dt)
  &&
  "rival_pressures_last3"
  %in%
  names(dt)
) {

  dt[
    ,
    pressure_share_america_last3 :=
      fifelse(
        (
          america_pressures_last3
          +
          rival_pressures_last3
        ) > 0,

        america_pressures_last3
        /
        (
          america_pressures_last3
          +
          rival_pressures_last3
        ),

        NA_real_
      )
  ]
}


# ============================================================
# 10. CAMBIOS RECIENTES
#
# Diferencia entre ventana corta y larga
# ============================================================

trend_candidates <- c(
  "america_field_tilt",
  "america_possession_context",
  "entries_final_third_diff",
  "entries_box_diff",
  "progressive_passes_diff",
  "pressures_diff",
  "recovery_height_diff"
)


trend_candidates <- trend_candidates[
  trend_candidates
  %in%
  base_metrics
]


for (metric in trend_candidates) {

  col3 <- paste0(
    metric,
    "_last3"
  )

  col5 <- paste0(
    metric,
    "_last5"
  )

  trend_col <- paste0(
    metric,
    "_trend"
  )


  if (
    col3
    %in%
    names(dt)
    &&
    col5
    %in%
    names(dt)
  ) {

    dt[
      ,
      (trend_col) :=
        get(col3)
        -
        get(col5)
    ]
  }
}


# ============================================================
# 11. VALIDACIÓN DE FILAS USABLES
# ============================================================

rolling_columns <- grep(
  "_last3$|_last5$|_trend$",
  names(dt),
  value = TRUE
)


dt[
  ,
  rolling_complete :=
    rowSums(
      !is.na(
        .SD
      )
    )
    >
    0,
  .SDcols =
    rolling_columns
]


usable_rows <- dt[
  rolling_complete == TRUE
]


# ============================================================
# 12. VALIDACIÓN
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACION ROLLING FEATURES\n")
cat("============================================\n\n")


cat(
  "Filas totales:",
  nrow(dt),
  "\n"
)


cat(
  "Filas con rolling:",
  nrow(usable_rows),
  "\n"
)


cat(
  "Partidos:",
  uniqueN(
    dt$match_id
  ),
  "\n"
)


cat(
  "Columnas totales:",
  ncol(dt),
  "\n"
)


cat(
  "Features rolling creadas:",
  length(
    rolling_columns
  ),
  "\n\n"
)


cat(
  "Ejemplo de columnas rolling:\n"
)


print(
  head(
    rolling_columns,
    25
  )
)


if (
  "america_field_tilt_last5"
  %in%
  names(dt)
) {

  cat(
    "\nResumen field tilt last5:\n"
  )


  print(
    summary(
      dt$america_field_tilt_last5
    )
  )
}


if (
  "entries_final_third_diff_last5"
  %in%
  names(dt)
) {

  cat(
    "\nResumen entradas último tercio diff last5:\n"
  )


  print(
    summary(
      dt$entries_final_third_diff_last5
    )
  )
}


if (
  "progressive_passes_diff_last5"
  %in%
  names(dt)
) {

  cat(
    "\nResumen pases progresivos diff last5:\n"
  )


  print(
    summary(
      dt$progressive_passes_diff_last5
    )
  )
}


# ============================================================
# 13. EXPORTAR
# ============================================================

fwrite(
  dt,
  output_file
)


cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Archivo creado:\n",
  output_file,
  "\n"
)