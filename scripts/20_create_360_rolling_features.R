library(data.table)

# ============================================================
# 1. LEER DATASET CON 360
# ============================================================

dt <- fread(
  "data/processed/america_male/model_context_360.csv",
  encoding = "UTF-8"
)

setorder(
  dt,
  match_id,
  minute
)

cat("Filas:", nrow(dt), "\n")
cat("Columnas:", ncol(dt), "\n")
cat("Partidos:", uniqueN(dt$match_id), "\n")


# ============================================================
# 2. EVENTOS 360 EN LOS ÚLTIMOS 5 MINUTOS
# ============================================================

dt[
  ,
  america_360_events_last5 :=
    frollsum(
      america_360_events,
      n = 5,
      align = "right",
      fill = NA
    ),
  by = match_id
]

dt[
  ,
  opponent_360_events_last5 :=
    frollsum(
      opponent_360_events,
      n = 5,
      align = "right",
      fill = NA
    ),
  by = match_id
]


# ============================================================
# 3. FUNCIÓN PARA PROMEDIO ROLLING PONDERADO
#
# Minutos con más eventos 360 tienen mayor peso.
# ============================================================

weighted_roll_5 <- function(
  DT,
  variable,
  weight,
  new_name
) {

  temp_num <- paste0("__num_", new_name)
  temp_den <- paste0("__den_", new_name)

  # Numerador:
  # métrica * número de eventos
  DT[
    ,
    (temp_num) :=
      fifelse(
        is.na(get(variable)),
        0,
        get(variable) * get(weight)
      )
  ]

  # Denominador:
  # solo contamos eventos cuando la métrica existe
  DT[
    ,
    (temp_den) :=
      fifelse(
        is.na(get(variable)),
        0,
        get(weight)
      )
  ]

  # Rolling numerator
  DT[
    ,
    paste0(temp_num, "_roll") :=
      frollsum(
        get(temp_num),
        n = 5,
        align = "right",
        fill = NA
      ),
    by = match_id
  ]

  # Rolling denominator
  DT[
    ,
    paste0(temp_den, "_roll") :=
      frollsum(
        get(temp_den),
        n = 5,
        align = "right",
        fill = NA
      ),
    by = match_id
  ]

  num_roll <- paste0(
    temp_num,
    "_roll"
  )

  den_roll <- paste0(
    temp_den,
    "_roll"
  )

  # Resultado final
  DT[
    ,
    (new_name) :=
      fifelse(
        get(den_roll) > 0,
        get(num_roll) /
          get(den_roll),
        NA_real_
      )
  ]

  # Borrar columnas temporales
  DT[
    ,
    c(
      temp_num,
      temp_den,
      num_roll,
      den_roll
    ) := NULL
  ]
}


# ============================================================
# 4. VARIABLES DE AMÉRICA
# ============================================================

america_metrics <- c(

  "america_visible_teammates",
  "america_visible_opponents",

  "america_teammates_within_5",
  "america_opponents_within_5",

  "america_teammates_within_10",
  "america_opponents_within_10",

  "america_local_superiority_5",
  "america_local_superiority_10",

  "america_nearest_opponent_distance",
  "america_avg_opponent_distance",
  "america_avg_teammate_distance",

  "america_pressure_share_5",
  "america_pressure_share_10",

  "america_crowded_share_10",
  "america_support_share_10"
)


for (variable in america_metrics) {

  if (variable %in% names(dt)) {

    new_name <- paste0(
      variable,
      "_last5"
    )

    weighted_roll_5(
      DT = dt,
      variable = variable,
      weight = "america_360_events",
      new_name = new_name
    )
  }
}


# ============================================================
# 5. VARIABLES DEL RIVAL
# ============================================================

opponent_metrics <- c(

  "opponent_visible_teammates",
  "opponent_visible_opponents",

  "opponent_teammates_within_5",
  "opponent_opponents_within_5",

  "opponent_teammates_within_10",
  "opponent_opponents_within_10",

  "opponent_local_superiority_5",
  "opponent_local_superiority_10",

  "opponent_nearest_opponent_distance",
  "opponent_avg_opponent_distance",
  "opponent_avg_teammate_distance",

  "opponent_pressure_share_5",
  "opponent_pressure_share_10",

  "opponent_crowded_share_10",
  "opponent_support_share_10"
)


for (variable in opponent_metrics) {

  if (variable %in% names(dt)) {

    new_name <- paste0(
      variable,
      "_last5"
    )

    weighted_roll_5(
      DT = dt,
      variable = variable,
      weight = "opponent_360_events",
      new_name = new_name
    )
  }
}


# ============================================================
# 6. DIFERENCIAS AMÉRICA VS RIVAL
# ============================================================

dt[
  ,
  local_superiority_diff_10_last5 :=
    america_local_superiority_10_last5 -
    opponent_local_superiority_10_last5
]

dt[
  ,
  nearest_opponent_distance_diff_last5 :=
    america_nearest_opponent_distance_last5 -
    opponent_nearest_opponent_distance_last5
]

dt[
  ,
  pressure_share_diff_5_last5 :=
    america_pressure_share_5_last5 -
    opponent_pressure_share_5_last5
]

dt[
  ,
  pressure_share_diff_10_last5 :=
    america_pressure_share_10_last5 -
    opponent_pressure_share_10_last5
]

dt[
  ,
  crowded_share_diff_10_last5 :=
    america_crowded_share_10_last5 -
    opponent_crowded_share_10_last5
]

dt[
  ,
  support_share_diff_10_last5 :=
    america_support_share_10_last5 -
    opponent_support_share_10_last5
]


# ============================================================
# 7. DIFERENCIA EN DENSIDAD DE RIVALES
# ============================================================

dt[
  ,
  opponents_within_5_diff_last5 :=
    america_opponents_within_5_last5 -
    opponent_opponents_within_5_last5
]

dt[
  ,
  opponents_within_10_diff_last5 :=
    america_opponents_within_10_last5 -
    opponent_opponents_within_10_last5
]


# ============================================================
# 8. INDICADOR DE HISTORIA COMPLETA
#
# Como usamos ventana de 5 minutos,
# a partir del minuto 4 ya existe una ventana completa.
# ============================================================

dt[
  ,
  has_full_360_history :=
    minute >= 4
]


# ============================================================
# 9. LIMPIAR NaN / Inf
# ============================================================

numeric_cols <- names(dt)[
  sapply(dt, is.numeric)
]

for (col in numeric_cols) {

  idx <- which(
    is.nan(dt[[col]]) |
    is.infinite(dt[[col]])
  )

  if (length(idx) > 0) {

    set(
      dt,
      idx,
      col,
      NA_real_
    )
  }
}


# ============================================================
# 10. GUARDAR
# ============================================================

fwrite(
  dt,
  "data/processed/america_male/model_context_360_rolling.csv",
  bom = TRUE
)


# ============================================================
# 11. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("360 ROLLING FEATURES TERMINADAS\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(dt),
  "\n"
)

cat(
  "Columnas:",
  ncol(dt),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(dt$match_id),
  "\n"
)