library(data.table)
library(jsonlite)

# ============================================================
# 1. LEER EVENTS LIMPIOS
# ============================================================

events <- fread(
  "data/processed/america_male/events_america_clean.csv",
  encoding = "UTF-8"
)

cat("Eventos:", nrow(events), "\n")
cat("Columnas:", ncol(events), "\n")


# ============================================================
# 2. FUNCIÓN PARA EXTRAER COORDENADAS
#
# StatsBomb pitch:
# X = 0 a 120
# Y = 0 a 80
# ============================================================

parse_location <- function(x) {

  if (is.na(x) || x == "" || x == "null") {
    return(c(NA_real_, NA_real_))
  }

  # Intentar JSON
  out <- tryCatch(
    fromJSON(x),
    error = function(e) NULL
  )

  if (is.null(out) || length(out) < 2) {
    return(c(NA_real_, NA_real_))
  }

  c(
    as.numeric(out[1]),
    as.numeric(out[2])
  )
}


# ============================================================
# 3. LOCATION INICIAL
# ============================================================

loc <- t(
  vapply(
    events$location,
    parse_location,
    numeric(2)
  )
)

events[, x := loc[, 1]]
events[, y := loc[, 2]]


# ============================================================
# 4. PASS END LOCATION
# ============================================================

pass_end_col <- grep(
  "^pass\\.end_location$",
  names(events),
  value = TRUE
)

if (length(pass_end_col) == 1) {

  pass_loc <- t(
    vapply(
      events[[pass_end_col]],
      parse_location,
      numeric(2)
    )
  )

  events[, pass_end_x := pass_loc[, 1]]
  events[, pass_end_y := pass_loc[, 2]]

} else {

  events[, pass_end_x := NA_real_]
  events[, pass_end_y := NA_real_]
}


# ============================================================
# 5. CARRY END LOCATION
# ============================================================

carry_end_col <- grep(
  "^carry\\.end_location$",
  names(events),
  value = TRUE
)

if (length(carry_end_col) == 1) {

  carry_loc <- t(
    vapply(
      events[[carry_end_col]],
      parse_location,
      numeric(2)
    )
  )

  events[, carry_end_x := carry_loc[, 1]]
  events[, carry_end_y := carry_loc[, 2]]

} else {

  events[, carry_end_x := NA_real_]
  events[, carry_end_y := NA_real_]
}


# ============================================================
# 6. IDENTIFICAR AMÉRICA
# ============================================================

events[, is_america := team.name == "América"]


# ============================================================
# 7. TERCIO FINAL
#
# StatsBomb:
# último tercio ≈ x >= 80
# ============================================================

events[
  ,
  in_final_third :=
    !is.na(x) &
    x >= 80
]


# ============================================================
# 8. ÁREA
#
# Aproximación StatsBomb:
# x >= 102
# y entre 18 y 62
# ============================================================

events[
  ,
  in_box :=
    !is.na(x) &
    !is.na(y) &
    x >= 102 &
    y >= 18 &
    y <= 62
]


# ============================================================
# 9. ENTRADAS AL ÚLTIMO TERCIO POR PASE
# ============================================================

events[
  ,
  pass_into_final_third :=
    type.name == "Pass" &
    !is.na(x) &
    !is.na(pass_end_x) &
    x < 80 &
    pass_end_x >= 80
]


# ============================================================
# 10. ENTRADAS AL ÁREA POR PASE
# ============================================================

events[
  ,
  pass_into_box :=
    type.name == "Pass" &
    !is.na(pass_end_x) &
    !is.na(pass_end_y) &
    pass_end_x >= 102 &
    pass_end_y >= 18 &
    pass_end_y <= 62
]


# ============================================================
# 11. PROGRESIONES POR PASE
#
# Aproximación:
# avance longitudinal >= 15 unidades
# ============================================================

events[
  ,
  progressive_pass :=
    type.name == "Pass" &
    !is.na(x) &
    !is.na(pass_end_x) &
    (pass_end_x - x) >= 15
]


# ============================================================
# 12. PROGRESIONES POR CONDUCCIÓN
# ============================================================

events[
  ,
  progressive_carry :=
    type.name == "Carry" &
    !is.na(x) &
    !is.na(carry_end_x) &
    (carry_end_x - x) >= 10
]


# ============================================================
# 13. RECUPERACIONES Y ALTURA
# ============================================================

events[
  ,
  is_recovery :=
    type.name %in% c(
      "Ball Recovery",
      "Interception"
    )
]


# ============================================================
# 14. ZONAS LATERALES
#
# izquierda / centro / derecha
# usando eje Y
# ============================================================

events[
  ,
  lane := fifelse(
    is.na(y),
    NA_character_,
    fifelse(
      y < 26.67,
      "Left",
      fifelse(
        y > 53.33,
        "Right",
        "Center"
      )
    )
  )
]


# ============================================================
# 15. FEATURES ESPACIALES AMÉRICA POR MINUTO
# ============================================================

spatial_america <- events[
  is_america == TRUE,
  .(

    # Altura de recuperación
    america_recovery_height =
      mean(
        x[is_recovery],
        na.rm = TRUE
      ),

    # Acciones en tercio final
    america_final_third_events =
      sum(
        in_final_third,
        na.rm = TRUE
      ),

    # Entradas al último tercio
    america_entries_final_third =
      sum(
        pass_into_final_third,
        na.rm = TRUE
      ),

    # Entradas al área
    america_entries_box =
      sum(
        pass_into_box,
        na.rm = TRUE
      ),

    # Progresiones
    america_progressive_passes =
      sum(
        progressive_pass,
        na.rm = TRUE
      ),

    america_progressive_carries =
      sum(
        progressive_carry,
        na.rm = TRUE
      ),

    # Anchura media
    america_avg_width =
      mean(
        abs(y - 40),
        na.rm = TRUE
      ),

    # Distribución por carril
    america_left_events =
      sum(
        lane == "Left",
        na.rm = TRUE
      ),

    america_center_events =
      sum(
        lane == "Center",
        na.rm = TRUE
      ),

    america_right_events =
      sum(
        lane == "Right",
        na.rm = TRUE
      )

  ),
  by = .(
    match_id,
    minute
  )
]


# ============================================================
# 16. FEATURES ESPACIALES RIVAL POR MINUTO
# ============================================================

spatial_opponent <- events[
  is_america == FALSE &
  !is.na(team.name),
  .(

    opponent_recovery_height =
      mean(
        x[is_recovery],
        na.rm = TRUE
      ),

    opponent_final_third_events =
      sum(
        in_final_third,
        na.rm = TRUE
      ),

    opponent_entries_final_third =
      sum(
        pass_into_final_third,
        na.rm = TRUE
      ),

    opponent_entries_box =
      sum(
        pass_into_box,
        na.rm = TRUE
      ),

    opponent_progressive_passes =
      sum(
        progressive_pass,
        na.rm = TRUE
      ),

    opponent_progressive_carries =
      sum(
        progressive_carry,
        na.rm = TRUE
      ),

    opponent_avg_width =
      mean(
        abs(y - 40),
        na.rm = TRUE
      ),

    opponent_left_events =
      sum(
        lane == "Left",
        na.rm = TRUE
      ),

    opponent_center_events =
      sum(
        lane == "Center",
        na.rm = TRUE
      ),

    opponent_right_events =
      sum(
        lane == "Right",
        na.rm = TRUE
      )

  ),
  by = .(
    match_id,
    minute
  )
]


# ============================================================
# 17. UNIR CON MODELO CONTEXT
# ============================================================

model_context <- fread(
  "data/processed/america_male/model_context.csv",
  encoding = "UTF-8"
)

model_spatial <- merge(
  model_context,
  spatial_america,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)

model_spatial <- merge(
  model_spatial,
  spatial_opponent,
  by = c(
    "match_id",
    "minute"
  ),
  all.x = TRUE
)


# ============================================================
# 18. REEMPLAZAR CONTEOS NA POR 0
# ============================================================

count_cols <- grep(
  "final_third|entries|progressive|left_events|center_events|right_events",
  names(model_spatial),
  value = TRUE
)

for (col in count_cols) {

  set(
    model_spatial,
    which(is.na(model_spatial[[col]])),
    col,
    0
  )
}


# ============================================================
# 19. FIELD TILT PROXY
#
# América final third events /
# total final third events
# ============================================================

model_spatial[
  ,
  total_final_third_events :=
    america_final_third_events +
    opponent_final_third_events
]

model_spatial[
  ,
  america_field_tilt_proxy :=
    fifelse(
      total_final_third_events > 0,
      america_final_third_events /
        total_final_third_events,
      NA_real_
    )
]


# ============================================================
# 20. DIFERENCIAS ESPACIALES
# ============================================================

model_spatial[
  ,
  recovery_height_diff :=
    america_recovery_height -
    opponent_recovery_height
]

model_spatial[
  ,
  progressive_pass_diff :=
    america_progressive_passes -
    opponent_progressive_passes
]

model_spatial[
  ,
  entries_final_third_diff :=
    america_entries_final_third -
    opponent_entries_final_third
]

model_spatial[
  ,
  entries_box_diff :=
    america_entries_box -
    opponent_entries_box
]


# ============================================================
# 21. GUARDAR
# ============================================================

fwrite(
  model_spatial,
  "data/processed/america_male/model_context_spatial.csv",
  bom = TRUE
)

cat("\n=====================================\n")
cat("SPATIAL FEATURES CREADAS\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(model_spatial),
  "\n"
)

cat(
  "Columnas:",
  ncol(model_spatial),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(model_spatial$match_id),
  "\n"
)



