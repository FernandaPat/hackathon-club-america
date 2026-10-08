library(data.table)

# ============================================================
# 1. LEER FEATURES 360
# ============================================================

features_360 <- fread(
  "data/processed/america_male/features_360.csv",
  encoding = "UTF-8"
)

cat("360 filas:", nrow(features_360), "\n")


# ============================================================
# 2. LEER EVENTS
# ============================================================

events <- fread(
  "data/processed/america_male/events_america_clean.csv",
  encoding = "UTF-8"
)

cat("Events filas:", nrow(events), "\n")


# ============================================================
# 3. PREPARAR EVENTOS
#
# En Events el identificador del evento es "id"
# En 360 es "event_uuid"
# ============================================================

events_info <- events[
  ,
  .(
    match_id,
    event_uuid = id,
    minute,
    second,
    period,
    team_name = team.name,
    player_name = player.name,
    event_type = type.name,
    possession,
    possession_team = possession_team.name,
    under_pressure,
    counterpress,
    location
  )
]


# ============================================================
# 4. ELIMINAR DUPLICADOS POR EVENT UUID
# ============================================================

events_info <- unique(
  events_info,
  by = c(
    "match_id",
    "event_uuid"
  )
)

cat(
  "Eventos únicos para unión:",
  nrow(events_info),
  "\n"
)


# ============================================================
# 5. UNIR 360 CON EVENTS
# ============================================================

features_360_events <- merge(
  features_360,
  events_info,
  by = c(
    "match_id",
    "event_uuid"
  ),
  all.x = TRUE
)


# ============================================================
# 6. IDENTIFICAR AMÉRICA
# ============================================================

features_360_events[
  ,
  is_america := team_name == "América"
]


# ============================================================
# 7. VALIDAR MATCH
# ============================================================

cat("\n=====================================\n")
cat("JOIN 360 + EVENTS\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(features_360_events),
  "\n"
)

cat(
  "Eventos con minute:",
  sum(!is.na(features_360_events$minute)),
  "\n"
)

cat(
  "Eventos sin minute:",
  sum(is.na(features_360_events$minute)),
  "\n"
)

cat(
  "Porcentaje matched:",
  round(
    mean(!is.na(features_360_events$minute)) * 100,
    2
  ),
  "%\n"
)

cat(
  "Partidos:",
  uniqueN(features_360_events$match_id),
  "\n"
)


# ============================================================
# 8. GUARDAR
# ============================================================

fwrite(
  features_360_events,
  "data/processed/america_male/features_360_events.csv",
  bom = TRUE
)

cat(
  "\nfeatures_360_events.csv guardado correctamente\n"
)