library(data.table)
library(dplyr)

# ============================================
# RUTA DE LOS EVENTS
# ============================================

ruta_events <- "data/raw/america_male/events"

archivos <- list.files(
  ruta_events,
  pattern = "\\.csv$",
  full.names = TRUE
)

cat("Archivos encontrados:", length(archivos), "\n")


# ============================================
# REVISAR CUÁNTOS TIENEN DATOS
# ============================================

conteos <- sapply(
  archivos,
  function(archivo) {
    tryCatch(
      nrow(fread(archivo)),
      error = function(e) 0
    )
  }
)

cat("Partidos con eventos:", sum(conteos > 0), "\n")
cat("Partidos sin eventos:", sum(conteos == 0), "\n")


# ============================================
# QUEDARNOS SOLO CON ARCHIVOS CON EVENTOS
# ============================================

archivos_validos <- archivos[conteos > 0]

cat(
  "Archivos que se van a unir:",
  length(archivos_validos),
  "\n"
)


# ============================================
# LEER TODOS LOS EVENTS
# ============================================

lista_events <- lapply(
  archivos_validos,
  function(archivo) {

    datos <- fread(
      archivo,
      encoding = "UTF-8"
    )

    datos
  }
)


# ============================================
# UNIR TODO
# ============================================

events_america <- rbindlist(
  lista_events,
  fill = TRUE,
  use.names = TRUE
)

cat(
  "Total de eventos:",
  nrow(events_america),
  "\n"
)

cat(
  "Total de columnas:",
  ncol(events_america),
  "\n"
)


# ============================================
# GUARDAR BASE PROCESADA
# ============================================

fwrite(
  events_america,
  "data/processed/america_male/events_america.csv"
)

cat(
  "events_america.csv guardado correctamente\n"
)