library(StatsBombR)
library(dplyr)
library(jsonlite)

# ============================================
# CREDENCIALES
# ============================================

username <- "itam_hackathon@hudl.com"
password <- "NDeXcBUp"

# ============================================
# LEER PARTIDOS DE AMÉRICA
# ============================================

america_matches <- read.csv(
  "data/processed/america_male/america_matches.csv"
)

# Nos quedamos solo con IDs válidos
match_ids <- unique(america_matches$match_id)

match_ids <- match_ids[!is.na(match_ids)]

cat(
  "Partidos encontrados:",
  length(match_ids),
  "\n"
)

# ============================================
# CARPETA DE SALIDA
# ============================================

dir.create(
  "data/raw/america_male/events",
  showWarnings = FALSE,
  recursive = TRUE
)

# ============================================
# FUNCIÓN PARA CONVERTIR COLUMNAS LISTA
# ============================================

preparar_para_csv <- function(df) {

  for (col in names(df)) {

    if (is.list(df[[col]])) {

      df[[col]] <- sapply(
        df[[col]],
        function(x) {

          if (is.null(x)) {

            return(NA_character_)

          }

          toJSON(
            x,
            auto_unbox = TRUE,
            null = "null"
          )
        }
      )
    }
  }

  df
}

# ============================================
# DESCARGAR EVENTOS PARTIDO POR PARTIDO
# ============================================

for (i in seq_along(match_ids)) {

  match_id <- match_ids[i]

  archivo <- paste0(
    "data/raw/america_male/events/events_",
    match_id,
    ".csv"
  )

  # ------------------------------------------
  # SI YA EXISTE, NO VOLVER A DESCARGAR
  # ------------------------------------------

  if (file.exists(archivo)) {

    cat(
      "[", i, "/", length(match_ids), "]",
      "Ya existe:",
      match_id,
      "\n"
    )

    next
  }

  # ------------------------------------------
  # DESCARGAR
  # ------------------------------------------

  cat(
    "[", i, "/", length(match_ids), "]",
    "Descargando match:",
    match_id,
    "\n"
  )

  tryCatch({

    eventos <- get.events(
      username,
      password,
      match_id
    )

    # Agregar ID del partido por seguridad
    eventos$match_id_request <- match_id

    # Preparar columnas anidadas para CSV
    eventos_csv <- preparar_para_csv(eventos)

    # Guardar
    write.csv(
      eventos_csv,
      archivo,
      row.names = FALSE,
      fileEncoding = "UTF-8"
    )

    cat(
      "OK -",
      match_id,
      "-",
      nrow(eventos_csv),
      "eventos guardados\n"
    )

  },

  error = function(e) {

    cat(
      "ERROR - match:",
      match_id,
      "-",
      e$message,
      "\n"
    )

  })

}

cat("\nDESCARGA DE EVENTS TERMINADA\n")