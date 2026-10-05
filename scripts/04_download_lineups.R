library(httr)
library(jsonlite)

# ============================================
# CREDENCIALES
# ============================================

username <- Sys.getenv("STATSBOMB_USERNAME")
password <- Sys.getenv("STATSBOMB_PASSWORD")

# ============================================
# PARTIDOS DE AMÉRICA
# ============================================

america_matches <- read.csv(
  "data/processed/america_male/america_matches.csv"
)

match_ids <- unique(america_matches$match_id)
match_ids <- match_ids[!is.na(match_ids)]

cat(
  "Partidos encontrados:",
  length(match_ids),
  "\n"
)

# ============================================
# CARPETA
# ============================================

dir.create(
  "data/raw/america_male/lineups",
  showWarnings = FALSE,
  recursive = TRUE
)

# ============================================
# DESCARGAR LINEUPS
# ============================================

for (i in seq_along(match_ids)) {

  match_id <- match_ids[i]

  archivo <- paste0(
    "data/raw/america_male/lineups/lineup_",
    match_id,
    ".json"
  )

  # Si ya existe, saltarlo
  if (file.exists(archivo)) {

    cat(
      "[", i, "/", length(match_ids), "]",
      "Ya existe:",
      match_id,
      "\n"
    )

    next
  }

  cat(
    "[", i, "/", length(match_ids), "]",
    "Descargando lineup:",
    match_id,
    "\n"
  )

  tryCatch({

    url <- paste0(
      "https://data.statsbomb.com/api/v5/lineups/",
      match_id
    )

    respuesta <- GET(
      url,
      authenticate(username, password)
    )

    stop_for_status(respuesta)

    texto <- content(
      respuesta,
      as = "text",
      encoding = "UTF-8"
    )

    # Evitar guardar respuestas vacías
    datos <- fromJSON(
      texto,
      simplifyVector = FALSE
    )

    if (length(datos) == 0) {

      cat(
        "SIN DATOS -",
        match_id,
        "\n"
      )

      next
    }

    writeLines(
      texto,
      archivo,
      useBytes = TRUE
    )

    cat(
      "OK -",
      match_id,
      "\n"
    )

  },

  error = function(e) {

    cat(
      "ERROR -",
      match_id,
      "-",
      e$message,
      "\n"
    )

  })

}

cat("\nDESCARGA DE LINEUPS TERMINADA\n")