library(httr)
library(jsonlite)
library(dplyr)

# ============================================
# CARGAR CREDENCIALES
# ============================================

if (Sys.getenv("STATSBOMB_USERNAME") == "") {
  readRenviron(".Renviron")
}

username <- Sys.getenv("STATSBOMB_USERNAME")
password <- Sys.getenv("STATSBOMB_PASSWORD")

if (username == "" || password == "") {
  stop("No se encontraron las credenciales en .Renviron")
}

cat("Credenciales cargadas correctamente\n")


# ============================================
# LEER PARTIDOS DE AMÉRICA
# ============================================

america_matches <- read.csv(
  "data/processed/america_male/america_matches.csv"
)

# ============================================
# USAR SOLO PARTIDOS CON 360 DISPONIBLE
# ============================================

if ("match_status_360" %in% names(america_matches)) {

  matches_360 <- america_matches %>%
    filter(
      !is.na(match_status_360),
      tolower(match_status_360) == "available"
    )

  match_ids <- unique(matches_360$match_id)

} else {

  # Si la columna no existe, intentamos todos
  match_ids <- unique(america_matches$match_id)
}

match_ids <- match_ids[!is.na(match_ids)]

cat(
  "Partidos candidatos para 360:",
  length(match_ids),
  "\n"
)


# ============================================
# CREAR CARPETA
# ============================================

dir.create(
  "data/raw/america_male/frames_360",
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================
# DESCARGAR 360 PARTIDO POR PARTIDO
# ============================================

for (i in seq_along(match_ids)) {

  match_id <- match_ids[i]

  archivo <- paste0(
    "data/raw/america_male/frames_360/360_",
    match_id,
    ".json"
  )

  # ------------------------------------------
  # NO DESCARGAR DE NUEVO SI YA EXISTE
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

  cat(
    "[", i, "/", length(match_ids), "]",
    "Descargando 360:",
    match_id,
    "\n"
  )


  # ------------------------------------------
  # CONSULTAR API
  # ------------------------------------------

  tryCatch({

    url <- paste0(
      "https://data.statsbomb.com/api/v2/360-frames/",
      match_id
    )

    respuesta <- GET(
      url,
      authenticate(username, password)
    )

    # Si no existe 360 para ese partido
    if (status_code(respuesta) == 404) {

      cat(
        "SIN 360 -",
        match_id,
        "\n"
      )

      next
    }

    stop_for_status(respuesta)

    texto <- content(
      respuesta,
      as = "text",
      encoding = "UTF-8"
    )

    datos <- fromJSON(
      texto,
      simplifyVector = FALSE
    )

    # Evitar archivos vacíos
    if (length(datos) == 0) {

      cat(
        "SIN DATOS -",
        match_id,
        "\n"
      )

      next
    }

    # Guardar JSON completo
    writeLines(
      texto,
      archivo,
      useBytes = TRUE
    )

    cat(
      "OK -",
      match_id,
      "-",
      length(datos),
      "frames guardados\n"
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

cat("\nDESCARGA DE 360 TERMINADA\n")