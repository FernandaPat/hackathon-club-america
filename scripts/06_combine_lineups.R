library(jsonlite)
library(data.table)

# ============================================================
# RUTAS
# ============================================================

ruta_lineups <- "data/raw/america_male/lineups"

archivos <- list.files(
  ruta_lineups,
  pattern = "\\.json$",
  full.names = TRUE
)

cat("Archivos de lineups encontrados:", length(archivos), "\n")


# ============================================================
# LISTA PARA GUARDAR RESULTADOS
# ============================================================

lineups_list <- list()


# ============================================================
# LEER CADA PARTIDO
# ============================================================

for (i in seq_along(archivos)) {

  archivo <- archivos[i]

  match_id <- gsub(
    "lineup_|\\.json",
    "",
    basename(archivo)
  )

  cat(
    "[", i, "/", length(archivos), "]",
    "Procesando:",
    match_id,
    "\n"
  )

  tryCatch({

    datos <- fromJSON(
      archivo,
      flatten = TRUE
    )

    # Si viene una lista por equipo
    if (is.data.frame(datos)) {

      datos$match_id <- as.integer(match_id)

      lineups_list[[length(lineups_list) + 1]] <- datos

    }

  },

  error = function(e) {

    cat(
      "ERROR:",
      match_id,
      "-",
      e$message,
      "\n"
    )

  })

}


# ============================================================
# UNIR
# ============================================================

lineups_full <- rbindlist(
  lineups_list,
  fill = TRUE,
  use.names = TRUE
)

cat("\nFilas:", nrow(lineups_full), "\n")
cat("Columnas:", ncol(lineups_full), "\n")


# ============================================================
# CONVERTIR COLUMNAS LISTA A JSON
# ============================================================

for (col in names(lineups_full)) {

  if (is.list(lineups_full[[col]])) {

    lineups_full[[col]] <- sapply(
      lineups_full[[col]],
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


# ============================================================
# GUARDAR
# ============================================================

fwrite(
  lineups_full,
  "data/processed/america_male/lineups_america.csv"
)

cat(
  "\nlineups_america.csv guardado correctamente\n"
)

