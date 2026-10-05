library(StatsBombR)
library(data.table)
library(dplyr)
library(jsonlite)

# ============================================================
# 1. CREDENCIALES
# ============================================================

if (Sys.getenv("STATSBOMB_USERNAME") == "") {
  readRenviron(".Renviron")
}

username <- Sys.getenv("STATSBOMB_USERNAME")
password <- Sys.getenv("STATSBOMB_PASSWORD")

if (username == "" || password == "") {
  stop("No se encontraron las credenciales de StatsBomb.")
}

cat("Credenciales cargadas correctamente\n")


# ============================================================
# 2. LEER COMPETICIONES
# ============================================================

competitions_data <- fread(
  "data/raw/america_male/competitions.csv",
  encoding = "UTF-8"
)

# Solo Liga MX masculina
liga_mx <- competitions_data[
  competition_id == 73
]

cat(
  "Temporadas Liga MX encontradas:",
  nrow(liga_mx),
  "\n"
)


# ============================================================
# 3. DESCARGAR TODAS LAS TEMPORADAS
# ============================================================

matches_list <- list()

for (i in seq_len(nrow(liga_mx))) {

  competition_id <- liga_mx$competition_id[i]
  season_id <- liga_mx$season_id[i]

  cat(
    "[", i, "/", nrow(liga_mx), "]",
    "Descargando season_id:",
    season_id,
    "\n"
  )

  tryCatch({

    temp_matches <- get.matches(
      username,
      password,
      season_id = season_id,
      competition_id = competition_id
    )

    if (!is.null(temp_matches) && nrow(temp_matches) > 0) {

      # Agregar IDs por seguridad
      temp_matches$competition_id_request <- competition_id
      temp_matches$season_id_request <- season_id

      matches_list[[length(matches_list) + 1]] <- temp_matches

      cat(
        "OK -",
        nrow(temp_matches),
        "partidos\n"
      )

    } else {

      cat(
        "SIN DATOS - season:",
        season_id,
        "\n"
      )

    }

  },

  error = function(e) {

    cat(
      "ERROR season:",
      season_id,
      "-",
      e$message,
      "\n"
    )

  })

}


# ============================================================
# 4. UNIR TODAS LAS TEMPORADAS
# ============================================================

matches_full <- rbindlist(
  matches_list,
  fill = TRUE,
  use.names = TRUE
)

cat("\n=====================================\n")
cat("LIGA MX COMPLETA\n")
cat("=====================================\n")

cat(
  "Partidos:",
  nrow(matches_full),
  "\n"
)

cat(
  "Match IDs únicos:",
  uniqueN(matches_full$match_id),
  "\n"
)

cat(
  "Columnas:",
  ncol(matches_full),
  "\n"
)


# ============================================================
# 5. CONVERTIR COLUMNAS TIPO LIST
# ============================================================

for (col in names(matches_full)) {

  if (is.list(matches_full[[col]])) {

    matches_full[[col]] <- sapply(
      matches_full[[col]],
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
# 6. GUARDAR BACKUP ANTES DE REEMPLAZAR
# ============================================================

archivo_actual <- "data/raw/america_male/matches_liga_mx.csv"

if (file.exists(archivo_actual)) {

  file.copy(
    archivo_actual,
    "data/raw/america_male/matches_liga_mx_broken_backup.csv",
    overwrite = TRUE
  )

  cat(
    "Backup del archivo anterior creado\n"
  )

}


# ============================================================
# 7. GUARDAR ARCHIVO COMPLETO
# ============================================================

fwrite(
  matches_full,
  archivo_actual,
  bom = TRUE
)

cat(
  "\nmatches_liga_mx.csv reconstruido correctamente\n"
)