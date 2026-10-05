library(httr)
library(jsonlite)
library(data.table)

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

competitions <- fread(
  "data/raw/america_male/competitions.csv",
  encoding = "UTF-8"
)

# Solo Liga MX varonil
liga_mx <- competitions[
  competition_id == 73
]

cat(
  "Temporadas Liga MX encontradas:",
  nrow(liga_mx),
  "\n"
)


# ============================================================
# 3. CREAR CARPETA RAW
# ============================================================

dir.create(
  "data/raw/america_male/team_season_stats",
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# 4. DESCARGAR TEAM SEASON STATS
# ============================================================

stats_list <- list()

for (i in seq_len(nrow(liga_mx))) {

  competition_id <- liga_mx$competition_id[i]
  season_id <- liga_mx$season_id[i]

  cat(
    "[", i, "/", nrow(liga_mx), "]",
    "Descargando season:",
    season_id,
    "\n"
  )

  archivo_json <- paste0(
    "data/raw/america_male/team_season_stats/",
    "team_stats_",
    competition_id,
    "_",
    season_id,
    ".json"
  )

  tryCatch({

    # --------------------------------------------------------
    # DESCARGAR SOLO SI NO EXISTE
    # --------------------------------------------------------

    if (!file.exists(archivo_json)) {

      url <- paste0(
        "https://data.statsbomb.com/api/v4/competitions/",
        competition_id,
        "/seasons/",
        season_id,
        "/team-stats"
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

      writeLines(
        texto,
        archivo_json,
        useBytes = TRUE
      )

    }


    # --------------------------------------------------------
    # LEER JSON
    # --------------------------------------------------------

    datos <- fromJSON(
      archivo_json,
      flatten = TRUE
    )

    if (!is.data.frame(datos)) {
      datos <- as.data.frame(datos)
    }

    datos$competition_id_request <- competition_id
    datos$season_id_request <- season_id

    stats_list[[length(stats_list) + 1]] <- datos

    cat(
      "OK season:",
      season_id,
      "-",
      nrow(datos),
      "equipos\n"
    )

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
# 5. UNIR TODAS LAS TEMPORADAS
# ============================================================

team_season_stats <- rbindlist(
  stats_list,
  fill = TRUE,
  use.names = TRUE
)

cat("\n=====================================\n")
cat("TEAM SEASON STATS\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(team_season_stats),
  "\n"
)

cat(
  "Columnas:",
  ncol(team_season_stats),
  "\n"
)

cat(
  "Temporadas:",
  uniqueN(team_season_stats$season_id_request),
  "\n"
)


# ============================================================
# 6. CONVERTIR LISTAS A JSON PARA CSV
# ============================================================

for (col in names(team_season_stats)) {

  if (is.list(team_season_stats[[col]])) {

    team_season_stats[[col]] <- sapply(
      team_season_stats[[col]],
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
# 7. GUARDAR
# ============================================================

fwrite(
  team_season_stats,
  "data/processed/america_male/team_season_stats.csv",
  bom = TRUE
)

cat(
  "\nteam_season_stats.csv guardado correctamente\n"
)

grep(
  "team|xg|possession|pressure|obv|pass|shot|direct",
  names(team_season_stats),
  value = TRUE,
  ignore.case = TRUE
)