# ============================================================
# 42_female_matches.R
# CLUB AMÉRICA FEMENIL
# ============================================================

library(data.table)
library(StatsBombR)


# ============================================================
# CONFIGURACIÓN
# ============================================================

competition_id <- 1438

season_ids <- c(
  317,
  318,
  351
)

team_name <- "América W"


# ============================================================
# RUTAS
# ============================================================

project_root <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)

output_dir <- file.path(
  project_root,
  "data",
  "processed",
  "america_female"
)

if (!dir.exists(output_dir)) {
  dir.create(
    output_dir,
    recursive = TRUE
  )
}


# ============================================================
# CREDENCIALES
# ============================================================

username <- Sys.getenv("SB_USERNAME")
password <- Sys.getenv("SB_PASSWORD")

if (username == "") {
  username <- Sys.getenv("STATSBOMB_USERNAME")
}

if (password == "") {
  password <- Sys.getenv("STATSBOMB_PASSWORD")
}

if (username == "" || password == "") {

  renviron_path <- file.path(
    project_root,
    ".Renviron"
  )

  if (file.exists(renviron_path)) {

    readRenviron(renviron_path)

    username <- Sys.getenv("SB_USERNAME")
    password <- Sys.getenv("SB_PASSWORD")

    if (username == "") {
      username <- Sys.getenv("STATSBOMB_USERNAME")
    }

    if (password == "") {
      password <- Sys.getenv("STATSBOMB_PASSWORD")
    }
  }
}

if (username == "" || password == "") {
  stop(
    "No se encontraron las credenciales de StatsBomb."
  )
}


# ============================================================
# INICIO
# ============================================================

cat("\n")
cat("============================================\n")
cat("COACHLENS - CLUB AMERICA FEMENIL\n")
cat("============================================\n\n")

cat("Competition ID:", competition_id, "\n")
cat("Equipo:", team_name, "\n\n")


# ============================================================
# DESCARGAR TEMPORADAS
# ============================================================

all_seasons <- list()


for (season_id in season_ids) {

  cat(
    "Descargando season_id:",
    season_id,
    "...\n"
  )

  season_matches <- tryCatch(

    StatsBombR::get.matches(
      username = username,
      password = password,
      season_id = season_id,
      competition_id = competition_id
    ),

    error = function(e) {

      cat(
        "ERROR en temporada",
        season_id,
        ":",
        conditionMessage(e),
        "\n"
      )

      NULL
    }
  )


  if (
    is.null(season_matches) ||
    nrow(season_matches) == 0
  ) {

    cat(
      "Sin partidos para season_id:",
      season_id,
      "\n\n"
    )

    next
  }


  season_matches <- as.data.table(
    season_matches
  )


  season_matches[
    ,
    source_season_id := season_id
  ]


  # ==========================================================
  # IMPORTANTE:
  # usamos append() y NO usamos [[ ]]
  # ==========================================================

  all_seasons <- append(
    all_seasons,
    list(season_matches)
  )


  cat(
    "Partidos descargados:",
    nrow(season_matches),
    "\n\n"
  )
}


# ============================================================
# VALIDAR DESCARGA
# ============================================================

if (length(all_seasons) == 0) {
  stop(
    "No se descargó ninguna temporada."
  )
}


# ============================================================
# UNIR TEMPORADAS
# ============================================================

league_matches <- rbindlist(
  all_seasons,
  fill = TRUE,
  use.names = TRUE
)


cat("\n")
cat("============================================\n")
cat(
  "PARTIDOS TOTALES LIGA MX FEMENIL:",
  nrow(league_matches),
  "\n"
)
cat("============================================\n\n")


# ============================================================
# REVISAR COLUMNAS
# ============================================================

cat("Columnas disponibles:\n\n")

print(
  names(league_matches)
)

cat("\n")


# ============================================================
# COLUMNAS DE EQUIPOS
# ============================================================

home_team_col <- "home_team.home_team_name"
away_team_col <- "away_team.away_team_name"


if (!home_team_col %in% names(league_matches)) {

  stop(
    paste(
      "No existe la columna:",
      home_team_col
    )
  )
}


if (!away_team_col %in% names(league_matches)) {

  stop(
    paste(
      "No existe la columna:",
      away_team_col
    )
  )
}


# ============================================================
# LISTA DE EQUIPOS
# ============================================================

home_teams <- league_matches[
  ,
  get(home_team_col)
]


away_teams <- league_matches[
  ,
  get(away_team_col)
]


all_teams <- sort(
  unique(
    c(
      home_teams,
      away_teams
    )
  )
)


cat(
  "Equipos encontrados:\n\n"
)

print(all_teams)

cat("\n")


# ============================================================
# VALIDAR AMÉRICA W
# ============================================================

if (!team_name %in% all_teams) {

  america_candidates <- all_teams[
    grepl(
      "america",
      iconv(
        all_teams,
        from = "",
        to = "ASCII//TRANSLIT"
      ),
      ignore.case = TRUE
    )
  ]


  cat(
    "No se encontró exactamente:",
    team_name,
    "\n\n"
  )


  cat(
    "Posibles nombres encontrados:\n"
  )

  print(
    america_candidates
  )


  stop(
    "No se encontró América W."
  )
}


cat(
  "Equipo encontrado correctamente:",
  team_name,
  "\n\n"
)


# ============================================================
# FILTRAR AMÉRICA W
# ============================================================

america_matches <- league_matches[
  get(home_team_col) == team_name |
  get(away_team_col) == team_name
]


cat(
  "Partidos de América W encontrados:",
  nrow(america_matches),
  "\n\n"
)


if (nrow(america_matches) == 0) {

  stop(
    "Se encontraron 0 partidos de América W."
  )
}


# ============================================================
# VARIABLES LIMPIAS
# ============================================================

america_matches[
  ,
  home_team := get(home_team_col)
]


america_matches[
  ,
  away_team := get(away_team_col)
]


america_matches[
  ,
  opponent := fifelse(
    home_team == team_name,
    away_team,
    home_team
  )
]


america_matches[
  ,
  venue := fifelse(
    home_team == team_name,
    "HOME",
    "AWAY"
  )
]


# ============================================================
# FECHA
# ============================================================

if ("match_date" %in% names(america_matches)) {

  america_matches[
    ,
    match_date := as.IDate(match_date)
  ]
}


# ============================================================
# MARCADOR
# ============================================================

if ("home_score" %in% names(america_matches)) {

  america_matches[
    ,
    home_score_clean := as.numeric(home_score)
  ]

} else {

  america_matches[
    ,
    home_score_clean := NA_real_
  ]
}


if ("away_score" %in% names(america_matches)) {

  america_matches[
    ,
    away_score_clean := as.numeric(away_score)
  ]

} else {

  america_matches[
    ,
    away_score_clean := NA_real_
  ]
}


# ============================================================
# GOLES DE AMÉRICA Y RIVAL
# ============================================================

america_matches[
  ,
  america_score := fifelse(
    venue == "HOME",
    home_score_clean,
    away_score_clean
  )
]


america_matches[
  ,
  opponent_score := fifelse(
    venue == "HOME",
    away_score_clean,
    home_score_clean
  )
]


# ============================================================
# PARTIDOS JUGADOS
# ============================================================

played_matches <- copy(
  america_matches
)


if ("match_date" %in% names(played_matches)) {

  played_matches <- played_matches[
    match_date <= Sys.Date()
  ]
}


played_matches <- played_matches[
  !is.na(home_score_clean) &
  !is.na(away_score_clean)
]


cat(
  "Partidos con marcador final:",
  nrow(played_matches),
  "\n\n"
)


# ============================================================
# RESUMEN POR TEMPORADA
# ============================================================

cat(
  "PARTIDOS POR TEMPORADA:\n"
)


season_summary <- america_matches[
  ,
  .(
    partidos = .N
  ),
  by = source_season_id
]


setorder(
  season_summary,
  source_season_id
)


print(
  season_summary
)


# ============================================================
# LOCAL / VISITANTE
# ============================================================

cat(
  "\nLOCAL / VISITANTE:\n"
)


venue_summary <- america_matches[
  ,
  .(
    partidos = .N
  ),
  by = venue
]


print(
  venue_summary
)


# ============================================================
# RIVALES
# ============================================================

cat(
  "\nRIVALES:\n"
)


opponent_summary <- america_matches[
  ,
  .(
    partidos = .N
  ),
  by = opponent
]


setorder(
  opponent_summary,
  -partidos
)


print(
  opponent_summary
)


# ============================================================
# TABLA FINAL
# ============================================================

columns_to_keep <- c(
  "match_id",
  "match_date",
  "source_season_id",
  "home_team",
  "away_team",
  "opponent",
  "venue",
  "home_score_clean",
  "away_score_clean",
  "america_score",
  "opponent_score"
)


columns_to_keep <- columns_to_keep[
  columns_to_keep %in% names(america_matches)
]


america_matches_clean <- america_matches[
  ,
  ..columns_to_keep
]


played_matches_clean <- played_matches[
  ,
  ..columns_to_keep
]


# ============================================================
# RENOMBRAR SCORES
# ============================================================

if (
  "home_score_clean"
  %in%
  names(america_matches_clean)
) {

  setnames(
    america_matches_clean,
    "home_score_clean",
    "home_score"
  )
}


if (
  "away_score_clean"
  %in%
  names(america_matches_clean)
) {

  setnames(
    america_matches_clean,
    "away_score_clean",
    "away_score"
  )
}


if (
  "home_score_clean"
  %in%
  names(played_matches_clean)
) {

  setnames(
    played_matches_clean,
    "home_score_clean",
    "home_score"
  )
}


if (
  "away_score_clean"
  %in%
  names(played_matches_clean)
) {

  setnames(
    played_matches_clean,
    "away_score_clean",
    "away_score"
  )
}


# ============================================================
# ORDENAR
# ============================================================

if ("match_date" %in% names(america_matches_clean)) {

  setorder(
    america_matches_clean,
    match_date
  )
}


if ("match_date" %in% names(played_matches_clean)) {

  setorder(
    played_matches_clean,
    match_date
  )
}


# ============================================================
# EXPORTAR
# ============================================================

file_america <- file.path(
  output_dir,
  "america_female_matches.csv"
)

file_played <- file.path(
  output_dir,
  "america_female_matches_played.csv"
)


# ------------------------------------------------------------
# Comprobar que las tablas finales NO tengan list columns
# ------------------------------------------------------------

america_list_cols <- names(
  america_matches_clean
)[
  vapply(
    america_matches_clean,
    is.list,
    logical(1)
  )
]


played_list_cols <- names(
  played_matches_clean
)[
  vapply(
    played_matches_clean,
    is.list,
    logical(1)
  )
]


if (length(america_list_cols) > 0) {

  stop(
    paste(
      "america_matches_clean todavía contiene columnas tipo lista:",
      paste(
        america_list_cols,
        collapse = ", "
      )
    )
  )
}


if (length(played_list_cols) > 0) {

  stop(
    paste(
      "played_matches_clean todavía contiene columnas tipo lista:",
      paste(
        played_list_cols,
        collapse = ", "
      )
    )
  )
}


# ------------------------------------------------------------
# Guardar únicamente tablas procesadas
# ------------------------------------------------------------

fwrite(
  america_matches_clean,
  file_america
)


fwrite(
  played_matches_clean,
  file_played
)


# ============================================================
# RESUMEN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("EXPORTACION COMPLETADA\n")
cat("============================================\n\n")


cat(
  "Partidos América W totales:",
  nrow(
    america_matches_clean
  ),
  "\n"
)


cat(
  "Partidos América W jugados:",
  nrow(
    played_matches_clean
  ),
  "\n\n"
)


cat(
  "Archivo historial completo:\n",
  file_america,
  "\n\n"
)


cat(
  "Archivo solo partidos jugados:\n",
  file_played,
  "\n\n"
)


cat(
  "Rango de fechas:\n"
)


if (
  "match_date"
  %in%
  names(
    america_matches_clean
  )
) {

  print(
    range(
      america_matches_clean$match_date,
      na.rm = TRUE
    )
  )
}


cat(
  "\nPartidos por temporada:\n"
)


print(
  america_matches[
    ,
    .(
      partidos = .N
    ),
    by = source_season_id
  ][
    order(
      source_season_id
    )
  ]
)


cat(
  "\nPartidos por rival:\n"
)


opponent_summary_final <- america_matches_clean[
  ,
  .(
    partidos = .N
  ),
  by = opponent
]


setorder(
  opponent_summary_final,
  -partidos
)


print(
  opponent_summary_final
)


cat(
  "\nPrimeros partidos:\n"
)


print(
  head(
    america_matches_clean,
    10
  )
)


cat(
  "\nÚltimos partidos:\n"
)


print(
  tail(
    america_matches_clean,
    10
  )
)


cat(
  "\nTODO CORRECTO.\n"
)