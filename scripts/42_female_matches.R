library(data.table)
library(StatsBombR)
library(jsonlite)

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

source(
  "scripts/00_config.R",
  echo = FALSE
)

if (team_gender != "female") {
  stop("Este script debe ejecutarse con team_gender = 'female'.")
}


# ============================================================
# 2. CREDENCIALES
# ============================================================

if (Sys.getenv("STATSBOMB_USERNAME") == "") {
  readRenviron(".Renviron")
}

username <- Sys.getenv("STATSBOMB_USERNAME")
password <- Sys.getenv("STATSBOMB_PASSWORD")

if (
  username == "" ||
  password == ""
) {
  stop("No se encontraron credenciales de StatsBomb.")
}


# ============================================================
# 3. DESCARGAR COMPETICIONES
# ============================================================

competitions_data <- competitions(
  username,
  password
)

competitions_dt <- as.data.table(
  competitions_data
)


# ============================================================
# 4. ENCONTRAR TEMPORADAS DE LIGA MX FEMENIL
# ============================================================

female_seasons <- competitions_dt[
  competition_id == competition_id,
  .(
    competition_id,
    season_id,
    season_name
  )
]

# Evitar conflicto entre variable y columna
female_seasons <- competitions_dt[
  competition_id == 1438,
  .(
    competition_id,
    season_id,
    season_name
  )
]

if (
  nrow(female_seasons) == 0
) {
  stop(
    "No se encontraron temporadas para competition_id = 1438."
  )
}

setorder(
  female_seasons,
  season_id
)


# ============================================================
# 5. TABLA DE TEMPORADAS
# ============================================================

View(
  female_seasons
)


# ============================================================
# 6. DESCARGAR TODOS LOS PARTIDOS
# ============================================================

all_matches_list <- list()

counter <- 1


for (
  i in seq_len(
    nrow(female_seasons)
  )
) {

  season_id_current <-
    female_seasons$season_id[i]

  season_name_current <-
    female_seasons$season_name[i]


  cat(
    "\nDescargando temporada:",
    season_name_current,
    "| season_id:",
    season_id_current,
    "\n"
  )


  temp_matches <- tryCatch(

    get.matches(
      username,
      password,
      season_id_current,
      competition_id
    ),

    error = function(e) {

      cat(
        "Error temporada",
        season_id_current,
        ":",
        conditionMessage(e),
        "\n"
      )

      NULL
    }
  )


  if (
    is.null(temp_matches)
  ) {
    next
  }


  temp_dt <- as.data.table(
    temp_matches
  )


  if (
    nrow(temp_dt) == 0
  ) {
    next
  }


  temp_dt[
    ,
    season_id_request :=
      season_id_current
  ]


  temp_dt[
    ,
    season_name_request :=
      season_name_current
  ]


  all_matches_list[[counter]] <-
    temp_dt


  counter <- counter + 1
}


# ============================================================
# 7. VALIDAR DESCARGA
# ============================================================

if (
  length(all_matches_list) == 0
) {

  stop(
    "No se descargaron partidos femeniles."
  )
}


matches_full <- rbindlist(
  all_matches_list,
  use.names = TRUE,
  fill = TRUE
)


cat(
  "\nPartidos Liga MX Femenil descargados:",
  nrow(matches_full),
  "\n"
)


# ============================================================
# 8. IDENTIFICAR COLUMNAS DE EQUIPO
# ============================================================

home_candidates <- c(
  "home_team.home_team_name",
  "home_team_name",
  "home_team"
)

away_candidates <- c(
  "away_team.away_team_name",
  "away_team_name",
  "away_team"
)


home_found <- home_candidates[
  home_candidates %in%
    names(matches_full)
]

away_found <- away_candidates[
  away_candidates %in%
    names(matches_full)
]


if (
  length(home_found) == 0 ||
  length(away_found) == 0
) {

  stop(
    paste(
      "No encontré columnas home/away team.",
      "Columnas disponibles:",
      paste(
        names(matches_full),
        collapse = ", "
      )
    )
  )
}


home_col <- home_found[1]
away_col <- away_found[1]


# ============================================================
# 9. CREAR COLUMNAS LIMPIAS
# ============================================================

matches_full[
  ,
  home_team_clean :=
    as.character(
      get(home_col)
    )
]


matches_full[
  ,
  away_team_clean :=
    as.character(
      get(away_col)
    )
]


# ============================================================
# 10. REVISAR NOMBRES CON "AMERICA"
# ============================================================

america_name_check <- unique(
  c(

    matches_full[
      grepl(
        "Am",
        home_team_clean,
        ignore.case = TRUE
      ),
      home_team_clean
    ],

    matches_full[
      grepl(
        "Am",
        away_team_clean,
        ignore.case = TRUE
      ),
      away_team_clean
    ]
  )
)


america_name_table <- data.table(
  detected_name =
    america_name_check
)


View(
  america_name_table
)


# ============================================================
# 11. FILTRAR AMÉRICA
# ============================================================

america_matches <- matches_full[
  home_team_clean == team_name |
    away_team_clean == team_name
]


if (
  nrow(america_matches) == 0
) {

  stop(
    paste(
      "No encontré partidos con team_name =",
      team_name,
      ". Revisa america_name_table."
    )
  )
}


# ============================================================
# 12. HOME / AWAY
# ============================================================

america_matches[
  ,
  venue :=
    fifelse(
      home_team_clean == team_name,
      "HOME",
      "AWAY"
    )
]


america_matches[
  ,
  opponent :=
    fifelse(
      home_team_clean == team_name,
      away_team_clean,
      home_team_clean
    )
]


# ============================================================
# 13. SCORES
# ============================================================

home_score_candidates <- c(
  "home_score",
  "home_team_score"
)

away_score_candidates <- c(
  "away_score",
  "away_team_score"
)


home_score_found <- home_score_candidates[
  home_score_candidates %in%
    names(america_matches)
]

away_score_found <- away_score_candidates[
  away_score_candidates %in%
    names(america_matches)
]


if (
  length(home_score_found) > 0 &&
  length(away_score_found) > 0
) {

  home_score_col <-
    home_score_found[1]

  away_score_col <-
    away_score_found[1]


  america_matches[
    ,
    america_score :=
      fifelse(
        home_team_clean == team_name,
        as.numeric(
          get(home_score_col)
        ),
        as.numeric(
          get(away_score_col)
        )
      )
  ]


  america_matches[
    ,
    opponent_score :=
      fifelse(
        home_team_clean == team_name,
        as.numeric(
          get(away_score_col)
        ),
        as.numeric(
          get(home_score_col)
        )
      )
  ]


  america_matches[
    ,
    goal_difference :=
      america_score -
      opponent_score
  ]
}


# ============================================================
# 14. SEASON
# ============================================================

if (
  "season.season_name" %in%
    names(america_matches)
) {

  america_matches[
    ,
    season :=
      as.character(
        season.season_name
      )
  ]

} else if (
  "season_name" %in%
    names(america_matches)
) {

  america_matches[
    ,
    season :=
      as.character(
        season_name
      )
  ]

} else {

  america_matches[
    ,
    season :=
      as.character(
        season_name_request
      )
  ]
}


# ============================================================
# 15. MATCH DATE
# ============================================================

if (
  "match_date" %in%
    names(america_matches)
) {

  america_matches[
    ,
    match_date :=
      as.IDate(
        match_date
      )
  ]
}


# ============================================================
# 16. CONVERTIR LIST COLUMNS A JSON
# ============================================================

prepare_for_csv <- function(dt) {

  output <- copy(dt)


  list_cols <- names(output)[
    vapply(
      output,
      is.list,
      logical(1)
    )
  ]


  for (
    col in list_cols
  ) {

    output[
      ,
      (col) :=
        vapply(
          get(col),
          function(x) {

            if (
              is.null(x) ||
              length(x) == 0
            ) {

              return(
                NA_character_
              )
            }


            jsonlite::toJSON(
              x,
              auto_unbox = TRUE,
              null = "null"
            )
          },
          character(1)
        )
    ]
  }


  output
}


matches_full_csv <- prepare_for_csv(
  matches_full
)

america_matches_csv <- prepare_for_csv(
  america_matches
)


# ============================================================
# 17. GUARDAR RAW
# ============================================================

fwrite(
  matches_full_csv,
  file.path(
    raw_dir,
    "matches_liga_mx_femenil.csv"
  ),
  bom = TRUE
)


# ============================================================
# 18. GUARDAR AMÉRICA
# ============================================================

fwrite(
  america_matches_csv,
  file.path(
    processed_dir,
    "america_matches.csv"
  ),
  bom = TRUE
)


# ============================================================
# 19. FECHAS
# ============================================================

first_match <- NA_character_
last_match <- NA_character_


if (
  "match_date" %in%
    names(america_matches)
) {

  valid_dates <- america_matches[
    !is.na(match_date),
    match_date
  ]


  if (
    length(valid_dates) > 0
  ) {

    first_match <- as.character(
      min(
        valid_dates
      )
    )


    last_match <- as.character(
      max(
        valid_dates
      )
    )
  }
}


# ============================================================
# 20. RESUMEN
# ============================================================

female_matches_summary <- data.table(

  metric = c(
    "Liga MX Femenil matches",
    "America matches",
    "Unique America match IDs",
    "Seasons",
    "First match",
    "Last match"
  ),

  value = c(

    as.character(
      nrow(matches_full)
    ),

    as.character(
      nrow(america_matches)
    ),

    as.character(
      uniqueN(
        america_matches$match_id
      )
    ),

    as.character(
      uniqueN(
        america_matches$season
      )
    ),

    first_match,

    last_match
  )
)


# ============================================================
# 21. SAMPLE
# ============================================================

sample_cols <- c(
  "match_id",
  "match_date",
  "season",
  "venue",
  "opponent"
)


available_sample_cols <- sample_cols[
  sample_cols %in%
    names(america_matches)
]


female_matches_sample <- america_matches[
  ,
  ..available_sample_cols
][
  1:min(
    20,
    .N
  )
]


# ============================================================
# 22. SOLO DOS VISTAS
# ============================================================

View(
  female_matches_summary
)


View(
  female_matches_sample
)


# ============================================================
# 23. TERMINAL
# ============================================================

cat(
  "\n=====================================\n"
)

cat(
  "FEMALE MATCHES PIPELINE TERMINADO\n"
)

cat(
  "=====================================\n"
)


cat(
  "Liga MX Femenil matches:",
  nrow(matches_full),
  "\n"
)


cat(
  "America matches:",
  nrow(america_matches),
  "\n"
)


cat(
  "Unique America match IDs:",
  uniqueN(
    america_matches$match_id
  ),
  "\n"
)