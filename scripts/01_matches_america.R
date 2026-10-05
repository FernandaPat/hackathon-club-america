# ============================================================
# LIBRERÍAS
# ============================================================

library(StatsBombR)
library(dplyr)
library(jsonlite)


# ============================================================
# 1. CREDENCIALES
# ============================================================

username <- Sys.getenv("STATSBOMB_USERNAME")
password <- Sys.getenv("STATSBOMB_PASSWORD")


# ============================================================
# 2. CREAR CARPETAS
# ============================================================

dir.create("data", showWarnings = FALSE)
dir.create("data/raw", showWarnings = FALSE)
dir.create("data/processed", showWarnings = FALSE)


# ============================================================
# 3. DESCARGAR COMPETICIONES DISPONIBLES
# ============================================================

competitions_data <- competitions(
  username,
  password
)

write.csv(
  competitions_data,
  "data/raw/competitions.csv",
  row.names = FALSE
)

cat("competitions.csv guardado\n")


# ============================================================
# 4. FILTRAR LIGA MX MASCULINA
# Competition ID = 73
# ============================================================

liga_mx <- competitions_data %>%
  filter(competition_id == 73)

print(
  liga_mx %>%
    select(
      competition_id,
      season_id,
      competition_name,
      season_name
    )
)


# ============================================================
# 5. DESCARGAR PARTIDOS DE TODAS LAS TEMPORADAS
# ============================================================

all_matches <- list()

for (i in 1:nrow(liga_mx)) {

  comp_id <- liga_mx$competition_id[i]
  season_id <- liga_mx$season_id[i]

  cat(
    "Descargando temporada:",
    season_id,
    "\n"
  )

  temp_matches <- tryCatch(
    {

      get.matches(
        username,
        password,
        season_id = season_id,
        competition_id = comp_id
      )

    },
    error = function(e) {

      cat(
        "Error en season_id",
        season_id,
        ":",
        e$message,
        "\n"
      )

      NULL
    }
  )

  if (!is.null(temp_matches)) {

    all_matches[[length(all_matches) + 1]] <- temp_matches

  }
}


# ============================================================
# 6. UNIR TODOS LOS PARTIDOS
# ============================================================

matches_full <- bind_rows(all_matches)

write.csv(
  matches_full,
  "data/raw/matches_liga_mx.csv",
  row.names = FALSE
)

cat(
  "matches_liga_mx.csv guardado con",
  nrow(matches_full),
  "partidos\n"
)


# ============================================================
# 7. LIMPIAR NOMBRES DE COLUMNAS IMPORTANTES
# ============================================================

matches_full <- matches_full %>%
  rename(
    home_team = home_team.home_team_name,
    away_team = away_team.away_team_name,
    competition = competition.competition_name,
    season = season.season_name
  )


# ============================================================
# 8. FILTRAR PARTIDOS DEL CLUB AMÉRICA
# ============================================================

america_matches <- matches_full %>%
  filter(
    home_team == "América" |
    away_team == "América"
  )


# ============================================================
# 9. CREAR VARIABLES DESDE LA PERSPECTIVA DEL AMÉRICA
# ============================================================

america_matches <- america_matches %>%
  mutate(

    # Rival
    opponent = ifelse(
      home_team == "América",
      away_team,
      home_team
    ),

    # Local o visitante
    venue = ifelse(
      home_team == "América",
      "Home",
      "Away"
    ),

    # Goles del América
    america_score = ifelse(
      home_team == "América",
      home_score,
      away_score
    ),

    # Goles del rival
    opponent_score = ifelse(
      home_team == "América",
      away_score,
      home_score
    ),

    # Diferencia de goles
    goal_difference = america_score - opponent_score
  )


# ============================================================
# 10. REVISAR RESULTADO
# ============================================================

america_matches %>%
  select(
    match_id,
    match_date,
    season,
    opponent,
    venue,
    america_score,
    opponent_score,
    goal_difference
  ) %>%
  head(20)


# ============================================================
# 11. CONVERTIR COLUMNAS TIPO LISTA A JSON
# Para evitar problemas al guardar el CSV
# ============================================================

america_matches_csv <- america_matches

for (col in names(america_matches_csv)) {

  if (is.list(america_matches_csv[[col]])) {

    america_matches_csv[[col]] <- sapply(
      america_matches_csv[[col]],
      function(x) {

        if (is.null(x)) {

          NA_character_

        } else {

          toJSON(
            x,
            auto_unbox = TRUE,
            null = "null"
          )

        }
      }
    )
  }
}


# ============================================================
# 12. GUARDAR PARTIDOS DEL AMÉRICA
# ============================================================

write.csv(
  america_matches_csv,
  "data/processed/america_matches.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat(
  "america_matches.csv guardado con",
  nrow(america_matches_csv),
  "partidos del América\n"
)