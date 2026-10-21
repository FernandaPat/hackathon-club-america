# ============================================================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# ============================================================

# Cambia únicamente esta línea según el pipeline
team_gender <- "female"
# team_gender <- "male"


# ============================================================
# IDS STATSBOMB
# ============================================================

if (team_gender == "male") {

  competition_id <- 73

  raw_dir <-
    "data/raw/america_male"

  processed_dir <-
    "data/processed/america_male"

  team_name <-
    "América"

} else if (team_gender == "female") {

  competition_id <- 1438

  raw_dir <-
    "data/raw/america_female"

  processed_dir <-
    "data/processed/america_female"

  team_name <-
    "América"

} else {

  stop(
    "team_gender debe ser 'male' o 'female'."
  )
}


# ============================================================
# CARPETAS
# ============================================================

dir.create(
  raw_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  processed_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(
    raw_dir,
    "events"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(
    raw_dir,
    "lineups"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(
    raw_dir,
    "frames_360"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(
    raw_dir,
    "team_season_stats"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)


cat(
  "\n=====================================\n"
)

cat(
  "PROJECT CONFIGURATION\n"
)

cat(
  "=====================================\n"
)

cat(
  "Gender:",
  team_gender,
  "\n"
)

cat(
  "Competition ID:",
  competition_id,
  "\n"
)

cat(
  "Raw directory:",
  raw_dir,
  "\n"
)

cat(
  "Processed directory:",
  processed_dir,
  "\n"
)