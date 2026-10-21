library(data.table)

# ============================================================
# RUTAS
# ============================================================

INPUT_PATH <- "data/processed/america_male/america_matches.csv"

OUTPUT_PATH <- "dashboard/data/match_history.csv"


# ============================================================
# CARGAR PARTIDOS
# ============================================================

matches <- fread(
  INPUT_PATH
)


# ============================================================
# REVISAR COLUMNAS
# ============================================================

View(
  data.frame(
    columna = names(matches)
  )
)


# ============================================================
# CONVERTIR FECHA
# ============================================================

matches[
  ,
  match_date := as.IDate(match_date)
]


# ============================================================
# QUEDARNOS SOLO CON PARTIDOS YA JUGADOS
#
# Un partido se considera jugado si tiene:
# - home_score
# - away_score
#
# Así no contamos partidos futuros/programados.
# ============================================================

matches_played <- matches[
  !is.na(home_score)
  &
  !is.na(away_score)
]


# ============================================================
# ORDENAR
# ============================================================

setorder(
  matches_played,
  match_date
)


# ============================================================
# GUARDAR ARCHIVO LIGERO PARA STREAMLIT
# ============================================================

fwrite(
  matches_played,
  OUTPUT_PATH
)


# ============================================================
# COMPROBACIÓN
# ============================================================

cat(
  "\nPartidos jugados exportados:",
  uniqueN(matches_played$match_id),
  "\n"
)


cat(
  "Archivo creado:",
  OUTPUT_PATH,
  "\n"
)


# ============================================================
# COMPROBAR ATLAS
# ============================================================

atlas_matches <- matches_played[
  grepl(
    "atlas",
    home_team,
    ignore.case = TRUE
  )
  |
  grepl(
    "atlas",
    away_team,
    ignore.case = TRUE
  )
]


View(
  atlas_matches
)


cat(
  "\nPartidos JUGADOS vs Atlas:",
  uniqueN(atlas_matches$match_id),
  "\n"
)


# ============================================================
# COMPROBAR MAZATLÁN
# ============================================================

mazatlan_matches <- matches_played[
  grepl(
    "mazatl",
    home_team,
    ignore.case = TRUE
  )
  |
  grepl(
    "mazatl",
    away_team,
    ignore.case = TRUE
  )
]


View(
  mazatlan_matches
)


cat(
  "\nPartidos JUGADOS vs Mazatlán:",
  uniqueN(mazatlan_matches$match_id),
  "\n"
)