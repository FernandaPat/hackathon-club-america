# ============================================================
# 43_female_events.R
# CLUB AMÉRICA FEMENIL
# DESCARGA DE EVENTOS STATSBOMB
# ============================================================


library(data.table)
library(StatsBombR)


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

team_name <- "América W"


# ============================================================
# 2. RUTAS
# ============================================================

project_root <- normalizePath(
  ".",
  winslash = "/",
  mustWork = TRUE
)


input_dir <- file.path(
  project_root,
  "data",
  "processed",
  "america_female"
)


output_dir <- input_dir


matches_file <- file.path(
  input_dir,
  "america_female_matches_played.csv"
)


events_file <- file.path(
  output_dir,
  "events_america_female.csv"
)


events_clean_file <- file.path(
  output_dir,
  "events_america_female_clean.csv"
)


# ============================================================
# 3. VALIDAR ARCHIVO DE PARTIDOS
# ============================================================

if (!file.exists(matches_file)) {

  stop(
    paste(
      "No existe:",
      matches_file
    )
  )
}


matches <- fread(
  matches_file
)


if (
  !"match_id"
  %in%
  names(matches)
) {

  stop(
    "El archivo de partidos no contiene match_id."
  )
}


# ============================================================
# 4. CREDENCIALES
# ============================================================

username <- Sys.getenv(
  "SB_USERNAME"
)


password <- Sys.getenv(
  "SB_PASSWORD"
)


if (username == "") {

  username <- Sys.getenv(
    "STATSBOMB_USERNAME"
  )
}


if (password == "") {

  password <- Sys.getenv(
    "STATSBOMB_PASSWORD"
  )
}


if (
  username == ""
  ||
  password == ""
) {

  renviron_path <- file.path(
    project_root,
    ".Renviron"
  )


  if (file.exists(renviron_path)) {

    readRenviron(
      renviron_path
    )


    username <- Sys.getenv(
      "SB_USERNAME"
    )


    password <- Sys.getenv(
      "SB_PASSWORD"
    )


    if (username == "") {

      username <- Sys.getenv(
        "STATSBOMB_USERNAME"
      )
    }


    if (password == "") {

      password <- Sys.getenv(
        "STATSBOMB_PASSWORD"
      )
    }
  }
}


if (
  username == ""
  ||
  password == ""
) {

  stop(
    "No se encontraron credenciales StatsBomb."
  )
}


# ============================================================
# 5. MATCH IDS
# ============================================================

match_ids <- unique(
  matches$match_id
)


match_ids <- match_ids[
  !is.na(match_ids)
]


cat("\n")
cat("============================================\n")
cat("COACHLENS - AMÉRICA FEMENIL EVENTS\n")
cat("============================================\n\n")


cat(
  "Partidos a descargar:",
  length(match_ids),
  "\n\n"
)


# ============================================================
# 6. DESCARGAR EVENTOS
# ============================================================

events_list <- list()


success_count <- 0

error_count <- 0


for (
  i
  in
  seq_along(match_ids)
) {

  match_id <- match_ids[i]


  cat(
    "[",
    i,
    "/",
    length(match_ids),
    "] match_id:",
    match_id,
    "\n"
  )


  match_events <- tryCatch(

    {

      StatsBombR::get.events(
        username = username,
        password = password,
        match_id = match_id
      )

    },

    error = function(e) {

      cat(
        "ERROR:",
        conditionMessage(e),
        "\n\n"
      )


      return(NULL)
    }
  )


  if (
    is.null(match_events)
    ||
    nrow(match_events) == 0
  ) {

    error_count <- error_count + 1

    next
  }


  match_events <- as.data.table(
    match_events
  )


  # ----------------------------------------------
  # asegurarnos de tener match_id
  # ----------------------------------------------

  if (
    !"match_id"
    %in%
    names(match_events)
  ) {

    match_events[
      ,
      match_id := match_id
    ]
  }


  events_list <- append(
    events_list,
    list(match_events)
  )


  success_count <- success_count + 1


  cat(
    "Eventos:",
    nrow(match_events),
    "\n\n"
  )
}


# ============================================================
# 7. VALIDAR RESULTADO
# ============================================================

if (
  length(events_list) == 0
) {

  stop(
    "No se descargaron eventos de ningún partido."
  )
}


# ============================================================
# 8. UNIR EVENTOS
# ============================================================

events <- rbindlist(
  events_list,
  fill = TRUE,
  use.names = TRUE
)


cat("\n")
cat("============================================\n")

cat(
  "PARTIDOS DESCARGADOS:",
  success_count,
  "\n"
)

cat(
  "PARTIDOS CON ERROR:",
  error_count,
  "\n"
)

cat(
  "EVENTOS TOTALES:",
  nrow(events),
  "\n"
)

cat(
  "PARTIDOS ÚNICOS:",
  uniqueN(events$match_id),
  "\n"
)

cat("============================================\n\n")


# ============================================================
# 9. ELIMINAR COLUMNAS TIPO LISTA PARA CSV
# ============================================================

list_columns <- names(events)[
  vapply(
    events,
    is.list,
    logical(1)
  )
]


cat(
  "Columnas tipo lista detectadas:",
  length(list_columns),
  "\n"
)


if (
  length(list_columns) > 0
) {

  cat(
    "No se exportarán directamente estas columnas:\n"
  )

  print(
    list_columns
  )
}


# ============================================================
# 10. TABLA ATÓMICA PARA GUARDAR
# ============================================================

atomic_columns <- names(events)[
  !vapply(
    events,
    is.list,
    logical(1)
  )
]


events_atomic <- events[
  ,
  ..atomic_columns
]


# ============================================================
# 11. NORMALIZAR NOMBRES IMPORTANTES
# ============================================================

team_candidates <- c(
  "team.name",
  "team_name",
  "team"
)


type_candidates <- c(
  "type.name",
  "type_name",
  "type"
)


player_candidates <- c(
  "player.name",
  "player_name",
  "player"
)


team_col <- team_candidates[
  team_candidates
  %in%
  names(events_atomic)
][1]


type_col <- type_candidates[
  type_candidates
  %in%
  names(events_atomic)
][1]


player_col <- player_candidates[
  player_candidates
  %in%
  names(events_atomic)
][1]


# ============================================================
# 12. CREAR COLUMNAS ESTÁNDAR
# ============================================================

if (
  length(team_col) > 0
  &&
  !is.na(team_col)
) {

  events_atomic[
    ,
    team_name :=
      as.character(
        get(team_col)
      )
  ]

} else {

  events_atomic[
    ,
    team_name :=
      NA_character_
  ]
}


if (
  length(type_col) > 0
  &&
  !is.na(type_col)
) {

  events_atomic[
    ,
    event_type :=
      as.character(
        get(type_col)
      )
  ]

} else {

  events_atomic[
    ,
    event_type :=
      NA_character_
  ]
}


if (
  length(player_col) > 0
  &&
  !is.na(player_col)
) {

  events_atomic[
    ,
    player_name :=
      as.character(
        get(player_col)
      )
  ]

} else {

  events_atomic[
    ,
    player_name :=
      NA_character_
  ]
}


# ============================================================
# 13. VARIABLES BÁSICAS
# ============================================================

if (
  !"minute"
  %in%
  names(events_atomic)
) {

  stop(
    "Los eventos no contienen minute."
  )
}


events_atomic[
  ,
  minute :=
    as.numeric(
      minute
    )
]


if (
  "second"
  %in%
  names(events_atomic)
) {

  events_atomic[
    ,
    second :=
      as.numeric(
        second
      )
  ]
}


# ============================================================
# 14. IDENTIFICAR EVENTOS DE AMÉRICA W
# ============================================================

events_atomic[
  ,
  is_america :=
    team_name
    ==
    team_name
]


# CORRECCIÓN IMPORTANTE:
# la línea anterior usa el mismo nombre de variable/columna,
# así que la reemplazamos correctamente aquí.

events_atomic[
  ,
  is_america :=
    get("team_name")
    ==
    "América W"
]


# ============================================================
# 15. AGREGAR CONTEXTO DEL PARTIDO
# ============================================================

match_context <- matches[
  ,
  .(
    match_id,
    match_date,
    opponent,
    venue,
    america_score,
    opponent_score
  )
]


events_atomic <- merge(
  events_atomic,
  match_context,
  by = "match_id",
  all.x = TRUE
)


# ============================================================
# 16. FILTRAR SOLO MATCHES DE AMÉRICA FEMENIL
# ============================================================

events_clean <- events_atomic[
  !is.na(opponent)
]


# ============================================================
# 17. ORDENAR
# ============================================================

order_columns <- c(
  "match_id",
  "period",
  "minute",
  "second"
)


order_columns <- order_columns[
  order_columns
  %in%
  names(events_clean)
]


if (
  length(order_columns) > 0
) {

  setorderv(
    events_clean,
    order_columns
  )
}


# ============================================================
# 18. EXPORTAR
# ============================================================

fwrite(
  events_atomic,
  events_file
)


fwrite(
  events_clean,
  events_clean_file
)


# ============================================================
# 19. VALIDACIONES
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACIÓN FINAL\n")
cat("============================================\n\n")


cat(
  "Eventos totales:",
  nrow(events_clean),
  "\n"
)


cat(
  "Partidos únicos:",
  uniqueN(
    events_clean$match_id
  ),
  "\n"
)


cat(
  "Eventos de América W:",
  sum(
    events_clean$is_america,
    na.rm = TRUE
  ),
  "\n"
)


cat(
  "Eventos rivales:",
  sum(
    !events_clean$is_america,
    na.rm = TRUE
  ),
  "\n\n"
)


cat(
  "Rango de minutos:\n"
)


print(
  range(
    events_clean$minute,
    na.rm = TRUE
  )
)


cat(
  "\nTipos de evento más frecuentes:\n"
)


event_summary <- events_clean[
  ,
  .(
    eventos = .N
  ),
  by = event_type
]


setorder(
  event_summary,
  -eventos
)


print(
  head(
    event_summary,
    20
  )
)


cat(
  "\nEventos por partido:\n"
)


match_summary <- events_clean[
  ,
  .(
    eventos = .N
  ),
  by = .(
    match_id,
    opponent
  )
]


setorder(
  match_summary,
  -eventos
)


print(
  match_summary
)


cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Archivo:\n",
  events_clean_file,
  "\n"
)