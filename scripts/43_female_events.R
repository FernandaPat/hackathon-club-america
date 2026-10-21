# ============================================================
# 43_female_events.R
# CLUB AMÉRICA FEMENIL
# DESCARGA Y LIMPIEZA DE EVENTOS STATSBOMB
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

matches_file <- file.path(
  input_dir,
  "america_female_matches_played.csv"
)

events_clean_file <- file.path(
  input_dir,
  "events_america_female_clean.csv"
)


# ============================================================
# 3. VALIDAR PARTIDOS
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

if (!"match_id" %in% names(matches)) {

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
    "No se encontraron credenciales de StatsBomb."
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
cat("COACHLENS - AMERICA FEMENIL EVENTS\n")
cat("============================================\n\n")

cat(
  "Partidos a descargar:",
  length(match_ids),
  "\n\n"
)


# ============================================================
# 6. FUNCIÓN PARA EXTRAER COORDENADAS
# ============================================================

get_coord <- function(
  value,
  position
) {

  if (is.null(value)) {
    return(NA_real_)
  }

  if (length(value) < position) {
    return(NA_real_)
  }

  result <- suppressWarnings(
    as.numeric(
      value[[position]]
    )
  )

  if (length(result) == 0) {
    return(NA_real_)
  }

  result[1]
}


extract_coordinate_column <- function(
  list_column,
  position
) {

  vapply(
    list_column,
    FUN = function(value) {

      get_coord(
        value,
        position
      )
    },
    FUN.VALUE = numeric(1)
  )
}


# ============================================================
# 7. DESCARGAR EVENTOS
# ============================================================

events_list <- list()

success_count <- 0
error_count <- 0


for (
  i
  in
  seq_along(match_ids)
) {

  current_match_id <- match_ids[i]

  cat(
    "[",
    i,
    "/",
    length(match_ids),
    "] match_id:",
    current_match_id,
    "\n"
  )


  match_events <- tryCatch(

    StatsBombR::get.events(
      username = username,
      password = password,
      match_id = current_match_id
    ),

    error = function(e) {

      cat(
        "ERROR:",
        conditionMessage(e),
        "\n\n"
      )

      NULL
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


  if (!"match_id" %in% names(match_events)) {

    match_events[
      ,
      match_id := current_match_id
    ]
  }


  # ==========================================================
  # EXTRAER LOCATION
  # ==========================================================

  if (
    "location"
    %in%
    names(match_events)
  ) {

    match_events[
      ,
      x := extract_coordinate_column(
        location,
        1
      )
    ]

    match_events[
      ,
      y := extract_coordinate_column(
        location,
        2
      )
    ]

  } else {

    match_events[
      ,
      x := NA_real_
    ]

    match_events[
      ,
      y := NA_real_
    ]
  }


  # ==========================================================
  # PASS END LOCATION
  # ==========================================================

  if (
    "pass.end_location"
    %in%
    names(match_events)
  ) {

    match_events[
      ,
      pass_end_x :=
        extract_coordinate_column(
          get(
            "pass.end_location"
          ),
          1
        )
    ]

    match_events[
      ,
      pass_end_y :=
        extract_coordinate_column(
          get(
            "pass.end_location"
          ),
          2
        )
    ]

  } else {

    match_events[
      ,
      pass_end_x := NA_real_
    ]

    match_events[
      ,
      pass_end_y := NA_real_
    ]
  }


  # ==========================================================
  # CARRY END LOCATION
  # ==========================================================

  if (
    "carry.end_location"
    %in%
    names(match_events)
  ) {

    match_events[
      ,
      carry_end_x :=
        extract_coordinate_column(
          get(
            "carry.end_location"
          ),
          1
        )
    ]

    match_events[
      ,
      carry_end_y :=
        extract_coordinate_column(
          get(
            "carry.end_location"
          ),
          2
        )
    ]

  } else {

    match_events[
      ,
      carry_end_x := NA_real_
    ]

    match_events[
      ,
      carry_end_y := NA_real_
    ]
  }


  # ==========================================================
  # SHOT END LOCATION
  # ==========================================================

  if (
    "shot.end_location"
    %in%
    names(match_events)
  ) {

    match_events[
      ,
      shot_end_x :=
        extract_coordinate_column(
          get(
            "shot.end_location"
          ),
          1
        )
    ]

    match_events[
      ,
      shot_end_y :=
        extract_coordinate_column(
          get(
            "shot.end_location"
          ),
          2
        )
    ]

    match_events[
      ,
      shot_end_z :=
        extract_coordinate_column(
          get(
            "shot.end_location"
          ),
          3
        )
    ]

  } else {

    match_events[
      ,
      shot_end_x := NA_real_
    ]

    match_events[
      ,
      shot_end_y := NA_real_
    ]

    match_events[
      ,
      shot_end_z := NA_real_
    ]
  }


  events_list <- append(
    events_list,
    list(match_events)
  )


  success_count <- (
    success_count + 1
  )


  cat(
    "Eventos:",
    nrow(match_events),
    "\n\n"
  )
}


# ============================================================
# 8. VALIDAR DESCARGA
# ============================================================

if (
  length(events_list) == 0
) {

  stop(
    "No se descargaron eventos."
  )
}


# ============================================================
# 9. UNIR
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
  "PARTIDOS UNICOS:",
  uniqueN(events$match_id),
  "\n"
)

cat("============================================\n\n")


# ============================================================
# 10. IDENTIFICAR COLUMNAS IMPORTANTES
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
  names(events)
][1]


type_col <- type_candidates[
  type_candidates
  %in%
  names(events)
][1]


player_col <- player_candidates[
  player_candidates
  %in%
  names(events)
][1]


# ============================================================
# 11. COLUMNAS NORMALIZADAS
# ============================================================

if (
  length(team_col) > 0
  &&
  !is.na(team_col)
) {

  events[
    ,
    team_name_clean :=
      as.character(
        get(team_col)
      )
  ]

} else {

  stop(
    "No se encontró la columna de equipo."
  )
}


if (
  length(type_col) > 0
  &&
  !is.na(type_col)
) {

  events[
    ,
    event_type :=
      as.character(
        get(type_col)
      )
  ]

} else {

  stop(
    "No se encontró la columna de tipo de evento."
  )
}


if (
  length(player_col) > 0
  &&
  !is.na(player_col)
) {

  events[
    ,
    player_name_clean :=
      as.character(
        get(player_col)
      )
  ]

} else {

  events[
    ,
    player_name_clean :=
      NA_character_
  ]
}


# ============================================================
# 12. EVENTO DE AMÉRICA
# ============================================================

events[
  ,
  is_america :=
    team_name_clean
    ==
    team_name
]


# ============================================================
# 13. VARIABLES NUMÉRICAS
# ============================================================

if (!"minute" %in% names(events)) {

  stop(
    "No existe la columna minute."
  )
}


events[
  ,
  minute :=
    as.numeric(
      minute
    )
]


if (
  "second"
  %in%
  names(events)
) {

  events[
    ,
    second :=
      as.numeric(
        second
      )
  ]
}


if (
  "period"
  %in%
  names(events)
) {

  events[
    ,
    period :=
      as.numeric(
        period
      )
  ]
}


# ============================================================
# 14. CONTEXTO DE PARTIDO
# ============================================================

match_context <- matches[
  ,
  .(
    match_id,
    match_date,
    opponent,
    venue,
    home_score,
    away_score,
    america_score,
    opponent_score
  )
]


events <- merge(
  events,
  match_context,
  by = "match_id",
  all.x = TRUE
)


events <- events[
  !is.na(opponent)
]


# ============================================================
# 15. PROGRESIONES
# ============================================================

events[
  ,
  pass_progress :=
    pass_end_x
    -
    x
]


events[
  ,
  carry_progress :=
    carry_end_x
    -
    x
]


# ============================================================
# 16. ELIMINAR COLUMNAS TIPO LISTA
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
    "Se eliminarán después de extraer coordenadas:\n"
  )

  print(
    list_columns
  )
}


atomic_columns <- names(events)[
  !vapply(
    events,
    is.list,
    logical(1)
  )
]


events_clean <- events[
  ,
  ..atomic_columns
]


# ============================================================
# 17. ORDEN
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
# 18. VALIDAR COORDENADAS
# ============================================================

cat("\n")
cat("COORDENADAS EXTRAIDAS:\n\n")


cat(
  "Eventos con x/y:",
  events_clean[
    !is.na(x)
    &
    !is.na(y),
    .N
  ],
  "\n"
)


cat(
  "Pases con destino:",
  events_clean[
    !is.na(pass_end_x)
    &
    !is.na(pass_end_y),
    .N
  ],
  "\n"
)


cat(
  "Conducciones con destino:",
  events_clean[
    !is.na(carry_end_x)
    &
    !is.na(carry_end_y),
    .N
  ],
  "\n"
)


cat(
  "Tiros con destino:",
  events_clean[
    !is.na(shot_end_x),
    .N
  ],
  "\n\n"
)


# ============================================================
# 19. EXPORTAR
# ============================================================

fwrite(
  events_clean,
  events_clean_file
)


# ============================================================
# 20. VALIDACIÓN FINAL
# ============================================================

cat("\n")
cat("============================================\n")
cat("VALIDACION FINAL\n")
cat("============================================\n\n")


cat(
  "Eventos:",
  nrow(events_clean),
  "\n"
)


cat(
  "Partidos:",
  uniqueN(
    events_clean$match_id
  ),
  "\n"
)


cat(
  "Eventos America W:",
  events_clean[
    is_america == TRUE,
    .N
  ],
  "\n"
)


cat(
  "Eventos rivales:",
  events_clean[
    is_america == FALSE,
    .N
  ],
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
  "\nTipos de eventos principales:\n"
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


cat("\n")
cat("============================================\n")
cat("TODO CORRECTO.\n")
cat("============================================\n\n")


cat(
  "Archivo creado:\n",
  events_clean_file,
  "\n"
)