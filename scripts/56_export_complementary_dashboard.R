# ============================================================
# COACHLENS
# EXPORTAR ANALISIS COMPLEMENTARIO
#
# Genera:
# - complementary_substitutions.csv
# - complementary_set_pieces.csv
# - complementary_set_piece_summary.csv
# ============================================================

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

suppressPackageStartupMessages(
  library(data.table)
)


cat("\n")
cat("============================================================\n")
cat(" COACHLENS - ANALISIS COMPLEMENTARIO\n")
cat("============================================================\n\n")


# ============================================================
# RAIZ DEL PROYECTO
# ============================================================

args <- commandArgs(
  trailingOnly = FALSE
)

file_arg <- grep(
  "^--file=",
  args,
  value = TRUE
)

script_path <- normalizePath(
  sub(
    "^--file=",
    "",
    file_arg[1]
  )
)

project_root <- normalizePath(
  file.path(
    dirname(script_path),
    ".."
  )
)

cat(
  "Proyecto:",
  project_root,
  "\n\n"
)


# ============================================================
# RUTAS
# ============================================================

male_events_path <- file.path(
  project_root,
  "data",
  "processed",
  "america_male",
  "events_america_clean.csv"
)

female_events_path <- file.path(
  project_root,
  "data",
  "processed",
  "america_female",
  "events_america_female_clean.csv"
)

dashboard_dir <- file.path(
  project_root,
  "dashboard",
  "data"
)

dir.create(
  dashboard_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# LEER SOLO COLUMNAS NECESARIAS
# ============================================================

read_events <- function(path) {

  if (!file.exists(path)) {
    stop(
      paste(
        "No existe:",
        path
      )
    )
  }

  headers <- names(
    fread(
      path,
      nrows = 0,
      showProgress = FALSE
    )
  )

  wanted <- c(
    "match_id",
    "minute",
    "second",
    "team.name",
    "player.name",
    "type.name",
    "play_pattern.name",
    "pass.type.name",
    "shot.type.name",
    "shot.outcome.name",
    "substitution.replacement.name",
    "substitution.outcome.name"
  )

  available <- intersect(
    wanted,
    headers
  )

  cat(
    "Leyendo:",
    basename(path),
    "\n"
  )

  cat(
    "Columnas utilizadas:",
    length(available),
    "\n"
  )

  data <- fread(
    path,
    select = available,
    showProgress = FALSE
  )

  if ("minute" %in% names(data)) {

    data[
      ,
      minute := suppressWarnings(
        as.numeric(minute)
      )
    ]
  }

  return(data)
}


# ============================================================
# CONTAR EVENTOS EN UNA VENTANA
# ============================================================

window_metrics <- function(
  events,
  selected_match,
  selected_team,
  start_minute,
  end_minute
) {

  window <- events[
    as.character(match_id) == selected_match &
    team.name == selected_team &
    minute >= start_minute &
    minute < end_minute
  ]

  if (nrow(window) == 0) {

    return(
      list(
        shots = 0L,
        pressures = 0L,
        recoveries = 0L,
        passes = 0L
      )
    )
  }

  return(
    list(

      shots = sum(
        window$type.name == "Shot",
        na.rm = TRUE
      ),

      pressures = sum(
        window$type.name == "Pressure",
        na.rm = TRUE
      ),

      recoveries = sum(
        window$type.name == "Ball Recovery",
        na.rm = TRUE
      ),

      passes = sum(
        window$type.name == "Pass",
        na.rm = TRUE
      )
    )
  )
}


# ============================================================
# SUSTITUCIONES
# ============================================================

build_substitutions <- function(
  events,
  team_name,
  squad_name
) {

  cat(
    "\nProcesando sustituciones:",
    squad_name,
    "\n"
  )

  required <- c(
    "match_id",
    "minute",
    "team.name",
    "player.name",
    "type.name",
    "substitution.replacement.name"
  )

  missing <- setdiff(
    required,
    names(events)
  )

  if (length(missing) > 0) {

    stop(
      paste(
        "Faltan columnas para sustituciones:",
        paste(
          missing,
          collapse = ", "
        )
      )
    )
  }

  substitutions <- events[
    type.name == "Substitution" &
    team.name == team_name
  ]

  cat(
    "Sustituciones:",
    nrow(substitutions),
    "\n"
  )

  if (nrow(substitutions) == 0) {
    return(data.table())
  }

  rows <- vector(
    "list",
    nrow(substitutions)
  )

  for (i in seq_len(nrow(substitutions))) {

    selected_match <- as.character(
      substitutions$match_id[i]
    )

    selected_minute <- suppressWarnings(
      as.numeric(
        substitutions$minute[i]
      )
    )

    player_out <- as.character(
      substitutions$player.name[i]
    )

    player_in <- as.character(
      substitutions$`substitution.replacement.name`[i]
    )

    if (is.na(selected_minute)) {
      next
    }

    before_start <- max(
      0,
      selected_minute - 10
    )

    before_end <- selected_minute

    after_start <- selected_minute

    after_end <- min(
      120,
      selected_minute + 10
    )


    before <- window_metrics(
      events = events,
      selected_match = selected_match,
      selected_team = team_name,
      start_minute = before_start,
      end_minute = before_end
    )


    after <- window_metrics(
      events = events,
      selected_match = selected_match,
      selected_team = team_name,
      start_minute = after_start,
      end_minute = after_end
    )


    rows[[i]] <- data.table(

      squad = squad_name,

      match_id = selected_match,

      minute = selected_minute,

      player_out = player_out,

      player_in = player_in,

      shots_before = before$shots,

      shots_after = after$shots,

      shots_change =
        after$shots -
        before$shots,

      pressures_before =
        before$pressures,

      pressures_after =
        after$pressures,

      pressures_change =
        after$pressures -
        before$pressures,

      recoveries_before =
        before$recoveries,

      recoveries_after =
        after$recoveries,

      recoveries_change =
        after$recoveries -
        before$recoveries,

      passes_before =
        before$passes,

      passes_after =
        after$passes,

      passes_change =
        after$passes -
        before$passes
    )
  }

  result <- rbindlist(
    rows,
    fill = TRUE
  )

  return(result)
}


# ============================================================
# BALON PARADO
# ============================================================

build_set_pieces <- function(
  events,
  team_name,
  squad_name
) {

  cat(
    "\nProcesando balón parado:",
    squad_name,
    "\n"
  )


  # ----------------------------------------------------------
  # EJECUCIONES
  # ----------------------------------------------------------

  if (
    "pass.type.name" %in% names(events)
  ) {

    executions <- events[
      pass.type.name %in%
        c(
          "Corner",
          "Free Kick"
        )
    ]

    executions[
      ,
      set_piece := ifelse(
        pass.type.name == "Corner",
        "CÓRNER",
        "TIRO LIBRE"
      )
    ]

  } else {

    executions <- data.table(
      match_id = character(),
      team.name = character(),
      set_piece = character()
    )
  }


  # ----------------------------------------------------------
  # TIROS DERIVADOS
  # ----------------------------------------------------------

  if (
    "play_pattern.name" %in% names(events) &&
    "type.name" %in% names(events)
  ) {

    shots <- events[
      type.name == "Shot" &
      play_pattern.name %in%
        c(
          "From Corner",
          "From Free Kick"
        )
    ]

    shots[
      ,
      set_piece := ifelse(
        play_pattern.name == "From Corner",
        "CÓRNER",
        "TIRO LIBRE"
      )
    ]

  } else {

    shots <- data.table(
      match_id = character(),
      team.name = character(),
      set_piece = character()
    )
  }


  # ----------------------------------------------------------
  # RESUMIR EJECUCIONES
  # ----------------------------------------------------------

  if (nrow(executions) > 0) {

    execution_summary <- executions[
      ,
      .(
        executions = .N
      ),
      by = .(
        match_id,
        team.name,
        set_piece
      )
    ]

  } else {

    execution_summary <- data.table(
      match_id = character(),
      team.name = character(),
      set_piece = character(),
      executions = integer()
    )
  }


  # ----------------------------------------------------------
  # RESUMIR TIROS Y GOLES
  # ----------------------------------------------------------

  if (nrow(shots) > 0) {

    if (
      "shot.outcome.name" %in%
      names(shots)
    ) {

      shots[
        ,
        is_goal :=
          shot.outcome.name == "Goal"
      ]

    } else {

      shots[
        ,
        is_goal := FALSE
      ]
    }


    shot_summary <- shots[
      ,
      .(
        shots = .N,

        goals = sum(
          is_goal,
          na.rm = TRUE
        )
      ),
      by = .(
        match_id,
        team.name,
        set_piece
      )
    ]

  } else {

    shot_summary <- data.table(
      match_id = character(),
      team.name = character(),
      set_piece = character(),
      shots = integer(),
      goals = integer()
    )
  }


  # ----------------------------------------------------------
  # UNIR
  # ----------------------------------------------------------

  result <- merge(
    execution_summary,
    shot_summary,
    by = c(
      "match_id",
      "team.name",
      "set_piece"
    ),
    all = TRUE
  )


  result[
    is.na(executions),
    executions := 0L
  ]

  result[
    is.na(shots),
    shots := 0L
  ]

  result[
    is.na(goals),
    goals := 0L
  ]


  # ----------------------------------------------------------
  # OFENSIVO VS DEFENSIVO
  # ----------------------------------------------------------

  result[
    ,
    phase := ifelse(
      team.name == team_name,
      "OFENSIVO",
      "DEFENSIVO"
    )
  ]


  result[
    ,
    squad := squad_name
  ]


  # ----------------------------------------------------------
  # TASAS
  # ----------------------------------------------------------

  result[
    ,
    shot_rate := ifelse(
      executions > 0,
      shots / executions,
      NA_real_
    )
  ]


  result[
    ,
    goal_rate := ifelse(
      executions > 0,
      goals / executions,
      NA_real_
    )
  ]


  result[
    ,
    match_id := as.character(
      match_id
    )
  ]


  return(result)
}


# ============================================================
# VARONIL
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" VARONIL\n")
cat("============================================================\n")


male_events <- read_events(
  male_events_path
)


male_substitutions <- build_substitutions(
  events = male_events,
  team_name = "América",
  squad_name = "Varonil"
)


male_set_pieces <- build_set_pieces(
  events = male_events,
  team_name = "América",
  squad_name = "Varonil"
)


rm(male_events)

gc()


# ============================================================
# FEMENIL
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" FEMENIL\n")
cat("============================================================\n")


female_events <- read_events(
  female_events_path
)


female_substitutions <- build_substitutions(
  events = female_events,
  team_name = "América W",
  squad_name = "Femenil"
)


female_set_pieces <- build_set_pieces(
  events = female_events,
  team_name = "América W",
  squad_name = "Femenil"
)


rm(female_events)

gc()


# ============================================================
# COMBINAR
# ============================================================

all_substitutions <- rbindlist(
  list(
    male_substitutions,
    female_substitutions
  ),
  fill = TRUE
)


all_set_pieces <- rbindlist(
  list(
    male_set_pieces,
    female_set_pieces
  ),
  fill = TRUE
)


# ============================================================
# RESUMEN GLOBAL DE BALON PARADO
# ============================================================

set_piece_summary <- all_set_pieces[
  ,
  .(
    executions = sum(
      executions,
      na.rm = TRUE
    ),

    shots = sum(
      shots,
      na.rm = TRUE
    ),

    goals = sum(
      goals,
      na.rm = TRUE
    )
  ),
  by = .(
    squad,
    phase,
    set_piece
  )
]


set_piece_summary[
  ,
  shot_rate := ifelse(
    executions > 0,
    shots / executions,
    NA_real_
  )
]


set_piece_summary[
  ,
  goal_rate := ifelse(
    executions > 0,
    goals / executions,
    NA_real_
  )
]


# ============================================================
# EXPORTAR
# ============================================================

substitution_path <- file.path(
  dashboard_dir,
  "complementary_substitutions.csv"
)

set_piece_path <- file.path(
  dashboard_dir,
  "complementary_set_pieces.csv"
)

summary_path <- file.path(
  dashboard_dir,
  "complementary_set_piece_summary.csv"
)


fwrite(
  all_substitutions,
  substitution_path
)


fwrite(
  all_set_pieces,
  set_piece_path
)


fwrite(
  set_piece_summary,
  summary_path
)


# ============================================================
# VALIDACION
# ============================================================

cat("\n")
cat("============================================================\n")
cat(" RESULTADOS\n")
cat("============================================================\n")


cat(
  "Sustituciones exportadas:",
  nrow(all_substitutions),
  "\n"
)


cat(
  "Registros de balón parado:",
  nrow(all_set_pieces),
  "\n"
)


cat(
  "Filas resumen:",
  nrow(set_piece_summary),
  "\n\n"
)


cat(
  "RESUMEN BALON PARADO:\n"
)

print(
  set_piece_summary
)


cat(
  "\nPRIMERAS SUSTITUCIONES:\n"
)

print(
  head(
    all_substitutions,
    10
  )
)


cat("\nArchivos creados:\n")

cat(
  substitution_path,
  "\n"
)

cat(
  set_piece_path,
  "\n"
)

cat(
  summary_path,
  "\n"
)


cat("\n")
cat("============================================================\n")
cat(" TODO CORRECTO\n")
cat("============================================================\n\n")