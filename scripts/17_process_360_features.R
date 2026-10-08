library(jsonlite)
library(data.table)

# ============================================================
# 1. RUTA DE ARCHIVOS 360
# ============================================================

ruta_360 <- "data/raw/america_male/frames_360"

archivos <- list.files(
  ruta_360,
  pattern = "\\.json$",
  full.names = TRUE
)

cat("Archivos 360 encontrados:", length(archivos), "\n")


# ============================================================
# 2. FUNCIÓN DE DISTANCIA
# ============================================================

distancia <- function(x1, y1, x2, y2) {
  sqrt((x1 - x2)^2 + (y1 - y2)^2)
}


# ============================================================
# 3. LISTA DE RESULTADOS
# ============================================================

features_360_list <- list()
contador <- 1


# ============================================================
# 4. PROCESAR ARCHIVO POR ARCHIVO
# ============================================================

for (i in seq_along(archivos)) {

  archivo <- archivos[i]

  match_id <- as.integer(
    gsub(
      "360_|\\.json",
      "",
      basename(archivo)
    )
  )

  cat(
    "[", i, "/", length(archivos), "]",
    "Procesando match:",
    match_id,
    "\n"
  )

  tryCatch({

    datos <- fromJSON(
      archivo,
      simplifyVector = FALSE
    )

    if (length(datos) == 0) {
      next
    }

    for (frame in datos) {

      event_uuid <- frame$event_uuid
      freeze <- frame$freeze_frame

      if (
        is.null(freeze) ||
        length(freeze) == 0
      ) {
        next
      }

      # ======================================================
      # IDENTIFICAR AL ACTOR
      # ======================================================

      actor_idx <- which(
        sapply(
          freeze,
          function(x) {
            !is.null(x$actor) &&
              isTRUE(x$actor)
          }
        )
      )

      if (length(actor_idx) == 0) {
        next
      }

      actor <- freeze[[actor_idx[1]]]

      if (
        is.null(actor$location) ||
        length(actor$location) < 2
      ) {
        next
      }

      actor_x <- as.numeric(actor$location[[1]])
      actor_y <- as.numeric(actor$location[[2]])

      # ======================================================
      # CONTADORES
      # ======================================================

      visible_teammates <- 0
      visible_opponents <- 0

      teammates_within_5 <- 0
      teammates_within_10 <- 0

      opponents_within_5 <- 0
      opponents_within_10 <- 0

      teammate_distances <- c()
      opponent_distances <- c()

      # ======================================================
      # RECORRER JUGADORES VISIBLES
      # ======================================================

      for (player in freeze) {

        if (isTRUE(player$actor)) {
          next
        }

        if (
          is.null(player$location) ||
          length(player$location) < 2
        ) {
          next
        }

        px <- as.numeric(player$location[[1]])
        py <- as.numeric(player$location[[2]])

        d <- distancia(
          actor_x,
          actor_y,
          px,
          py
        )

        # ====================================================
        # COMPAÑERO
        # ====================================================

        if (isTRUE(player$teammate)) {

          visible_teammates <- visible_teammates + 1

          teammate_distances <- c(
            teammate_distances,
            d
          )

          if (d <= 5) {
            teammates_within_5 <- teammates_within_5 + 1
          }

          if (d <= 10) {
            teammates_within_10 <- teammates_within_10 + 1
          }

        } else {

          # ==================================================
          # RIVAL
          # ==================================================

          visible_opponents <- visible_opponents + 1

          opponent_distances <- c(
            opponent_distances,
            d
          )

          if (d <= 5) {
            opponents_within_5 <- opponents_within_5 + 1
          }

          if (d <= 10) {
            opponents_within_10 <- opponents_within_10 + 1
          }
        }
      }


      # ======================================================
      # MÉTRICAS DE DISTANCIA
      # ======================================================

      avg_teammate_distance <- if (
        length(teammate_distances) > 0
      ) {
        mean(teammate_distances)
      } else {
        NA_real_
      }

      avg_opponent_distance <- if (
        length(opponent_distances) > 0
      ) {
        mean(opponent_distances)
      } else {
        NA_real_
      }

      nearest_teammate_distance <- if (
        length(teammate_distances) > 0
      ) {
        min(teammate_distances)
      } else {
        NA_real_
      }

      nearest_opponent_distance <- if (
        length(opponent_distances) > 0
      ) {
        min(opponent_distances)
      } else {
        NA_real_
      }


      # ======================================================
      # GUARDAR FILA
      # ======================================================

      features_360_list[[contador]] <- data.table(

        match_id = match_id,

        event_uuid = event_uuid,

        actor_x = actor_x,
        actor_y = actor_y,

        visible_teammates = visible_teammates,
        visible_opponents = visible_opponents,

        teammates_within_5 = teammates_within_5,
        opponents_within_5 = opponents_within_5,

        teammates_within_10 = teammates_within_10,
        opponents_within_10 = opponents_within_10,

        local_superiority_5 =
          teammates_within_5 -
          opponents_within_5,

        local_superiority_10 =
          teammates_within_10 -
          opponents_within_10,

        avg_teammate_distance =
          avg_teammate_distance,

        avg_opponent_distance =
          avg_opponent_distance,

        nearest_teammate_distance =
          nearest_teammate_distance,

        nearest_opponent_distance =
          nearest_opponent_distance
      )

      contador <- contador + 1
    }

  },

  error = function(e) {

    cat(
      "ERROR match:",
      match_id,
      "-",
      e$message,
      "\n"
    )

  })
}


# ============================================================
# 5. VALIDAR QUE SÍ HAYA DATOS
# ============================================================

if (length(features_360_list) == 0) {

  stop(
    "No se pudieron extraer features 360."
  )
}


# ============================================================
# 6. UNIR
# ============================================================

features_360 <- rbindlist(
  features_360_list,
  fill = TRUE,
  use.names = TRUE
)


# ============================================================
# 7. RESUMEN
# ============================================================

cat("\n=====================================\n")
cat("360 FEATURES CREADAS\n")
cat("=====================================\n")

cat(
  "Filas:",
  nrow(features_360),
  "\n"
)

cat(
  "Columnas:",
  ncol(features_360),
  "\n"
)

cat(
  "Partidos:",
  uniqueN(features_360$match_id),
  "\n"
)


# ============================================================
# 8. GUARDAR
# ============================================================

fwrite(
  features_360,
  "data/processed/america_male/features_360.csv",
  bom = TRUE
)

cat(
  "\nfeatures_360.csv guardado correctamente\n"
)