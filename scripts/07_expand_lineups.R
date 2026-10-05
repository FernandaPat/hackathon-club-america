library(data.table)

# ============================================================
# 1. LEER EVENTS ORIGINAL
# ============================================================

ruta_entrada <- "data/processed/america_male/events_america.csv"

events_america <- fread(
  ruta_entrada,
  encoding = "UTF-8"
)

cat("Filas:", nrow(events_america), "\n")
cat("Columnas:", ncol(events_america), "\n")


# ============================================================
# 2. FUNCIÓN PARA CORREGIR NOMBRES MAL CODIFICADOS
#
# Ejemplos:
# MartÃ­nez   -> Martínez
# JesÃºs      -> Jesús
# SÃ¡nchez    -> Sánchez
# Ãlvaro     -> Álvaro
# FernÃ¡ndez  -> Fernández
# ============================================================

corregir_texto <- function(x) {

  # Trabajar únicamente con texto
  x <- as.character(x)

  # Detectar textos con señales típicas de UTF-8 mal interpretado
  problema <- !is.na(x) & grepl(
    "Ã|Â|â|ð|�",
    x
  )

  # Corregir únicamente los valores afectados
  if (any(problema)) {

    corregido <- iconv(
      x[problema],
      from = "latin1",
      to = "UTF-8"
    )

    # No sustituir por NA si algún texto raro no pudo convertirse
    valido <- !is.na(corregido)

    posiciones <- which(problema)

    x[posiciones[valido]] <- corregido[valido]
  }

  x
}


# ============================================================
# 3. IDENTIFICAR TODAS LAS COLUMNAS DE TEXTO
# ============================================================

columnas_texto <- names(events_america)[
  sapply(events_america, is.character)
]

cat(
  "Columnas de texto encontradas:",
  length(columnas_texto),
  "\n"
)


# ============================================================
# 4. MOSTRAR COLUMNAS QUE TIENEN PROBLEMAS
# ============================================================

columnas_con_error <- columnas_texto[
  sapply(
    events_america[, ..columnas_texto],
    function(x) {
      any(
        grepl(
          "Ã|Â|â|ð|�",
          x,
          useBytes = TRUE
        ),
        na.rm = TRUE
      )
    }
  )
]

cat("\nColumnas con posible problema de codificación:\n")

print(columnas_con_error)


# ============================================================
# 5. CORREGIR TODAS LAS COLUMNAS DE TEXTO
# ============================================================

for (col in columnas_texto) {

  events_america[
    ,
    (col) := corregir_texto(get(col))
  ]

}

cat("\nCorrección de texto terminada.\n")


# ============================================================
# 6. COMPROBAR COLUMNAS RELACIONADAS CON JUGADORES
# ============================================================

columnas_jugador <- grep(
  "player|recipient|replacement",
  names(events_america),
  value = TRUE,
  ignore.case = TRUE
)

cat("\nColumnas relacionadas con jugadores:\n")

print(columnas_jugador)


# ============================================================
# 7. BUSCAR SI TODAVÍA QUEDAN CARACTERES EXTRAÑOS
# ============================================================

problemas_restantes <- sapply(
  events_america[, ..columnas_texto],
  function(x) {
    sum(
      grepl(
        "Ã|Â|â|ð|�",
        x,
        useBytes = TRUE
      ),
      na.rm = TRUE
    )
  }
)

problemas_restantes <- problemas_restantes[
  problemas_restantes > 0
]

cat("\nProblemas restantes:\n")

print(problemas_restantes)


# ============================================================
# 8. GUARDAR NUEVA VERSIÓN LIMPIA
#
# IMPORTANTE:
# No sobrescribimos el archivo original.
# ============================================================

ruta_salida <- paste0(
  "data/processed/america_male/",
  "events_america_clean.csv"
)

fwrite(
  events_america,
  ruta_salida,
  bom = TRUE
)

cat(
  "\nARCHIVO GUARDADO:\n",
  ruta_salida,
  "\n"
)


# ============================================================
# 9. VERIFICACIÓN FINAL
# ============================================================

cat("\nVerificación terminada.\n")

cat(
  "Filas guardadas:",
  nrow(events_america),
  "\n"
)

cat(
  "Columnas guardadas:",
  ncol(events_america),
  "\n"
)

grep(
  "player",
  names(events_america),
  value = TRUE,
  ignore.case = TRUE
)