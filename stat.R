setwd("/Users/janoskonig/Documents/shadematch/kankor")

data <- read.csv("patients.csv", sep=',', stringsAsFactors = FALSE)


# Variable selection and derived variables: -----

#denture type
#chewing_today_situation

#init_mai_huedegree
#final_mai_huedegree
#OHIP pontszámok összesítése —> új változó: OHIP_sum
#GOHAI pontszámok összesítése —> új változó: GOHAI_sum
#F1 mm
#F2 + F1_ívhossz_mm (L) —> új változó: új derived F2 / L^3 (dimenziónélküli)
#F3 mm
#F4 fok (szög)
#F5 faktor
#F6 fok (szög)
#F7 faktor
#F9 faktor
#A1_Kaan faktor
#A2_mag_mm
#A3_jobb faktor
#A3_bal faktor
#A4_jobb faktor
#A4_bal faktor
#A5_jobb faktor
#A5_bal faktor
#A6_jobb faktor
#A6_bal faktor
#A7_jobb faktor
#A7_bal faktor
#A8_jobb faktor
#A8_bal faktor
#A9_jobb faktor
#A9_bal faktor
#A10 fok (szög)
#A11
#A12
#A13


# ============================================================
# Segédfüggvények ---------------------------------------------
# ============================================================

# Néhány numerikus mező MongoDB-export formában érkezhet, pl.
# {"$decimal":"10.17"}. Ez a függvény ezt is és a sima számokat is
# numerikussá alakítja, az üres / hiányzó értékeket NA-ra állítja.
parse_decimal <- function(x) {
  x <- as.character(x)
  x[x == "" | x == "NA" | is.na(x)] <- NA
  out <- suppressWarnings(as.numeric(x))           # sima numerikus értékek
  is_json <- !is.na(x) & grepl("\\$decimal", x)    # {"$decimal":"..."} formátum
  out[is_json] <- suppressWarnings(as.numeric(
    sub('.*"\\$decimal"\\s*:\\s*"([0-9eE.+-]+)".*', "\\1", x[is_json])
  ))
  out
}

# Ordinális (5-fokú) magyar skála -> 1..5 numerikus érték.
# Nagyon rossz < Rossz < Átlagos < Jó < Kiváló
ordinal_5 <- function(x) {
  lvl <- c("Nagyon rossz" = 1, "Rossz" = 2, "Átlagos" = 3, "Jó" = 4, "Kiváló" = 5)
  x <- trimws(as.character(x))
  out <- unname(lvl[x])
  out
}

# Biztonságos sorösszeg: ha a sorban BÁRMELYIK tétel hiányzik, az
# összeg NA (a kérdőív-pontszámoknál ez a kívánt viselkedés).
row_sum_strict <- function(df) {
  m <- as.matrix(sapply(df, function(z) suppressWarnings(as.numeric(z))))
  apply(m, 1, function(r) if (any(is.na(r))) NA_real_ else sum(r))
}


# ============================================================
# Derived változók --------------------------------------------
# ============================================================

# OHIP_sum: az 5 OHIP tétel összege
ohip_items  <- paste0("OHIP_", 1:5)
data$OHIP_sum <- row_sum_strict(data[, ohip_items, drop = FALSE])

# GOHAI_sum: a 12 GOHAI tétel összege
gohai_items <- paste0("GOHAI_", 1:12)
data$GOHAI_sum <- row_sum_strict(data[, gohai_items, drop = FALSE])

# A $decimal formátumú mezők parse-olása
data$F1_mm        <- parse_decimal(data$F1)             # F1 hossz (mm)
data$F1_ivhossz   <- parse_decimal(data$F1_ivhossz_mm)  # L (ívhossz, mm)
data$A2_mag       <- parse_decimal(data$A2_mag_mm)       # A2 magasság (mm)

# F2 / L^3  — dimenziónélküli derived változó (L = F1_ívhossz_mm)
data$F2_num   <- parse_decimal(data$F2)
data$F2_per_L3 <- data$F2_num / (data$F1_ivhossz^3)

# Ordinális szöveges skála -> numerikus
data$chewing_num        <- ordinal_5(data$chewing_today_situation)
data$responsiveness_num <- ordinal_5(data$responsiveness_today_situation)

# denture_type: nominális (both / lower / upper). Dummy-kódolás
# (referenciaszint = "both"), hogy bekerülhessen a numerikus blokkba.
data$denture_type <- factor(data$denture_type, levels = c("both", "lower", "upper"))
data$denture_lower <- as.integer(data$denture_type == "lower")
data$denture_upper <- as.integer(data$denture_type == "upper")

# A "faktor" jelölésű ordinális pontszámokat numerikusként kezeljük.
factor_as_numeric <- c("F5", "F7", "F9", "A1_Kaan", "A11", "A12", "A13")
for (v in factor_as_numeric) data[[v]] <- suppressWarnings(as.numeric(data[[v]]))

# Szög / mm jellegű folytonos morfológiai változók (parse a $decimal miatt).
# Ezek a jelenlegi mintában ritkán mértek; lásd a Y blokk megjegyzését.
cont_morph <- c("F3", "F4", "F6", "A10")
for (v in cont_morph) data[[v]] <- parse_decimal(data[[v]])

# Bilaterális (jobb/bal) párok összevonása egy-egy bilaterális átlaggá.
# A3–A9 jobb + bal -> A3_mean ... A9_mean (14 változó -> 7).
# Ha az egyik oldal hiányzik, a meglévő oldalt vesszük; ha mindkettő NA -> NA.
bilateral_mean <- function(left, right) {
  m <- cbind(suppressWarnings(as.numeric(left)),
             suppressWarnings(as.numeric(right)))
  out <- rowMeans(m, na.rm = TRUE)
  out[is.nan(out)] <- NA_real_
  out
}
for (a in paste0("A", 3:9)) {
  data[[paste0(a, "_mean")]] <- bilateral_mean(data[[paste0(a, "_jobb")]],
                                               data[[paste0(a, "_bal")]])
}


# ============================================================
# Elemzési változócsoportok -----------------------------------
# ============================================================

# X blokk – beteg-orientált (szubjektív) kimenetek
# Csak a "both" (teljes felső+alsó) protézisű betegekre szűkítünk, ezért a
# denture_type a blokkból kimarad (konstans lenne).
X_vars <- c(
  "chewing_num",
  "init_mai_huedegree",
  "OHIP_sum", "GOHAI_sum"
)

# Y blokk – a legfontosabb morfológiai változók (reduktált).
# Stratégia: bilaterális párok átlaga + a jól (≈40 betegnél) mért változók.
# Kihagyva a ritkán mért folytonos F-mérések (F1_mm, F2_per_L3, F3, F4, F6),
# hogy a kanonikus korrelációhoz minél több teljes eset (n) maradjon.
Y_vars <- c(
  "F5", "F7",                                  # jól mért faktorok
  "A1_Kaan",                                   # Kaan-index
  "A3_mean", "A4_mean", "A5_mean", "A6_mean",  # bilaterális átlagok
  "A7_mean", "A8_mean", "A9_mean",
  "A11", "A12", "A13"
)

# Opcionális, klinikailag releváns, de a jelenlegi mintában ritkán mért
# változók (A2_mag ≈5, A10 ≈6 beteg). Erősen csökkentik a teljes esetek
# számát; ha a valós adatban jól kitöltöttek, told be őket a Y blokkba:
# Y_vars <- c(Y_vars, "A2_mag", "A10")

# init/final_mai_huedegree már numerikus a CSV-ben, de biztos, ami biztos:
data$init_mai_huedegree  <- suppressWarnings(as.numeric(data$init_mai_huedegree))
data$final_mai_huedegree <- suppressWarnings(as.numeric(data$final_mai_huedegree))

# Csak a "both" protézisű betegek
data <- data[!is.na(data$denture_type) & data$denture_type == "both", ]
cat("\n'both' protézisű betegek száma:", nrow(data), "\n")

analysis <- data[, c(X_vars, Y_vars)]


# ============================================================
# Adatminőség / hiányzó értékek áttekintése -------------------
# ============================================================

cat("\n--- Elemszám változónként (nem-hiányzó) ---\n")
print(sort(colSums(!is.na(analysis))))

# Konstans (zéró-varianciájú) változók kiszűrése – CCA-ban használhatatlanok
non_constant <- sapply(analysis, function(z) {
  v <- z[!is.na(z)]
  length(unique(v)) > 1
})
dropped_const <- names(non_constant)[!non_constant]
if (length(dropped_const)) {
  cat("\nKonstans / üres változók kihagyva:\n  ",
      paste(dropped_const, collapse = ", "), "\n")
}
X_use <- intersect(X_vars, names(non_constant)[non_constant])
Y_use <- intersect(Y_vars, names(non_constant)[non_constant])

# Csak a teljes esetek (complete cases) használhatók a kanonikus korrelációhoz
cc <- complete.cases(analysis[, c(X_use, Y_use)])
cat("\nTeljes esetek száma (mindkét blokk hiánytalan):", sum(cc), "\n")


# ============================================================
# Kanonikus korreláció ----------------------------------------
# ============================================================
# FIGYELEM: a kanonikus korreláció megbízhatóan csak akkor fut, ha a
# teljes esetek száma (n) jóval nagyobb, mint a két blokk változóinak
# száma (p_x + p_y). Kevés betegnél ez ritkán teljesül -> a kód ezt
# ellenőrzi, és ha nincs elég adat, érdemi fallbackot ad.

run_cca <- function(df, xv, yv) {
  X <- as.matrix(df[, xv, drop = FALSE])
  Y <- as.matrix(df[, yv, drop = FALSE])
  n <- nrow(X); px <- ncol(X); py <- ncol(Y)
  cat(sprintf("\nCCA: n=%d, X változók=%d, Y változók=%d\n", n, px, py))

  if (n <= max(px, py) + 1) {
    cat("** Nincs elég megfigyelés a kanonikus korrelációhoz",
        "(n <= max(p_x, p_y)). **\n",
        "Lehetséges teendők:\n",
        "  - változók további összevonása / dimenziócsökkentés (PCA),\n",
        "  - blokkonként kevesebb, elméletileg fontos változó,\n",
        "  - regularizált CCA (pl. CCA::rcc vagy PMA::CCA).\n")
    # Fallback: páronkénti X–Y korrelációs mátrix (ami az adatból kijön)
    cat("\nFallback – X és Y változók közötti páronkénti korrelációk:\n")
    print(round(cor(X, Y, use = "pairwise.complete.obs"), 3))
    return(invisible(NULL))
  }

  cc_res <- cancor(X, Y)
  cat("\nKanonikus korrelációk:\n")
  print(round(cc_res$cor, 4))
  cat("\nX együtthatók (xcoef):\n"); print(round(cc_res$xcoef, 4))
  cat("\nY együtthatók (ycoef):\n"); print(round(cc_res$ycoef, 4))
  invisible(cc_res)
}

if (sum(cc) > 0 && length(X_use) > 0 && length(Y_use) > 0) {
  cca_result <- run_cca(analysis[cc, ], X_use, Y_use)
} else {
  cat("\nNincs egyetlen teljes eset sem mindkét blokkra – a CCA nem futtatható.\n")
}


# ============================================================
# Hipotézisgeneráló blokk-elemzés -----------------------------
#   1) RV-együttható mátrix: mely változócsoport korrelál
#      legjobban a többivel (0..1 asszociáció, p-értékkel).
#   2) Sparse CCA a legerősebb blokkpárra: mely konkrét
#      változók hajtják a kapcsolatot (változószelekcióval).
# ============================================================
suppressPackageStartupMessages({
  library(FactoMineR)
  library(PMA)
})

# Értelmes változócsoportok. Az item-szintű kérdőívek multivariáns
# blokként informatívabbak, mint az összegük.
# OHIP és GOHAI egyetlen QoL (életminőség) blokkba vonva – mindkettő
# ugyanazt a konstruktumot méri, így külön blokkként redundáns lenne.
blocks <- list(
  QoL     = c(paste0("OHIP_", 1:5), paste0("GOHAI_", 1:12)),
  Funkcio = c("chewing_num", "responsiveness_num"),
  Szin    = "init_mai_huedegree",
  F_morf  = c("F5", "F7"),
  A_morf  = c("A1_Kaan", paste0("A", 3:9, "_mean"), "A11", "A12", "A13")
)

# Blokk-mátrix kinyerése numerikusként (megtartja az oszlopneveket).
get_block <- function(df, vars) {
  m <- vapply(df[, vars, drop = FALSE],
              function(z) suppressWarnings(as.numeric(z)),
              numeric(nrow(df)))
  matrix(m, nrow = nrow(df), dimnames = list(NULL, vars))
}

# Konstans (zéró-varianciájú) oszlopok eldobása az adott eseteken.
drop_constant <- function(m) {
  keep <- apply(m, 2, function(z) {
    z <- z[is.finite(z)]
    length(z) > 0 && sd(z) > 0
  })
  m[, keep, drop = FALSE]
}

# --- 1) RV-együttható mátrix -------------------------------------
bn  <- names(blocks); nb <- length(bn)
RV  <- diag(nb); dimnames(RV) <- list(bn, bn)
RVp <- matrix(NA_real_, nb, nb, dimnames = list(bn, bn))

for (i in 1:(nb - 1)) for (j in (i + 1):nb) {
  Xi <- get_block(data, blocks[[i]])
  Xj <- get_block(data, blocks[[j]])
  keep <- complete.cases(Xi, Xj)                 # páronkénti teljes esetek
  Xi <- drop_constant(Xi[keep, , drop = FALSE])
  Xj <- drop_constant(Xj[keep, , drop = FALSE])
  if (sum(keep) >= 4 && ncol(Xi) >= 1 && ncol(Xj) >= 1) {
    res <- FactoMineR::coeffRV(scale(Xi), scale(Xj))
    RV[i, j]  <- RV[j, i]  <- res$rv
    RVp[i, j] <- RVp[j, i] <- res$p.value
  }
}

cat("\n--- RV-együttható mátrix (blokkok közötti asszociáció, 0..1) ---\n")
print(round(RV, 3))
cat("\n--- RV p-értékek (permutációs) ---\n")
print(round(RVp, 4))

# A legjobban kapcsolódó blokk: legnagyobb átlagos RV a többivel.
mean_rv <- (rowSums(RV, na.rm = TRUE) - 1) / (nb - 1)
cat("\nÁtlagos RV a többi blokkal (rangsor):\n")
print(round(sort(mean_rv, decreasing = TRUE), 3))

# --- 2) Sparse CCA a legerősebb (multivariáns) blokkpárra --------
# A globálisan legerősebb pár tartalmazhat egyváltozós blokkot (pl.
# Szín = init_mai_huedegree); a sparse CCA-hoz mindkét blokknak ≥2
# (nem konstans) változóra van szüksége. Ezért a legerősebb OLYAN
# párt választjuk, ahol mindkét blokk multivariáns.
off <- RV; diag(off) <- NA
if (all(is.na(off))) {
  cat("\nNincs elég adat egyetlen blokkpár RV-jéhez sem – sparse CCA kihagyva.\n")
} else {
  gmax <- which(off == max(off, na.rm = TRUE), arr.ind = TRUE)[1, ]
  cat(sprintf("\nGlobálisan legerősebb blokkpár: %s <-> %s (RV = %.3f)\n",
              bn[gmax[1]], bn[gmax[2]], off[gmax[1], gmax[2]]))

  # Multivariáns blokkpárokra szűkített RV, sparse CCA-hoz
  multivar <- lengths(blocks) >= 2
  eff <- off
  eff[!multivar, ] <- NA; eff[, !multivar] <- NA

  if (all(is.na(eff))) {
    cat("Nincs két multivariáns blokk – sparse CCA kihagyva.\n")
    bi <- bj <- NA
  } else {
    idx <- which(eff == max(eff, na.rm = TRUE), arr.ind = TRUE)[1, ]
    bi <- bn[idx[1]]; bj <- bn[idx[2]]
    if (!(bi %in% bn[gmax] && bj %in% bn[gmax])) {
      cat(sprintf("Sparse CCA a legerősebb multivariáns páron: %s <-> %s (RV = %.3f)\n",
                  bi, bj, off[idx[1], idx[2]]))
    }
  }

  Xi <- if (!is.na(bi)) get_block(data, blocks[[bi]]) else NULL
  Xj <- if (!is.na(bj)) get_block(data, blocks[[bj]]) else NULL
  keep <- if (!is.null(Xi)) complete.cases(Xi, Xj) else logical(0)
  if (!is.null(Xi)) {
    Xi <- drop_constant(Xi[keep, , drop = FALSE])
    Xj <- drop_constant(Xj[keep, , drop = FALSE])
  }

  if (!is.null(Xi) && sum(keep) >= 5 && ncol(Xi) >= 2 && ncol(Xj) >= 2) {
    Xi <- scale(Xi); Xj <- scale(Xj)
    set.seed(1)
    perm <- PMA::CCA.permute(Xi, Xj, typex = "standard", typez = "standard",
                             nperms = 50, trace = FALSE)
    fit <- PMA::CCA(Xi, Xj, typex = "standard", typez = "standard",
                    penaltyx = perm$bestpenaltyx, penaltyz = perm$bestpenaltyz,
                    K = 1, trace = FALSE)
    cat(sprintf("Sparse CCA kanonikus korreláció (1. komponens): %.3f\n",
                fit$cors))
    selx <- colnames(Xi)[fit$u[, 1] != 0]
    selz <- colnames(Xj)[fit$v[, 1] != 0]
    cat(sprintf("  %s blokkból kiválasztva: %s\n", bi,
                if (length(selx)) paste(selx, collapse = ", ") else "(egyik sem)"))
    cat(sprintf("  %s blokkból kiválasztva: %s\n", bj,
                if (length(selz)) paste(selz, collapse = ", ") else "(egyik sem)"))
  } else {
    cat("Túl kevés teljes eset / változó a sparse CCA-hoz ezen a blokkpáron.\n")
  }
}


# ============================================================
# 3) Lasso regresszió: QoL itemek -> init_mai_huedegree -------
# A legerősebb (és szignifikáns) RV-kapcsolat a Szín ↔ QoL, de a
# Szín egyváltozós, ezért itt nem CCA, hanem regularizált (lasso)
# regresszió a megfelelő hipotézisgeneráló eszköz: mely QoL itemek
# jelzik előre a kezdeti színértéket. A lasso a kevés betegnél is
# robusztus, mert változót szelektál (a felesleges együtthatókat 0-ra húzza).
# ============================================================
suppressPackageStartupMessages(library(glmnet))

y_var <- "init_mai_huedegree"
x_vars <- blocks$QoL

Xq <- get_block(data, x_vars)
yq <- suppressWarnings(as.numeric(data[[y_var]]))
keep <- complete.cases(Xq, yq)
Xq <- drop_constant(Xq[keep, , drop = FALSE])
yq <- yq[keep]

cat(sprintf("\n--- Lasso: %s itemek -> %s ---\n", "QoL", y_var))
cat("Teljes esetek:", length(yq), " | prediktorok:", ncol(Xq), "\n")

if (length(yq) >= 5 && ncol(Xq) >= 2) {
  # Kevés betegnél leave-one-out CV a lambda választásához.
  set.seed(1)
  cvfit <- glmnet::cv.glmnet(Xq, yq, alpha = 1, nfolds = length(yq),
                             grouped = FALSE, standardize = TRUE)

  for (lab in c("lambda.1se", "lambda.min")) {
    co <- coef(cvfit, s = cvfit[[lab]])
    nz <- co[co[, 1] != 0, , drop = FALSE]
    sel <- setdiff(rownames(nz), "(Intercept)")
    # Magyarázott variancia az adott lambdánál (CV-n kívüli, in-sample R^2)
    yhat <- as.numeric(predict(cvfit, newx = Xq, s = cvfit[[lab]]))
    r2 <- 1 - sum((yq - yhat)^2) / sum((yq - mean(yq))^2)
    cat(sprintf("\n[%s = %.3f]  kiválasztott itemek (%d):\n", lab,
                cvfit[[lab]], length(sel)))
    if (length(sel)) {
      print(round(nz[sel, , drop = FALSE], 3))
    } else {
      cat("  (egyik sem – csak az intercept marad)\n")
    }
    cat(sprintf("  in-sample R^2 = %.3f\n", r2))
  }
} else {
  cat("Túl kevés teljes eset / prediktor a lasso regresszióhoz.\n")
}
