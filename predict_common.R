# ============================================================
# predict_common.R
# Közös adat-előkészítés és segédfüggvények a PREDICT
# keresztmetszeti hipotézisvizsgálatokhoz (H1–H3).
#
# Forrás: kutatasterv_PREDICT_keresztmetszeti.xlsx (hipotézisek),
#         patients.csv (adat), templates/questionnaire1.html (kódkönyv).
#
# Használat a hipotézis-szkriptek elején:
#   source("predict_common.R")
# A szkripteket a repó gyökeréből futtasd (ott van a patients.csv).
#
# Adatvédelmi megjegyzés: a szkriptek a beteg-azonosítókat (név, TAJ,
# képútvonalak) beolvasás után azonnal eldobják; kimenetükben beteg
# csak sor-indexszel jelenhet meg.
# ============================================================

DATA_FILE <- "patients.csv"
OUT_DIR   <- "stat_output"
SEED      <- 20260807
N_PERM    <- 10000   # permutációs ismétlésszám (elsődleges tesztek)
N_BOOT    <- 5000    # bootstrap ismétlésszám

dir.create(OUT_DIR, showWarnings = FALSE)

# ------------------------------------------------------------
# Segédfüggvények (parse) -------------------------------------
# ------------------------------------------------------------

# MongoDB-exportból jövő {"$decimal":"10.17"} formátum és sima számok
# egységes numerikussá alakítása; üres / hiányzó -> NA.
parse_decimal <- function(x) {
  x <- as.character(x)
  x[x == "" | x == "NA" | is.na(x)] <- NA
  out <- suppressWarnings(as.numeric(x))
  is_json <- !is.na(x) & grepl("\\$decimal", x)
  out[is_json] <- suppressWarnings(as.numeric(
    sub('.*"\\$decimal"\\s*:\\s*"([0-9eE.+-]+)".*', "\\1", x[is_json])
  ))
  out
}

# Szigorú sorösszeg: ha a sor bármelyik tétele hiányzik, az összeg NA
# (kérdőív-összpontszámoknál ez a kívánt viselkedés).
row_sum_strict <- function(df) {
  m <- as.matrix(sapply(df, function(z) suppressWarnings(as.numeric(z))))
  apply(m, 1, function(r) if (any(is.na(r))) NA_real_ else sum(r))
}

# ------------------------------------------------------------
# Adat beolvasása és azonosítók eldobása ----------------------
# ------------------------------------------------------------

dat <- read.csv(DATA_FILE, stringsAsFactors = FALSE)
stopifnot("A patients.csv beolvasása hiányos" = nrow(dat) > 10)

id_cols <- c("TAJ", "paciens_neve", "data_uploader",
             grep("image_path|_gerincelvonal|athajlas", names(dat), value = TRUE))
dat <- dat[, setdiff(names(dat), id_cols)]

# ------------------------------------------------------------
# Származtatott változók --------------------------------------
# ------------------------------------------------------------

# Életkor a felvétel időpontjában. A nyilvánvalóan hibás születési
# dátumokból számolt életkort (>105 vagy <18 év) NA-ra állítjuk.
rec  <- as.Date(dat$record_datetime)
born <- as.Date(dat$birthdate)
dat$age <- as.numeric(difftime(rec, born, units = "days")) / 365.25
n_bad_age <- sum(!is.na(dat$age) & (dat$age > 105 | dat$age < 18))
dat$age[!is.na(dat$age) & (dat$age > 105 | dat$age < 18)] <- NA

# QoL összpontszámok.
# OHIP-5: 5 tétel, 0–4 (0 = soha ... 4 = nagyon gyakran) -> 0–20,
#         MAGASABB = ROSSZABB életminőség.
# GOHAI:  12 tétel, 1–5 (1 = mindig panasz ... 5 = soha)  -> 12–60,
#         MAGASABB = JOBB életminőség.
dat$OHIP_sum  <- row_sum_strict(dat[, paste0("OHIP_", 1:5)])
dat$GOHAI_sum <- row_sum_strict(dat[, paste0("GOHAI_", 1:12)])

# Objektív rágóképesség: kevert-szín vizsgálat, huedegree-módszer.
# MAGASABB init_MAI = ROSSZABB színkeverés = rosszabb rágóképesség
# (a kutatásterv konvenciója szerint).
dat$mai <- suppressWarnings(as.numeric(dat$init_mai_huedegree))

# Önbevallott rágóképesség (5-fokú: 1 = nagyon rossz ... 5 = kiváló) —
# a MAI irány-konvenciójának empirikus horgonyaként használjuk.
ordinal_5 <- function(x) {
  lvl <- c("Nagyon rossz" = 1, "Rossz" = 2, "Átlagos" = 3, "Jó" = 4, "Kiváló" = 5)
  unname(lvl[trimws(as.character(x))])
}
dat$chewing_num <- ordinal_5(dat$chewing_today_situation)

# Fogatlanság érintettsége (protetikai ellátás alá vont állcsontok).
dat$denture_type <- factor(dat$denture_type, levels = c("both", "lower", "upper"))

# ------------------------------------------------------------
# Klinikai F/A tételek kódkönyve ------------------------------
# ------------------------------------------------------------
# A "fav" mező az adott tétel KEDVEZŐ kategóriáit adja meg; minden más
# megfigyelt érték kedvezőtlennek számít. A kódolás a questionnaire1.html
# válaszopcióin és klinikai megfontoláson alapul:
#   F5  lötyögő gerinc:        1 = nincs (kedvező)
#   F7  torus palatinus:       1 = nincs (kedvező)
#   A1  Kaán-osztály:          1–5 monoton súlyosbodó atrófia;
#                              elsődlegesen 1–2 = kedvező, 3–5 = kedvezőtlen
#   A3  buccinator tasak:      2 = szájnyitáskor KISZÉLESEDŐ a kedvező;
#                              1 (beszűkülő) és 3 (lebenyezett) kedvezőtlen
#                              -> NEM monoton kódolás, irány-feltevés! (ambiguous)
#   A4  torus mandibularis:    1 = nincs (kedvező); 2 = nagy, 3 = kicsi
#                              -> NEM monoton (2 rosszabb, mint 3)
#   A5  lingualis tasak:       1 = izmok nem szűkítik (kedvező)
#   A6  feszes íny:            1 = az egész tuberculumot fedi (kedvező)
#   A7  tuberculum alakja:     1 = fordított körte (kedvező)
#   A8  inklinációs szög:      1 = nincs eltérés (kedvező); 3 = plicaszerű,
#                              nem vizsgálható -> irány-feltevés (ambiguous)
#   A9  tuberculum mozgása:    1 = nem változik (kedvező)
#   A11 szublingvális tájék:   1 = nem elődomborodó (kedvező)
#   A12 spinae mentales:       1 = nem tapintható (kedvező)
#   A13 TMI funkció:           1 = panaszmentes (kedvező)
# A "monotone" mező jelzi, hogy a nyers 1..k kód súlyossági sorrendként
# értelmezhető-e (csak ezeknél van értelme item-szintű Spearman-nak).

ITEMS <- list(
  F5      = list(jaw = "F", bilateral = FALSE, fav = 1,      monotone = FALSE, ambiguous = FALSE, label = "F5 lötyögő gerinc"),
  F7      = list(jaw = "F", bilateral = FALSE, fav = 1,      monotone = FALSE, ambiguous = FALSE, label = "F7 torus palatinus"),
  A1_Kaan = list(jaw = "A", bilateral = FALSE, fav = c(1,2), monotone = TRUE,  ambiguous = FALSE, label = "A1 Kaán-gerincforma"),
  A3      = list(jaw = "A", bilateral = TRUE,  fav = 2,      monotone = FALSE, ambiguous = TRUE,  label = "A3 buccinator tasak"),
  A4      = list(jaw = "A", bilateral = TRUE,  fav = 1,      monotone = FALSE, ambiguous = FALSE, label = "A4 torus mandibularis"),
  A5      = list(jaw = "A", bilateral = TRUE,  fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A5 lingualis tasak"),
  A6      = list(jaw = "A", bilateral = TRUE,  fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A6 feszes íny hiánya"),
  A7      = list(jaw = "A", bilateral = TRUE,  fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A7 tuberculum alakja"),
  A8      = list(jaw = "A", bilateral = TRUE,  fav = 1,      monotone = FALSE, ambiguous = TRUE,  label = "A8 inklinációs szög"),
  A9      = list(jaw = "A", bilateral = TRUE,  fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A9 tuberculum mozgékonysága"),
  A11     = list(jaw = "A", bilateral = FALSE, fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A11 szublingvális tájék"),
  A12     = list(jaw = "A", bilateral = FALSE, fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A12 spinae mentales"),
  A13     = list(jaw = "A", bilateral = FALSE, fav = 1,      monotone = TRUE,  ambiguous = FALSE, label = "A13 TMI funkció")
)

A_ITEMS <- names(ITEMS)[sapply(ITEMS, function(m) m$jaw == "A")]
F_ITEMS <- names(ITEMS)[sapply(ITEMS, function(m) m$jaw == "F")]

# ------------------------------------------------------------
# Kedvezőtlen-indikátorok és kumulatív rizikópontszám ---------
# ------------------------------------------------------------

# 0/1 kedvezőtlen-indikátor egy értékvektorra
unfav01 <- function(v, fav) ifelse(is.na(v), NA_integer_, as.integer(!(v %in% fav)))

# Pontszám-építő. collapse = "worst": bilaterális tételnél bármelyik
# oldal kedvezőtlensége kedvezőtlennek számít (klinikailag a rosszabb
# oldal a mérvadó); "mean": a két oldal átlaga (0/0.5/1) — érzékenységi
# variáns. fav_override névvel felülírja egyes tételek kedvező halmazát,
# drop_items kihagy tételeket (szintén érzékenységi elemzésekhez).
build_scores <- function(d, items = ITEMS, collapse = c("worst", "mean"),
                         fav_override = list(), drop_items = character(0)) {
  collapse <- match.arg(collapse)
  items <- items[setdiff(names(items), drop_items)]
  u <- data.frame(row.names = seq_len(nrow(d)))
  o <- data.frame(row.names = seq_len(nrow(d)))   # ordinális (monoton) értékek
  for (nm in names(items)) {
    meta <- items[[nm]]
    fav <- if (!is.null(fav_override[[nm]])) fav_override[[nm]] else meta$fav
    if (meta$bilateral) {
      vj <- suppressWarnings(as.numeric(d[[paste0(nm, "_jobb")]]))
      vb <- suppressWarnings(as.numeric(d[[paste0(nm, "_bal")]]))
      uj <- unfav01(vj, fav); ub <- unfav01(vb, fav)
      both_na <- is.na(uj) & is.na(ub)
      u_val <- if (collapse == "worst") pmax(uj, ub, na.rm = TRUE)
               else rowMeans(cbind(uj, ub), na.rm = TRUE)
      u_val[both_na] <- NA
      u[[nm]] <- u_val
      if (meta$monotone) {
        o_val <- pmax(vj, vb, na.rm = TRUE)   # rosszabb oldal
        o_val[is.na(vj) & is.na(vb)] <- NA
        o[[nm]] <- o_val
      }
    } else {
      v <- suppressWarnings(as.numeric(d[[nm]]))
      u[[nm]] <- unfav01(v, fav)
      if (meta$monotone) o[[nm]] <- v
    }
  }
  a_names <- intersect(names(u), A_ITEMS)
  f_names <- intersect(names(u), F_ITEMS)
  prop_if <- function(m, min_k) {
    k <- rowSums(!is.na(m))
    p <- rowMeans(m, na.rm = TRUE)
    p[k < min_k] <- NA
    p
  }
  res <- data.frame(
    A_prop   = prop_if(u[, a_names, drop = FALSE], min_k = max(1, ceiling(0.75 * length(a_names)))),
    F_prop   = prop_if(u[, f_names, drop = FALSE], min_k = length(f_names)),
    all_prop = prop_if(u[, c(a_names, f_names), drop = FALSE],
                       min_k = max(1, ceiling(0.75 * (length(a_names) + length(f_names)))))
  )
  # Saját állcsont pontszáma: both -> összesített, lower -> A, upper -> F
  res$own_prop <- ifelse(d$denture_type == "both",  res$all_prop,
                  ifelse(d$denture_type == "lower", res$A_prop, res$F_prop))
  list(unfav = u, ordinal = o, scores = res)
}

sc <- build_scores(dat)
dat <- cbind(dat, sc$scores)
UNFAV   <- sc$unfav     # tételenkénti 0/1 kedvezőtlen-indikátorok
ORDINAL <- sc$ordinal   # monoton tételek ordinális (rosszabb oldali) értékei

# ------------------------------------------------------------
# Statisztikai segédfüggvények --------------------------------
# ------------------------------------------------------------

# Spearman-korreláció permutációs p-értékkel (két- és egyoldali).
# A p-érték a szokásos (1 + #extrém) / (n_perm + 1) formula.
spearman_perm <- function(x, y, n_perm = N_PERM, seed = SEED) {
  ok <- !is.na(x) & !is.na(y)
  x <- x[ok]; y <- y[ok]
  n <- length(x)
  if (n < 4 || length(unique(x)) < 2 || length(unique(y)) < 2) {
    return(list(n = n, rho = NA, p_perm_two = NA, p_perm_greater = NA,
                p_perm_less = NA, p_asymp = NA))
  }
  rho <- suppressWarnings(cor(x, y, method = "spearman"))
  set.seed(seed)
  rho_p <- replicate(n_perm, suppressWarnings(cor(x, sample(y), method = "spearman")))
  p_two <- (1 + sum(abs(rho_p) >= abs(rho))) / (n_perm + 1)
  p_gt  <- (1 + sum(rho_p >= rho)) / (n_perm + 1)
  p_lt  <- (1 + sum(rho_p <= rho)) / (n_perm + 1)
  p_asy <- suppressWarnings(cor.test(x, y, method = "spearman")$p.value)
  list(n = n, rho = rho, p_perm_two = p_two, p_perm_greater = p_gt,
       p_perm_less = p_lt, p_asymp = p_asy)
}

# Bootstrap percentilis CI tetszőleges kétváltozós statisztikára.
boot_ci_xy <- function(x, y, stat_fun, n_boot = N_BOOT, seed = SEED,
                       conf = 0.95) {
  ok <- !is.na(x) & !is.na(y)
  x <- x[ok]; y <- y[ok]
  n <- length(x)
  set.seed(seed)
  bs <- replicate(n_boot, {
    i <- sample.int(n, replace = TRUE)
    stat_fun(x[i], y[i])
  })
  bs <- bs[is.finite(bs)]
  a <- (1 - conf) / 2
  c(lower = unname(quantile(bs, a)), upper = unname(quantile(bs, 1 - a)))
}

spearman_boot_ci <- function(x, y, ...) {
  boot_ci_xy(x, y, function(a, b) suppressWarnings(cor(a, b, method = "spearman")), ...)
}

# Leave-one-out (LOO) érzékenység: stat_fun(df) statisztikát minden
# beteg kihagyásával újraszámoljuk. Visszaadja a teljes mintás értéket,
# a LOO-értékek terjedelmét és az előjelváltások számát.
loo_sensitivity <- function(df, stat_fun) {
  full <- stat_fun(df)
  loo <- vapply(seq_len(nrow(df)),
                function(i) stat_fun(df[-i, , drop = FALSE]),
                numeric(1))
  list(full = full, loo = loo,
       min = min(loo, na.rm = TRUE), max = max(loo, na.rm = TRUE),
       sign_flips = sum(sign(loo) != sign(full), na.rm = TRUE),
       max_abs_change = max(abs(loo - full), na.rm = TRUE))
}

print_loo <- function(name, ls) {
  cat(sprintf("  %-38s teljes: %+.3f | LOO-terjedelem: [%+.3f, %+.3f] | előjelváltás: %d/%d | max |Δ|: %.3f\n",
              name, ls$full, ls$min, ls$max, ls$sign_flips, length(ls$loo),
              ls$max_abs_change))
}

# Vargha–Delaney A hatásméret (P(X1 > X2) + 0.5 P(X1 = X2));
# 0.5 = nincs különbség, >0.5: az 1. csoport értékei nagyobbak.
vd_A <- function(x1, x2) {
  r <- rank(c(x1, x2))
  n1 <- length(x1); n2 <- length(x2)
  (sum(r[seq_len(n1)]) / n1 - (n1 + 1) / 2) / n2
}

fmt_p <- function(p) ifelse(is.na(p), "NA",
                            ifelse(p < 0.0001, "<0.0001", sprintf("%.4f", p)))

section <- function(title) {
  cat("\n", strrep("=", 64), "\n", title, "\n", strrep("=", 64), "\n", sep = "")
}

# ------------------------------------------------------------
# Rövid adat-összefoglaló a forrásoláskor ---------------------
# ------------------------------------------------------------
cat(sprintf("PREDICT adat betöltve: n = %d beteg (both: %d, lower: %d, upper: %d)\n",
            nrow(dat), sum(dat$denture_type == "both", na.rm = TRUE),
            sum(dat$denture_type == "lower", na.rm = TRUE),
            sum(dat$denture_type == "upper", na.rm = TRUE)))
cat(sprintf("  init MAI (huedegree) elérhető: %d | OHIP_sum: %d | GOHAI_sum: %d\n",
            sum(!is.na(dat$mai)), sum(!is.na(dat$OHIP_sum)),
            sum(!is.na(dat$GOHAI_sum))))
cat(sprintf("  kumulatív rizikópontszám (all_prop): %d | saját állcsont (own_prop): %d\n",
            sum(!is.na(dat$all_prop)), sum(!is.na(dat$own_prop))))
if (n_bad_age > 0)
  cat(sprintf("  FIGYELEM: %d értelmezhetetlen életkor (>105 vagy <18 év) NA-ra állítva.\n",
              n_bad_age))
