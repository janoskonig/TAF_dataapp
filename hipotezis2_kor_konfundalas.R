# ============================================================
# hipotezis2_kor_konfundalas.R
#
# H2 (rivális hipotézis): a rosszabb rágóképességet nem a gerinc
# anatómiája, hanem az életkorral járó neuromuszkuláris hanyatlás
# okozza — az anatómia csak együtt jár a hosszú fogatlansággal
# (konfundálás).
#
#   P4: korra (és fogsortípusra) korrigálva az anatómia–MAI összefüggés
#       eltűnik; a kor önállóan magyarázza az init MAI-t.
#       h0: béta(anatómia) = 0 a korrigált modellben
#       h1: béta(anatómia) > 0 a korrekció után is  -> H2 kizárható
#
# Elsődleges elemzés: parciális Spearman (rang-alapú, kis mintára
# robusztus) + többszörös lineáris regresszió a kutatásterv szerint.
#
# MÓDSZER-VALIDÁCIÓ (legalább két független úton):
#   V1  Freedman–Lane permutációs p a korrigált hatásra
#   V2  bootstrap CI (percentilis és BCa) a korrigált együtthatóra
#   V3  leave-one-out: korrigált hatás betegenkénti kihagyással + dfbeta
#   V4  specifikációs görbe: kimenet (nyers/rang) × pontszám-variánsok ×
#       korrekciós készletek — az következtetés modellválasztás-függése
#   V5  kollinearitás (VIF) és a kor–pontszám együttjárás ellenőrzése
# ============================================================

source("predict_common.R")
suppressPackageStartupMessages({
  library(boot)   # BCa bootstrap
  library(MASS)   # rlm robusztus regresszió
})
set.seed(SEED)

# Elsődleges populáció: 'both' betegek, homogén 13 tételes pontszámmal.
# (Itt a denture_type konstans, így az nem kovariáns; a teljes mintás,
# fogsortípussal is korrigált modell a másodlagos elemzés.)
prim <- dat[dat$denture_type == "both" &
            !is.na(dat$all_prop) & !is.na(dat$mai) & !is.na(dat$age), ]
full <- dat[!is.na(dat$own_prop) & !is.na(dat$mai) & !is.na(dat$age), ]

# ------------------------------------------------------------
section("Leíró kép: kor, anatómiai pontszám, MAI páronkénti kapcsolatai")
# ------------------------------------------------------------
s_age_mai   <- spearman_perm(prim$age, prim$mai)
s_age_score <- spearman_perm(prim$age, prim$all_prop)
s_score_mai <- spearman_perm(prim$all_prop, prim$mai)
cat(sprintf("\n'both' populáció, n=%d (életkor-hiány miatt kizárva: %d):\n\n",
            nrow(prim),
            sum(dat$denture_type == "both" & !is.na(dat$all_prop) &
                !is.na(dat$mai) & is.na(dat$age))))
cat(sprintf("  rho(kor, MAI)          = %+.3f  (p_perm = %s)\n",
            s_age_mai$rho, fmt_p(s_age_mai$p_perm_two)))
cat(sprintf("  rho(kor, pontszám)     = %+.3f  (p_perm = %s)\n",
            s_age_score$rho, fmt_p(s_age_score$p_perm_two)))
cat(sprintf("  rho(pontszám, MAI)     = %+.3f  (p_perm = %s)\n",
            s_score_mai$rho, fmt_p(s_score_mai$p_perm_greater)))
cat("\n  A konfundáláshoz az kellene, hogy a kor MINDKETTŐVEL erősen járjon\n")
cat("  együtt. Ha a kor egyikkel sem korrelál, H2 eleve gyengén áll.\n")

# ------------------------------------------------------------
section("P4 | Korrigált modellek: eltűnik-e az anatómia-hatás?")
# ------------------------------------------------------------

# --- Elsődleges: parciális Spearman (rang-alapú korrekció) ----------
# rho(pontszám, MAI | kor): mindkét változóból kiregresszáljuk a kor
# rang-hatását, és a reziduálisokat korreláljuk.
partial_spearman <- function(x, y, z) {
  ok <- !is.na(x) & !is.na(y) & !is.na(z)
  rx <- resid(lm(rank(x[ok]) ~ rank(z[ok])))
  ry <- resid(lm(rank(y[ok]) ~ rank(z[ok])))
  suppressWarnings(cor(rx, ry))
}
rho_marg <- s_score_mai$rho
rho_part <- partial_spearman(prim$all_prop, prim$mai, prim$age)
cat(sprintf("\nParciális Spearman ('both', n=%d):\n", nrow(prim)))
cat(sprintf("  korrigálatlan rho(pontszám, MAI)      = %+.3f\n", rho_marg))
cat(sprintf("  korra korrigált rho(pontszám, MAI|kor) = %+.3f  (változás: %+.3f)\n",
            rho_part, rho_part - rho_marg))

# --- Lineáris modellek (a kutatásterv szerinti forma) ---------------
m0 <- lm(mai ~ all_prop, data = prim)
m1 <- lm(mai ~ all_prop + age, data = prim)
m0f <- lm(mai ~ own_prop, data = full)
m1f <- lm(mai ~ own_prop + age, data = full)
m2f <- lm(mai ~ own_prop + age + denture_type, data = full)

coef_line <- function(fit, term, label) {
  ct <- summary(fit)$coefficients
  ci <- confint(fit)[term, ]
  cat(sprintf("  %-42s béta=%+7.2f  95%% CI [%+7.2f, %+7.2f]  p=%s\n",
              label, ct[term, 1], ci[1], ci[2], fmt_p(ct[term, 4])))
  invisible(ct[term, 1])
}
cat("\nOLS modellek — az anatómiai pontszám együtthatója:\n\n")
b0 <- coef_line(m0, "all_prop", "M0  'both': MAI ~ pontszám")
b1 <- coef_line(m1, "all_prop", "M1  'both': MAI ~ pontszám + kor")
cat("\n")
coef_line(m0f, "own_prop", "M0f teljes minta: MAI ~ saját pontszám")
coef_line(m1f, "own_prop", "M1f teljes minta: + kor")
b2f <- coef_line(m2f, "own_prop", "M2f teljes minta: + kor + fogsortípus")
cat("\nA kor önálló hatása a legteljesebb modellekben:\n\n")
coef_line(m1, "age", "M1  'both': kor")
coef_line(m2f, "age", "M2f teljes minta: kor")
cat(sprintf("\n  Az anatómia-együttható változása a korrekcióra ('both'): %+.1f%%\n",
            100 * (b1 - b0) / abs(b0)))
cat("  Értelmezés: ha a korrigált béta(pontszám) érdemben pozitív marad,\n")
cat("  a tiszta konfundálás (H2) kizárható; ha nullára esik, H1 gyengül.\n")

# --- Robusztus (rlm) variáns — kilógó MAI-értékekre érzéketlenebb ---
r1 <- MASS::rlm(mai ~ all_prop + age, data = prim, maxit = 100)
cat(sprintf("\nRobusztus (Huber) regresszió, 'both': béta(pontszám) = %+.2f\n",
            coef(r1)["all_prop"]))

# ------------------------------------------------------------
section("V1 | Validáció 1: Freedman–Lane permutációs p a korrigált hatásra")
# ------------------------------------------------------------
# A kovariánst (kor) tiszteletben tartó permutációs séma: a redukált
# modell (MAI ~ kor) reziduálisait permutáljuk, így a kor-MAI kapcsolat
# megmarad, a pontszám-hatás nullhipotézise alatt szimulálunk.
freedman_lane <- function(df, yvar, xvar, zvars, n_perm = N_PERM,
                          rank_based = TRUE) {
  tr <- if (rank_based) function(v) rank(v) else identity
  Y <- tr(df[[yvar]]); X <- tr(df[[xvar]])
  Z <- sapply(df[, zvars, drop = FALSE], tr)
  red <- lm(Y ~ Z)
  fitted_red <- fitted(red); res_red <- resid(red)
  t_obs <- summary(lm(Y ~ X + Z))$coefficients["X", 3]
  set.seed(SEED)
  t_perm <- replicate(n_perm, {
    Y_star <- fitted_red + sample(res_red)
    summary(lm(Y_star ~ X + Z))$coefficients["X", 3]
  })
  list(t = t_obs,
       p_two = (1 + sum(abs(t_perm) >= abs(t_obs))) / (n_perm + 1),
       p_greater = (1 + sum(t_perm >= t_obs)) / (n_perm + 1))
}
fl_rank <- freedman_lane(prim, "mai", "all_prop", "age", rank_based = TRUE)
fl_raw  <- freedman_lane(prim, "mai", "all_prop", "age", rank_based = FALSE)
cat(sprintf("\n  rang-alapú:  t=%+.2f  p_perm(1old)=%s\n", fl_rank$t, fmt_p(fl_rank$p_greater)))
cat(sprintf("  nyers skálán: t=%+.2f  p_perm(1old)=%s\n", fl_raw$t, fmt_p(fl_raw$p_greater)))
# Az aszimptotikus egyoldali p-t a TESZTELT irány (h1: béta>0) felső
# farkából kell számolni, nem a kétoldali p felezésével (az negatív
# becslésnél a rossz farkot adná).
p_asymp_1old <- pt(summary(m1)$coefficients["all_prop", 3],
                   df.residual(m1), lower.tail = FALSE)
cat(sprintf("  (Aszimptotikus egyoldali OLS p ugyanerre (M1, h1: béta>0): %s.\n",
            fmt_p(p_asymp_1old)))
if (is.finite(p_asymp_1old) && abs(p_asymp_1old - fl_raw$p_greater) < 0.05) {
  cat("   A nyers skálás permutációs p-vel egybevág — az inferencia nem múlik\n")
  cat("   a normalitási feltevésen. A rang-alapú FL más — rang-transzformált —\n")
  cat("   modellen fut, azzal pontos egyezés nem várható.)\n")
} else {
  cat("   A permutációs és az aszimptotikus p eltér — az aszimptotikus\n")
  cat("   közelítés itt óvatosan kezelendő.)\n")
}

# ------------------------------------------------------------
section("V2 | Validáció 2: bootstrap CI a korrigált hatásra")
# ------------------------------------------------------------
stat_part <- function(d, i) {
  db <- d[i, ]
  c(partial_spearman(db$all_prop, db$mai, db$age),
    coef(lm(mai ~ all_prop + age, data = db))["all_prop"])
}
set.seed(SEED)
bt <- boot(prim, stat_part, R = N_BOOT)
ci_p_perc <- boot.ci(bt, index = 1, type = "perc")$percent[4:5]
ci_p_bca  <- tryCatch(boot.ci(bt, index = 1, type = "bca")$bca[4:5],
                      error = function(e) c(NA, NA))
ci_b_perc <- boot.ci(bt, index = 2, type = "perc")$percent[4:5]
ci_b_bca  <- tryCatch(boot.ci(bt, index = 2, type = "bca")$bca[4:5],
                      error = function(e) c(NA, NA))
cat(sprintf("\n  parciális rho:  percentilis CI [%+.3f, %+.3f] | BCa CI [%+.3f, %+.3f]\n",
            ci_p_perc[1], ci_p_perc[2], ci_p_bca[1], ci_p_bca[2]))
cat(sprintf("  OLS béta:       percentilis CI [%+.2f, %+.2f] | BCa CI [%+.2f, %+.2f]\n",
            ci_b_perc[1], ci_b_perc[2], ci_b_bca[1], ci_b_bca[2]))
cat("  (Ha a kétféle bootstrap CI és a parametrikus CI ugyanarra a döntésre\n")
cat("   vezet, az eredmény nem a CI-konstrukció műterméke.)\n")

# ------------------------------------------------------------
section("V3 | Validáció 3: leave-one-out és befolyásos betegek")
# ------------------------------------------------------------
loo_part <- loo_sensitivity(prim, function(df)
  partial_spearman(df$all_prop, df$mai, df$age))
print_loo("parciális rho(pontszám, MAI | kor)", loo_part)
loo_beta <- loo_sensitivity(prim, function(df)
  coef(lm(mai ~ all_prop + age, data = df))["all_prop"])
print_loo("OLS béta(pontszám | kor)", loo_beta)

dfb <- dfbeta(m1)[, "all_prop"]
infl <- which(abs(dfb) > 2 * sd(dfb))
# names(infl) a prim sorNEVE = az eredeti (patients.csv fejléc utáni)
# sorszám — a szűrt részminta pozíciója itt félrevezető lenne.
cat(sprintf("\n  Befolyásos betegek (|dfbeta| > 2 SD): %s\n",
            ifelse(length(infl) == 0, "nincs",
                   paste0("patients.csv szerinti adatsor ",
                          paste(names(infl), collapse = ", "),
                          " — érdemes adatellenőrzést futtatni rájuk"))))

# ------------------------------------------------------------
section("V4 | Validáció 4: specifikációs görbe")
# ------------------------------------------------------------
# Minden ésszerű modellváltozatot lefuttatunk; a következtetés akkor
# megbízható, ha nem egy-egy önkényes választáson múlik.
score_variants <- list(
  "worst"    = build_scores(dat),
  "mean"     = build_scores(dat, collapse = "mean"),
  "noambig"  = build_scores(dat, drop_items = c("A3", "A8")),
  "strictA1" = build_scores(dat, fav_override = list(A1_Kaan = 1))
)
spec <- data.frame()
for (sv in names(score_variants)) {
  d2 <- dat
  d2$all_prop <- score_variants[[sv]]$scores$all_prop
  d2$own_prop <- score_variants[[sv]]$scores$own_prop
  pb <- d2[d2$denture_type == "both" & !is.na(d2$all_prop) & !is.na(d2$mai) & !is.na(d2$age), ]
  fb <- d2[!is.na(d2$own_prop) & !is.na(d2$mai) & !is.na(d2$age), ]
  for (outc in c("nyers", "rang")) {
    tr <- if (outc == "rang") rank else identity
    fits <- list(
      "both: +kor" = lm(tr(mai) ~ all_prop + age, data = pb),
      "teljes: +kor+típus" = lm(tr(mai) ~ own_prop + age + denture_type, data = fb)
    )
    for (fn in names(fits)) {
      term <- if (grepl("both", fn)) "all_prop" else "own_prop"
      ct <- summary(fits[[fn]])$coefficients
      spec <- rbind(spec, data.frame(pontszam = sv, kimenet = outc, modell = fn,
                                     t = ct[term, 3], p_1old = pt(ct[term, 3],
                                     df.residual(fits[[fn]]), lower.tail = FALSE)))
    }
  }
}
cat(sprintf("\n  %d specifikáció; a korrigált anatómia-hatás t-statisztikái:\n\n",
            nrow(spec)))
cat(sprintf("  pozitív előjelű: %d/%d | p<0.05 (egyoldali): %d/%d | t-terjedelem: [%+.2f, %+.2f]\n",
            sum(spec$t > 0), nrow(spec), sum(spec$p_1old < 0.05), nrow(spec),
            min(spec$t), max(spec$t)))
write.csv(spec, file.path(OUT_DIR, "h2_v4_specifikacios_gorbe.csv"), row.names = FALSE)
cat("  Részletek: stat_output/h2_v4_specifikacios_gorbe.csv\n")

# ------------------------------------------------------------
section("V5 | Validáció 5: kollinearitás-ellenőrzés (VIF)")
# ------------------------------------------------------------
vif_manual <- function(fit) {
  X <- model.matrix(fit)[, -1, drop = FALSE]
  sapply(seq_len(ncol(X)), function(j) {
    r2 <- summary(lm(X[, j] ~ X[, -j, drop = FALSE]))$r.squared
    1 / (1 - r2)
  }) |> setNames(colnames(X))
}
cat("\n  VIF (M2f, teljes minta):\n")
print(round(vif_manual(m2f), 2))
cat("  VIF < 5: a kor és a pontszám hatása szétválasztható; magas VIF\n")
cat("  esetén a korrigált együtthatók instabilak — óvatos értelmezés!\n")

# ------------------------------------------------------------
section("H2 ÖSSZEFOGLALÁS")
# ------------------------------------------------------------
# Döntési ágak:
#  (1) a korrigált anatómia-hatás a várt (+) irányban szignifikáns
#      -> a tiszta konfundálás (H2) kizárható;
#  (2) a hatás a korrekcióval érdemben zsugorodik ÉS a kor önállóan
#      magyarázza a MAI-t -> az adat H2-vel konzisztens;
#  (3) a korrekció gyakorlatilag nem változtat, ÉS a kor önállóan sem
#      magyaráz -> kor-konfundálásra nincs jel, de a várt irányú
#      anatómia-hatás sem áll -> H1 és H2 egyaránt támogatás nélkül.
p_age  <- summary(m1)$coefficients["age", 4]
pos_sig <- !is.na(fl_rank$p_greater) && fl_rank$p_greater < 0.05 &&
           !is.na(ci_p_perc[1]) && ci_p_perc[1] > 0
shrunk  <- abs(rho_part) < 0.5 * abs(rho_marg)
dontes <- if (pos_sig) {
  "az anatómia-hatás a korrekció UTÁN is fennáll a várt irányban -> a tiszta konfundálás (H2) kizárható."
} else if (shrunk && p_age < 0.05) {
  "a korrekció az anatómia-hatást érdemben elnyeli ÉS a kor önállóan magyarázza a MAI-t -> az adat H2-vel konzisztens; H1 gyengül."
} else if (!shrunk && p_age >= 0.05) {
  "a kor-korrekció gyakorlatilag semmit nem változtat az anatómia–MAI kapcsolaton, és a kor önállóan SEM magyarázza a MAI-t -> kor-konfundálásra (H2) nincs jel; ugyanakkor a várt irányú anatómia-hatás (P4 h1) sem igazolódott."
} else {
  "köztes kép: a hatás részben gyengül — a minta ereje kevés az egyértelmű döntéshez (ld. H1 V3 erő-szimuláció)."
}
cat(sprintf("
Korrigálatlan rho(pontszám, MAI) = %+.3f -> korra korrigálva %+.3f.
Freedman–Lane permutációs p (rang, egyoldali, h1: béta>0) = %s.
Kor önálló hatása (M1): p = %s.

Döntés P4-ről: %s
", rho_marg, rho_part, fmt_p(fl_rank$p_greater), fmt_p(p_age), dontes))
if (!is.na(rho_part) && rho_part < 0) cat("
FIGYELEM — irány: a korrigált anatómia–MAI kapcsolat NEGATÍV (a H1-
predikcióval ellentétes), és ezt a kor-korrekció nem magyarázza meg.
Az irány-kérdés részleteit a H1-szkript összefoglalója tárgyalja.\n")
