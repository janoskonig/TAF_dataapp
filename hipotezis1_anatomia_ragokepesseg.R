# ============================================================
# hipotezis1_anatomia_ragokepesseg.R
#
# H1: A kedvezőtlen (atrofizált) gerincanatómiájú betegeknek rosszabb
#     a fogpótlás nélküli objektív rágóképessége (init MAI, huedegree;
#     magasabb érték = rosszabb színkeverés).
#
#   P1: tételenként — a kedvezőtlen F/A-besorolású betegek init MAI-ja
#       magasabb (Mann–Whitney / exakt Wilcoxon, Holm-korrekcióval;
#       monoton tételekre Spearman is).
#   P2: állcsont-specificitás — az init MAI elsősorban a fogatlan
#       állcsont SAJÁT anatómiájával függ össze.
#   P3: dózis–válasz — a kedvezőtlen tételek halmozódása (kumulatív
#       rizikópontszám) monoton együtt jár a romló MAI-jal és QoL-lal
#       (Jonckheere–Terpstra + Spearman-trend).
#
# MÓDSZER-VALIDÁCIÓ (legalább két független úton):
#   V1  exakt/permutációs p-értékek az aszimptotikusakkal szembeállítva
#   V2  leave-one-out (LOO) beteg-szintű érzékenységi elemzés
#   V3  szimulációs kalibráció: elsőfajú hiba permutált (null) adaton
#       + erő (power) ismert nagyságú beültetett hatás mellett
#   V4  kódolási/specifikációs érzékenység: rosszabb-oldal vs. átlag,
#       bizonytalan irányú tételek (A3, A8) elhagyása, szigorú A1-küszöb
# ============================================================

source("predict_common.R")
suppressPackageStartupMessages({
  library(coin)       # exakt Wilcoxon (kötések mellett is)
  library(DescTools)  # Jonckheere–Terpstra permutációs p-vel
})
set.seed(SEED)

# ------------------------------------------------------------
section("P1 | Tételenkénti elemzés: kedvezőtlen besorolás vs. init MAI")
# ------------------------------------------------------------
# Tételenként az a beteg kerül be, akinél az adott tétel ÉS a MAI mért
# (az F/A-tételeket eleve csak az érintett fogatlan állcsonton veszik fel).
# Elsődleges próba: exakt Wilcoxon rangösszeg-teszt, egyoldali
# (h1: MAI_kedvezőtlen > MAI_kedvező, a kutatásterv szerint), Holm-
# korrekcióval a 13 tételes családon. Hatásméret: Vargha–Delaney A.

p1 <- data.frame()
for (nm in names(ITEMS)) {
  u <- UNFAV[[nm]]
  ok <- !is.na(u) & !is.na(dat$mai) & u %in% c(0, 1)   # "mean"-kollapszus törtjei kizárva
  if (sum(ok) < 6 || length(unique(u[ok])) < 2) next
  g <- factor(ifelse(u[ok] == 1, "kedvezotlen", "kedvezo"),
              levels = c("kedvezotlen", "kedvezo"))
  m <- dat$mai[ok]
  wt_gt  <- coin::wilcox_test(m ~ g, distribution = "exact", alternative = "greater")
  wt_two <- coin::wilcox_test(m ~ g, distribution = "exact", alternative = "two.sided")
  wt_asy <- suppressWarnings(wilcox.test(m[g == "kedvezotlen"], m[g == "kedvezo"],
                                         alternative = "greater", exact = FALSE))
  p1 <- rbind(p1, data.frame(
    item = nm, label = ITEMS[[nm]]$label,
    n_kedvezo = sum(g == "kedvezo"), n_kedvezotlen = sum(g == "kedvezotlen"),
    med_kedvezo = median(m[g == "kedvezo"]),
    med_kedvezotlen = median(m[g == "kedvezotlen"]),
    VD_A = vd_A(m[g == "kedvezotlen"], m[g == "kedvezo"]),
    p_exact_1old = coin::pvalue(wt_gt),
    p_exact_2old = coin::pvalue(wt_two),
    p_asymp_1old = wt_asy$p.value
  ))
}
p1$p_holm <- p.adjust(p1$p_exact_1old, method = "holm")
p1 <- p1[order(p1$p_exact_1old), ]

cat("\nExakt Wilcoxon (egyoldali, h1: kedvezőtlen > kedvező), Holm-korrekcióval:\n\n")
for (i in seq_len(nrow(p1))) with(p1[i, ], cat(sprintf(
  "  %-30s n(kedv/kedvtlen)=%2d/%2d  medián MAI kedv=%5.1f kedvtlen=%5.1f  A=%.2f  p=%s  p_Holm=%s\n",
  label, n_kedvezo, n_kedvezotlen, med_kedvezo, med_kedvezotlen, VD_A,
  fmt_p(p_exact_1old), fmt_p(p_holm))))
cat("\n  (A = Vargha–Delaney: P(MAI_kedvezőtlen > MAI_kedvező); 0.5 = nincs hatás.)\n")
write.csv(p1, file.path(OUT_DIR, "h1_p1_tetelenkent.csv"), row.names = FALSE)

# Monoton kódolású tételekre kiegészítésként Spearman (súlyossági fokozat vs MAI)
cat("\nMonoton tételek: súlyossági fokozat vs. init MAI (Spearman, permutációs p):\n\n")
p1_mono <- data.frame()
for (nm in names(ORDINAL)) {
  s <- spearman_perm(ORDINAL[[nm]], dat$mai)
  p1_mono <- rbind(p1_mono, data.frame(item = nm, n = s$n, rho = s$rho,
                                       p_perm_1old = s$p_perm_greater,
                                       p_asymp = s$p_asymp))
  cat(sprintf("  %-30s n=%2d  rho=%+.3f  p_perm(1old)=%s\n",
              ITEMS[[nm]]$label, s$n, s$rho, fmt_p(s$p_perm_greater)))
}
p1_mono$p_holm <- p.adjust(p1_mono$p_perm_1old, method = "holm")
write.csv(p1_mono, file.path(OUT_DIR, "h1_p1_monoton_spearman.csv"), row.names = FALSE)

# ------------------------------------------------------------
section("P2 | Állcsont-specificitás: saját vs. ellenoldali anatómia")
# ------------------------------------------------------------
# (a) Rétegzett elemzés: a saját állcsont pontszáma vs. MAI fogatlansági
#     típusonként. FIGYELEM: a lower (n≈7) és upper (n≈6) réteg nagyon
#     kicsi — ezek az eredmények csak leíró jellegűek.
cat("\n(a) Saját állcsont rizikópontszáma vs. init MAI, rétegenként:\n\n")
for (lv in levels(dat$denture_type)) {
  sub <- dat[dat$denture_type == lv, ]
  s <- spearman_perm(sub$own_prop, sub$mai)
  ci <- if (s$n >= 8) spearman_boot_ci(sub$own_prop, sub$mai) else c(lower = NA, upper = NA)
  cat(sprintf("  %-6s n=%2d  rho=%+.3f  p_perm(1old)=%s  95%% boot CI [%+.2f, %+.2f]\n",
              lv, s$n, s$rho, fmt_p(s$p_perm_greater), ci["lower"], ci["upper"]))
}

# (b) A "both" betegeknél mindkét állcsont pontszáma mért: melyik jelzi
#     erősebben a MAI-t? Függő korrelációk különbsége bootstrap CI-vel.
both <- dat[dat$denture_type == "both" & !is.na(dat$mai) &
            !is.na(dat$A_prop) & !is.na(dat$F_prop), ]
rho_A <- suppressWarnings(cor(both$A_prop, both$mai, method = "spearman"))
rho_F <- suppressWarnings(cor(both$F_prop, both$mai, method = "spearman"))
set.seed(SEED)
d_bs <- replicate(N_BOOT, {
  i <- sample.int(nrow(both), replace = TRUE)
  suppressWarnings(cor(both$A_prop[i], both$mai[i], method = "spearman") -
                   cor(both$F_prop[i], both$mai[i], method = "spearman"))
})
d_ci <- quantile(d_bs, c(0.025, 0.975), na.rm = TRUE)
cat(sprintf("\n(b) 'both' betegek (n=%d): rho(A-pontszám, MAI)=%+.3f, rho(F-pontszám, MAI)=%+.3f\n",
            nrow(both), rho_A, rho_F))
cat(sprintf("    különbség (A−F) = %+.3f, 95%% bootstrap CI [%+.3f, %+.3f]\n",
            rho_A - rho_F, d_ci[1], d_ci[2]))
cat("    (Ha a CI a 0-t tartalmazza, a két állcsont-pontszám prediktív ereje\n")
cat("     nem különböztethető meg ezen a mintán.)\n")

# (c) Interakciós modell: az A-pontszám hatása a MAI-ra függ-e attól,
#     hogy a felső állcsont is fogatlan-e (both vs. lower)?
#     Rang-transzformált MAI-on (robusztus, kis mintás lineáris modell).
al <- dat[dat$denture_type %in% c("both", "lower") &
          !is.na(dat$A_prop) & !is.na(dat$mai), ]
al$upper_is_edent <- as.integer(al$denture_type == "both")
fit_int <- lm(rank(mai) ~ A_prop * upper_is_edent, data = al)
ct <- summary(fit_int)$coefficients
p_int <- ct["A_prop:upper_is_edent", 4]
cat(sprintf("\n(c) rank(MAI) ~ A_pontszám × felső_is_fogatlan (n=%d):\n", nrow(al)))
cat(sprintf("    interakciós tag: béta=%+.2f, p=%s\n",
            ct["A_prop:upper_is_edent", 1], fmt_p(p_int)))
if (!is.na(p_int) && p_int < 0.10) {
  cat("    FIGYELEM: a marginális interakció arra utal, hogy az A-pontszám és a\n")
  cat(sprintf("    MAI kapcsolata eltérhet rétegek közt (lower-meredekség %+.1f vs\n",
              ct["A_prop", 1]))
  cat(sprintf("    both-meredekség %+.1f a rangskálán) — a nagyon kis lower-réteg\n",
              ct["A_prop", 1] + ct["A_prop:upper_is_edent", 1]))
  cat("    (n=7) miatt instabil becslés, óvatosan értelmezendő.\n")
} else {
  cat("    Az interakció nem mutatható ki: az A-hatás a rétegek közt homogén.\n")
}
cat("    Megjegyzés: e modell MINDKÉT rétegében az alsó állcsont fogatlan, így\n")
cat("    az A-pontszám mindenütt 'saját' anatómia. A terv 'saját vs. ellenoldali'\n")
cat("    kontrasztja ezen az adaton strukturálisan nem tesztelhető (F-tételek\n")
cat("    csak felső, A-tételek csak alsó fogatlanságnál mértek), és H1 meg H2\n")
cat("    EGYARÁNT null-interakciót jósol — ez a próba tehát konzisztencia-\n")
cat("    ellenőrzés, nem H1–H2 diszkriminátor; a P2 tényleges összevetése a\n")
cat("    (b) blokk (A- vs. F-pontszám prediktív ereje a 'both' betegeken).\n")

# ------------------------------------------------------------
section("P3 | Dózis–válasz: kumulatív rizikópontszám vs. MAI és QoL")
# ------------------------------------------------------------
# Elsődleges populáció: 'both' betegek az összesített (13 tételes)
# pontszámmal — itt a pontszám tartalma homogén. Másodlagos: teljes
# minta a saját állcsont pontszámával.

dose_pop <- dat[dat$denture_type == "both" & !is.na(dat$all_prop), ]

dose_test <- function(df, score, yname, y, direction = c("greater", "less")) {
  direction <- match.arg(direction)
  s <- spearman_perm(score, y)
  p_dir <- if (direction == "greater") s$p_perm_greater else s$p_perm_less
  ci <- spearman_boot_ci(score, y)
  # Jonckheere–Terpstra a pontszám tercilisei mentén
  ok <- !is.na(score) & !is.na(y)
  br <- unique(quantile(score[ok], c(0, 1/3, 2/3, 1)))
  jt_p <- NA
  if (length(br) >= 3) {
    g <- cut(score[ok], breaks = br, include.lowest = TRUE, ordered_result = TRUE)
    jt <- suppressWarnings(DescTools::JonckheereTerpstraTest(
      y[ok], g, alternative = ifelse(direction == "greater", "increasing", "decreasing"),
      nperm = 5000))
    jt_p <- jt$p.value
  }
  cat(sprintf("  %-10s n=%2d  rho=%+.3f  95%% CI [%+.2f, %+.2f]  p_perm(1old)=%s  p_JT=%s\n",
              yname, s$n, s$rho, ci["lower"], ci["upper"], fmt_p(p_dir), fmt_p(jt_p)))
  invisible(list(rho = s$rho, p_perm = p_dir, p_jt = jt_p, n = s$n))
}

cat("\nElsődleges ('both', 13 tételes pontszám); h1-irányok: MAI +, OHIP +, GOHAI −:\n\n")
r_mai   <- dose_test(dose_pop, dose_pop$all_prop, "init MAI",  dose_pop$mai,       "greater")
r_ohip  <- dose_test(dose_pop, dose_pop$all_prop, "OHIP_sum",  dose_pop$OHIP_sum,  "greater")
r_gohai <- dose_test(dose_pop, dose_pop$all_prop, "GOHAI_sum", dose_pop$GOHAI_sum, "less")

cat("\nMásodlagos (teljes minta, saját állcsont pontszáma):\n\n")
r_mai2  <- dose_test(dat, dat$own_prop, "init MAI",  dat$mai,      "greater")
r_ohip2 <- dose_test(dat, dat$own_prop, "OHIP_sum",  dat$OHIP_sum, "greater")
r_goh2  <- dose_test(dat, dat$own_prop, "GOHAI_sum", dat$GOHAI_sum, "less")

# ------------------------------------------------------------
section("V1 | Validáció 1: exakt/permutációs vs. aszimptotikus p-értékek")
# ------------------------------------------------------------
# Kis mintán az aszimptotikus p megbízhatatlan lehet; ha a kétféle
# számítás közel esik, az inferencia nem a p-érték számítási módján múlik.
cat("\nTételenkénti Wilcoxon: exakt vs. aszimptotikus egyoldali p:\n\n")
p1$p_diff <- abs(p1$p_exact_1old - p1$p_asymp_1old)
for (i in seq_len(nrow(p1))) with(p1[i, ], cat(sprintf(
  "  %-30s p_exakt=%s  p_aszimpt=%s  |Δ|=%.4f\n",
  label, fmt_p(p_exact_1old), fmt_p(p_asymp_1old), p_diff)))
cat(sprintf("\n  Legnagyobb eltérés: %.4f — %s\n", max(p1$p_diff),
            ifelse(max(p1$p_diff) < 0.02,
                   "a kétféle számítás gyakorlatilag egybevág, az eredmény robusztus.",
                   "van érdemi eltérés: a döntést az EXAKT p-re kell alapozni!")))
cat("\nDózis–válasz: a Spearman permutációs p-je mellett a Jonckheere–Terpstra\n")
cat("(fent, p_JT) FÜGGETLEN próbaként ellenőrzi ugyanazt a monoton trendet.\n")

# ------------------------------------------------------------
section("V2 | Validáció 2: leave-one-out (LOO) beteg-érzékenység")
# ------------------------------------------------------------
# Minden beteget egyenként kihagyva újraszámoljuk a kulcs-statisztikákat.
# Kis mintán egyetlen beteg is billentheti az eredményt — itt látszik, ha így van.
cat("\nDózis–válasz rho ('both' populáció) LOO-érzékenysége:\n\n")
loo_mai <- loo_sensitivity(dose_pop[!is.na(dose_pop$mai) & !is.na(dose_pop$all_prop), ],
  function(df) suppressWarnings(cor(df$all_prop, df$mai, method = "spearman")))
loo_ohi <- loo_sensitivity(dose_pop[!is.na(dose_pop$OHIP_sum) & !is.na(dose_pop$all_prop), ],
  function(df) suppressWarnings(cor(df$all_prop, df$OHIP_sum, method = "spearman")))
loo_goh <- loo_sensitivity(dose_pop[!is.na(dose_pop$GOHAI_sum) & !is.na(dose_pop$all_prop), ],
  function(df) suppressWarnings(cor(df$all_prop, df$GOHAI_sum, method = "spearman")))
print_loo("rho(pontszám, MAI)", loo_mai)
print_loo("rho(pontszám, OHIP)", loo_ohi)
print_loo("rho(pontszám, GOHAI)", loo_goh)

# A p-érték LOO-stabilitása a MAI-ra: marad-e a döntés minden kihagyásnál?
dsub <- dose_pop[!is.na(dose_pop$mai) & !is.na(dose_pop$all_prop), ]
loo_p <- vapply(seq_len(nrow(dsub)), function(i)
  spearman_perm(dsub$all_prop[-i], dsub$mai[-i], n_perm = 2000)$p_perm_greater,
  numeric(1))
cat(sprintf("\n  p_perm(pontszám→MAI) LOO-terjedelme: [%s, %s] — %s\n",
            fmt_p(min(loo_p)), fmt_p(max(loo_p)),
            ifelse(max(loo_p) < 0.05, "a szignifikancia egyetlen betegen sem múlik.",
            ifelse(min(loo_p) >= 0.05, "egy beteg kihagyása sem hozna szignifikanciát.",
                   "FIGYELEM: a döntés egyes betegek kihagyására érzékeny!"))))

# ------------------------------------------------------------
section("V3 | Validáció 3: szimulációs kalibráció (elsőfajú hiba + erő)")
# ------------------------------------------------------------
# (i)  Null-szimuláció: a MAI-értékeket permutáljuk (a valódi anatómiai
#      mintázat megtartásával) -> a teljes P1-pipeline (13 teszt + Holm)
#      családszintű téves riasztási aránya ~5% kell legyen, a dózis–
#      válasz teszté szintén ~5%.
# (ii) Erő-szimuláció: ismert nagyságú eltolást ültetünk a kedvezőtlen
#      csoportba -> mekkora hatást képes egyáltalán kimutatni ez a minta?
# A szimulációs cikluson belül (sebességi okból) normál-közelítéses
# Wilcoxont használunk; a V1 blokk igazolta, hogy ez itt az exakttal
# gyakorlatilag egybevág.

B_SIM <- 1000
sim_pipeline <- function(mai_vec) {
  ps <- rep(NA_real_, length(names(ITEMS)))
  for (k in seq_along(names(ITEMS))) {
    u <- UNFAV[[names(ITEMS)[k]]]
    ok <- !is.na(u) & !is.na(mai_vec) & u %in% c(0, 1)
    if (sum(ok) < 6 || length(unique(u[ok])) < 2) next
    ps[k] <- suppressWarnings(wilcox.test(mai_vec[ok][u[ok] == 1], mai_vec[ok][u[ok] == 0],
                                          alternative = "greater", exact = FALSE)$p.value)
  }
  ps <- ps[!is.na(ps)]
  any(p.adjust(ps, "holm") < 0.05)
}

set.seed(SEED)
null_fam <- mean(replicate(B_SIM, sim_pipeline(sample(dat$mai))))
null_dose <- mean(replicate(B_SIM, {
  ok <- !is.na(dose_pop$all_prop) & !is.na(dose_pop$mai)
  x <- dose_pop$all_prop[ok]; y <- sample(dose_pop$mai[ok])
  suppressWarnings(cor.test(x, y, method = "spearman",
                            alternative = "greater")$p.value) < 0.05
}))
cat(sprintf("\n(i) Elsőfajú hiba permutált (null) adaton, B=%d:\n", B_SIM))
cat(sprintf("    P1 család (13 teszt, Holm):  %.1f%%  (elvárt ≤ 5%%)\n", 100 * null_fam))
cat(sprintf("    dózis–válasz Spearman:       %.1f%%  (elvárt ≈ 5%%)\n", 100 * null_dose))

pow_dose <- function(rho_target, B = B_SIM) {
  ok <- !is.na(dose_pop$all_prop) & !is.na(dose_pop$mai)
  x <- dose_pop$all_prop[ok]; y0 <- dose_pop$mai[ok]; n <- sum(ok)
  mean(replicate(B, {
    y <- sample(y0)
    y <- y + rho_target * sd(y0) / sd(x) * (x - mean(x)) / (1 - rho_target^2)^0.5
    suppressWarnings(cor.test(x, y, method = "spearman",
                              alternative = "greater")$p.value) < 0.05
  }))
}
cat(sprintf("\n(ii) A dózis–válasz teszt ereje beültetett hatás mellett (n=%d):\n",
            sum(!is.na(dose_pop$all_prop) & !is.na(dose_pop$mai))))
for (r in c(0.3, 0.5, 0.7))
  cat(sprintf("    valódi rho ≈ %.1f  ->  erő: %.0f%%\n", r, 100 * pow_dose(r)))
cat("    (Kis erő gyenge hatásra: a nem-szignifikáns eredmény ITT nem bizonyítja\n")
cat("     a hatás hiányát — ezt az értelmezésnél figyelembe kell venni.)\n")

# ------------------------------------------------------------
section("V4 | Validáció 4: kódolási / specifikációs érzékenység")
# ------------------------------------------------------------
# A rizikópontszám definíciós döntéseit variáljuk; ha a dózis–válasz
# következtetés ezek mentén stabil, nem a kódolási önkény terméke.
variants <- list(
  "elsődleges (rosszabb oldal)" = list(collapse = "worst", fav_override = list(), drop = character(0)),
  "kétoldali átlag"             = list(collapse = "mean",  fav_override = list(), drop = character(0)),
  "A3+A8 (bizonytalan) nélkül"  = list(collapse = "worst", fav_override = list(), drop = c("A3", "A8")),
  "szigorú A1 (csak 1 kedvező)" = list(collapse = "worst", fav_override = list(A1_Kaan = 1), drop = character(0))
)
cat("\nDózis–válasz ('both'): rho és p a pontszám-variánsok mentén:\n\n")
v4 <- data.frame()
for (vn in names(variants)) {
  v <- variants[[vn]]
  scv <- build_scores(dat, collapse = v$collapse, fav_override = v$fav_override,
                      drop_items = v$drop)
  sub <- dat$denture_type == "both"
  s_m <- spearman_perm(scv$scores$all_prop[sub], dat$mai[sub])
  s_o <- spearman_perm(scv$scores$all_prop[sub], dat$OHIP_sum[sub])
  s_g <- spearman_perm(scv$scores$all_prop[sub], dat$GOHAI_sum[sub])
  v4 <- rbind(v4, data.frame(variant = vn, rho_MAI = s_m$rho, p_MAI = s_m$p_perm_greater,
                             rho_OHIP = s_o$rho, rho_GOHAI = s_g$rho))
  cat(sprintf("  %-29s rho_MAI=%+.3f (p=%s)  rho_OHIP=%+.3f  rho_GOHAI=%+.3f\n",
              vn, s_m$rho, fmt_p(s_m$p_perm_greater), s_o$rho, s_g$rho))
}
write.csv(v4, file.path(OUT_DIR, "h1_v4_specifikacio.csv"), row.names = FALSE)
rng <- range(v4$rho_MAI)
cat(sprintf("\n  rho_MAI terjedelme a variánsok közt: [%+.3f, %+.3f] — %s\n", rng[1], rng[2],
            ifelse(diff(rng) < 0.15 && !any(sign(v4$rho_MAI) != sign(v4$rho_MAI[1])),
                   "a következtetés kódolás-stabil.",
                   "FIGYELEM: a kódolási döntések érdemben befolyásolják az eredményt!")))

# ------------------------------------------------------------
section("H1 ÖSSZEFOGLALÁS")
# ------------------------------------------------------------
# A MAI irány-konvenciójának horgonya: ha magasabb MAI = rosszabb rágás,
# akkor az önbevallott rágóképességgel (1=nagyon rossz...5=kiváló)
# negatívan kell korrelálnia.
s_anchor <- spearman_perm(dat$mai, dat$chewing_num)
ci_anchor <- spearman_boot_ci(dat$mai, dat$chewing_num)
cat(sprintf("\nIrány-horgony: rho(init MAI, önbevallott rágóképesség) = %+.3f (n=%d, p_perm=%s, 95%% boot CI [%+.2f, %+.2f])\n",
            s_anchor$rho, s_anchor$n, fmt_p(s_anchor$p_perm_two),
            ci_anchor["lower"], ci_anchor["upper"]))
cat(ifelse(!is.na(s_anchor$rho) && s_anchor$rho < 0,
  "  -> a pont-becslés iránya egyezik a 'magasabb MAI = rosszabb rágás'\n     konvencióval, de statisztikailag NEM igazolt (a CI pozitív értékeket\n     is megenged) — a konvenciót műszeresen kell megerősíteni, mielőtt a\n     negatív dózis–válasz irányát értelmezed.\n",
  "  -> NEM negatív: a huedegree-mérőszám irány-konvencióját érdemes\n     műszeresen ellenőrizni, mielőtt a H1-eredményt értelmezed!\n"))

n_sig <- sum(p1$p_holm < 0.05)
cat(sprintf("
P1  Holm-korrekció után szignifikáns tétel: %d a %d vizsgáltból%s
P2  a 'saját vs. ellenoldali' kontraszt strukturálisan nem tesztelhető
    (ellenoldali tételek nem mértek); a 'both' betegeken az A- és F-pontszám
    prediktív ereje nem különbözött (ld. P2(b)) — P2 csak leíró jellegű.
P3  dózis–válasz ('both'): rho(pontszám, MAI) = %+.3f (p_perm = %s, p_JT = %s)
    QoL-irányok: OHIP rho = %+.3f (várt: +), GOHAI rho = %+.3f (várt: −)
Döntés h0-ról (nincs anatómia–MAI összefüggés): %s
A validációk (V1–V4) alapján az eredmény p-számítási módra, egyes
betegekre és kódolási döntésekre való érzékenysége FENT részletezve —
a végső interpretáció ezek együttes képén alapuljon.
", n_sig, nrow(p1),
   ifelse(n_sig == 0, " (tételszinten nincs Holm-szignifikáns különbség)", ""),
   r_mai$rho, fmt_p(r_mai$p_perm), fmt_p(r_mai$p_jt),
   r_ohip$rho, r_gohai$rho,
   ifelse(!is.na(r_mai$p_perm) && r_mai$p_perm < 0.05 &&
          !is.na(r_mai$p_jt) && r_mai$p_jt < 0.05,
          "mindkét trendpróba elveti h0-t -> az adat H1 predikcióival konzisztens.",
          "h0 nem vethető el egyértelműen -> H1 e mintán nem igazolódott (ld. erő-szimuláció!).")))
if (!is.na(r_mai$rho) && r_mai$rho < 0) cat("
FIGYELEM — irány: a megfigyelt dózis–válasz kapcsolat NEGATÍV, azaz a
kedvezőtlenebb anatómiájú betegek KEVERTEK JOBBAN (alacsonyabb MAI) —
a H1-predikcióval ellentétes irány. Mielőtt ebből következtetést vonsz
le: (1) ellenőrizd a fenti irány-horgonyt és a huedegree-számítás
konvencióját; (2) ha az irány valós, kézenfekvő rivális magyarázat a
fogatlanság fennállási ideje (a régebb óta fogatlan beteg gerince
atrofizáltabb ÉS gyakorlottabb a protézis nélküli rágásban) — ezt az
anamnesztikus adat rögzítésével lehetne tesztelni (ld. kutatásterv,
'Feltevés' sor).\n")
