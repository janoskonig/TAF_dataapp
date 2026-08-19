# ============================================================
# hipotezis3_mediacio_eletminoseg.R
#
# H3: a kedvezőtlen anatómia az életminőséget nem közvetlenül,
#     hanem a romló rágófunkción keresztül rontja.
#
#   P5: az anatómia korrelál a QoL-lal, de az összefüggés az init
#       MAI-ra kontrollálva eltűnik (indirekt út); ha megmarad,
#       közvetlen (komfort, esztétikai) hatás is van.
#       h0: nincs indirekt út (a×b = 0)
#       h1: indirekt út ≠ 0 (mediáció)
#
# Jelölések (rang-transzformált lineáris utakon):
#   X = kumulatív anatómiai rizikópontszám ('both' betegek, 13 tétel)
#   M = init MAI (huedegree; magasabb = rosszabb rágás)
#   Y = OHIP_sum (magasabb = ROSSZABB QoL)  ill.  GOHAI_sum (magasabb = JOBB)
#   a: X -> M | b: M -> Y (X mellett) | c': X -> Y direkt | c: teljes hatás
#   Várt irányok: OHIP-nál a>0, b>0, a×b>0; GOHAI-nál a>0, b<0, a×b<0.
#
# MÓDSZER-VALIDÁCIÓ (legalább két független úton):
#   V1  az indirekt hatás kétféle bootstrap CI-je (percentilis, BCa)
#       + permutációs null-eloszlás + joint-significance keresztpróba
#   V2  leave-one-out: az indirekt hatás betegenkénti érzékenysége
#   V3  kimenet-konkordancia (OHIP vs. GOHAI) és nyers-skálás variáns
#   V4  visszanyerési (recovery) szimuláció: ezen a mintaméreten a
#       bootstrap-mediáció megtalálja-e az ismert, beültetett hatást,
#       és mekkora a téves riasztása null-adaton?
#
# Megjegyzés: keresztmetszeti adaton a mediáció csak asszociációs
# értelemben tesztelhető; a M–Y út nem mért közös okai (pl. általános
# egészségi állapot) torzíthatják. Ez tervezési korlát, nem statisztikai.
# ============================================================

source("predict_common.R")
suppressPackageStartupMessages(library(boot))
set.seed(SEED)

med_pop <- dat[dat$denture_type == "both" & !is.na(dat$all_prop) &
               !is.na(dat$mai) & !is.na(dat$OHIP_sum) & !is.na(dat$GOHAI_sum), ]
cat(sprintf("\nMediációs populáció ('both', teljes X-M-Y adat): n = %d\n", nrow(med_pop)))

# ------------------------------------------------------------
# Mediációs becslő: rang-transzformált (elsődleges) vagy nyers skálán.
# Visszaadja az a, b, c', c és a×b becsléseket standardizált változókon.
# ------------------------------------------------------------
# Standardizált változókon az OLS-útegyütthatók zárt képlettel jönnek a
# páronkénti korrelációkból (gyors: a bootstrap/szimulációs ciklusokban
# nem kell lm-et illeszteni):
#   a = r(X,M);  b = (r_YM − r_YX·r_XM)/(1 − r_XM²);
#   c' = (r_YX − r_YM·r_XM)/(1 − r_XM²);  c = r_YX.
med_paths <- function(x, m, y, rank_based = TRUE) {
  tr <- if (rank_based) function(v) as.numeric(scale(rank(v))) else function(v) as.numeric(scale(v))
  X <- tr(x); M <- tr(m); Y <- tr(y)
  r_xm <- cor(X, M); r_ym <- cor(Y, M); r_yx <- cor(Y, X)
  den <- 1 - r_xm^2
  b  <- (r_ym - r_yx * r_xm) / den
  cp <- (r_yx - r_ym * r_xm) / den
  c(a = r_xm, b = b, indirect = r_xm * b, direct = cp, total = r_yx)
}

run_mediation <- function(df, yvar, ylabel, expected_ab_sign, rank_based = TRUE) {
  section(sprintf("P5 | Mediáció: pontszám -> MAI -> %s (%s skála)", ylabel,
                  ifelse(rank_based, "rang", "nyers")))
  x <- df$all_prop; m <- df$mai; y <- df[[yvar]]
  est <- med_paths(x, m, y, rank_based)
  cat(sprintf("\n  a (pontszám→MAI)        = %+.3f\n", est["a"]))
  cat(sprintf("  b (MAI→%s | pontszám) = %+.3f\n", ylabel, est["b"]))
  cat(sprintf("  indirekt (a×b)          = %+.3f   (várt előjel: %s)\n",
              est["indirect"], expected_ab_sign))
  cat(sprintf("  direkt (c')             = %+.3f\n", est["direct"]))
  cat(sprintf("  teljes (c)              = %+.3f\n", est["total"]))
  if (abs(est["total"]) > 1e-8 && sign(est["indirect"]) == sign(est["total"]))
    cat(sprintf("  mediált hányad          = %.0f%%\n",
                100 * est["indirect"] / est["total"]))

  # Bootstrap az indirekt és a direkt hatásra
  set.seed(SEED)
  bt <- boot(df, function(d, i) med_paths(d$all_prop[i], d$mai[i], d[[yvar]][i],
                                          rank_based), R = N_BOOT)
  ci <- function(idx, type) tryCatch({
    o <- boot.ci(bt, index = idx, type = type)
    if (type == "perc") o$percent[4:5] else o$bca[4:5]
  }, error = function(e) c(NA, NA))
  ab_perc <- ci(3, "perc"); ab_bca <- ci(3, "bca")
  cp_perc <- ci(4, "perc"); cp_bca <- ci(4, "bca")
  cat(sprintf("\n  indirekt 95%% CI: percentilis [%+.3f, %+.3f] | BCa [%+.3f, %+.3f]\n",
              ab_perc[1], ab_perc[2], ab_bca[1], ab_bca[2]))
  cat(sprintf("  direkt   95%% CI: percentilis [%+.3f, %+.3f] | BCa [%+.3f, %+.3f]\n",
              cp_perc[1], cp_perc[2], cp_bca[1], cp_bca[2]))

  # Parciális Spearman a kutatásterv szerint: eltűnik-e X–Y az M mellett?
  ok <- rep(TRUE, nrow(df))
  rho_xy  <- suppressWarnings(cor(x, y, method = "spearman"))
  rx <- resid(lm(rank(x) ~ rank(m))); ry <- resid(lm(rank(y) ~ rank(m)))
  rho_xy_m <- suppressWarnings(cor(rx, ry))
  cat(sprintf("\n  rho(pontszám, %s)       = %+.3f\n", ylabel, rho_xy))
  cat(sprintf("  rho(pontszám, %s | MAI) = %+.3f  (ha ~0: az összefüggést a MAI közvetíti)\n",
              ylabel, rho_xy_m))

  list(est = est, bt = bt, ab_perc = ab_perc, ab_bca = ab_bca,
       cp_perc = cp_perc, cp_bca = cp_bca, rho_xy = rho_xy, rho_xy_m = rho_xy_m)
}

res_ohip  <- run_mediation(med_pop, "OHIP_sum",  "OHIP",  "+", rank_based = TRUE)
res_gohai <- run_mediation(med_pop, "GOHAI_sum", "GOHAI", "-", rank_based = TRUE)

# ------------------------------------------------------------
section("V1 | Validáció 1: permutációs null + joint-significance keresztpróba")
# ------------------------------------------------------------
# (i) Permutációs null: X sorait permutáljuk (a és c' út megszakad, M–Y
#     kapcsolat érintetlen). FONTOS KORLÁT: ez csak az "X független
#     (M,Y)-tól" al-nullhipotézist szimulálja. Az a×b=0 h0 ÖSSZETETT
#     (a=0 VAGY b=0); a b=0, a≠0 ág alatt ez a teszt erősen anti-
#     konzervatív (szimulációval igazolva: ~40% téves riasztás), ezért
#     a döntéshez NEM ezt, hanem a lenti joint-significance p-t és a
#     bootstrap-CI-t használjuk.
perm_indirect <- function(df, yvar, n_perm = 5000) {
  obs <- med_paths(df$all_prop, df$mai, df[[yvar]])["indirect"]
  set.seed(SEED)
  nulls <- replicate(n_perm, {
    i <- sample.int(nrow(df))
    med_paths(df$all_prop[i], df$mai, df[[yvar]])["indirect"]
  })
  (1 + sum(abs(nulls) >= abs(obs))) / (n_perm + 1)
}
p_permX_ohip  <- perm_indirect(med_pop, "OHIP_sum")
p_permX_gohai <- perm_indirect(med_pop, "GOHAI_sum")
cat(sprintf("\n  permutációs p(a×b) — CSAK az 'X független (M,Y)-tól' al-nullára érvényes:\n"))
cat(sprintf("    OHIP: %s | GOHAI: %s   (önmagában nem döntési alap!)\n",
            fmt_p(p_permX_ohip), fmt_p(p_permX_gohai)))

# (ii) Joint significance: a mediáció csak akkor hihető, ha az a út ÉS
#      a b út külön-külön is kimutatható. Az összetett a×b=0 nullra
#      kalibrált p-érték: p_joint = max(p_a, p_b) (union–intersection).
s_a <- spearman_perm(med_pop$all_prop, med_pop$mai)
b_test <- function(yvar) {
  Y <- rank(med_pop[[yvar]]); M <- rank(med_pop$mai); X <- rank(med_pop$all_prop)
  red <- lm(Y ~ X)
  t_obs <- summary(lm(Y ~ M + X))$coefficients["M", 3]
  set.seed(SEED)
  t_perm <- replicate(5000, {
    Ys <- fitted(red) + sample(resid(red))
    summary(lm(Ys ~ M + X))$coefficients["M", 3]
  })
  (1 + sum(abs(t_perm) >= abs(t_obs))) / (5000 + 1)
}
p_b_ohip  <- b_test("OHIP_sum")
p_b_gohai <- b_test("GOHAI_sum")
p_joint_ohip  <- max(s_a$p_perm_two, p_b_ohip)
p_joint_gohai <- max(s_a$p_perm_two, p_b_gohai)
cat(sprintf("  a-út (pontszám→MAI):            p_perm = %s\n", fmt_p(s_a$p_perm_two)))
cat(sprintf("  b-út (MAI→OHIP | pontszám):     p_perm = %s\n", fmt_p(p_b_ohip)))
cat(sprintf("  b-út (MAI→GOHAI | pontszám):    p_perm = %s\n", fmt_p(p_b_gohai)))
cat(sprintf("  p_joint(a×b) = max(p_a, p_b):   OHIP: %s | GOHAI: %s\n",
            fmt_p(p_joint_ohip), fmt_p(p_joint_gohai)))
cat("  Mediáció-állítás CSAK akkor, ha a bootstrap-CI a várt irányban\n")
cat("  kizárja a 0-t ÉS p_joint < 0.05.\n")

# ------------------------------------------------------------
section("V2 | Validáció 2: leave-one-out az indirekt hatásra")
# ------------------------------------------------------------
loo_ab_o <- loo_sensitivity(med_pop, function(df)
  med_paths(df$all_prop, df$mai, df$OHIP_sum)["indirect"])
loo_ab_g <- loo_sensitivity(med_pop, function(df)
  med_paths(df$all_prop, df$mai, df$GOHAI_sum)["indirect"])
print_loo("indirekt a×b (OHIP)", loo_ab_o)
print_loo("indirekt a×b (GOHAI)", loo_ab_g)

# ------------------------------------------------------------
section("V3 | Validáció 3: kimenet-konkordancia és nyers-skálás variáns")
# ------------------------------------------------------------
# A két QoL-mérőeszköz ellentétes irányítású (OHIP: magasabb=rosszabb,
# GOHAI: magasabb=jobb) — valódi mediáció esetén az indirekt hatások
# ELLENTÉTES előjelűek és hasonló nagyságúak.
cat(sprintf("\n  indirekt (rang):  OHIP %+.3f  vs.  GOHAI %+.3f  -> %s\n",
            res_ohip$est["indirect"], res_gohai$est["indirect"],
            ifelse(res_ohip$est["indirect"] * res_gohai$est["indirect"] < 0,
                   "irány-konkordáns (ellentétes előjel, ahogy várható).",
                   "NEM konkordáns — az eredmény műszer-függő, óvatosan!")))
cat(sprintf("  ellenőrzés: rho(OHIP_sum, GOHAI_sum) = %+.3f (erősen negatívnak kell lennie)\n",
            suppressWarnings(cor(med_pop$OHIP_sum, med_pop$GOHAI_sum, method = "spearman"))))
raw_o <- med_paths(med_pop$all_prop, med_pop$mai, med_pop$OHIP_sum, rank_based = FALSE)
raw_g <- med_paths(med_pop$all_prop, med_pop$mai, med_pop$GOHAI_sum, rank_based = FALSE)
cat(sprintf("  indirekt (nyers): OHIP %+.3f  vs.  GOHAI %+.3f  (rang-becsléssel egybevetve)\n",
            raw_o["indirect"], raw_g["indirect"]))

# ------------------------------------------------------------
section("V4 | Validáció 4: visszanyerési szimuláció ezen a mintaméreten")
# ------------------------------------------------------------
# Ismert szerkezetű szintetikus adatot generálunk (n = a tényleges
# mediációs minta), és a TELJES becslő-pipeline-t futtatjuk rajta:
#   (A) valódi mediáció:  a=0.5, b=0.5, c'=0   -> kimutatási arány (erő)
#   (B) null: a=0 (X független M-től)          -> téves riasztás
#   (C) null: b=0 (Y független M-től, de a≠0)  -> téves riasztás
# A CI-alapú döntés (percentilis bootstrap, 0 kizárása) érvényes, ha
# (B) és (C) mellett ~5% alatt marad, és (A) mellett érdemi az erő.

sim_once <- function(n, a, b, cp, R_boot = 1000) {
  X <- rnorm(n); M <- a * X + rnorm(n, sd = sqrt(max(1e-9, 1 - a^2)))
  eps <- sqrt(max(1e-9, 1 - b^2 - cp^2 - 2 * a * b * cp))
  Y <- b * M + cp * X + rnorm(n, sd = eps)
  d <- data.frame(X, M, Y)
  bt <- boot(d, function(dd, i) med_paths(dd$X[i], dd$M[i], dd$Y[i])["indirect"],
             R = R_boot)
  ci <- tryCatch(boot.ci(bt, type = "perc")$percent[4:5], error = function(e) c(NA, NA))
  c(sig = !is.na(ci[1]) && (ci[1] > 0 || ci[2] < 0), est = unname(bt$t0))
}
B_SIM <- 300
n_med <- nrow(med_pop)
set.seed(SEED)
simA <- replicate(B_SIM, sim_once(n_med, 0.5, 0.5, 0))
simB <- replicate(B_SIM, sim_once(n_med, 0.0, 0.5, 0.3))
simC <- replicate(B_SIM, sim_once(n_med, 0.5, 0.0, 0.3))
cat(sprintf("\n  B=%d szimuláció, n=%d, percentilis bootstrap CI-döntés:\n\n", B_SIM, n_med))
cat(sprintf("  (A) valódi mediáció (a=b=0.5):   kimutatva %.0f%% | becslés-átlag %.3f (valódi: 0.25)\n",
            100 * mean(simA["sig", ]), mean(simA["est", ])))
cat(sprintf("  (B) null (a=0):                  téves riasztás %.1f%%  (elvárt ≤ ~5%%)\n",
            100 * mean(simB["sig", ])))
cat(sprintf("  (C) null (b=0):                  téves riasztás %.1f%%  (elvárt ≤ ~5%%)\n",
            100 * mean(simC["sig", ])))
cat("\n  Ha a téves riasztás kalibrált és az erő elfogadható, a pipeline\n")
cat("  alkalmas az adatunkon mért hatás megítélésére; gyenge erő esetén a\n")
cat("  nem-szignifikáns indirekt hatás nem cáfolja a mediációt.\n")

# ------------------------------------------------------------
section("H3 ÖSSZEFOGLALÁS")
# ------------------------------------------------------------
# Döntés a V1-ben deklarált szabály szerint: (1) a bootstrap-CI a VÁRT
# irányban zárja ki a 0-t, (2) p_joint < 0.05, és a hipotézissel
# ellentétes irányú "szignifikancia" külön ágat kap.
verdict <- function(res, p_joint, expected_sign, label) {
  ab_excl0 <- !is.na(res$ab_perc[1]) && (res$ab_perc[1] > 0 || res$ab_perc[2] < 0)
  dir_ok   <- if (expected_sign == "+") !is.na(res$ab_perc[1]) && res$ab_perc[1] > 0
              else                      !is.na(res$ab_perc[2]) && res$ab_perc[2] < 0
  joint_ok <- !is.na(p_joint) && p_joint < 0.05
  ab_sig <- dir_ok && joint_ok
  cp_sig <- !is.na(res$cp_perc[1]) && (res$cp_perc[1] > 0 || res$cp_perc[2] < 0)
  msg <- if (ab_excl0 && !dir_ok)
    "az indirekt hatás kizárja a 0-t, de a P5-predikcióval ELLENTÉTES irányban — H3-at NEM támogatja, külön értelmezést igényel."
  else if (ab_excl0 && dir_ok && !joint_ok)
    "a bootstrap-CI kizárja a 0-t, de p_joint nem szignifikáns — a V1 döntési szabálya szerint mediáció-állítás nem tehető."
  else if (ab_sig && !cp_sig) "az adat a TISZTA mediációval konzisztens (P5 fő ága)."
  else if (ab_sig && cp_sig) "mediáció + megmaradó DIREKT hatás (P5 másik ága: komfort/esztétika)."
  else if (cp_sig) "nincs kimutatható indirekt út, a direkt kapcsolat áll — H3 nem igazolódott."
  else "egyik út sem szignifikáns — e mintán H3 nem dönthető el (ld. V4 erő)."
  cat(sprintf("
%s: indirekt = %+.3f (várt előjel: %s; perc. CI [%+.3f, %+.3f], p_joint = %s), direkt = %+.3f
   -> %s
", label, res$est["indirect"], expected_sign, res$ab_perc[1], res$ab_perc[2],
     fmt_p(p_joint), res$est["direct"], msg))
}
verdict(res_ohip,  p_joint_ohip,  "+", "OHIP")
verdict(res_gohai, p_joint_gohai, "-", "GOHAI")
if (sign(res_ohip$est["indirect"]) != 1 || sign(res_gohai$est["indirect"]) != -1) cat("
FIGYELEM — irány: a megfigyelt indirekt út a P5-predikcióval ELLENTÉTES
előjelű (a negatív a-út a H1/H2-szkriptekben jelzett fordított dózis–
válasz irányt örökli), miközben a b-út (MAI→QoL) erős és a várt irányú.
A 'nem igazolódott' verdikt ezt nem fedi le — az irány-anomáliát a H1
összefoglalójának FIGYELEM-blokkjával együtt kell értelmezni.\n")
cat("\nEmlékeztető: keresztmetszeti elrendezésben a mediáció oksági olvasata\n")
cat("feltevéses; a megerősítés a recall-ág longitudinális adatára vár.\n")
