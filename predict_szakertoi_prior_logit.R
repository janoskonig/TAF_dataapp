#!/usr/bin/env Rscript
# =============================================================================
# PREDICT – szakértői elicitáció → logit-skálájú priorok (v2 kérdőív)
# -----------------------------------------------------------------------------
# A kérdőív (expert_priors.py, FORM_VERSION v2.0) tételenként a következőket
# kérdezi: irány (A / B rosszabb, közepes a legjobb, nincs különbség, nem tudom),
# irány-bizonyosság (p_irany, hitfok), pólusonként „100 betegből hány sikeres”
# legvalószínűbb érték + alsó/felső határ (húsz becslésből tizenkilenc ≈ 95 %),
# az optimum alakú tételeknél (F2, F6) a közepes forgatókönyv is, és a
# „nagyságát nem tudom megbecsülni” jelölés.
#
# Ez a szkript a sikerarányokból közvetlenül a modellparaméter (a sikertelenség
# esélyhányadosának logaritmusa a B pólus és az A pólus között) priorját állítja
# elő szakértőnként, majd egyenlő súlyú lineáris véleménykeveréssel egyesíti:
#   * irányos válasz pontbecsléssel és tartománnyal: pólusonként béta-eloszlás
#     (momentum-illesztés: várható érték = legvalószínűbb szám, szórás =
#     (felső − alsó) / 3,92), β = logit(1 − p_B) − logit(1 − p_A) Monte-Carlóval;
#     β > 0: a B (az elődök szerint kedvezőtlen) pólus mellett nagyobb a
#     sikertelenség esélye, azaz a válasz egyezik az elődökkel.
#   * „nagyságát nem tudom” vagy hiányzó számok: csak irány-prior — előjel a
#     p_irany hitfokból, |β| ~ félnormális(0; 1) (OR nagyjából 1 … 5).
#   * „nincs érdemi különbség”: β ~ N(0; 0,15) (OR 90 %-ban 0,78 … 1,28).
#   * „közepes a legjobb”: görbület-paraméter, c = logit(1−p_M) − átlag(logit(1−p_A),
#     logit(1−p_B)) (c < 0: a közepes mellett kisebb a sikertelenség), külön táblában.
#   * „nem tudom megítélni”: kimarad (információhiány, nem vélemény).
# Az állított irány-bizonyosság (p_irany) és a pólusokból implikált P(β > 0)
# összevetése kalibrációs jelzés (nem kettős beszámítás).
# Kimenetek: stat_output/szakertoi_prior_logit_R/ (táblák, ábrák, összefoglaló).
# Környezet: PREDICT_EXPERT_CSV (app export), PREDICT_EXPERT_HATTER_CSV,
#   PREDICT_INCLUDE_SIM=1 (szimulált sorok), PREDICT_INCLUDE_PI=0 (PI-sorok ki),
#   PREDICT_EXPERT_ROLE (fogorvos | fogtechnikus | all; alap: fogorvos),
#   PREDICT_OUT_SUBDIR, PREDICT_SIM_N (a demó betegszáma, alap 150).
# =============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(tibble); library(ggplot2); library(scales)
})
set.seed(20260906)
ROOT <- if (nzchar(Sys.getenv("PREDICT_ROOT"))) Sys.getenv("PREDICT_ROOT") else getwd()
OUT_DIR <- file.path(ROOT, "stat_output", Sys.getenv("PREDICT_OUT_SUBDIR", unset = "szakertoi_prior_logit_R"))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
N_MC <- 4000L
INCLUDE_PI <- Sys.getenv("PREDICT_INCLUDE_PI", unset = "1") != "0"
INCLUDE_SIM <- Sys.getenv("PREDICT_INCLUDE_SIM", unset = "0") == "1"
EXPERT_ROLE <- Sys.getenv("PREDICT_EXPERT_ROLE", unset = "fogorvos")
PI_IDS <- c("VV-2026-08-19")
PAL <- list(blue = "#2a78d6", orange = "#eb6834", aqua = "#1baf7a", red = "#e34948", gray = "#8a8f98",
            ink = "#1f2933", ink2 = "#52514e", grid = "#e6e8eb", surface = "#ffffff")
theme_p <- function(base = 10) theme_minimal(base_size = base) + theme(
  plot.background = element_rect(fill = PAL$surface, colour = NA), panel.grid.minor = element_blank(),
  panel.grid.major = element_line(colour = PAL$grid, linewidth = 0.4), plot.title = element_text(face = "bold", colour = PAL$ink),
  plot.subtitle = element_text(colour = PAL$ink2), plot.caption = element_text(colour = PAL$ink2, hjust = 0), legend.position = "bottom")
save_png <- function(name, plot, w, h) {
  path <- file.path(OUT_DIR, name)
  if (requireNamespace("ragg", quietly = TRUE)) ragg::agg_png(path, width = w, height = h, units = "cm", res = 200) else png(path, width = w, height = h, units = "cm", res = 200)
  print(plot); dev.off(); invisible(path)
}
write_csv_utf8 <- function(df, name) { con <- file(file.path(OUT_DIR, name), open = "w", encoding = "UTF-8"); writeLines("﻿", con, sep = ""); write.csv(df, con, row.names = FALSE, na = ""); close(con) }
read_csv_utf8 <- function(path) { df <- read.csv(path, check.names = FALSE, encoding = "UTF-8", na.strings = c("", "NA")); names(df) <- sub("^﻿", "", names(df)); df }

# --- Tételregiszter (azonos az applikáció ITEMS / POLE_VALUES regiszterével) ---------
items <- tribble(
  ~tetel, ~cimke, ~csoport, ~polus_A, ~polus_B, ~polus_M, ~egyseg,
  "F1",  "F1 felső gerincmagasság",      "Felső állcsont", 10,  5,   NA, "mm",
  "F2",  "F2 alámenősség (optimum)",     "Felső állcsont", NA,  NA,  NA, "",
  "F3",  "F3 szájpadboltozat",           "Felső állcsont", 25,  17,  NA, "mm",
  "F4",  "F4 gerincív szöge",            "Felső állcsont", 140, 125, NA, "°",
  "F5",  "F5 lötyögő gerinc",            "Felső állcsont", NA,  NA,  NA, "",
  "F6",  "F6 gerincélek eltérése (optimum)", "Felső állcsont", 2.5, 20, 10, "° eltérés 90°-tól",
  "F7",  "F7 torus palatinus",           "Felső állcsont", NA,  NA,  NA, "",
  "F8",  "F8 antagonista",               "Felső állcsont", NA,  NA,  NA, "",
  "A1",  "A1 alsó gerinc (Kaán)",        "Alsó állcsont",  NA,  NA,  NA, "",
  "A3",  "A3 buccinator-tasak",          "Alsó állcsont",  NA,  NA,  NA, "",
  "A4",  "A4 torus mandibularis",        "Alsó állcsont",  NA,  NA,  NA, "",
  "A5",  "A5 lingualis tasak",           "Alsó állcsont",  NA,  NA,  NA, "",
  "TUB", "A6–A9 tuberculum",             "Alsó állcsont",  NA,  NA,  NA, "",
  "A10", "A10 sagittális reláció",       "Alsó állcsont",  NA,  NA,  NA, "",
  "A11", "A11 szájfenék",                "Alsó állcsont",  NA,  NA,  NA, "",
  "A12", "A12 spinae mentales",          "Alsó állcsont",  NA,  NA,  NA, "",
)

# --- Bemenet ---------------------------------------------------------------------------
sources <- c(file.path(ROOT, "predict_expert_priorok.csv"), Sys.getenv("PREDICT_EXPERT_CSV"),
             if (INCLUDE_SIM) file.path(ROOT, "predict_expert_priorok_SZIMULACIO.csv"))
bg_sources <- c(file.path(ROOT, "predict_expert_hatter.csv"), Sys.getenv("PREDICT_EXPERT_HATTER_CSV"),
                if (INCLUDE_SIM) file.path(ROOT, "predict_expert_hatter_SZIMULACIO.csv"))
read_many <- function(paths) bind_rows(lapply(unique(paths[nzchar(paths) & file.exists(paths)]), function(p) { d <- read_csv_utf8(p); d$forras <- basename(p); d }))
raw <- read_many(sources)
stopifnot(nrow(raw) > 0)
bg <- read_many(bg_sources)
num_cols <- c("p_irany", "siker_A", "siker_B", "siker_M", "siker_A_min", "siker_A_max", "siker_B_min", "siker_B_max", "siker_M_min", "siker_M_max", "nagysag_nem_tudom")
for (cc in num_cols) raw[[cc]] <- if (cc %in% names(raw)) suppressWarnings(as.numeric(raw[[cc]])) else NA_real_
if (!"szerep" %in% names(raw)) raw$szerep <- "fogorvos"
raw$szerep[is.na(raw$szerep) | raw$szerep == ""] <- "fogorvos"
if (!INCLUDE_PI) raw <- raw[!raw$szakerto_id %in% PI_IDS, ]
raw <- raw[raw$tetel %in% items$tetel, ]

# --- Egy sor → β-minta -----------------------------------------------------------------
clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)
beta_from <- function(point, lo, hi) {
  m <- clamp(point / 100, 0.01, 0.99)
  s <- if (is.finite(lo) && is.finite(hi) && hi > lo) (hi - lo) / 100 / 3.92 else NA_real_
  if (!is.finite(s)) s <- 0.08                          # tartomány nélkül: mérsékelt helyőrző szórás
  s <- min(s, sqrt(m * (1 - m)) * 0.95)                 # béta-eloszlással elérhető szórás
  k <- m * (1 - m) / s^2 - 1
  c(a = max(m * k, 0.5), b = max((1 - m) * k, 0.5))
}
logit <- function(p) log(p / (1 - p))
row_samples <- function(r) {
  irany <- as.character(r$irany)
  if (is.na(irany) || irany %in% c("", "nem_tudom")) return(NULL)
  p_dir <- if (is.finite(r$p_irany)) clamp(r$p_irany / 100, 0.5, 0.995) else 0.5
  p_pos <- switch(irany, B_kedvezotlenebb = p_dir, A_kedvezotlenebb = 1 - p_dir, 0.5)
  unknown <- isTRUE(r$nagysag_nem_tudom == 1)
  has_pts <- is.finite(r$siker_A) && is.finite(r$siker_B)
  if (irany == "nincs_kulonbseg") {
    centre <- if (has_pts) logit(1 - clamp(r$siker_B / 100, .01, .99)) - logit(1 - clamp(r$siker_A / 100, .01, .99)) else 0
    return(list(tipus = "nincs érdemi különbség", beta = rnorm(N_MC, centre, 0.15), p_pos_implikalt = NA_real_, p_pos_allitott = NA_real_))
  }
  if (irany == "nem_monoton") {
    if (unknown || !has_pts || !is.finite(r$siker_M)) return(list(tipus = "görbület (nagyság nem becsült)", beta = NULL, gorbulet = -abs(rnorm(N_MC, 0, 0.5)), p_pos_implikalt = NA_real_, p_pos_allitott = NA_real_))
    pa <- rbeta(N_MC, beta_from(r$siker_A, r$siker_A_min, r$siker_A_max)[1], beta_from(r$siker_A, r$siker_A_min, r$siker_A_max)[2])
    pb <- rbeta(N_MC, beta_from(r$siker_B, r$siker_B_min, r$siker_B_max)[1], beta_from(r$siker_B, r$siker_B_min, r$siker_B_max)[2])
    pm <- rbeta(N_MC, beta_from(r$siker_M, r$siker_M_min, r$siker_M_max)[1], beta_from(r$siker_M, r$siker_M_min, r$siker_M_max)[2])
    lin <- logit(1 - pb) - logit(1 - pa)
    curv <- logit(1 - pm) - (logit(1 - pa) + logit(1 - pb)) / 2
    return(list(tipus = "görbület (három forgatókönyv)", beta = lin, gorbulet = curv, p_pos_implikalt = mean(lin > 0), p_pos_allitott = NA_real_))
  }
  # irányos válasz
  if (unknown || !has_pts) {
    sign <- ifelse(runif(N_MC) < p_pos, 1, -1)
    return(list(tipus = "csak irány (félnormális nagyság)", beta = sign * abs(rnorm(N_MC, 0, 1)), p_pos_implikalt = p_pos, p_pos_allitott = p_pos))
  }
  A <- beta_from(r$siker_A, r$siker_A_min, r$siker_A_max); B <- beta_from(r$siker_B, r$siker_B_min, r$siker_B_max)
  pa <- rbeta(N_MC, A[1], A[2]); pb <- rbeta(N_MC, B[1], B[2])
  beta <- logit(1 - pb) - logit(1 - pa)
  tipus <- if (is.finite(r$siker_A_min) && is.finite(r$siker_B_min)) "pólusok tartománnyal" else "pólusok tartomány nélkül"
  list(tipus = tipus, beta = beta, p_pos_implikalt = mean(beta > 0), p_pos_allitott = p_pos)
}

q <- function(x, p) if (length(x)) unname(quantile(x, p, na.rm = TRUE)) else NA_real_
per_expert <- list(); samples <- list(); curvature <- list()
for (i in seq_len(nrow(raw))) {
  r <- raw[i, ]
  sm <- row_samples(r)
  if (is.null(sm)) {
    per_expert[[length(per_expert) + 1]] <- tibble(szakerto_id = r$szakerto_id, szerep = r$szerep, tetel = r$tetel, tipus = "nem tudom megítélni (kimarad)",
                                                  beta_atlag = NA_real_, beta_sd = NA_real_, P_beta_pozitiv = NA_real_, OR_q05 = NA_real_, OR_q50 = NA_real_, OR_q95 = NA_real_,
                                                  P_irany_allitott = NA_real_, P_irany_implikalt = NA_real_, kalibracios_jelzes = "")
    next
  }
  b <- sm$beta
  if (!is.null(b)) {
    samples[[length(samples) + 1]] <- tibble(szakerto_id = r$szakerto_id, szerep = r$szerep, tetel = r$tetel, beta = b)
  }
  if (!is.null(sm$gorbulet)) curvature[[length(curvature) + 1]] <- tibble(szakerto_id = r$szakerto_id, szerep = r$szerep, tetel = r$tetel, gorbulet = sm$gorbulet)
  flag <- if (is.finite(sm$p_pos_allitott) && is.finite(sm$p_pos_implikalt) && abs(sm$p_pos_allitott - sm$p_pos_implikalt) > 0.25) "az állított és a számokból implikált irány-bizonyosság eltér" else ""
  per_expert[[length(per_expert) + 1]] <- tibble(
    szakerto_id = r$szakerto_id, szerep = r$szerep, tetel = r$tetel, tipus = sm$tipus,
    beta_atlag = if (is.null(b)) NA_real_ else mean(b), beta_sd = if (is.null(b)) NA_real_ else sd(b),
    P_beta_pozitiv = if (is.null(b)) NA_real_ else mean(b > 0),
    OR_q05 = exp(q(b, .05)), OR_q50 = exp(q(b, .5)), OR_q95 = exp(q(b, .95)),
    P_irany_allitott = sm$p_pos_allitott, P_irany_implikalt = sm$p_pos_implikalt, kalibracios_jelzes = flag)
}
per_expert <- bind_rows(per_expert) |> left_join(items |> select(tetel, cimke, csoport), by = "tetel")
write_csv_utf8(per_expert, "01_szakertonkenti_prior_logit.csv")
samples <- bind_rows(samples)
curvature <- bind_rows(curvature)

# --- Egyesített (lineáris pool) prior tételenként, a kiválasztott szerep(ek)ből ---------
pool_samples <- if (EXPERT_ROLE == "all") samples else samples |> filter(szerep == EXPERT_ROLE)
pooled <- pool_samples |>
  group_by(tetel) |>
  summarise(n_szakerto = n_distinct(szakerto_id), beta_atlag = mean(beta), beta_sd = sd(beta), P_beta_pozitiv = mean(beta > 0),
            OR_q05 = exp(quantile(beta, .05)), OR_q25 = exp(quantile(beta, .25)), OR_q50 = exp(quantile(beta, .5)),
            OR_q75 = exp(quantile(beta, .75)), OR_q95 = exp(quantile(beta, .95)), .groups = "drop") |>
  left_join(items, by = "tetel") |>
  mutate(beta_egysegenkent = ifelse(is.finite(polus_A) & is.finite(polus_B), beta_atlag / abs(polus_B - polus_A), NA_real_),
         beta_sd_egysegenkent = ifelse(is.finite(polus_A) & is.finite(polus_B), beta_sd / abs(polus_B - polus_A), NA_real_))
counts <- per_expert |> (\(d) if (EXPERT_ROLE == "all") d else d |> filter(szerep == EXPERT_ROLE))() |>
  group_by(tetel) |>
  summarise(n_valasz = n(), n_nem_tudom = sum(tipus == "nem tudom megítélni (kimarad)"), n_csak_irany = sum(grepl("csak irány", tipus)),
            n_nincs_kulonbseg = sum(tipus == "nincs érdemi különbség"), n_gorbulet = sum(grepl("görbület", tipus)), .groups = "drop")
pooled <- items |> select(tetel, cimke, csoport) |> left_join(pooled |> select(-cimke, -csoport), by = "tetel") |> left_join(counts, by = "tetel") |>
  mutate(szerep_pool = EXPERT_ROLE)
write_csv_utf8(pooled, "02_egyesitett_prior_logit.csv")
if (nrow(curvature)) {
  curv_tab <- curvature |> group_by(tetel, szerep) |> summarise(n_szakerto = n_distinct(szakerto_id), gorbulet_atlag = mean(gorbulet), gorbulet_sd = sd(gorbulet), P_kozepes_jobb = mean(gorbulet < 0), .groups = "drop")
  write_csv_utf8(curv_tab, "02b_gorbulet_prior.csv")
}

# --- Prior prediktív ellenőrzés: implikált sikerarány vs. B1 alapráta ------------------
base_rate <- if (nrow(bg) && "alap_siker_100" %in% names(bg)) suppressWarnings(as.numeric(bg$alap_siker_100)) else numeric(0)
implied <- raw |> filter(is.finite(siker_A), is.finite(siker_B), !(nagysag_nem_tudom %in% 1)) |>
  mutate(implikalt_50_50 = (siker_A + siker_B) / 2) |>
  group_by(tetel) |> summarise(n = n(), implikalt_sikerarany = mean(implikalt_50_50), .groups = "drop") |>
  mutate(B1_alaprata_atlag = if (length(base_rate)) mean(base_rate, na.rm = TRUE) else NA_real_,
         elteres = implikalt_sikerarany - B1_alaprata_atlag) |>
  left_join(items |> select(tetel, cimke), by = "tetel")
write_csv_utf8(implied, "03_prior_prediktiv.csv")

# --- Szerep szerinti bontás -------------------------------------------------------------
by_role <- samples |> group_by(tetel, szerep) |>
  summarise(n_szakerto = n_distinct(szakerto_id), beta_atlag = mean(beta), P_beta_pozitiv = mean(beta > 0), OR_q05 = exp(quantile(beta, .05)), OR_q50 = exp(quantile(beta, .5)), OR_q95 = exp(quantile(beta, .95)), .groups = "drop") |>
  left_join(items |> select(tetel, cimke, csoport), by = "tetel")
write_csv_utf8(by_role, "04_szerep_szerint.csv")

# --- 1. ábra: egyesített prior esélyhányados-skálán, szakértőnkénti pontokkal -----------
# Sorrend: a regiszter sorrendje (felső, majd alsó állcsont); a tengelyfelirat hordozza az n-t és a P(B rosszabb)-ot.
group_levels <- c("Felső állcsont", "Alsó állcsont")
pool_labels <- pooled |> mutate(cimke2 = ifelse(is.finite(P_beta_pozitiv), paste0(cimke, "\nn = ", n_szakerto, " · P(B rosszabb) = ", percent(P_beta_pozitiv, 1)), cimke))
lab_levels <- rev(pool_labels$cimke2[match(items$tetel, pool_labels$tetel)])
plot_pool <- pool_labels |> filter(is.finite(OR_q50)) |> mutate(cimke2 = factor(cimke2, levels = lab_levels), csoport = factor(csoport, levels = group_levels))
plot_ind <- per_expert |> filter(is.finite(OR_q50), if (EXPERT_ROLE == "all") TRUE else szerep == EXPERT_ROLE) |>
  left_join(pool_labels |> select(tetel, cimke2), by = "tetel") |> mutate(cimke2 = factor(cimke2, levels = lab_levels), csoport = factor(csoport, levels = group_levels))
wrap <- function(x, w = 150) paste(strwrap(x, w), collapse = "\n")
if (nrow(plot_pool)) {
  p1 <- ggplot() +
    geom_vline(xintercept = 1, colour = PAL$ink2, linewidth = 0.5) +
    geom_segment(data = plot_pool, aes(x = OR_q05, xend = OR_q95, y = cimke2, yend = cimke2), colour = PAL$blue, linewidth = 1.6, alpha = 0.35) +
    geom_segment(data = plot_pool, aes(x = OR_q25, xend = OR_q75, y = cimke2, yend = cimke2), colour = PAL$blue, linewidth = 3) +
    geom_point(data = plot_ind, aes(x = OR_q50, y = cimke2, shape = tipus), colour = PAL$orange, size = 2.2, alpha = 0.8, position = position_jitter(height = 0.18, width = 0, seed = 1)) +
    geom_point(data = plot_pool, aes(x = OR_q50, y = cimke2), colour = PAL$ink, fill = PAL$surface, shape = 21, size = 3, stroke = 1.2) +
    facet_grid(csoport ~ ., scales = "free_y", space = "free_y") +
    scale_x_log10(breaks = c(0.2, 0.5, 1, 2, 5, 10, 20), labels = c("0,2", "0,5", "1", "2", "5", "10", "20")) +
    scale_shape_manual(values = c(16, 17, 15, 18, 8, 4), name = "szakértőnkénti válasz típusa") +
    labs(title = "Szakértői prior a sikertelenség esélyhányadosára: B pólus az A-hoz képest",
         subtitle = wrap(paste0("Egyenlő súlyú keverék (", if (EXPERT_ROLE == "all") "fogorvosok és fogtechnikusok" else EXPERT_ROLE, "). Kék sáv: 5–95 % és 25–75 %, karika: medián. Narancs jel: egy-egy szakértő mediánja."), 110),
         x = "esélyhányados (OR), logaritmikus skála · OR > 1: a B változat mellett több a sikertelen fogsor", y = NULL,
         caption = wrap("β = logit(1 − p_B) − logit(1 − p_A); pólusonként béta-eloszlás a legvalószínűbb számból és a 19/20-os határokból. A „nem tudom megítélni” válaszok nem szerepelnek; a „nincs érdemi különbség” szűk, 1 körüli eloszlás.", 130)) +
    theme_p(10) + theme(strip.text.y = element_text(angle = 0), legend.direction = "vertical", axis.text.y = element_text(size = 8, lineheight = 0.9))
  save_png("abra_01_egyesitett_prior_OR.png", p1, 22, 17)
}

# --- 2. ábra: fogorvosok vs. fogtechnikusok ----------------------------------------------
if (n_distinct(by_role$szerep) >= 2) {
  p2 <- by_role |> mutate(cimke = factor(cimke, levels = rev(items$cimke)), csoport = factor(csoport, levels = group_levels), szerep = factor(szerep, levels = c("fogorvos", "fogtechnikus"))) |>
    ggplot(aes(y = cimke, colour = szerep)) +
    geom_vline(xintercept = 1, colour = PAL$ink2, linewidth = 0.5) +
    geom_linerange(aes(xmin = OR_q05, xmax = OR_q95, group = szerep), position = position_dodge(width = 0.6), linewidth = 1.2, alpha = 0.6) +
    geom_point(aes(x = OR_q50, shape = szerep, group = szerep), position = position_dodge(width = 0.6), size = 2.8) +
    facet_grid(csoport ~ ., scales = "free_y", space = "free_y") +
    scale_x_log10(breaks = c(0.2, 0.5, 1, 2, 5, 10, 20), labels = c("0,2", "0,5", "1", "2", "5", "10", "20")) +
    scale_colour_manual(values = c(PAL$blue, PAL$orange), name = NULL) + scale_shape_manual(values = c(16, 17), name = NULL) +
    labs(title = "Fogorvosok és fogtechnikusok priorja egymás mellett", x = "esélyhányados (OR), logaritmikus skála", y = NULL,
         caption = "Szerepenként külön egyesített keverék; 5–95 %-os sáv és medián.") +
    theme_p(10) + theme(strip.text.y = element_text(angle = 0))
  save_png("abra_02_fogorvos_vs_fogtechnikus.png", p2, 18, 13)
}

# --- 3. ábra: prior prediktív (implikált sikerarány vs. alapráta) -------------------------
if (nrow(implied)) {
  p3 <- implied |> mutate(cimke = factor(cimke, levels = rev(items$cimke))) |>
    ggplot(aes(y = cimke)) +
    { if (is.finite(implied$B1_alaprata_atlag[1])) geom_vline(xintercept = implied$B1_alaprata_atlag[1], colour = PAL$orange, linewidth = 0.8, linetype = "dashed") } +
    geom_point(aes(x = implikalt_sikerarany), colour = PAL$blue, size = 3) +
    scale_x_continuous(limits = c(0, 100)) +
    labs(title = "Prior prediktív ellenőrzés: a pólusokból következő átlagos sikerarány",
         subtitle = "Pont: az A és B pólus sikerarányának átlaga (50/50 keverék); szaggatott: a szakértők B1 alaprátája (100 betegből hány sikeres)",
         x = "sikeres fogsor 100 betegből", y = NULL,
         caption = "Nagy, egyirányú eltérés a pont és a vonal között azt jelzi, hogy a szakértők a pólusokat nem az átlagos beteghez viszonyítva számolták.") +
    theme_p(10)
  save_png("abra_03_prior_prediktiv.png", p3, 18, 12)
}

# --- Demó szimulált adaton: Bayes-i logisztikus regresszió az egyesített priorral ---------
demo <- NULL
if (requireNamespace("MCMCpack", quietly = TRUE) && nrow(plot_pool) >= 3) {
  n_sim <- as.integer(Sys.getenv("PREDICT_SIM_N", unset = "150"))
  use <- plot_pool |> filter(is.finite(beta_sd), beta_sd > 0) |> arrange(tetel) |> slice_head(n = 8)
  K <- nrow(use)
  X <- matrix(rbinom(n_sim * K, 1, 0.4), n_sim, K); colnames(X) <- use$tetel   # B pólus jelenléte (0/1)
  true_beta <- rnorm(K, use$beta_atlag, use$beta_sd)
  eta <- qlogis(0.3) + X %*% true_beta                                     # sikertelenség log-esélye (alapráta 30 %)
  y <- rbinom(n_sim, 1, plogis(eta))
  dat <- data.frame(y = y, X)
  form <- as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))
  b0 <- c(0, use$beta_atlag); B0 <- diag(c(1e-4, 1 / use$beta_sd^2))
  fit_prior <- MCMCpack::MCMClogit(form, data = dat, burnin = 1000, mcmc = 6000, thin = 3, b0 = b0, B0 = B0, verbose = 0)
  fit_flat <- MCMCpack::MCMClogit(form, data = dat, burnin = 1000, mcmc = 6000, thin = 3, b0 = 0, B0 = 1e-4, verbose = 0)
  summ <- function(fit, label) {
    m <- as.matrix(fit)[, -1, drop = FALSE]
    tibble(tetel = colnames(m), modell = label, post_atlag = colMeans(m), post_q05 = apply(m, 2, quantile, .05), post_q95 = apply(m, 2, quantile, .95))
  }
  demo <- bind_rows(summ(fit_prior, "szakértői prior"), summ(fit_flat, "lapos prior")) |>
    left_join(tibble(tetel = use$tetel, valodi_beta = true_beta, prior_atlag = use$beta_atlag, prior_q05 = use$beta_atlag - 1.645 * use$beta_sd, prior_q95 = use$beta_atlag + 1.645 * use$beta_sd), by = "tetel") |>
    mutate(n_beteg = n_sim)
  write_csv_utf8(demo, "05_szimulalt_modell_demo.csv")
  p4 <- demo |> mutate(tetel = factor(tetel, levels = use$tetel)) |>
    ggplot(aes(y = tetel)) +
    geom_vline(xintercept = 0, colour = PAL$ink2, linewidth = 0.5) +
    geom_linerange(aes(xmin = prior_q05, xmax = prior_q95), colour = PAL$gray, linewidth = 3, alpha = 0.5) +
    geom_linerange(aes(xmin = post_q05, xmax = post_q95, colour = modell, group = modell), position = position_dodge(width = 0.5), linewidth = 1.3) +
    geom_point(aes(x = post_atlag, colour = modell, group = modell), position = position_dodge(width = 0.5), size = 2.5) +
    geom_point(aes(x = valodi_beta), shape = 4, size = 3, stroke = 1.2, colour = PAL$red) +
    scale_colour_manual(values = c(PAL$blue, PAL$orange), name = NULL) +
    labs(title = paste0("Demó szimulált adaton (n = ", n_sim, "): a szakértői prior hatása a posteriorra"),
         subtitle = wrap("Szürke: egyesített prior 90 %-os sávja; színes: posterior 90 %-os sávja; piros ×: a szimulációban használt valódi együttható", 110),
         x = "β (log-esélyhányados, sikertelenség)", y = NULL,
         caption = wrap("A modell: Bayes-i logisztikus regresszió (MCMCpack::MCMClogit), normális priorral az egyesített keverék átlagából és szórásából. Csak a folyamat bemutatására; nem valódi beteg-adat.", 120)) +
    theme_p(10)
  save_png("abra_04_szimulalt_posterior.png", p4, 18, 11)
}

# --- Összefoglaló -------------------------------------------------------------------------
md <- c(
  "# Szakértői elicitáció → logit-priorok (v2 kérdőív)", "",
  paste0("Forrás(ok): ", paste(unique(raw$forras), collapse = "; "), "; szakértők: ", n_distinct(raw$szakerto_id),
         " (", paste(names(table(distinct(raw, szakerto_id, szerep)$szerep)), table(distinct(raw, szakerto_id, szerep)$szerep), sep = ": ", collapse = ", "), ")"),
  paste0("Pool: ", if (EXPERT_ROLE == "all") "minden szerep együtt" else paste0("csak ", EXPERT_ROLE), "; N_MC = ", N_MC, " minta / szakértő / tétel."), "",
  "## Egyesített prior tételenként (OR a sikertelenségre, B vs. A)", "",
  "| tétel | n | csak irány | nem tudom | OR medián | OR 5–95 % | P(B rosszabb) |", "|---|---|---|---|---|---|---|",
  apply(pooled, 1, function(r) sprintf("| %s | %s | %s | %s | %s | %s–%s | %s |", r[["cimke"]], r[["n_szakerto"]], r[["n_csak_irany"]], r[["n_nem_tudom"]],
                                       ifelse(is.na(r[["OR_q50"]]), "–", sprintf("%.2f", as.numeric(r[["OR_q50"]]))),
                                       ifelse(is.na(r[["OR_q05"]]), "–", sprintf("%.2f", as.numeric(r[["OR_q05"]]))), ifelse(is.na(r[["OR_q95"]]), "–", sprintf("%.2f", as.numeric(r[["OR_q95"]]))),
                                       ifelse(is.na(r[["P_beta_pozitiv"]]), "–", sprintf("%.0f %%", 100 * as.numeric(r[["P_beta_pozitiv"]]))))),
  "", "## Kalibrációs jelzések", "",
  if (any(per_expert$kalibracios_jelzes != "")) paste0("- ", per_expert$szakerto_id[per_expert$kalibracios_jelzes != ""], " · ", per_expert$tetel[per_expert$kalibracios_jelzes != ""], ": ", per_expert$kalibracios_jelzes[per_expert$kalibracios_jelzes != ""]) else "- nincs",
  "", "## Fájlok", "",
  "- 01_szakertonkenti_prior_logit.csv · 02_egyesitett_prior_logit.csv · 02b_gorbulet_prior.csv (ha van) · 03_prior_prediktiv.csv · 04_szerep_szerint.csv · 05_szimulalt_modell_demo.csv (ha van MCMCpack)",
  "- abra_01_egyesitett_prior_OR.png · abra_02_fogorvos_vs_fogtechnikus.png (két szerepnél) · abra_03_prior_prediktiv.png · abra_04_szimulalt_posterior.png"
)
writeLines(md, file.path(OUT_DIR, "09_osszefoglalo.md"), useBytes = TRUE)
cat(paste(md, collapse = "\n"), "\n")
cat("\nKimenet:", OUT_DIR, "\n")
