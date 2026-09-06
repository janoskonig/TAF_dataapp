#!/usr/bin/env Rscript

# =============================================================================
# PREDICT – Bayes-i keretű FELTÁRÓ elemzés: az elődök tapasztalata (prior)
# és a hat teljes longitudinális eset (adat)
# =============================================================================
#
# Kérdés
# ------
# Az elődeink tapasztalati alapon jegyezték fel az egyes anatómiai képletek
# mellé, hogy a teljes lemezes fogpótlás sikeressége szempontjából előnyösek
# vagy hátrányosak. Mit mond erről a PREDICT hat tökéletesen teljes
# (kiindulás + utánkövetés) esete – és mennyit képes hat beteg adata
# elmozdítani egy tapasztalati (tankönyvi/szakértői) meggyőződésen?
#
# Tervezési elvek
# ---------------
#   * TELJESEN FELTÁRÓ: nincs p-érték, nincs szignifikanciateszt, nincs
#     többszörösség-korrekció. Minden szám leíró hatásnagyság vagy
#     posterior valószínűség, széles bizonytalansággal.
#   * A Bayes-rész Stan nélkül, RÁCSALAPÚ (grid) egzakt posteriorral készül:
#       – prediktoronként egy egyváltozós, standardizált (normal-score)
#         regresszió; a tengelymetszetet és a szórást Jeffreys-priorral
#         analitikusan integráljuk ki, így a béta marginális likelihoodja
#         zárt alakú: L(β) ∝ (1 − 2βr + β²)^(−(n−1)/2);
#       – prior a [−1; 1] korrelációs skálán: (a) SEMLEGES, szimmetrikus
#         N(0; 0,5²); (b) ELŐDÖK/SZAKÉRTŐI: keverék-prior, amely az
#         iránybizonyosságot (P[várt irány]) és a hatásnagyságot külön
#         kezeli — a `predict_expert_priorok.csv` sorai alapján, szakértők
#         közti lineáris véleménykeveréssel (linear opinion pool).
#   * Iránykonvenció: x = kockázatirányított anatómia (nagyobb = az elődök
#     szerint kedvezőtlenebb), y = ROSSZABB kimenet (nagyobb = rosszabb), így
#     β > 0 ⇔ „a kedvezőtlen anatómia rosszabb eredménnyel jár” ⇔ egyezik az
#     elődök tapasztalatával.
#   * „Siker” = az ÚJ fogsorral elért állapot az utánkövetéskor (OHIP-5,
#     GOHAI, MAI, önbevallott rágóképesség, valamint ezek z-átlagából képzett
#     siker-index); a kiindulás→utánkövetés VÁLTOZÁS másodlagos nézet, mert a
#     kiindulási rágásteszt a régi fogpótlással készült.
#   * A lemorzsolódás leírásához a szkript csak olvasásra kapcsolódik az
#     adatbázishoz; ha az nem elérhető, ezt a blokkot kihagyja.
#
# Bemenetek
# ---------
#   analíziiiis/predict_elemzes_adatok_TISZTITOTT.csv  (a hat teljes eset)
#   predict_expert_priorok.csv                          (szakértői/tankönyvi
#       iránypriorok; jelenleg a vizsgálatvezető 2026-08-19-i kvalitatív
#       elicitációja — a hatásnagyság-oszlopok üresek → HN(0; 0,5) helyőrző)
#   .env → DATABASE_URL                                  (opcionális)
#
# Futtatás a projekt gyökeréből:  Rscript predict_bayes_feltaro.R
# Kimenetek: stat_output/bayes_feltaro_R/
# =============================================================================

options(stringsAsFactors = FALSE, scipen = 999, warn = 1)
if (!isTRUE(l10n_info()$`UTF-8`)) {
  invisible(suppressWarnings(Sys.setlocale("LC_ALL", "en_US.UTF-8")))
}

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0L) return(normalizePath(getwd(), mustWork = TRUE))
  normalizePath(dirname(sub("^--file=", "", file_arg[[1]])), mustWork = TRUE)
}
ROOT <- get_script_dir()
# Környezeti kapcsolók (szimulációhoz / érzékenységi futtatáshoz):
#   PREDICT_OUT_SUBDIR   kimeneti almappa a stat_output alatt (alap: bayes_feltaro_R)
#   PREDICT_EXPERT_CSV   szakértői prior-CSV útvonala (alap: predict_expert_priorok.csv)
#   PREDICT_EXPERT_DB    "0" → az adatbázis szakértői tábláját nem olvassa
#   PREDICT_INCLUDE_PI   "0" → a vizsgálatvezetői (VV-…) sorok kimaradnak a poolból
#   PREDICT_INCLUDE_SIM  "1" → a szimulált próbasorok (form_version v1.0-SZIMULACIO) is beszámítanak
OUT_DIR <- file.path(ROOT, "stat_output", Sys.getenv("PREDICT_OUT_SUBDIR", unset = "bayes_feltaro_R"))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
invisible(file.remove(list.files(OUT_DIR, full.names = TRUE)))

required_packages <- c("dplyr", "tidyr", "readr", "ggplot2", "scales", "patchwork", "ggrepel", "ragg")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0L) {
  stop("Hiányzó R-csomag(ok): ", paste(missing_packages, collapse = ", "))
}
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2)
  library(scales); library(patchwork); library(ggrepel)
})

EXPECTED_MAIN_N <- 6L
EXPECTED_IDS <- c(2L, 8L, 9L, 44L, 48L, 53L)

# -----------------------------------------------------------------------------
# 0. Vizuális rendszer (validált, színtévesztés-biztos alappaletta)
# -----------------------------------------------------------------------------
PAL <- list(
  blue = "#2a78d6", orange = "#eb6834", aqua = "#1baf7a",
  red = "#e34948", ink = "#0b0b0b", ink2 = "#52514e", muted = "#898781",
  grid = "#e1e0d9", axis = "#c3c2b7", surface = "#fcfcfb", mid = "#f0efec",
  de_emph = "#c9c8c2"
)
SEQ_BLUE <- c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQ_ORANGE <- c("#fde3d8", "#f9bda5", "#f39a76", "#eb6834", "#c94f21", "#9d3c17", "#6e2a0f")

theme_predict <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background = element_rect(fill = PAL$surface, colour = NA),
      panel.background = element_rect(fill = PAL$surface, colour = NA),
      plot.title = element_text(face = "bold", size = rel(1.18), colour = PAL$ink),
      plot.subtitle = element_text(colour = PAL$ink2, size = rel(0.92)),
      plot.caption = element_text(colour = PAL$muted, hjust = 0, size = rel(0.78)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = PAL$grid, linewidth = 0.4),
      axis.text = element_text(colour = PAL$ink2),
      axis.title = element_text(colour = PAL$ink2),
      strip.text = element_text(face = "bold", colour = PAL$ink, size = rel(0.9)),
      legend.title = element_text(colour = PAL$ink2, size = rel(0.85)),
      legend.text = element_text(colour = PAL$ink2, size = rel(0.85)),
      legend.position = "right",
      plot.title.position = "plot",
      plot.caption.position = "plot"
    )
}

save_png <- function(filename, plot, width, height, dpi = 200) {
  ggsave(file.path(OUT_DIR, filename), plot, device = ragg::agg_png,
         width = width, height = height, dpi = dpi, bg = PAL$surface, limitsize = FALSE)
}
write_csv_utf8 <- function(x, filename) {
  x <- dplyr::mutate(x, dplyr::across(dplyr::where(is.double), ~ round(.x, 4)))
  readr::write_excel_csv(x, file.path(OUT_DIR, filename), na = "")
}
fmt <- function(x, d = 2) ifelse(is.na(x), "–", formatC(x, digits = d, format = "f", decimal.mark = ","))
fmt_pct <- function(x) ifelse(is.na(x), "–", paste0(formatC(100 * x, digits = 0, format = "f"), "%"))

# -----------------------------------------------------------------------------
# 1. A hat teljes eset (tisztított CSV)
# -----------------------------------------------------------------------------
csv_path <- file.path(ROOT, "analíziiiis", "predict_elemzes_adatok_TISZTITOTT.csv")
if (!file.exists(csv_path)) stop("Nem található a tisztított CSV: ", csv_path)
raw6 <- read.csv(csv_path, check.names = FALSE, encoding = "UTF-8")
names(raw6) <- sub("^﻿", "", names(raw6))
if (nrow(raw6) != EXPECTED_MAIN_N) stop("A tisztított CSV ", nrow(raw6), " sort tartalmaz, a várt ", EXPECTED_MAIN_N, " helyett.")
if (!setequal(raw6$patient_id, EXPECTED_IDS)) stop("A tisztított CSV betegazonosítói eltérnek a várt hat teljes esettől.")

chew_levels <- c("Nagyon rossz", "Rossz", "Átlagos", "Jó", "Kiváló")
grc_levels <- c("Sokat romlott", "Kicsit romlott", "Változatlan maradt", "Kicsit javult", "Sokat javult")
to_ord <- function(x, levels) as.integer(factor(trimws(x), levels = levels))

d6 <- raw6 |>
  arrange(patient_id) |>
  mutate(
    study_id = sprintf("P%02d", row_number()),
    birthdate = as.Date(birthdate),
    kor_ev = NA_real_,                       # az adatbázisból pontosítjuk, ha elérhető
    nem = ifelse(gender == "Female", "nő", "férfi"),
    ragas_init = to_ord(chewing_today_init, chew_levels),
    ragas_fu = to_ord(chewing_today_followup, chew_levels),
    egeszseg_init = to_ord(responsiveness_init, chew_levels),
    egeszseg_fu = to_ord(responsiveness_followup, chew_levels),
    grc_ragas = to_ord(chewing_change, grc_levels) - 3L,
    grc_egeszseg = to_ord(responsiveness_change, grc_levels) - 3L,
    # --- siker-orientált ÁLLAPOT az új fogsorral (nagyobb = jobb)
    s_OHIP = -OHIP_sum_followup,
    s_GOHAI = GOHAI_sum_followup,
    s_MAI = -MAI_huedegree_followup,
    s_RAGAS = ragas_fu,
    # --- VÁLTOZÁS (nagyobb = nagyobb javulás)
    v_OHIP = OHIP_sum_init - OHIP_sum_followup,
    v_GOHAI = GOHAI_sum_followup - GOHAI_sum_init,
    v_MAI = MAI_huedegree_init - MAI_huedegree_followup,
    v_GRC = grc_ragas
  )
z <- function(x) (x - mean(x)) / sd(x)
d6$s_INDEX <- rowMeans(cbind(z(d6$s_OHIP), z(d6$s_GOHAI), z(d6$s_MAI), z(d6$s_RAGAS)))

# Kockázatirányított anatómia (nagyobb = az elődök szerint kedvezőtlenebb);
# az irány nélküli tételek nyers kóddal, „irány-semleges” jelöléssel.
a1_sat <- c(`1` = 0, `2` = 0.35, `3` = 0.70, `4` = 0.95, `5` = 1)
# Az A10 szög kategoriális változata (ideiglenes küszöb, a mért 4–12° tartomány felett): a
# kérdőív Angle I-et hasonlít az eltérő relációhoz, a nyers szög lineáris hatása más kérdés.
A10_ANGLE_THRESHOLD <- as.numeric(Sys.getenv("PREDICT_A10_THRESHOLD", unset = "10"))
d6 <- d6 |>
  mutate(
    r_F1 = -F1,
    r_F2 = F2,
    r_F2s = F2_standardizalt,
    r_F3 = -F3,
    r_F4 = -F4,
    r_F5 = as.numeric(as.integer(F5) != 1L),
    r_F6 = abs(F6 - 90),
    r_F7 = as.numeric(as.integer(F7) != 1L),
    r_F8 = as.numeric(F8),
    r_A1 = unname(a1_sat[as.character(A1_Kaan)]),
    r_A2 = -A2_atlag,
    r_A3 = as.numeric(A3 != "beszukulo"),
    r_A4 = unname(c(nincs = 0, kicsi = 0.5, nagy = 1)[A4]),
    r_A5 = as.numeric(A5) / 2,
    r_TUB = tuberculum_score,
    r_A10 = A10,
    r_A10k = as.numeric(A10 > A10_ANGLE_THRESHOLD),   # a kérdőív kontrasztja: szabályos vs. eltérő reláció
    r_A11 = unname(c(`2` = 0, `1` = 0.5, `3` = 1)[as.character(A11)]),
    r_A12 = as.numeric(as.integer(A12) != 1L)
  )

# -----------------------------------------------------------------------------
# 2. Regiszterek: anatómiai tételek + az elődök (tankönyvi) iránya; kimenetek
# -----------------------------------------------------------------------------
# p_tankonyv = a vizsgálatvezető 2026-08-19-i elicitációjának iránybizonyossága
# (gyenge 0,80 / mérsékelt 0,90 / erős 0,98; irány nélküli tétel 0,50).
predictors <- tribble(
  ~kod, ~var, ~cimke, ~rovid, ~csoport, ~tipus, ~polus_A, ~polus_B, ~p_tankonyv, ~allitas,
  "F1",  "r_F1",  "F1 · felső gerincmagasság (mm)",            "F1 gerincmagasság ↓",   "Felső állcsont", "folytonos",  "magas gerinc",                       "alacsony gerinc",                     0.98, "Magasabb gerinc → nagyobb fedett felszín, jobb retenció.",
  "F3",  "r_F3",  "F3 · szájpadboltozat-magasság (mm)",        "F3 boltozat ↓",         "Felső állcsont", "folytonos",  "magas boltozat",                     "lapos boltozat",                      0.90, "Magas, kúpos boltozat → jobb illeszkedés/retenció.",
  "F4",  "r_F4",  "F4 · felső gerincalak szöge (°)",           "F4 gerincalak-szög ↓",  "Felső állcsont", "folytonos",  "nagy szög (négyzetes ív)",           "kis szög (elkeskenyedő ív)",          0.90, "Nagyobb szög → a szemfogak a gerincélre állíthatók, stabilabb fogsor.",
  "F6",  "r_F6",  "F6 · interalveoláris szög eltérése 90°-tól", "F6 |szög−90°| ↑",      "Felső állcsont", "folytonos",  "≈ 90°",                              "90°-tól eltérő",                      0.98, "Eltérés → külpontos terhelés, instabil felső fogsor.",
  "F5",  "r_F5",  "F5 · lötyögő, csontmag nélküli gerinc",     "F5 lötyögő gerinc",     "Felső állcsont", "bináris",    "nincs",                              "van (bárhol)",                        0.80, "Jelenléte → instabil alátámasztás, nyomási fájdalom.",
  "F7",  "r_F7",  "F7 · torus palatinus",                      "F7 torus palatinus",    "Felső állcsont", "bináris",    "nincs",                              "van (plató/orsó)",                    0.98, "Bármely torus → akadályozza a szívóhatást, nyomásérzékeny.",
  "A1",  "r_A1",  "A1 · Kaán-féle gerincforma (telítődő 0–1)", "A1 Kaán-gerincforma ↑", "Alsó állcsont",  "ordinális",  "egészében megtartott",               "negatív / mélyült negatív",           0.98, "Romló gerincforma → romló alátámasztás és stabilitás (telítődő).",
  "A2",  "r_A2",  "A2 · alsó gerincmagasság, modellanalízis (mm)", "A2 gerincmagasság ↓", "Alsó állcsont", "folytonos", "magas gerinc",                      "alacsony gerinc",                     0.98, "Ugyanaz a konstruktum, mint A1, folytonos mérésként.",
  "A4",  "r_A4",  "A4 · torus mandibularis",                   "A4 torus mandibularis", "Alsó állcsont",  "ordinális",  "nincs",                              "van (kicsi < nagy)",                  0.98, "Jelenléte → nyomásérzékenység, szívóhatás akadálya.",
  "A5",  "r_A5",  "A5 · lingualis tasak (izomerő iránya)",     "A5 lingualis tasak ↑",  "Alsó állcsont",  "ordinális",  "mandibulához préseli",               "kifelé préseli",                      0.98, "Kifelé mutató izomerő → elmozdítja a fogsort.",
  "TUB", "r_TUB", "A6–A9 · tuberculum-konstruktum (0–1)",      "A6–A9 tuberculum ↑",    "Alsó állcsont",  "folytonos",  "feszes, fedett, stabil, jó alakú",    "fedetlen, plicaszerű, mozgékony",     0.98, "Stabil, fedett tuberculum → hatásos megtámasztás.",
  "A11", "r_A11", "A11 · szublingvális tájék / szájfenék",     "A11 szájfenék ↑",       "Alsó állcsont",  "ordinális",  "puhán elődomborodó",                 "tömött, elődomborodó",                0.98, "Puha szájfenék → szívóhatású alsó fogsor alakítható ki.",
  "A12", "r_A12", "A12 · spinae mentales",                     "A12 spinae mentales",   "Alsó állcsont",  "bináris",    "nem tapintható",                     "tapintható / nyomásérzékeny",         0.98, "Felszínes spina → sorvadás jele, terhelési fájdalom.",
  "F2",  "r_F2",  "F2 · felső alámenősség térfogata (mm³)",    "F2 alámenősség (nyers)", "Nincs tankönyvi irány", "folytonos", "–",                           "– (optimum feltételezett)",           0.50, "Nem monoton: közepes alámenősség optimális, a túl nagy fájdalmas.",
  "F2s", "r_F2s", "F2/L³ · méretstandardizált alámenősség",    "F2/L³ (nyers)",         "Nincs tankönyvi irány", "folytonos", "–",                           "– (optimum feltételezett)",           0.50, "Mint F2, ívhosszra standardizálva.",
  "F8",  "r_F8",  "F8 · antagonista fogazat (1→3)",            "F8 antagonista (nyers)", "Nincs tankönyvi irány", "ordinális", "–",                           "– (interakciófüggő)",                 0.50, "Az erők iránya számít, nem a kategória önmagában.",
  "A3",  "r_A3",  "A3 · buccinator tasak (beszűkülő vs. egyéb)", "A3 buccinator (nyers)", "Nincs tankönyvi irány", "bináris", "–",                            "–",                                   0.50, "Nincs előzetesen kedvező vagy kedvezőtlen forma.",
  "A10", "r_A10", "A10 · állcsontreláció szöge (°)",           "A10 állcsontreláció (nyers)", "Nincs tankönyvi irány", "folytonos", "–",                      "– (Angle-eltérés)",                   0.50, "Nem Angle I nehezebb eset, de kimeneti prior nincs.",
  "A10k", "r_A10k", "A10 · eltérő állcsontreláció (küszöb felett)", "A10 reláció (kategoriális)", "Nincs tankönyvi irány", "bináris", "–",                      "– (Angle-eltérés)",                   0.50, "Nem Angle I nehezebb eset, de kimeneti prior nincs."
) |>
  mutate(
    irany_van = p_tankonyv > 0.5,
    csoport = factor(csoport, levels = c("Felső állcsont", "Alsó állcsont", "Nincs tankönyvi irány"))
  )

outcomes <- tribble(
  ~kod, ~var, ~cimke, ~blokk, ~rovid,
  "s_INDEX", "s_INDEX", "Siker-index (4 kimenet z-átlaga)",         "Állapot az új fogsorral", "Siker-index",
  "s_GOHAI", "s_GOHAI", "GOHAI az utánkövetéskor (↑ jobb)",        "Állapot az új fogsorral", "GOHAI",
  "s_OHIP",  "s_OHIP",  "OHIP-5 az utánkövetéskor (↓ jobb)",       "Állapot az új fogsorral", "OHIP-5",
  "s_MAI",   "s_MAI",   "MAI az utánkövetéskor (↓ jobb)",          "Állapot az új fogsorral", "MAI",
  "s_RAGAS", "s_RAGAS", "Önbevallott rágóképesség a fogsorral",   "Állapot az új fogsorral", "Önbev. rágás",
  "v_GOHAI", "v_GOHAI", "GOHAI-javulás",                            "Változás (kiindulás → utánkövetés)", "ΔGOHAI",
  "v_OHIP",  "v_OHIP",  "OHIP-5-javulás",                           "Változás (kiindulás → utánkövetés)", "ΔOHIP-5",
  "v_MAI",   "v_MAI",   "MAI-javulás",                              "Változás (kiindulás → utánkövetés)", "ΔMAI",
  "v_GRC",   "v_GRC",   "Megélt rágásváltozás (GRC)",               "Változás (kiindulás → utánkövetés)", "GRC"
) |>
  mutate(blokk = factor(blokk, levels = c("Állapot az új fogsorral", "Változás (kiindulás → utánkövetés)")))

# Betegszintű anonim tábla (P01–P06; név/TAJ/adatbázis-id nem kerül fájlba)
patient_table <- d6 |>
  transmute(
    study_id, nem,
    OHIP_init = OHIP_sum_init, OHIP_fu = OHIP_sum_followup,
    GOHAI_init = GOHAI_sum_init, GOHAI_fu = GOHAI_sum_followup,
    MAI_init = round(MAI_huedegree_init, 1), MAI_fu = round(MAI_huedegree_followup, 1),
    ragas_init = chewing_today_init, ragas_fu = chewing_today_followup, GRC_ragas = chewing_change,
    siker_index = round(s_INDEX, 2),
    F1, F2, F2_L3 = signif(F2_standardizalt, 3), F3, F4, F5, F6, F7, F8,
    A1_Kaan, A2_atlag, A3, A4, A5, tuberculum_score, A10, A11, A12
  )

# -----------------------------------------------------------------------------
# 3. Lemorzsolódás és szelekció: a teljes kiindulási kohorsz (csak olvasás)
# -----------------------------------------------------------------------------
strip_outer_quotes <- function(x) {
  if (nchar(x) < 2L) return(x)
  f <- substr(x, 1L, 1L); l <- substr(x, nchar(x), nchar(x))
  if ((f == "\"" && l == "\"") || (f == "'" && l == "'")) return(substr(x, 2L, nchar(x) - 1L))
  x
}
read_env_value <- function(path, key) {
  if (!file.exists(path)) return(NA_character_)
  lines <- readLines(path, warn = FALSE)
  hit <- grep(paste0("^[[:space:]]*", key, "[[:space:]]*="), lines, value = TRUE)
  if (length(hit) == 0L) return(NA_character_)
  strip_outer_quotes(trimws(sub("^[^=]*=", "", hit[[1]])))
}
parse_postgres_url <- function(url) {
  core <- sub("^postgres(ql)?://", "", url)
  core <- strsplit(core, "?", fixed = TRUE)[[1]][[1]]
  at <- max(gregexpr("@", core, fixed = TRUE)[[1]])
  auth <- substr(core, 1L, at - 1L); host_db <- substr(core, at + 1L, nchar(core))
  ca <- regexpr(":", auth, fixed = TRUE)[[1]]
  user <- utils::URLdecode(substr(auth, 1L, ca - 1L)); password <- utils::URLdecode(substr(auth, ca + 1L, nchar(auth)))
  sl <- regexpr("/", host_db, fixed = TRUE)[[1]]
  host_port <- substr(host_db, 1L, sl - 1L); dbname <- utils::URLdecode(substr(host_db, sl + 1L, nchar(host_db)))
  ch <- regexpr(":", host_port, fixed = TRUE)[[1]]
  if (ch > 0) { host <- substr(host_port, 1L, ch - 1L); port <- as.integer(substr(host_port, ch + 1L, nchar(host_port))) } else { host <- host_port; port <- 5432L }
  list(host = host, port = port, dbname = dbname, user = user, password = password)
}

cohort <- NULL
DB_CON <- NULL
db_note <- "Adatbázis-kapcsolat nem volt elérhető; a lemorzsolódási blokk kimaradt."
db_ok <- FALSE
if (requireNamespace("DBI", quietly = TRUE) && requireNamespace("RPostgres", quietly = TRUE)) {
  database_url <- Sys.getenv("DATABASE_URL", unset = "")
  if (!nzchar(database_url)) database_url <- read_env_value(file.path(ROOT, ".env"), "DATABASE_URL")
  if (!is.na(database_url) && nzchar(database_url)) {
    cohort <- tryCatch({
      db <- parse_postgres_url(database_url)
      con <- DBI::dbConnect(RPostgres::Postgres(), host = db$host, port = db$port, dbname = db$dbname,
                            user = db$user, password = db$password, connect_timeout = 15)
      DB_CON <- con
      DBI::dbExecute(con, "SET default_transaction_read_only = on")
      sides <- sprintf('p."A%d_%s"::double precision AS a%d_%s', rep(4:9, each = 2), rep(c("jobb", "bal"), 6), rep(4:9, each = 2), rep(c("jobb", "bal"), 6))
      q <- paste0(
        'WITH latest AS (\n  SELECT DISTINCT ON ("TAJ") * FROM patients\n  WHERE "TAJ" IS NOT NULL\n',
        '    AND "TAJ" NOT IN (SELECT "TAJ" FROM patients WHERE "id" IN (3))\n  ORDER BY "TAJ", "id" DESC\n)\n',
        'SELECT p."id"::bigint AS patient_id, p."record_datetime"::text AS record_datetime, p."birthdate"::text AS birthdate,\n',
        '  p."gender" AS gender, LOWER(TRIM(p."denture_type")) AS denture_type,\n  ',
        paste(sprintf('p."OHIP_%d"::double precision AS ohip_%d', 1:5, 1:5), collapse = ", "), ",\n  ",
        paste(sprintf('p."GOHAI_%d"::double precision AS gohai_%d', 1:12, 1:12), collapse = ", "), ",\n",
        '  p."init_mai_huedegree"::double precision AS mai_baseline,\n',
        '  p."chewing_today_situation" AS ragas_baseline,\n',
        '  p."F5"::double precision AS f5, p."F7"::double precision AS f7, p."A1_Kaan"::double precision AS a1,\n  ',
        paste(sides, collapse = ", "), ",\n",
        '  p."A11"::double precision AS a11, p."A12"::double precision AS a12,\n',
        '  p."dropout" AS dropout, p."modellanalizis_megtortent" AS modell_kesz,\n',
        '  (p."F1_profil" IS NOT NULL AND LENGTH(p."F1_profil") > 2) AS f1_profil_van,\n',
        '  (UPPER(TRIM(p."A2_modszer")) = \'C\' AND p."A2_profil" IS NOT NULL AND LENGTH(p."A2_profil") > 2) AS a2c_profil_van,\n  ',
        paste(sprintf('COALESCE(f.ohip_%d_recall::double precision, p."OHIP_%d_recall"::double precision) AS ohip_%d_recall', 1:5, 1:5, 1:5), collapse = ", "), ",\n",
        '  COALESCE(f.final_mai_huedegree::double precision, p."final_mai_huedegree"::double precision) AS mai_final,\n',
        '  f.visit_status AS visit_status, f.months_since_delivery::text AS months_since_delivery\n',
        'FROM latest p\nLEFT JOIN followup_visits f ON f.patient_id = p."id" AND f.visit_round = 1\nORDER BY p."id"'
      )
      out <- DBI::dbGetQuery(con, q)
      if (anyDuplicated(out$patient_id)) stop("Join-duplikáció a kohorsz-lekérdezésben.")
      out
    }, error = function(e) { db_note <<- paste0("Adatbázis-hiba: ", conditionMessage(e)); NULL })
  }
}

strict_sum <- function(df) { m <- as.matrix(df); storage.mode(m) <- "numeric"; s <- rowSums(m, na.rm = TRUE); s[rowSums(!is.na(m)) != ncol(m)] <- NA; s }
ord_score <- function(x) x - 1
tub_from_sides <- function(df) {
  a6 <- (ord_score(df$a6_jobb) + ord_score(df$a6_bal)) / 2
  a7 <- (ord_score(df$a7_jobb) + ord_score(df$a7_bal)) / 2
  a8 <- (as.numeric(df$a8_jobb %in% c(2, 3)) + as.numeric(df$a8_bal %in% c(2, 3))) / 2
  a8[is.na(df$a8_jobb) | is.na(df$a8_bal)] <- NA
  a9 <- (ord_score(df$a9_jobb) + ord_score(df$a9_bal)) / 2
  (a6 / 2 + a7 / 2 + a8 + a9 / 2) / 4
}

if (!is.null(cohort)) {
  db_ok <- TRUE
  db_note <- "Adatbázis: csak olvasás; TAJ-onként a legutolsó rekord; a 3-as rekord kizárva (elveszett modellanalízis-forrás)."
  a4map <- function(x) unname(c(`1` = 0, `3` = 0.5, `2` = 1)[as.character(x)])
  cohort <- cohort |>
    mutate(
      teljes_eset = patient_id %in% EXPECTED_IDS,
      rec_date = as.Date(substr(record_datetime, 1, 10)),
      birth = as.Date(substr(birthdate, 1, 10)),
      kor_ev = as.numeric(rec_date - birth) / 365.25,
      kor_ev = ifelse(is.finite(kor_ev) & kor_ev > 18 & kor_ev < 105, kor_ev, NA_real_),
      nem = ifelse(gender == "Female", "nő", ifelse(gender == "Male", "férfi", NA)),
      ohip_baseline = strict_sum(pick(starts_with("ohip_") & !contains("recall"))),
      gohai_baseline = strict_sum(pick(starts_with("gohai_"))),
      ragas_baseline_ord = to_ord(ragas_baseline, chew_levels),
      r_A1 = unname(a1_sat[as.character(a1)]),
      r_TUB = tub_from_sides(pick(matches("^a[6-9]_(jobb|bal)$"))),
      r_A11 = unname(c(`2` = 0, `1` = 0.5, `3` = 1)[as.character(a11)]),
      r_A12 = as.numeric(a12 != 1),
      r_A4 = pmax(a4map(a4_jobb), a4map(a4_bal), na.rm = FALSE),
      r_F5 = as.numeric(f5 != 1), r_F7 = as.numeric(f7 != 1),
      recall_kerdoiv = rowSums(!is.na(pick(matches("^ohip_[1-5]_recall$")))) == 5L,
      recall_mai = is.finite(mai_final),
      visit_status = ifelse(is.na(visit_status), "nincs rekord", visit_status),
      recall_statusz = case_when(
        teljes_eset ~ "Teljes eset (mind a 3 kimenetpár + modellanalízis)",
        recall_kerdoiv & recall_mai ~ "Kérdőív + MAI recall, hiányos modellanalízis",
        recall_kerdoiv ~ "Csak kérdőíves recall",
        recall_mai ~ "Csak MAI recall",
        visit_status == "declined" ~ "Visszahívást elutasította",
        visit_status == "unreachable" ~ "Nem elérhető",
        visit_status == "no_show" ~ "Nem jelent meg",
        visit_status %in% c("contacted", "not_contacted") ~ "Kapcsolatfelvétel folyamatban / nem történt",
        visit_status == "completed" ~ "Vizit lezárva, kimenet hiányos",
        dropout %in% TRUE ~ "Kezelésből kiesett (dropout)",
        TRUE ~ "Nincs visszahívási rekord"
      )
    )
  # kor a hat esethez
  d6$kor_ev <- cohort$kor_ev[match(d6$patient_id, cohort$patient_id)]
  patient_table$kor_ev <- round(d6$kor_ev, 0)

  both <- cohort$denture_type == "both" & !is.na(cohort$denture_type)
  funnel <- tibble(
    lepes = c(
      "Deduplikált kiindulási kohorsz (3-as rekord kizárva)",
      "Visszahívási rekord létezik (1. kör)",
      "Utánkövetési kérdőív kitöltve",
      "Utánkövetési MAI mérés",
      "Kétállcsontos fogpótlás",
      "… és érvényes F1-profil",
      "… és érvényes A2-C-profil",
      "… és mindhárom kimenetpár teljes = FŐ KOHORSZ"
    ),
    n = c(
      nrow(cohort),
      sum(cohort$visit_status != "nincs rekord"),
      sum(cohort$recall_kerdoiv),
      sum(cohort$recall_mai),
      sum(both),
      sum(both & cohort$f1_profil_van %in% TRUE),
      sum(both & cohort$f1_profil_van %in% TRUE & cohort$a2c_profil_van %in% TRUE),
      sum(cohort$teljes_eset)
    ),
    ag = c("kohorsz", "recall", "recall", "recall", "modellanalízis", "modellanalízis", "modellanalízis", "fő kohorsz")
  )
  write_csv_utf8(funnel, "00_kohorsz_tolcser.csv")

  recall_breakdown <- cohort |> count(recall_statusz, name = "n") |> arrange(desc(n))
  write_csv_utf8(recall_breakdown, "01_recall_statusz_megoszlas.csv")

  # Kiindulási összehasonlítás: a 6 teljes eset vs. a kohorsz többi tagja
  comp_vars <- tribble(
    ~var, ~cimke, ~tipus,
    "kor_ev", "Életkor a kiinduláskor (év)", "folytonos",
    "ohip_baseline", "OHIP-5 kiindulás (↓ jobb)", "folytonos",
    "gohai_baseline", "GOHAI kiindulás (↑ jobb)", "folytonos",
    "mai_baseline", "MAI kiindulás, régi fogsorral (↓ jobb)", "folytonos",
    "ragas_baseline_ord", "Önbevallott rágás kiindulás (1–5)", "folytonos",
    "r_A1", "A1 Kaán-gerincforma (0–1, ↑ kedvezőtlen)", "folytonos",
    "r_TUB", "A6–A9 tuberculum (0–1, ↑ kedvezőtlen)", "folytonos",
    "r_A11", "A11 szájfenék (0–1, ↑ kedvezőtlen)", "folytonos",
    "r_A4", "A4 torus mandibularis (0–1)", "folytonos",
    "r_F5", "F5 lötyögő gerinc (0/1)", "folytonos",
    "r_F7", "F7 torus palatinus (0/1)", "folytonos",
    "r_A12", "A12 spinae mentales (0/1)", "folytonos"
  )
  summarise_group <- function(v, g) {
    x <- v[g]; x <- x[is.finite(x)]
    tibble(n = length(x), median = if (length(x)) median(x) else NA_real_,
           Q1 = if (length(x)) unname(quantile(x, .25)) else NA_real_,
           Q3 = if (length(x)) unname(quantile(x, .75)) else NA_real_,
           atlag = if (length(x)) mean(x) else NA_real_)
  }
  comparison <- bind_rows(lapply(seq_len(nrow(comp_vars)), function(i) {
    v <- cohort[[comp_vars$var[i]]]
    a <- summarise_group(v, cohort$teljes_eset); b <- summarise_group(v, !cohort$teljes_eset)
    tibble(valtozo = comp_vars$cimke[i],
           teljes_n = a$n, teljes_median = a$median, teljes_Q1 = a$Q1, teljes_Q3 = a$Q3, teljes_atlag = a$atlag,
           tobbi_n = b$n, tobbi_median = b$median, tobbi_Q1 = b$Q1, tobbi_Q3 = b$Q3, tobbi_atlag = b$atlag)
  }))
  gender_rows <- cohort |> group_by(teljes_eset) |> summarise(n = n(), no_arany = mean(nem == "nő", na.rm = TRUE), both_arany = mean(denture_type == "both", na.rm = TRUE), .groups = "drop")
  write_csv_utf8(comparison, "02_szelekcio_kiindulasi_osszehasonlitas.csv")
  write_csv_utf8(gender_rows, "02b_szelekcio_nem_fogsortipus.csv")
} else {
  funnel <- NULL; recall_breakdown <- NULL; comparison <- NULL; comp_vars <- NULL
}
write_csv_utf8(patient_table, "03_betegszintu_tabla_anonim.csv")

# -----------------------------------------------------------------------------
# 4. Leíró irány-egyezés: Spearman-ρ (kedvezőtlen anatómia ↔ rosszabb kimenet)
# -----------------------------------------------------------------------------
# ρ > 0: a kedvezőtlenebb anatómia rosszabb eredménnyel jár → egyezik az elődök
# tapasztalatával; ρ < 0: ellentmond. Irány nélküli tételnél ρ a nyers kódra
# vonatkozik (nagyobb nyers érték – rosszabb kimenet).
safe_cor <- function(x, y, method = "spearman") {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
  if (length(x) < 4L || length(unique(x)) < 2L || length(unique(y)) < 2L) return(NA_real_)
  suppressWarnings(cor(x, y, method = method))
}
normal_score <- function(x) {
  r <- rank(x, ties.method = "average"); q <- qnorm((r - 0.5) / length(r)); (q - mean(q)) / sd(q)
}
loo_range <- function(x, y) {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]; n <- length(x)
  if (n < 5L) return(c(NA_real_, NA_real_))
  v <- vapply(seq_len(n), function(i) safe_cor(x[-i], y[-i]), numeric(1))
  v <- v[is.finite(v)]; if (!length(v)) return(c(NA_real_, NA_real_)); range(v)
}

grid_rows <- tidyr::crossing(pred = predictors$kod, out = outcomes$kod)
assoc <- bind_rows(lapply(seq_len(nrow(grid_rows)), function(i) {
  p <- predictors[predictors$kod == grid_rows$pred[i], ]
  o <- outcomes[outcomes$kod == grid_rows$out[i], ]
  x <- d6[[p$var]]; y_worse <- -d6[[o$var]]
  ok <- is.finite(x) & is.finite(y_worse)
  rho <- safe_cor(x, y_worse)
  r_ns <- if (sum(ok) >= 4L && length(unique(x[ok])) > 1L && length(unique(y_worse[ok])) > 1L) cor(normal_score(x[ok]), normal_score(y_worse[ok])) else NA_real_
  lo <- loo_range(x, y_worse)
  tibble(
    pred = p$kod, pred_cimke = p$cimke, pred_rovid = p$rovid, csoport = as.character(p$csoport), irany_van = p$irany_van,
    out = o$kod, out_cimke = o$cimke, blokk = as.character(o$blokk), out_rovid = o$rovid,
    n = sum(ok), x_szintek = length(unique(x[ok])), rho = rho, loo_min = lo[1], loo_max = lo[2], r_normal_score = r_ns,
    egyezes = case_when(
      !is.finite(rho) ~ "nem becsülhető",
      !p$irany_van ~ "nincs tankönyvi irány",
      rho > 0.1 ~ "egyezik az elődökkel",
      rho < -0.1 ~ "ellentmond az elődöknek",
      TRUE ~ "semleges (|ρ| ≤ 0,1)"
    )
  )
}))
write_csv_utf8(assoc, "04_spearman_irany_egyezes.csv")

# -----------------------------------------------------------------------------
# 5. Bayes-i rács: likelihood, priorok (semleges / elődök–szakértői), posterior
# -----------------------------------------------------------------------------
BETA <- seq(-0.999, 0.999, length.out = 1999)
DBETA <- BETA[2] - BETA[1]
normalize <- function(w) { w[!is.finite(w)] <- 0; if (sum(w) <= 0) return(rep(1 / length(w), length(w))); w / sum(w) }
# Marginális likelihood standardizált x, y mellett (tengelymetszet és σ
# Jeffreys-priorral kiintegrálva): L(β) ∝ (1 − 2βr + β²)^(−(n−1)/2)
log_lik_beta <- function(r, n, beta = BETA) -((n - 1) / 2) * log(pmax(1 - 2 * beta * r + beta^2, 1e-12))
prior_neutral <- normalize(dnorm(BETA, 0, 0.5))
# Keverék-prior: P(β>0) = p_pos; a hatásnagyság |β| ~ N(mag_mean, mag_sd) az
# adott félegyenesre csonkolva (mag_mean = 0 → félnormális).
prior_mixture <- function(p_pos, mag_mean = 0, mag_sd = 0.5) {
  pos <- dnorm(BETA, mag_mean, mag_sd) * (BETA > 0)
  neg <- dnorm(BETA, -mag_mean, mag_sd) * (BETA < 0)
  p_pos * normalize(pos) + (1 - p_pos) * normalize(neg)
}
posterior_summary <- function(post) {
  cdf <- cumsum(post)
  q <- function(p) BETA[which(cdf >= p)[1]]
  tibble(p_pozitiv = sum(post[BETA > 0]), median = q(0.5), q05 = q(0.05), q95 = q(0.95))
}

# --- Szakértői/tankönyvi priorok beolvasása (predict_expert_priorok.csv) -----
# Oszlopok: szakerto_id, tetel, irany (B_kedvezotlenebb | A_kedvezotlenebb |
#   nincs_kulonbseg | nem_monoton | nem_tudom), p_irany (50–99, %),
#   siker_A, siker_B (100 betegből hány sikeres az A/B pólus mellett),
#   kulonbseg_min, kulonbseg_max (a különbség hihető tartománya, 100-ból),
#   alak, mechanizmus, kuszob, megjegyzes.
# Leképezés a korrelációs skálára: d = |Φ⁻¹(siker_A) − Φ⁻¹(siker_B)| (látens
# standardizált különbség), |r| = d / √(d² + 4); tartomány → szórás.
# Két forrás, azonos oszlopokkal: (1) a CSV-sablon (jelenleg a vizsgálatvezető
# 2026-08-19-i elicitációja); (2) az applikáció szakértői felmérő oldalán
# beküldött válaszok (expert_prior_responses tábla, status = 'submitted').
# INCLUDE_PI_PRIOR = FALSE esetén a vizsgálatvezetői sorok kimaradnak, és a
# prior kizárólag a külső szakértők véleménykeveréke.
INCLUDE_PI_PRIOR <- Sys.getenv("PREDICT_INCLUDE_PI", unset = "1") != "0"
USE_DB_EXPERTS <- Sys.getenv("PREDICT_EXPERT_DB", unset = "1") != "0"
INCLUDE_SIMULATED <- Sys.getenv("PREDICT_INCLUDE_SIM", unset = "0") == "1"
PI_EXPERT_IDS <- c("VV-2026-08-19")
# A kitöltő szerepe (fogorvos | fogtechnikus): a priorba alapból csak a
# fogorvosi válaszok kerülnek; PREDICT_EXPERT_ROLE = "all" mindkettőt egyesíti,
# "fogtechnikus" csak a technikusi véleményekből épít priort. A leíró
# konszenzus mindig szerep szerint is elkészül (05d, ábra 12).
EXPERT_ROLE <- Sys.getenv("PREDICT_EXPERT_ROLE", unset = "fogorvos")
EXPERT_COLS <- c("szakerto_id", "datum", "tetel", "irany", "p_irany", "siker_A", "siker_B",
                 "kulonbseg_min", "kulonbseg_max", "alak", "mechanizmus", "kuszob", "megjegyzes", "szerep",
                 "siker_A_min", "siker_A_max", "siker_B_min", "siker_B_max", "siker_M", "siker_M_min", "siker_M_max",
                 "nagysag_nem_tudom")
expert_path <- Sys.getenv("PREDICT_EXPERT_CSV", unset = file.path(ROOT, "predict_expert_priorok.csv"))
if (!grepl("^/", expert_path)) expert_path <- file.path(ROOT, expert_path)
expert_csv <- if (file.exists(expert_path)) read.csv(expert_path, check.names = FALSE, encoding = "UTF-8", na.strings = c("", "NA")) else NULL
if (!is.null(expert_csv)) {
  names(expert_csv) <- sub("^﻿", "", names(expert_csv))
  expert_csv$forras <- "CSV-sablon"
  if (!"szerep" %in% names(expert_csv)) expert_csv$szerep <- "fogorvos"
}
expert_db <- NULL
if (!is.null(DB_CON) && USE_DB_EXPERTS) {
  expert_db <- tryCatch({
    exists_row <- DBI::dbGetQuery(DB_CON, "SELECT to_regclass('public.expert_prior_responses') AS t")
    if (is.na(exists_row$t[1])) NULL else {
      resp <- DBI::dbGetQuery(DB_CON, paste0(
        "SELECT expert_code, submitted_at::text AS submitted_at, items::text AS items, ",
        "COALESCE(background->>'szerep', 'fogorvos') AS szerep FROM expert_prior_responses WHERE status = 'submitted'",
        if (INCLUDE_SIMULATED) "" else " AND form_version <> 'v1.0-SZIMULACIO'", " ORDER BY id"))
      if (nrow(resp) == 0L) NULL else bind_rows(lapply(seq_len(nrow(resp)), function(i) {
        items <- jsonlite::fromJSON(resp$items[i], simplifyVector = FALSE)
        bind_rows(lapply(predictors$kod, function(k) {
          a <- items[[k]]
          if (is.null(a)) return(NULL)   # az űrlapon nem szereplő tétel (pl. F2/L³): nincs sor
          g <- function(key) { v <- a[[key]]; if (is.null(v) || length(v) == 0L) NA else v }
          irany <- g("irany")
          tibble(
            szakerto_id = resp$expert_code[i], datum = substr(resp$submitted_at[i], 1, 10), tetel = k,
            irany = as.character(irany), p_irany = suppressWarnings(as.numeric(g("p_irany"))),
            siker_A = suppressWarnings(as.numeric(g("siker_A"))), siker_B = suppressWarnings(as.numeric(g("siker_B"))),
            kulonbseg_min = suppressWarnings(as.numeric(g("kulonbseg_min"))), kulonbseg_max = suppressWarnings(as.numeric(g("kulonbseg_max"))),
            alak = ifelse(identical(irany, "nem_monoton"), "optimum", NA_character_),
            mechanizmus = paste(unlist(a[["mechanizmus"]]), collapse = "; "), kuszob = as.character(g("kuszob")),
            megjegyzes = as.character(g("megjegyzes")), forras = "applikáció (szakértői felmérés)",
            szerep = as.character(resp$szerep[i]),
            siker_A_min = suppressWarnings(as.numeric(g("siker_A_min"))), siker_A_max = suppressWarnings(as.numeric(g("siker_A_max"))),
            siker_B_min = suppressWarnings(as.numeric(g("siker_B_min"))), siker_B_max = suppressWarnings(as.numeric(g("siker_B_max"))),
            siker_M = suppressWarnings(as.numeric(g("siker_M"))), siker_M_min = suppressWarnings(as.numeric(g("siker_M_min"))),
            siker_M_max = suppressWarnings(as.numeric(g("siker_M_max"))),
            nagysag_nem_tudom = as.integer(isTRUE(a[["nagysag_nem_tudom"]]))
          )
        }))
      }))
    }
  }, error = function(e) { message("Szakértői tábla olvasása sikertelen: ", conditionMessage(e)); NULL })
}
expert_raw <- bind_rows(
  if (!is.null(expert_csv)) expert_csv[, intersect(c(EXPERT_COLS, "forras"), names(expert_csv))] else NULL,
  expert_db
)
if (nrow(expert_raw) == 0L) expert_raw <- NULL
if (!is.null(expert_raw) && !INCLUDE_PI_PRIOR) expert_raw <- expert_raw[!expert_raw$szakerto_id %in% PI_EXPERT_IDS, , drop = FALSE]
if (!is.null(expert_raw) && nrow(expert_raw) == 0L) expert_raw <- NULL
if (!is.null(expert_raw)) {
  expert_raw$szerep <- as.character(expert_raw$szerep)
  expert_raw$szerep[is.na(expert_raw$szerep) | expert_raw$szerep == ""] <- "fogorvos"
}
# A priorba kerülő sorok (szerep szerint szűrve); a leíró bontás az összes sorból készül.
pool_raw <- if (is.null(expert_raw) || EXPERT_ROLE == "all") expert_raw else expert_raw[expert_raw$szerep == EXPERT_ROLE, , drop = FALSE]
if (!is.null(pool_raw) && nrow(pool_raw) == 0L) pool_raw <- NULL

clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)
# Egy szakértői sor → prior a korrelációs (β) skálán, a válasz típusa szerint:
#   nem_tudom → NULL (kimarad a poolból: nem információ, hanem információhiány);
#   nincs_kulonbseg → szűk, nulla körüli eloszlás (kis hatásra vonatkozó vélemény);
#   nem_monoton → NULL a lineáris poolból (görbületre vonatkozó tudás; a
#     logit-szkript kezeli a három forgatókönyvből), a konszenzusban megmarad;
#   irányos válasz pólusonkénti pontbecsléssel és tartománnyal → hatásnagyság a
#     látens különbségből, szórás a két pólus tartományából (húsz becslésből
#     tizenkilenc ≈ 95 %, ezért 3,92); tartomány nélkül vagy „nagyságát nem tudom”
#     → csak irány-prior (jelölt helyőrző nagysággal).
expert_row_to_prior <- function(row) {
  irany <- as.character(row$irany)
  if (is.na(irany) || irany == "" || irany == "nem_tudom" || irany == "nem_monoton") return(NULL)
  p <- suppressWarnings(as.numeric(row$p_irany)); p <- ifelse(is.finite(p), clamp(p / 100, 0.5, 0.995), 0.5)
  p_pos <- switch(irany, B_kedvezotlenebb = p, A_kedvezotlenebb = 1 - p, 0.5)
  if (irany == "nincs_kulonbseg") return(list(p_pos = 0.5, mag_mean = 0, mag_sd = 0.1, forras = "nincs érdemi különbség (szűk)"))
  num <- function(key) { v <- suppressWarnings(as.numeric(row[[key]])); if (length(v) == 0L) NA_real_ else v }
  unknown <- isTRUE(as.integer(num("nagysag_nem_tudom")) == 1L)
  sA <- num("siker_A"); sB <- num("siker_B")
  if (!unknown && is.finite(sA) && is.finite(sB)) {
    pa <- clamp(sA / 100, 0.02, 0.98); pb <- clamp(sB / 100, 0.02, 0.98)
    dlat <- abs(qnorm(pa) - qnorm(pb)); r_mag <- dlat / sqrt(dlat^2 + 4)
    delta <- abs(pa - pb)
    k <- if (delta > 0) r_mag / delta else 1.25
    sdA <- (num("siker_A_max") - num("siker_A_min")) / 100 / 3.92
    sdB <- (num("siker_B_max") - num("siker_B_min")) / 100 / 3.92
    if (is.finite(sdA) && is.finite(sdB)) {
      sd_r <- max(k * sqrt(sdA^2 + sdB^2), 0.05)
      return(list(p_pos = p_pos, mag_mean = r_mag, mag_sd = sd_r, forras = "elicitált hatásnagyság (pólusonkénti tartomány)"))
    }
    kmin <- num("kulonbseg_min"); kmax <- num("kulonbseg_max")   # v1 űrlap öröksége
    if (is.finite(kmin) && is.finite(kmax) && kmax > kmin) {
      return(list(p_pos = p_pos, mag_mean = r_mag, mag_sd = max((k * (kmax - kmin) / 100) / 3.92, 0.05), forras = "elicitált hatásnagyság (v1 különbség-tartomány)"))
    }
    return(list(p_pos = p_pos, mag_mean = r_mag, mag_sd = max(0.15, r_mag / 2), forras = "pontbecslés tartomány nélkül (szórás-helyőrző)"))
  }
  list(p_pos = p_pos, mag_mean = 0, mag_sd = 0.5, forras = "csak irány (nagyság nem becsült) + HN(0; 0,5) helyőrző")
}

# Az A2 (mért alsó gerincmagasság) ugyanazt a konstruktumot méri, mint az A1
# (Kaán-féle gerincforma): a kérdőív egyetlen tételként kérdezi (A1), ezért az
# A2 prediktor az A1 szakértői sorait örökli.
EXPERT_ITEM_ALIAS <- c(A2 = "A1", A10k = "A10")   # a kategoriális A10 az A10 szakértői sorait örökli
expert_pool <- lapply(predictors$kod, function(k) {
  source_code <- if (k %in% names(EXPERT_ITEM_ALIAS)) EXPERT_ITEM_ALIAS[[k]] else k
  rows <- if (!is.null(pool_raw)) pool_raw[pool_raw$tetel == source_code, , drop = FALSE] else NULL
  if (is.null(rows) || nrow(rows) == 0L) {
    p <- predictors$p_tankonyv[predictors$kod == k]
    return(list(prior = prior_mixture(p), n_expert = 0L, p_pos = p, forras = "regiszter (tankönyvi p) + HN(0; 0,5)", parts = NULL))
  }
  parts <- Filter(Negate(is.null), lapply(seq_len(nrow(rows)), function(i) expert_row_to_prior(rows[i, , drop = FALSE])))
  if (length(parts) == 0L) {
    p <- predictors$p_tankonyv[predictors$kod == k]
    return(list(prior = prior_mixture(p), n_expert = 0L, p_pos = p, forras = "regiszter (a szakértői válaszok mind „nem tudom” / görbület) + HN(0; 0,5)", parts = NULL))
  }
  dens <- Reduce(`+`, lapply(parts, function(pp) prior_mixture(pp$p_pos, pp$mag_mean, pp$mag_sd))) / length(parts)
  list(prior = dens, n_expert = length(parts), p_pos = sum(dens[BETA > 0]),
       forras = paste(unique(vapply(parts, function(pp) pp$forras, character(1))), collapse = "; "), parts = parts)
})
names(expert_pool) <- predictors$kod
prior_registry <- bind_rows(lapply(predictors$kod, function(k) {
  ep <- expert_pool[[k]]; s <- posterior_summary(ep$prior)
  tibble(tetel = k, cimke = predictors$cimke[predictors$kod == k], n_szakerto = ep$n_expert,
         prior_P_varhato_irany = s$p_pozitiv, prior_median = s$median, prior_q05 = s$q05, prior_q95 = s$q95, hatasnagysag_forras = ep$forras)
}))
write_csv_utf8(prior_registry, "05_prior_regiszter_pool.csv")
prior_caption <- if (is.null(pool_raw)) {
  "Prior: regiszter-alapú tankönyvi iránybizonyosság (gyenge 80% / mérsékelt 90% / erős 98%), hatásnagyság-helyőrzővel HN(0; 0,5)."
} else {
  paste0("Prior: ", length(unique(pool_raw$szakerto_id)), " szakértő egyenlő súlyú véleménykeveréke (",
         if (EXPERT_ROLE == "all") "fogorvosok és fogtechnikusok együtt" else paste0("szerep: ", EXPERT_ROLE),
         "; forrás: ", paste(unique(pool_raw$forras), collapse = " + "),
         "); irány a bizonyosságból, hatásnagyság a „100 beteg” válaszból, ahol elicitált, egyébként HN(0; 0,5).")
}
if (!is.null(expert_raw)) write_csv_utf8(expert_raw, "05b_szakertoi_prior_bemenet.csv")

# --- Szakértői konszenzus: tételenkénti megoszlás, egyet nem értés, pool ------
consensus <- NULL
if (!is.null(pool_raw)) {
  consensus <- bind_rows(lapply(predictors$kod, function(k) {
    source_code <- if (k %in% names(EXPERT_ITEM_ALIAS)) EXPERT_ITEM_ALIAS[[k]] else k
    rk <- pool_raw[pool_raw$tetel == source_code, , drop = FALSE]
    ep <- expert_pool[[k]]
    irany <- as.character(rk$irany)
    share <- function(v) if (length(irany)) mean(irany %in% v) else NA_real_
    dir_rows <- irany %in% c("A_kedvezotlenebb", "B_kedvezotlenebb")
    p_each <- if (is.null(ep$parts)) numeric(0) else vapply(ep$parts, function(pp) pp$p_pos, numeric(1))
    med_each <- if (is.null(ep$parts)) numeric(0) else vapply(ep$parts, function(pp) posterior_summary(prior_mixture(pp$p_pos, pp$mag_mean, pp$mag_sd))$median, numeric(1))
    s <- posterior_summary(ep$prior)
    tibble(
      tetel = k, cimke = predictors$rovid[predictors$kod == k], csoport = as.character(predictors$csoport[predictors$kod == k]),
      p_tankonyv = predictors$p_tankonyv[predictors$kod == k],
      n_szakerto = nrow(rk),
      B_arany = share("B_kedvezotlenebb"), A_arany = share("A_kedvezotlenebb"),
      optimum_arany = share("nem_monoton"), nincs_arany = share("nincs_kulonbseg"),
      nem_tudom_arany = share("nem_tudom"),
      p_irany_atlag = if (any(dir_rows)) mean(suppressWarnings(as.numeric(rk$p_irany[dir_rows])), na.rm = TRUE) else NA_real_,
      kulonbseg_atlag = mean(abs(suppressWarnings(as.numeric(rk$siker_A)) - suppressWarnings(as.numeric(rk$siker_B))), na.rm = TRUE),
      pool_P_varhato_irany = s$p_pozitiv, pool_median = s$median, pool_q05 = s$q05, pool_q95 = s$q95,
      egyet_nem_ertes_sd_P = if (length(p_each) > 1) sd(p_each) else NA_real_,
      egyeni_median_sd = if (length(med_each) > 1) sd(med_each) else NA_real_
    )
  }))
  consensus$kulonbseg_atlag[!is.finite(consensus$kulonbseg_atlag)] <- NA_real_
  write_csv_utf8(consensus, "05c_szakertoi_konszenzus.csv")
}

# --- Szerep szerinti bontás: fogorvosok vs. fogtechnikusok (leíró, minden sorból) ---
consensus_role <- NULL
if (!is.null(expert_raw)) {
  consensus_role <- bind_rows(lapply(sort(unique(expert_raw$szerep)), function(role) {
    rr <- expert_raw[expert_raw$szerep == role, , drop = FALSE]
    bind_rows(lapply(predictors$kod, function(k) {
      source_code <- if (k %in% names(EXPERT_ITEM_ALIAS)) EXPERT_ITEM_ALIAS[[k]] else k
      rk <- rr[rr$tetel == source_code, , drop = FALSE]
      irany <- as.character(rk$irany)
      share <- function(v) if (length(irany)) mean(irany %in% v) else NA_real_
      dir_rows <- irany %in% c("A_kedvezotlenebb", "B_kedvezotlenebb")
      tibble(
        szerep = role, tetel = k, cimke = predictors$rovid[predictors$kod == k],
        csoport = as.character(predictors$csoport[predictors$kod == k]),
        n_szakerto = nrow(rk), B_arany = share("B_kedvezotlenebb"), A_arany = share("A_kedvezotlenebb"),
        optimum_arany = share("nem_monoton"), nincs_arany = share("nincs_kulonbseg"), nem_tudom_arany = share("nem_tudom"),
        p_irany_atlag = if (any(dir_rows)) mean(suppressWarnings(as.numeric(rk$p_irany[dir_rows])), na.rm = TRUE) else NA_real_,
        kulonbseg_atlag = mean(abs(suppressWarnings(as.numeric(rk$siker_A)) - suppressWarnings(as.numeric(rk$siker_B))), na.rm = TRUE)
      )
    }))
  }))
  consensus_role$kulonbseg_atlag[!is.finite(consensus_role$kulonbseg_atlag)] <- NA_real_
  write_csv_utf8(consensus_role, "05d_szakertoi_konszenzus_szerep.csv")
}

# --- Posteriorok minden tétel × kimenet cellára -----------------------------
bayes <- bind_rows(lapply(seq_len(nrow(assoc)), function(i) {
  a <- assoc[i, ]
  if (!is.finite(a$r_normal_score)) {
    return(tibble(pred = a$pred, out = a$out, n = a$n, r_ns = NA_real_,
                  adat_P_pos = NA_real_, adat_median = NA_real_, adat_q05 = NA_real_, adat_q95 = NA_real_,
                  prior_P_pos = expert_pool[[a$pred]]$p_pos,
                  post_P_pos = NA_real_, post_median = NA_real_, post_q05 = NA_real_, post_q95 = NA_real_))
  }
  ll <- log_lik_beta(a$r_normal_score, a$n)
  lik <- normalize(exp(ll - max(ll)))
  post_neutral <- normalize(prior_neutral * lik)
  pr <- expert_pool[[a$pred]]$prior
  post_expert <- normalize(pr * lik)
  s1 <- posterior_summary(post_neutral); s2 <- posterior_summary(post_expert)
  tibble(pred = a$pred, out = a$out, n = a$n, r_ns = a$r_normal_score,
         adat_P_pos = s1$p_pozitiv, adat_median = s1$median, adat_q05 = s1$q05, adat_q95 = s1$q95,
         prior_P_pos = sum(pr[BETA > 0]),
         post_P_pos = s2$p_pozitiv, post_median = s2$median, post_q05 = s2$q05, post_q95 = s2$q95)
}))
bayes <- assoc |> select(pred, pred_cimke, pred_rovid, csoport, irany_van, out, out_cimke, blokk, out_rovid, rho) |>
  left_join(bayes, by = c("pred", "out")) |>
  mutate(elmozdulas_prior_to_post = post_P_pos - prior_P_pos)
write_csv_utf8(bayes, "06_bayes_posterior_osszes.csv")

# --- „Hány beteg kellene?” – a megfigyelt r-t rögzítve, n növelésével --------
N_GRID <- c(6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500)
n_needed <- bind_rows(lapply(seq_len(nrow(bayes)), function(i) {
  b <- bayes[i, ]
  if (!b$irany_van || !is.finite(b$r_ns) || !(b$blokk == "Állapot az új fogsorral")) return(NULL)
  pr <- expert_pool[[b$pred]]$prior
  curve <- vapply(N_GRID, function(n) {
    ll <- log_lik_beta(b$r_ns, n); lik <- normalize(exp(ll - max(ll)))
    c(sum(normalize(pr * lik)[BETA > 0]), sum(normalize(prior_neutral * lik)[BETA > 0]))
  }, numeric(2))
  p_exp <- curve[1, ]; p_neu <- curve[2, ]
  cross_expert <- if (b$r_ns < 0) N_GRID[which(p_exp < 0.5)[1]] else NA_integer_
  cross_neutral <- N_GRID[which(if (b$r_ns < 0) p_neu <= 0.05 else p_neu >= 0.95)[1]]
  tibble(pred = b$pred, pred_rovid = b$pred_rovid, csoport = b$csoport, out = b$out, out_rovid = b$out_rovid,
         r_ns = b$r_ns, n = N_GRID, P_expert_prior = p_exp, P_semleges_prior = p_neu,
         n_atbillenes_expert = cross_expert, n_95pct_semleges = cross_neutral)
}))
n_needed_summary <- n_needed |>
  group_by(pred, pred_rovid, csoport, out, out_rovid, r_ns, n_atbillenes_expert, n_95pct_semleges) |>
  summarise(P_expert_n6 = P_expert_prior[n == 6], P_expert_n30 = P_expert_prior[n == 30], P_expert_n100 = P_expert_prior[n == 100], .groups = "drop") |>
  mutate(
    adat_iranya = case_when(r_ns > 0.05 ~ "egyezik az elődökkel", r_ns < -0.05 ~ "ellentmond az elődöknek", TRUE ~ "nincs jel"),
    n_atbillenes_expert = ifelse(r_ns < -0.05, n_atbillenes_expert, NA_integer_),
    n_95pct_semleges = ifelse(abs(r_ns) > 0.05, n_95pct_semleges, NA_integer_),
    jelentes = "n_atbillenes_expert: csak ellentmondó jelnél – n, ahol az elődök priorjával a posterior 50% alá esik; n_95pct_semleges: n, ahol a semleges prioros posterior a MEGFIGYELT irányban eléri a 95%-ot"
  )
write_csv_utf8(n_needed, "07_hany_beteg_kellene_gorbek.csv")
write_csv_utf8(n_needed_summary, "07b_hany_beteg_kellene_osszefoglalo.csv")

# -----------------------------------------------------------------------------
# 6. Ábrák
# -----------------------------------------------------------------------------
label_pt <- function(x, d = 1) formatC(x, digits = d, format = "f", decimal.mark = ",")

## 6.1 Adatáttekintés: a hat beteg kiindulás → utánkövetés útja --------------
set.seed(11)
paired <- bind_rows(
  d6 |> transmute(study_id, grc = chewing_change, kimenet = "OHIP-5 (↓ jobb)", Kiindulás = OHIP_sum_init, Utánkövetés = OHIP_sum_followup),
  d6 |> transmute(study_id, grc = chewing_change, kimenet = "GOHAI (↑ jobb)", Kiindulás = GOHAI_sum_init, Utánkövetés = GOHAI_sum_followup),
  d6 |> transmute(study_id, grc = chewing_change, kimenet = "MAI hue-degree (↓ jobb)", Kiindulás = MAI_huedegree_init, Utánkövetés = MAI_huedegree_followup),
  d6 |> transmute(study_id, grc = chewing_change, kimenet = "Önbevallott rágóképesség, 1–5 (↑ jobb)", Kiindulás = ragas_init + runif(n(), -0.12, 0.12), Utánkövetés = ragas_fu + runif(n(), -0.12, 0.12))
) |>
  pivot_longer(c(Kiindulás, Utánkövetés), names_to = "idopont", values_to = "ertek") |>
  mutate(
    idopont = factor(idopont, levels = c("Kiindulás", "Utánkövetés")),
    kimenet = factor(kimenet, levels = c("OHIP-5 (↓ jobb)", "GOHAI (↑ jobb)", "MAI hue-degree (↓ jobb)", "Önbevallott rágóképesség, 1–5 (↑ jobb)")),
    cimke = ifelse(idopont == "Utánkövetés", paste0(study_id, " · ", grc), NA)
  )
p1 <- ggplot(paired, aes(x = idopont, y = ertek, group = study_id)) +
  geom_line(colour = PAL$blue, alpha = 0.75, linewidth = 0.8) +
  geom_point(shape = 21, fill = PAL$blue, colour = PAL$surface, size = 2.8, stroke = 0.9) +
  geom_text_repel(aes(label = cimke), data = paired |> filter(!is.na(cimke)), size = 2.6, colour = PAL$ink2,
                  nudge_x = 0.25, direction = "y", hjust = 0, segment.colour = PAL$axis, segment.size = 0.3, min.segment.length = 0.1) +
  facet_wrap(~kimenet, scales = "free_y", nrow = 1) +
  scale_x_discrete(expand = expansion(add = c(0.3, 1.5))) +
  labs(title = "A hat teljes eset: kiindulás → utánkövetés az új fogsorral",
       subtitle = "Minden vonal egy beteg (P01–P06); a címke a beteg saját globális rágásváltozás-ítéletét (GRC) mutatja. A kiindulási rágásteszt a RÉGI fogpótlással készült.",
       x = NULL, y = "Nyers pontszám",
       caption = "OHIP-5: 0–20; GOHAI: 12–60; MAI hue-degree: kisebb = jobb színkeverés; önbevallott rágóképesség: Nagyon rossz (1) … Kiváló (5), enyhe függőleges szórással a fedés ellen.") +
  theme_predict(11) + theme(panel.grid.major.x = element_blank())
save_png("abra_01_hat_beteg_kiindulas_utankovetes.png", p1, 15, 5.2)

## 6.2 Lemorzsolódás és szelekció ----------------------------------------------
if (db_ok) {
  fun <- funnel |> mutate(lepes = factor(lepes, levels = rev(lepes)), kiem = ag == "fő kohorsz")
  p2a <- ggplot(fun, aes(x = n, y = lepes, fill = kiem)) +
    geom_col(width = 0.62) +
    geom_text(aes(label = n), hjust = -0.35, size = 3.2, colour = PAL$ink) +
    scale_fill_manual(values = c(`FALSE` = PAL$blue, `TRUE` = PAL$orange), guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(title = "Tölcsér: 46 kiindulási betegből 6 teljes eset", x = "Betegek száma", y = NULL) +
    theme_predict(10.5) + theme(panel.grid.major.y = element_blank())
  rb <- recall_breakdown |> mutate(recall_statusz = factor(recall_statusz, levels = rev(recall_statusz)), kiem = grepl("^Teljes", recall_statusz))
  p2b <- ggplot(rb, aes(x = n, y = recall_statusz, fill = kiem)) +
    geom_col(width = 0.62) +
    geom_text(aes(label = n), hjust = -0.35, size = 3.2, colour = PAL$ink) +
    scale_fill_manual(values = c(`FALSE` = PAL$blue, `TRUE` = PAL$orange), guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(title = "Mi történt a többiekkel? (1. visszahívási kör)", x = "Betegek száma", y = NULL) +
    theme_predict(10.5) + theme(panel.grid.major.y = element_blank())
  strip_vars <- comp_vars$var[1:8]
  cohort_long <- cohort |>
    select(patient_id, teljes_eset, all_of(strip_vars)) |>
    pivot_longer(all_of(strip_vars), names_to = "var", values_to = "ertek") |>
    filter(is.finite(ertek)) |>
    mutate(valtozo = factor(comp_vars$cimke[match(var, comp_vars$var)], levels = comp_vars$cimke),
           csoport = ifelse(teljes_eset, "A 6 teljes eset", "A kohorsz többi tagja"),
           study_id = ifelse(teljes_eset, d6$study_id[match(patient_id, d6$patient_id)], NA))
  med <- cohort_long |> group_by(valtozo, csoport) |> summarise(med = median(ertek), .groups = "drop")
  p2c <- ggplot(cohort_long, aes(x = csoport, y = ertek)) +
    geom_jitter(data = cohort_long |> filter(!teljes_eset), width = 0.16, height = 0, colour = PAL$de_emph, size = 1.9, alpha = 0.9) +
    geom_point(data = cohort_long |> filter(teljes_eset), colour = PAL$blue, size = 2.4) +
    geom_errorbar(data = med, aes(x = csoport, ymin = med, ymax = med), width = 0.5, colour = PAL$ink, linewidth = 0.55, inherit.aes = FALSE) +
    facet_wrap(~valtozo, scales = "free_y", nrow = 2) +
    labs(title = "Hol ülnek a hatan a teljes kohorszban? (kiindulási jellemzők)",
         subtitle = "Kék: a 6 teljes eset; szürke: a kohorsz többi tagja; fekete vonal: csoportmedián. Leíró összevetés, teszt nélkül.",
         x = NULL, y = NULL) +
    theme_predict(10.5) + theme(axis.text.x = element_text(size = 8.2))
  p2 <- (p2a | p2b) / p2c + plot_layout(heights = c(1, 1.35)) +
    plot_annotation(caption = db_note, theme = theme_predict(10.5))
  save_png("abra_02_lemorzsolodas_szelekcio.png", p2, 15, 11)
}

## 6.3 Betegenkénti anatómiai térkép és siker ---------------------------------
raw_text <- function(k, i) {
  r <- d6[i, ]
  switch(k,
    F1 = paste0(label_pt(r$F1), " mm"), F3 = paste0(label_pt(r$F3), " mm"), F4 = paste0(label_pt(r$F4), "°"),
    F6 = paste0(label_pt(r$F6), "°"),
    F5 = c(`1` = "nincs", `2` = "tuber", `3` = "frontális")[as.character(r$F5)],
    F7 = c(`1` = "nincs", `2` = "plató", `3` = "orsó")[as.character(r$F7)],
    A1 = paste0("Kaán ", r$A1_Kaan), A2 = paste0(label_pt(r$A2_atlag), " mm"),
    A4 = as.character(r$A4), A5 = c(`0` = "préseli", `0.5` = "0,5", `1` = "nem szűkít", `1.5` = "1,5", `2` = "kifelé")[as.character(r$A5)],
    TUB = label_pt(r$tuberculum_score, 2),
    A11 = c(`2` = "puha", `1` = "nem elődomb.", `3` = "tömött")[as.character(r$A11)],
    A12 = c(`1` = "nem tap.", `2` = "tapintható", `3` = "érzékeny")[as.character(r$A12)],
    F2 = label_pt(r$F2, 0), F2s = formatC(r$F2_standardizalt, format = "e", digits = 1), F8 = as.character(r$F8),
    A3 = as.character(r$A3), A10 = paste0(label_pt(r$A10), "°"), "")
}
minmax <- function(x) if (diff(range(x)) == 0) rep(0.5, length(x)) else (x - min(x)) / (max(x) - min(x))
dir_preds <- predictors |> filter(irany_van)
order_ids <- d6$study_id[order(-d6$s_INDEX)]
map_df <- bind_rows(lapply(seq_len(nrow(dir_preds)), function(j) {
  k <- dir_preds$kod[j]; v <- d6[[dir_preds$var[j]]]
  tibble(study_id = d6$study_id, tetel = dir_preds$rovid[j], csoport = as.character(dir_preds$csoport[j]),
         kockazat = minmax(v), szoveg = vapply(seq_len(nrow(d6)), function(i) raw_text(k, i), character(1)))
})) |>
  mutate(study_id = factor(study_id, levels = order_ids), tetel = factor(tetel, levels = rev(dir_preds$rovid)),
         csoport = factor(csoport, levels = c("Felső állcsont", "Alsó állcsont")))
d6$anat_index <- vapply(seq_len(nrow(d6)), function(i) mean(map_df$kockazat[map_df$study_id == d6$study_id[i]]), numeric(1))
out_df <- bind_rows(
  d6 |> transmute(study_id, kimenet = "OHIP-5 (↓ jobb)", ertek = OHIP_sum_followup, z = z(s_OHIP), szoveg = as.character(OHIP_sum_followup)),
  d6 |> transmute(study_id, kimenet = "GOHAI (↑ jobb)", ertek = GOHAI_sum_followup, z = z(s_GOHAI), szoveg = as.character(GOHAI_sum_followup)),
  d6 |> transmute(study_id, kimenet = "MAI (↓ jobb)", ertek = MAI_huedegree_followup, z = z(s_MAI), szoveg = label_pt(MAI_huedegree_followup, 0)),
  d6 |> transmute(study_id, kimenet = "Önbev. rágás", ertek = ragas_fu, z = z(s_RAGAS), szoveg = chewing_today_followup),
  d6 |> transmute(study_id, kimenet = "Siker-index", ertek = s_INDEX, z = s_INDEX / sd(s_INDEX), szoveg = label_pt(s_INDEX, 2))
) |>
  mutate(study_id = factor(study_id, levels = order_ids),
         kimenet = factor(kimenet, levels = rev(c("OHIP-5 (↓ jobb)", "GOHAI (↑ jobb)", "MAI (↓ jobb)", "Önbev. rágás", "Siker-index"))))
p3a <- ggplot(map_df, aes(x = study_id, y = tetel, fill = kockazat)) +
  geom_tile(colour = PAL$surface, linewidth = 1.2) +
  geom_text(aes(label = szoveg, colour = kockazat > 0.6), size = 2.7) +
  facet_grid(csoport ~ ., scales = "free_y", space = "free_y") +
  scale_fill_gradientn(colours = SEQ_BLUE[1:6], limits = c(0, 1), name = "Kedvezőtlenség\na hat beteg között\n(0 = legkedvezőbb)") +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = PAL$ink), guide = "none") +
  scale_x_discrete(position = "top") +
  labs(title = "Anatómiai térkép: ki hordoz kedvezőtlen képleteket, és ki járt jól az új fogsorral?",
       subtitle = "Oszlopok: betegek a siker-index szerint csökkenő sorrendben (balra a legjobb kimenet).\nSötétebb kék = az elődök szerint kedvezőtlenebb változat (a hat betegen belül skálázva).",
       x = NULL, y = NULL) +
  theme_predict(10.5) + theme(panel.grid = element_blank(), legend.position = "right", strip.text.y = element_text(angle = 0))
p3b <- ggplot(out_df, aes(x = study_id, y = kimenet, fill = z)) +
  geom_tile(colour = PAL$surface, linewidth = 1.2) +
  geom_text(aes(label = szoveg, colour = z > 0.8), size = 2.7) +
  scale_fill_gradientn(colours = SEQ_ORANGE[1:6], name = "Siker (z-pont;\nsötétebb = jobb)") +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = PAL$ink), guide = "none") +
  labs(x = NULL, y = NULL, subtitle = "Állapot az új fogsorral (utánkövetés)") +
  theme_predict(10.5) + theme(panel.grid = element_blank(), axis.text.x = element_blank())
p3c <- ggplot(d6, aes(x = anat_index, y = s_INDEX)) +
  geom_hline(yintercept = 0, colour = PAL$axis, linewidth = 0.5) +
  geom_point(colour = PAL$blue, size = 3.2) +
  geom_text_repel(aes(label = study_id), size = 3, colour = PAL$ink2, seed = 3) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(title = "Összesített kép",
       subtitle = paste0("Spearman-ρ = ", fmt(safe_cor(d6$anat_index, d6$s_INDEX)), " (n = 6)"),
       x = "Anatómiai kedvezőtlenség-index (a 13 irányított tétel átlaga, 0–1)", y = "Siker-index (↑ jobb)") +
  theme_predict(10.5)
p3 <- (p3a / p3b + plot_layout(heights = c(13, 5))) | p3c + plot_layout(widths = c(2.1, 1))
save_png("abra_03_anatomiai_terkep_es_siker.png", p3, 16, 10.5)

## 6.4 Irány-egyezés hőtérkép -------------------------------------------------
hm <- assoc |>
  mutate(
    pred_rovid = factor(pred_rovid, levels = rev(predictors$rovid)),
    out_rovid = factor(out_rovid, levels = outcomes$rovid),
    csoport = factor(csoport, levels = levels(predictors$csoport)),
    blokk = factor(blokk, levels = levels(outcomes$blokk)),
    jel = case_when(!is.finite(rho) ~ "–", !irany_van ~ label_pt(rho, 2),
                    rho > 0.1 ~ paste0("✓ ", label_pt(rho, 2)), rho < -0.1 ~ paste0("✗ ", label_pt(rho, 2)), TRUE ~ label_pt(rho, 2)),
    sotet = is.finite(rho) & abs(rho) >= 0.6
  )
p4 <- ggplot(hm, aes(x = out_rovid, y = pred_rovid, fill = rho)) +
  geom_tile(colour = PAL$surface, linewidth = 1.2) +
  geom_text(aes(label = jel, colour = sotet), size = 2.9) +
  facet_grid(csoport ~ blokk, scales = "free", space = "free") +
  scale_fill_gradient2(low = PAL$red, mid = PAL$mid, high = PAL$blue, midpoint = 0, limits = c(-1, 1), na.value = "#eeeeee",
                       name = "Spearman-ρ\n(kedvezőtlen anatómia\nés rosszabb kimenet)") +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = PAL$ink), guide = "none") +
  scale_x_discrete(position = "top") +
  labs(title = "Egyezik-e a hat beteg adata az elődök tapasztalatával?",
       subtitle = "Kék, ρ > 0: a kedvezőtlennek tartott változat rosszabb eredménnyel járt (✓ egyezik). Piros, ρ < 0: fordítva (✗ ellentmond). n = 6 minden cellában; leíró rangkorreláció, teszt nélkül.",
       x = NULL, y = NULL,
       caption = "Az „irány nélküli” tételeknél ρ a nyers kódra vonatkozik (nagyobb nyers érték – rosszabb kimenet), ✓/✗ nélkül. A változás-blokkban a kiindulási MAI a régi fogpótlással készült.") +
  theme_predict(10.5) + theme(panel.grid = element_blank(), strip.text.y = element_text(angle = 0), axis.text.y = element_text(size = 8.6))
save_png("abra_04_irany_egyezes_hoterkep.png", p4, 14, 10)

## 6.5 Prior → adat → posterior ----------------------------------------------
prior_data_post_plot <- function(bdf, title, subtitle, facet = FALSE) {
  long <- bdf |>
    select(pred_rovid, csoport, out_rovid, prior_P_pos, adat_P_pos, post_P_pos) |>
    pivot_longer(c(prior_P_pos, adat_P_pos, post_P_pos), names_to = "tipus", values_to = "P") |>
    mutate(tipus = factor(tipus, levels = c("prior_P_pos", "adat_P_pos", "post_P_pos"),
                          labels = c("Prior (elődök / szakértői pool)", "Csak az adat (semleges priorral)", "Posterior = prior × hat beteg adata")),
           pred_rovid = factor(pred_rovid, levels = rev(dir_preds$rovid)),
           csoport = factor(csoport, levels = c("Felső állcsont", "Alsó állcsont")))
  seg <- bdf |> mutate(pred_rovid = factor(pred_rovid, levels = rev(dir_preds$rovid)), csoport = factor(csoport, levels = c("Felső állcsont", "Alsó állcsont")))
  g <- ggplot(long, aes(y = pred_rovid)) +
    geom_vline(xintercept = 0.5, colour = PAL$ink2, linewidth = 0.5) +
    geom_segment(data = seg, aes(x = prior_P_pos, xend = post_P_pos, y = pred_rovid, yend = pred_rovid), colour = PAL$axis, linewidth = 0.8, inherit.aes = FALSE) +
    geom_point(aes(x = P, shape = tipus, fill = tipus, colour = tipus), size = 3.1, stroke = 1) +
    scale_shape_manual(values = c(21, 21, 23), name = NULL) +
    scale_fill_manual(values = c(PAL$surface, PAL$blue, PAL$orange), name = NULL) +
    scale_colour_manual(values = c(PAL$ink2, PAL$blue, PAL$orange), name = NULL) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), labels = c("0", "25%", "50%", "75%", "100%")) +
    labs(title = title, subtitle = subtitle,
         x = "P(a kedvezőtlen változat rosszabb eredménnyel jár)  —  jobbra: az elődök iránya", y = NULL,
         caption = paste0(prior_caption, "\nLikelihood: standardizált (normal-score) egyváltozós regresszió, rácson egzaktul. A szürke szakasz a prior → posterior elmozdulás.")) +
    theme_predict(10.5) + theme(legend.position = "bottom", legend.direction = "vertical", strip.text.y = element_text(angle = 0))
  if (facet) g + facet_grid(csoport ~ out_rovid, scales = "free_y", space = "free_y") else g + facet_grid(csoport ~ ., scales = "free_y", space = "free_y")
}
b_index <- bayes |> filter(irany_van, out == "s_INDEX")
p5 <- prior_data_post_plot(b_index,
  "Mennyit mozdít hat beteg adata az elődök meggyőződésén? (siker-index)",
  "Üres kör: prior. Kék: mit mondana az adat önmagában. Narancs rombusz: posterior. Ahol a kék az 50%-tól balra van, a hat beteg az elődökkel ELLENTÉTES irányt mutat – de a posterior alig mozdul.")
save_png("abra_05_prior_adat_posterior_sikerindex.png", p5, 12.5, 8.5)
b_comp <- bayes |> filter(irany_van, out %in% c("s_GOHAI", "s_OHIP", "s_MAI", "s_RAGAS")) |>
  mutate(out_rovid = factor(out_rovid, levels = c("GOHAI", "OHIP-5", "MAI", "Önbev. rágás")))
p6 <- prior_data_post_plot(b_comp,
  "Ugyanez kimenetenként: GOHAI, OHIP-5, MAI és önbevallott rágóképesség az új fogsorral",
  "A megélt (GOHAI/OHIP/önbevallott) és az objektív (MAI) kimenet gyakran ellentétes irányban mozog ugyanannál a képletnél.", facet = TRUE)
save_png("abra_06_prior_adat_posterior_kimenetenkent.png", p6, 16, 9)

## 6.6 Prior–likelihood–posterior sűrűségek ----------------------------------
dens_items <- c("A1", "A11", "TUB", "A4", "F7", "F5")
dens_df <- bind_rows(lapply(dens_items, function(k) {
  b <- bayes |> filter(pred == k, out == "s_INDEX")
  if (!is.finite(b$r_ns)) return(NULL)
  ll <- log_lik_beta(b$r_ns, b$n); lik <- normalize(exp(ll - max(ll)))
  pr <- expert_pool[[k]]$prior; po <- normalize(pr * lik)
  tibble(tetel = paste0(b$pred_rovid, "\nprior ", fmt_pct(b$prior_P_pos), " → adat ", fmt_pct(b$adat_P_pos), " → posterior ", fmt_pct(b$post_P_pos)),
         beta = rep(BETA, 3), gorbe = rep(c("Elődök priorja", "Likelihood (hat beteg)", "Posterior"), each = length(BETA)),
         suruseg = c(pr, lik, po) / DBETA)
})) |> mutate(gorbe = factor(gorbe, levels = c("Elődök priorja", "Likelihood (hat beteg)", "Posterior")), tetel = factor(tetel, levels = unique(tetel)))
p7 <- ggplot(dens_df, aes(x = beta, y = suruseg, colour = gorbe)) +
  geom_vline(xintercept = 0, colour = PAL$axis, linewidth = 0.5) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~tetel, ncol = 3, scales = "free_y") +
  scale_colour_manual(values = c(PAL$muted, PAL$blue, PAL$orange), name = NULL) +
  labs(title = "Prior, likelihood és posterior a korrelációs skálán (siker-index)",
       subtitle = "β > 0: a kedvezőtlen anatómia rosszabb eredménnyel jár (az elődök iránya). Hat betegnél a likelihood lapos; a posterior alakját a prior adja.",
       x = "β (standardizált hatás, –1 … +1)", y = "Sűrűség",
       caption = prior_caption) +
  theme_predict(10.5) + theme(legend.position = "bottom")
save_png("abra_07_prior_likelihood_posterior_surusegek.png", p7, 13, 8)

## 6.7 Hány beteg kellene? ----------------------------------------------------
nn <- n_needed |> filter(out == "s_INDEX") |>
  pivot_longer(c(P_expert_prior, P_semleges_prior), names_to = "prior", values_to = "P") |>
  mutate(prior = factor(prior, levels = c("P_expert_prior", "P_semleges_prior"), labels = c("Elődök priorjával", "Semleges priorral")),
         panel = factor(paste0(pred_rovid, "  (r = ", label_pt(r_ns, 2), ")"), levels = unique(paste0(pred_rovid, "  (r = ", label_pt(r_ns, 2), ")"))))
ann <- n_needed_summary |> filter(out == "s_INDEX") |>
  mutate(panel = factor(paste0(pred_rovid, "  (r = ", label_pt(r_ns, 2), ")"), levels = levels(nn$panel)),
         szoveg = case_when(
           adat_iranya == "nincs jel" ~ "nincs irányjel",
           adat_iranya == "ellentmond az elődöknek" ~ paste0("elődök priorja átbillen: n ≈ ", ifelse(is.na(n_atbillenes_expert), "> 500", n_atbillenes_expert), "\nadat egyedül 95%: n ≈ ", ifelse(is.na(n_95pct_semleges), "> 500", n_95pct_semleges)),
           TRUE ~ paste0("adat egyedül 95%: n ≈ ", ifelse(is.na(n_95pct_semleges), "> 500", n_95pct_semleges))))
p8 <- ggplot(nn, aes(x = n, y = P, colour = prior)) +
  geom_hline(yintercept = c(0.5, 0.95), colour = PAL$axis, linewidth = 0.5) +
  geom_line(linewidth = 1) +
  geom_text(data = ann, aes(x = 500, y = 0.10, label = szoveg), hjust = 1, vjust = 0, size = 2.6, colour = PAL$ink2, lineheight = 0.95, inherit.aes = FALSE) +
  facet_wrap(~panel, ncol = 5) +
  scale_x_log10(breaks = c(6, 10, 20, 50, 100, 200, 500)) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  scale_colour_manual(values = c(PAL$orange, PAL$blue), name = NULL) +
  labs(title = "Hány beteg kellene, hogy az adat felülírja vagy megerősítse az elődök meggyőződését?",
       subtitle = "Gondolatkísérlet: a hat betegen megfigyelt r-t igaznak tekintve mekkora n-nél lenne az adat önmagában 95%-ig biztos az irányban (kék), és ellentmondó jelnél mikor billenne át az elődök priorja (narancs). A jelenlegi n = 6 a bal szél.",
       x = "Betegszám (log-skála)", y = "P(a kedvezőtlen változat rosszabb eredménnyel jár)",
       caption = "A számok a MEGFIGYELT hat betegre illesztett r-t extrapolálják; nem mintanagyság-becslés, hanem a prior súlyának szemléltetése.") +
  theme_predict(10.5) + theme(legend.position = "bottom")
save_png("abra_08_hany_beteg_kellene.png", p8, 15, 8.5)

## 6.8 Folytonos morfometria vs. siker-index ----------------------------------
cont <- tribble(
  ~kod, ~var, ~cimke,
  "F1", "F1", "F1 felső gerincmagasság (mm) — elődök: ↑ kedvezőbb",
  "F3", "F3", "F3 szájpadboltozat (mm) — elődök: ↑ kedvezőbb",
  "F4", "F4", "F4 gerincalak-szög (°) — elődök: ↑ kedvezőbb",
  "F6", "F6", "F6 interalveoláris szög (°) — elődök: 90° optimális",
  "A2", "A2_atlag", "A2 alsó gerincmagasság (mm) — elődök: ↑ kedvezőbb",
  "TUB", "tuberculum_score", "A6–A9 tuberculum (0–1) — elődök: ↓ kedvezőbb",
  "F2", "F2", "F2 alámenősség (mm³) — elődök: optimum, nem monoton",
  "F2s", "F2_standardizalt", "F2/L³ — elődök: optimum, nem monoton",
  "A10", "A10", "A10 állcsontreláció (°) — nincs tankönyvi irány"
)
cont_df <- bind_rows(lapply(seq_len(nrow(cont)), function(i) tibble(study_id = d6$study_id, panel = cont$cimke[i], x = d6[[cont$var[i]]], y = d6$s_INDEX))) |>
  mutate(panel = factor(panel, levels = cont$cimke))
cont_ann <- cont_df |> group_by(panel) |> summarise(rho = safe_cor(x, y), .groups = "drop") |> mutate(lab = paste0(panel, "\nρ(mérés, siker-index) = ", label_pt(rho, 2)))
cont_df <- cont_df |> mutate(panel2 = factor(cont_ann$lab[match(panel, cont_ann$panel)], levels = cont_ann$lab))
p9 <- ggplot(cont_df, aes(x = x, y = y)) +  # (ρ a fejlécben)
  geom_hline(yintercept = 0, colour = PAL$axis, linewidth = 0.5) +
  geom_point(colour = PAL$blue, size = 2.8) +
  geom_text_repel(aes(label = study_id), size = 2.6, colour = PAL$ink2, seed = 5) +
  facet_wrap(~panel2, scales = "free_x", ncol = 3) +
  labs(title = "Folytonos morfometria és a siker-index",
       subtitle = "Nyers mérések a vízszintes tengelyen; a fejlécben az elődök várt iránya. Hat pont — a mintázat csak hipotézisgenerálásra való.",
       x = "Mérés", y = "Siker-index (↑ jobb)") +
  theme_predict(10.5) + theme(strip.text = element_text(size = 8, lineheight = 0.95))
save_png("abra_09_folytonos_morfometria_sikerindex.png", p9, 14, 10)

## 6.9 Szakértői priorok tételenként: egyéni, egyesített, posterior ----------
if (!is.null(consensus) && any(consensus$n_szakerto > 0)) {
  beta_idx <- seq(1, length(BETA), by = 8)
  pool_long <- bind_rows(lapply(predictors$kod, function(k) {
    ep <- expert_pool[[k]]
    if (ep$n_expert == 0L) return(NULL)
    lab <- paste0(predictors$rovid[predictors$kod == k], "  (n = ", ep$n_expert, ")")
    ind <- if (is.null(ep$parts)) NULL else bind_rows(lapply(seq_along(ep$parts), function(i) {
      pp <- ep$parts[[i]]; d <- prior_mixture(pp$p_pos, pp$mag_mean, pp$mag_sd) / DBETA
      tibble(tetel = lab, szakerto = i, beta = BETA[beta_idx], suruseg = d[beta_idx], tipus = "egy-egy szakértő priorja")
    }))
    pooled <- tibble(tetel = lab, szakerto = 0L, beta = BETA[beta_idx], suruseg = ep$prior[beta_idx] / DBETA, tipus = "egyesített szakértői prior")
    b <- bayes |> filter(pred == k, out == "s_INDEX")
    post <- NULL
    if (nrow(b) == 1L && is.finite(b$r_ns)) {
      ll <- log_lik_beta(b$r_ns, b$n); lik <- normalize(exp(ll - max(ll))); po <- normalize(ep$prior * lik)
      post <- tibble(tetel = lab, szakerto = -1L, beta = BETA[beta_idx], suruseg = po[beta_idx] / DBETA, tipus = "posterior (hat beteg, siker-index)")
    }
    bind_rows(ind, pooled, post)
  })) |>
    mutate(tipus = factor(tipus, levels = c("egy-egy szakértő priorja", "egyesített szakértői prior", "posterior (hat beteg, siker-index)")),
           tetel = factor(tetel, levels = unique(tetel)))
  p10 <- ggplot() +
    geom_vline(xintercept = 0, colour = PAL$axis, linewidth = 0.5) +
    geom_line(data = pool_long |> filter(szakerto > 0), aes(x = beta, y = suruseg, group = szakerto, colour = tipus), linewidth = 0.4, alpha = 0.35) +
    geom_line(data = pool_long |> filter(szakerto == 0), aes(x = beta, y = suruseg, colour = tipus), linewidth = 1.2) +
    geom_line(data = pool_long |> filter(szakerto < 0), aes(x = beta, y = suruseg, colour = tipus), linewidth = 1) +
    facet_wrap(~tetel, ncol = 4, scales = "free_y") +
    scale_colour_manual(values = c(PAL$muted, PAL$orange, PAL$blue), name = NULL, drop = FALSE) +
    labs(title = "Szakértői priorok tételenként: egyéni vélemények, egyesített prior és posterior",
         subtitle = "β > 0: a kedvezőtlen változat rosszabb eredménnyel jár. Vékony szürke: egy-egy szakértő; narancs: egyenlő súlyú véleménykeverék; kék: az egyesített prior × a hat beteg adata.",
         x = "β (standardizált hatás, –1 … +1)", y = "Sűrűség",
         caption = "Egy szakértő priorja = irány-valószínűség (P[B pólus kedvezőtlenebb]) × hatásnagyság a „100 beteg” válaszból (probit-különbség → korrelációs skála), szórás a hihető tartományból.") +
    theme_predict(10) + theme(legend.position = "bottom", strip.text = element_text(size = 8))
  save_png("abra_10_szakertoi_priorok_tetelenkent.png", p10, 15, 12)

  share_long <- consensus |>
    select(cimke, csoport, B_arany, A_arany, optimum_arany, nincs_arany, nem_tudom_arany) |>
    pivot_longer(-c(cimke, csoport), names_to = "valasz", values_to = "arany") |>
    mutate(valasz = factor(valasz, levels = c("B_arany", "A_arany", "optimum_arany", "nincs_arany", "nem_tudom_arany"),
                           labels = c("B: az elődök szerinti kedvezőtlen pólus", "A: az ellenkező pólus", "optimum (nem monoton)", "nincs érdemi különbség", "nem tudom")),
           cimke = factor(cimke, levels = consensus$cimke[order(consensus$B_arany)]),
           csoport = factor(csoport, levels = levels(predictors$csoport)))
  p11a <- ggplot(share_long, aes(x = arany, y = cimke, fill = valasz)) +
    geom_col(width = 0.7, colour = PAL$surface, linewidth = 0.6) +
    facet_grid(csoport ~ ., scales = "free_y", space = "free_y") +
    scale_fill_manual(values = c(PAL$blue, PAL$red, PAL$aqua, PAL$de_emph, PAL$grid), name = NULL) +
    scale_x_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.02))) +
    labs(title = "Mit mondanak a szakértők az irányról?", x = "Válaszok aránya", y = NULL) +
    theme_predict(10) + theme(legend.position = "bottom", legend.direction = "vertical", strip.text.y = element_text(angle = 0), panel.grid.major.y = element_blank())
  cmp <- consensus |>
    mutate(cimke = factor(cimke, levels = levels(share_long$cimke)), csoport = factor(csoport, levels = levels(predictors$csoport))) |>
    select(cimke, csoport, p_tankonyv, pool_P_varhato_irany) |>
    pivot_longer(c(p_tankonyv, pool_P_varhato_irany), names_to = "forras", values_to = "P") |>
    mutate(forras = factor(forras, levels = c("p_tankonyv", "pool_P_varhato_irany"), labels = c("vizsgálatvezetői elicitáció (2026-08-19)", "szakértői véleménykeverék")))
  p11b <- ggplot(cmp, aes(x = P, y = cimke)) +
    geom_vline(xintercept = 0.5, colour = PAL$ink2, linewidth = 0.5) +
    geom_line(aes(group = cimke), colour = PAL$axis, linewidth = 0.8) +
    geom_point(aes(shape = forras, fill = forras, colour = forras), size = 3, stroke = 1) +
    facet_grid(csoport ~ ., scales = "free_y", space = "free_y") +
    scale_shape_manual(values = c(21, 23), name = NULL) +
    scale_fill_manual(values = c(PAL$surface, PAL$orange), name = NULL) +
    scale_colour_manual(values = c(PAL$ink2, PAL$orange), name = NULL) +
    scale_x_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = "P(az elődök iránya): Ön vs. a szakértők", x = "P(a kedvezőtlen változat rosszabb eredménnyel jár)", y = NULL) +
    theme_predict(10) + theme(legend.position = "bottom", legend.direction = "vertical", strip.text.y = element_text(angle = 0), axis.text.y = element_blank())
  p11 <- (p11a | p11b) + plot_layout(widths = c(1.35, 1)) +
    plot_annotation(caption = paste0("n = ", max(consensus$n_szakerto), " szakértő. A bal panel a nyers irányválaszok megoszlása; a jobb panel a keverék-priorból számított P(β > 0)."), theme = theme_predict(10))
  save_png("abra_11_szakertoi_konszenzus.png", p11, 15, 8.5)
}

# --- 12. ábra: fogorvosok és fogtechnikusok irányválaszai egymás mellett ------
if (!is.null(consensus_role) && length(unique(consensus_role$szerep)) >= 2) {
  role_long <- consensus_role |>
    filter(n_szakerto > 0) |>
    mutate(szerep = factor(szerep, levels = c("fogorvos", "fogtechnikus"), labels = c("fogorvosok", "fogtechnikusok")),
           cimke = factor(cimke, levels = consensus_role$cimke[consensus_role$szerep == "fogorvos"][order(consensus_role$B_arany[consensus_role$szerep == "fogorvos"])]),
           csoport = factor(csoport, levels = levels(predictors$csoport)))
  n_role <- expert_raw |> distinct(szakerto_id, szerep) |> count(szerep)
  p12 <- ggplot(role_long, aes(x = B_arany, y = cimke)) +
    geom_vline(xintercept = 0.5, colour = PAL$grid, linewidth = 0.5) +
    geom_line(aes(group = cimke), colour = PAL$axis, linewidth = 0.8) +
    geom_point(aes(colour = szerep, shape = szerep, size = nem_tudom_arany), fill = PAL$surface, stroke = 1.1) +
    facet_grid(csoport ~ ., scales = "free_y", space = "free_y") +
    scale_colour_manual(values = c(PAL$blue, PAL$orange), name = NULL) +
    scale_shape_manual(values = c(16, 17), name = NULL) +
    scale_size_continuous(range = c(2.2, 5), limits = c(0, 1), labels = percent_format(accuracy = 1), name = "„nem tudom megítélni” aránya") +
    scale_x_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
    labs(title = "Egyetértenek-e a fogtechnikusok a fogorvosokkal?",
         subtitle = "Azok aránya, akik szerint az elődök által kedvezőtlennek tartott változat (B) a rosszabb",
         x = "B-t rosszabbnak jelölők aránya", y = NULL,
         caption = paste0(paste0(n_role$szerep, ": n = ", n_role$n, collapse = " · "),
                          ". A pont mérete a „nem tudom megítélni” válaszok arányát mutatja; a priorba ",
                          if (EXPERT_ROLE == "all") "mindkét csoport" else paste0("csak a(z) ", EXPERT_ROLE, " csoport"), " került (PREDICT_EXPERT_ROLE).")) +
    theme_predict(10) + theme(legend.position = "bottom", legend.box = "vertical", strip.text.y = element_text(angle = 0), panel.grid.major.y = element_blank())
  save_png("abra_12_fogorvos_vs_fogtechnikus.png", p12, 15, 10)
}

# -----------------------------------------------------------------------------
# 7. QA, összefoglaló, sessionInfo
# -----------------------------------------------------------------------------
qa <- tibble(
  ellenorzes = c("Fő kohorsz elemszáma", "Betegazonosítók = a hat teljes eset", "OHIP tartomány 0–20", "GOHAI tartomány 12–60", "MAI ≥ 0",
                 "Siker-index számítható", "Szakértői prior-CSV", "Adatbázis-blokk", "Kor a hat esetnél"),
  statusz = c(ifelse(nrow(d6) == EXPECTED_MAIN_N, "OK", "HIBA"), ifelse(setequal(d6$patient_id, EXPECTED_IDS), "OK", "HIBA"),
              ifelse(all(c(d6$OHIP_sum_init, d6$OHIP_sum_followup) >= 0 & c(d6$OHIP_sum_init, d6$OHIP_sum_followup) <= 20), "OK", "HIBA"),
              ifelse(all(c(d6$GOHAI_sum_init, d6$GOHAI_sum_followup) >= 12 & c(d6$GOHAI_sum_init, d6$GOHAI_sum_followup) <= 60), "OK", "HIBA"),
              ifelse(all(c(d6$MAI_huedegree_init, d6$MAI_huedegree_followup) >= 0), "OK", "HIBA"),
              ifelse(all(is.finite(d6$s_INDEX)), "OK", "HIBA"),
              ifelse(is.null(expert_raw), "NINCS (regiszter-alapú tankönyvi prior)", paste0("OK (", length(unique(expert_raw$szakerto_id)), " szakértő, ", nrow(expert_raw), " sor; forrás: ", paste(unique(expert_raw$forras), collapse = " + "), ")")),
              ifelse(db_ok, "OK", "KIHAGYVA"), ifelse(all(is.finite(d6$kor_ev)), "OK", "hiányzik (DB nélkül)")),
  reszlet = c(paste0(nrow(d6), " beteg"), paste(sort(d6$patient_id), collapse = ", ") |> (\(x) "P01–P06 (adatbázis-id nem kerül fájlba)")(),
              "", "", "", paste0("tartomány ", fmt(min(d6$s_INDEX)), " … ", fmt(max(d6$s_INDEX))),
              ifelse(is.null(expert_raw), expert_path, paste(unique(expert_raw$szakerto_id), collapse = "; ")), db_note,
              ifelse(all(is.finite(d6$kor_ev)), paste0(fmt(min(d6$kor_ev), 0), "–", fmt(max(d6$kor_ev), 0), " év"), ""))
)
write_csv_utf8(qa, "08_QA_ellenorzes.csv")

egyezes_tab <- assoc |> filter(irany_van, blokk == "Állapot az új fogsorral", is.finite(rho)) |>
  group_by(out_rovid) |> summarise(egyezik = sum(rho > 0.1), ellentmond = sum(rho < -0.1), semleges = sum(abs(rho) <= 0.1), .groups = "drop")
top_index <- bayes |> filter(irany_van, out == "s_INDEX", is.finite(rho)) |> arrange(desc(abs(rho)))

summary_lines <- c(
  "# PREDICT – Bayes-i keretű feltáró elemzés: az elődök tapasztalata vs. hat teljes eset",
  "",
  paste0("Futás: ", format(Sys.time(), "%Y-%m-%d %H:%M"), " · fő kohorsz n = ", nrow(d6), " · ", db_note),
  "",
  "## Kimenetek az új fogsorral (állapot) és a változás",
  "",
  paste0("- OHIP-5 utánkövetés: medián ", fmt(median(d6$OHIP_sum_followup), 1), " (tartomány ", min(d6$OHIP_sum_followup), "–", max(d6$OHIP_sum_followup), "); ΔOHIP javult/változatlan/romlott = ",
         sum(d6$v_OHIP > 0), "/", sum(d6$v_OHIP == 0), "/", sum(d6$v_OHIP < 0), "."),
  paste0("- GOHAI utánkövetés: medián ", fmt(median(d6$GOHAI_sum_followup), 1), " (", min(d6$GOHAI_sum_followup), "–", max(d6$GOHAI_sum_followup), "); ΔGOHAI = ",
         sum(d6$v_GOHAI > 0), "/", sum(d6$v_GOHAI == 0), "/", sum(d6$v_GOHAI < 0), "."),
  paste0("- MAI utánkövetés: medián ", fmt(median(d6$MAI_huedegree_followup), 1), " (", fmt(min(d6$MAI_huedegree_followup), 1), "–", fmt(max(d6$MAI_huedegree_followup), 1), "); ΔMAI = ",
         sum(d6$v_MAI > 0), "/", sum(d6$v_MAI == 0), "/", sum(d6$v_MAI < 0), "."),
  paste0("- Önbevallott rágás az új fogsorral: ", paste(names(table(d6$chewing_today_followup)), table(d6$chewing_today_followup), sep = " ×", collapse = ", "), "; GRC: ",
         paste(names(table(d6$chewing_change)), table(d6$chewing_change), sep = " ×", collapse = ", "), "."),
  "",
  "## Irány-egyezés az elődökkel (13 irányított tétel, állapot-kimenetek)",
  "",
  paste0("- ", egyezes_tab$out_rovid, ": egyezik ", egyezes_tab$egyezik, " · ellentmond ", egyezes_tab$ellentmond, " · semleges ", egyezes_tab$semleges, "."),
  "",
  paste0("- Anatómiai kedvezőtlenség-index ↔ siker-index: Spearman-ρ = ", fmt(safe_cor(d6$anat_index, d6$s_INDEX)), "."),
  "",
  "## Siker-index: prior → adat → posterior (|ρ| szerint)",
  "",
  paste0("- ", top_index$pred_rovid, ": ρ = ", fmt(top_index$rho), "; P(elődök iránya) prior ", fmt_pct(top_index$prior_P_pos),
         " → adat ", fmt_pct(top_index$adat_P_pos), " → posterior ", fmt_pct(top_index$post_P_pos), "."),
  "",
  "## Értelmezési korlát",
  "",
  "Hat beteg, leíró rangkorrelációk és rácsalapú posteriorok teszt nélkül. A prior hatásnagyság-része",
  "helyőrző (HN 0; 0,5) a szakértői interjúkig; a posterior ezért döntően a priort tükrözi. A kiindulási",
  "MAI a régi fogpótlással készült; szelektált recall-minta; okozati állítás nem tehető."
)
writeLines(summary_lines, file.path(OUT_DIR, "09_osszefoglalo.md"), useBytes = TRUE)
writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "10_sessionInfo.txt"), useBytes = TRUE)
if (!is.null(DB_CON)) try(DBI::dbDisconnect(DB_CON), silent = TRUE)

cat("\n=== PREDICT Bayes-i feltáró elemzés kész ===\n")
cat("Kimeneti könyvtár: ", OUT_DIR, " (", length(list.files(OUT_DIR)), " fájl)\n", sep = "")
cat(db_note, "\n")
cat("\n--- Betegszintű tábla ---\n"); print(as.data.frame(patient_table |> select(study_id, nem, kor_ev, OHIP_init, OHIP_fu, GOHAI_init, GOHAI_fu, MAI_init, MAI_fu, ragas_fu, GRC_ragas, siker_index)), row.names = FALSE)
if (db_ok) { cat("\n--- Tölcsér ---\n"); print(as.data.frame(funnel |> select(lepes, n)), row.names = FALSE)
  cat("\n--- Recall-státusz ---\n"); print(as.data.frame(recall_breakdown), row.names = FALSE)
  cat("\n--- Szelekció (medián: teljes 6 vs. többi) ---\n"); print(as.data.frame(comparison |> transmute(valtozo, teljes_n, teljes_median = round(teljes_median, 2), tobbi_n, tobbi_median = round(tobbi_median, 2))), row.names = FALSE)
  print(as.data.frame(gender_rows), row.names = FALSE) }
cat("\n--- Irány-egyezés (Spearman-ρ; + = egyezik az elődökkel) ---\n")
print(as.data.frame(assoc |> select(pred_rovid, out_rovid, rho) |> mutate(rho = round(rho, 2)) |> pivot_wider(names_from = out_rovid, values_from = rho)), row.names = FALSE)
if (!is.null(consensus)) { cat("\n--- Szakértői konszenzus (pool) ---\n"); print(as.data.frame(consensus |> transmute(cimke, n = n_szakerto, B = round(B_arany, 2), A = round(A_arany, 2), opt = round(optimum_arany, 2), nincs = round(nincs_arany, 2), nt = round(nem_tudom_arany, 2), p_atlag = round(p_irany_atlag, 0), diff_atlag = round(kulonbseg_atlag, 1), pool_P = round(pool_P_varhato_irany, 3), pool_med = round(pool_median, 2), q05 = round(pool_q05, 2), q95 = round(pool_q95, 2), sdP = round(egyet_nem_ertes_sd_P, 2))), row.names = FALSE) }
cat("\n--- Bayes: siker-index ---\n")
print(as.data.frame(bayes |> filter(out == "s_INDEX") |> transmute(pred_rovid, rho = round(rho, 2), prior = round(prior_P_pos, 3), adat = round(adat_P_pos, 3), posterior = round(post_P_pos, 3), post_median = round(post_median, 2), post_q05 = round(post_q05, 2), post_q95 = round(post_q95, 2))), row.names = FALSE)
cat("\n--- Hány beteg kellene? (siker-index) ---\n")
print(as.data.frame(n_needed_summary |> filter(out == "s_INDEX") |> transmute(pred_rovid, r = round(r_ns, 2), adat_iranya, P_n6 = round(P_expert_n6, 2), P_n30 = round(P_expert_n30, 2), P_n100 = round(P_expert_n100, 2), n_atbillenes_expert, n_95pct_semleges)), row.names = FALSE)
cat("\n--- QA ---\n"); print(as.data.frame(qa), row.names = FALSE)
