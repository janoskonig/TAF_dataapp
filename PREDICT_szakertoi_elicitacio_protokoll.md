# PREDICT szakértői elicitáció – protokoll (v2.0, 2026-09-06)

Strukturált szakértői becslés az anatómiai adottságok és a teljes lemezes
fogsor sikere közötti kapcsolatról, Bayes-i priorok előállítására és önálló
közlésre. A módszertan a SHELF (Sheffield Elicitation Framework) elveit követi:
felkészítés, egyértelműen definiált mennyiségek, egyéni, egymástól független
becslés, kimondott lefedettségű bizonytalanság, ellenőrzés és egyenlő súlyú
véleménykeverés. Az EFSA útmutatójának megfelelően a betegek közötti
változatosságot és a tudás bizonytalanságát külön kérdezzük.

## 1. Cél és becsülendő mennyiségek

* **Elsődleges kimenet (vizsgálatvezetői döntés, 2026-09-06):** a
  siker-index az OHIP-5-, GOHAI- és MAI-változásból, három hónappal az
  átadás után (kiindulás a régi fogsorral → 3 hónap az újjal). Folytonos
  alak: a három javulás standardizált átlaga (Δ-siker-index). Kétértékű alak
  a szakértői kérdéshez: sikeres a fogsor, ha a beteg a régi fogsorához képest
  klinikailag érzékelhetően javult: jobban rág a rágástesztben, és jobb a
  szájegészséggel kapcsolatos életminősége mindkét kérdőívben; előzetes
  küszöb: a három változás MCID-egységben kifejezett átlaga ≥ 1 (a MCID-
  elemzés alapján véglegesítendő). A fogtechnikusi változat ugyanezt a mért
  meghatározást használja, kiegészítve azzal, hogy ezt a betegen mérjük, nem a
  laborba visszakerülésből.
* **Paraméter tételenként:** a sikertelenség esélyhányadosának logaritmusa a
  B pólus (az elődök szerint kedvezőtlen változat) és az A pólus között, a többi
  adottság azonos értéke mellett (feltételes hatás): β = logit(1 − p_B) −
  logit(1 − p_A). β > 0: a B változat mellett több a sikertelen fogsor, azaz a
  válasz egyezik az elődök tanításával.
* **Az optimum alakú tételeknél** (F2 alámenősség, F6 gerincélek eltérése) a
  görbület is: c = logit(1 − p_M) − ½·[logit(1 − p_A) + logit(1 − p_B)]; c < 0:
  a közepes forgatókönyv a legjobb.
* A másodlagos kimenetekre (OHIP, GOHAI, MAI) nem elicitálunk; ott gyengén
  informatív prior, a szakértői prior legfeljebb érzékenységi elemzésként.

## 2. Résztvevők

* Fogorvosok: legalább tíz év teljes fogsoros gyakorlat, lehetőleg egyetemi és
  magánpraxis vegyesen; cél 5–10 fő.
* Fogtechnikusok: évtizedek óta főleg kivehető pótlást készítők, egyetemi és
  magánlabor vegyesen; cél 3–5 fő. Külön csoport, nem egyenlő súllyal a
  fogorvosi poolban; a becslés forrását (rendszeres fogorvosi visszajelzés,
  visszakerülő munkák, közvetlen betegkapcsolat) rögzítjük.
* Egyéni, távoli, egymástól független kitöltés; a válaszokat kód azonosítja.

## 3. Felkészítés (SHELF „training”)

A kérdőív előtt kötelezően megjelenő oldal: (1) a kétféle bizonytalanság
elkülönítése; (2) a határok jelentése: húsz hasonló becslésből tizenkilenc
ezen belül; (3) az irány-bizonyosság mint hitfok, nem betegarány; (4)
kidolgozott példa; (5) három gyakorlókérdés ismert válasszal (Budapest–Bécs
távolság, Semmelweis életkora, 65 év felettiek aránya), azonnali
visszajelzéssel a lefedettségről, és egy irány-bizonyossági gyakorlat
(Duna–Rajna). A gyakorlat válaszai tárolódnak (calibration.gyak_*), a
kalibráció leírásához.

## 4. Az eszköz

Tizenhat tétel (F1–F8 felső, A1–A12 alsó állcsont; a tuberculum négy
jellemzője egy tételben). Tételenként:

1. **Irány:** A rosszabb / B rosszabb / a közepes a legjobb (optimum tételek) /
   nincs érdemi különbség / nem tudom megítélni.
2. **Irány-bizonyosság** (hitfok): 50–99 %; csak irányos válasznál.
3. **Száz beteg** pólusonként: legvalószínűbb szám és alsó–felső határ (19/20
   lefedettség); optimum tételeknél a közepes forgatókönyv is.
4. **„A nagyságát nem tudom megbecsülni”** jelölés: ekkor csak irány-prior
   készül, a hiány tudatos válasz.
5. Küszöbkérdés, résztételek, megjegyzés (nem kötelező).

A folytonos tételek pólusai konkrét mérési értékek (a kohorsz alsó és felső
ötödéből kerekítve, a vizsgálatvezető által módosítható):

| tétel | A pólus | közepes | B pólus | egység |
|---|---|---|---|---|
| F1 felső gerincmagasság | 10 | – | 5 | mm |
| F3 szájpadboltozat | 25 | – | 17 | mm |
| F4 gerincív szöge | 140 | – | 125 | ° |
| F6 gerincélek eltérése | ≤ 5 (2,5) | 10 | ≥ 20 | ° eltérés 90°-tól |

Az A10 kategoriális (Angle I. vs. II./III.); az elemzésben a mért szög
küszöbölt változata (A10k) felel meg neki. Az F2 alámenősség szóbeli
forgatókönyvekkel (kevés / közepes / nagy).

Háttér: fogorvosnál diploma éve, szakvizsga; fogtechnikusnál képesítés éve,
mesterfogtechnikusi vizsga, a visszajelzés forrása; mindkettőnél gyakorlati
évek, összes és évi teljes fogsor, oktatás, munkahely. Általános kérdések:
B1 alapráta (100 betegből hány sikeres), B2 az adottságok súlya, B3 állcsontok
súlya, B4 implantátumos fedőlemezes pótlás javallata, B5 korábbi fogsor.
Záró: öt legfontosabb tétel, legrosszabb páros, kiegyenlítő adottság,
hiányzó adottság, önértékelés.

## 5. Konzisztencia-szabályok (beküldést gátló)

* A számok iránya nem mondhat ellent a jelölt iránynak.
* A legvalószínűbb szám essen az alsó és a felső határ közé; alsó ≤ felső.
* „Nincs érdemi különbség” mellett a két szám legfeljebb 9 beteggel térhet el.
* „A közepes a legjobb” mellett a közepes szám a legmagasabb.
* Irányos és optimum válasznál a pontbecslés és a határok kötelezők, kivéve
  ha a nagyságot nem tudja megbecsülni.

Az irány-bizonyosság és a pólusokból implikált P(β > 0) eltérését az elemzés
kalibrációs jelzésként rögzíti (nem gátol, nem számít kétszer).

## 6. Elemzés (predict_szakertoi_prior_logit.R)

1. Pólusonként béta-eloszlás momentum-illesztéssel (várható érték = legvalószínűbb
   szám; szórás = (felső − alsó) / 3,92). Monte-Carlo (4000 minta) → β.
2. Választípusonként: irányos + számok → β-minta; csak irány → előjel a
   hitfokból, |β| ~ félnormális(0; 1); nincs különbség → N(0; 0,15); optimum →
   lineáris és görbület-komponens; nem tudom → kimarad.
3. Egyenlő súlyú lineáris pool tételenként; kimenet: β átlag és szórás
   (normális közelítés a modellhez), P(β > 0), OR-kvantilisek; a mérési
   pólusokkal egységnyi meredekség is.
4. Prior prediktív ellenőrzés: a pólusok 50/50 keverékéből következő sikerarány
   vs. a B1 alapráta.
5. Szerep szerinti bontás (fogorvos vs. fogtechnikus), PREDICT_EXPERT_ROLE
   választja a poolba kerülő csoportot (alap: fogorvos).
6. Modell a betegadaton: a kétértékű sikerre Bayes-i logisztikus regresszió, a
   folytonos Δ-siker-indexre lineáris modell, mindkettő az összes tétellel,
   a koefficienseken az egyesített normális priorral (MCMCpack::MCMClogit;
   Stan-alapú változat a fordítási környezet javítása után), semleges és
   szakértői priorral egyaránt (érzékenység), posterior prediktív ellenőrzés.
   A szkript szimulált adaton mutatja be a folyamatot (abra_04).
7. A kis elemszámú feltáró elemzés (predict_bayes_feltaro.R) ugyanezekből a
   válaszokból korrelációs skálájú priort készít; a választípusok ott is külön
   kezeltek, és a szakértői prior csak az elsődleges kimenetre (Δ-siker-index)
   kerül rá, a többi kimenet semleges priorral fut (PREDICT_PRIOR_ALL_OUTCOMES=1:
   érzékenységi futás).

## 7. Adatkezelés, riportálás

* Név csak a vizsgálatvezetőnél; export kóddal (SZnn), szereppel, minden
  elicitált mennyiséggel (predict_expert_priorok export, 26 oszlop) és a
  háttérrel (hatter export).
* Közléshez: résztvevők leírása, felkészítés, az eszköz (PREDICT_szakertoi_
  prior_urlap.docx / EN / fogtechnikusi változatok), a válaszok megoszlása,
  egyetértés (P(β > 0) tételenként, szakértők közötti szórás), kalibrációs
  gyakorlat eredménye, összevetés az elődök tanításával, fogorvos–fogtechnikus
  különbség, érzékenység a pool-szabályra.
* Verziók: v1.2 (2026-09-06 délelőtt, különbség-tartomány), v2.0 (pólusonkénti
  tartomány, mérési pólusok, konzisztencia-ellenőrzés, felkészítő, fogtechnikusi
  változat). v1 válasz nem érkezett.
