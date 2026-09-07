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
  a szakértői kérdéshez (a kérdőívben és a kódban azonos szabály): sikeres a
  fogsor, ha a beteg a régi fogsorához képest együttesen, klinikailag
  érzékelhetően javult a rágástesztben és a két életminőség-kérdőívben, és
  egyik mérésben sem romlott érdemben. Számítás: a három változás MCID-
  egységben kifejezett átlaga ≥ 1 ÉS egyik mérés sem rosszabb −0,5 MCID-nél
  (MCID: OHIP-5 4,5; GOHAI 16; MAI 8,6 hue-fok; a MCID-elemzés alapján
  véglegesítendő). Az összesített javulás tehát kompenzálhat, de érdemi
  romlás egy mérésben kizárja a sikert. A fogtechnikusi változat ugyanezt a mért
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
* **Hólabda-toborzás:** a meghívottak a beküldés után (és a saját válaszuk
  oldalán) legfeljebb három kollégát ajánlhatnak névvel és e-mail-címmel; a
  rendszer személyes linkkel küldi nekik a felkérést, az ajánló kódját az
  ajánlott háttéradatában rögzíti (`ajanlo_kod`, export oszlop), egy
  e-mail-cím csak egyszer hívható meg. A résztvevői folyamatábrán a közvetlen
  és az ajánlott meghívottak külön szerepelnek.
* **Határidő:** a felkérő levél küldésének napjától számított két hét
  (EXPERT_DEADLINE_DAYS), a levélben a nyelv szerint formázva; az emlékeztető
  ugyanezt a dátumot ismétli.

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
3. **Száz beteg**, az iránytól függő körben: irányos válasznál pólusonként
   (A, B), „a közepes a legjobb” válasznál a közepes forgatókönyvvel is, „nincs
   érdemi különbség” válasznál egyetlen közös számmal (mindkét változatnál),
   „nem tudom megítélni” válasznál szám nélkül; mindenhol legvalószínűbb szám
   és alsó–felső határ (19/20 lefedettség).
4. **„A nagyságát nem tudom megbecsülni”** jelölés: ekkor csak irány-prior
   készül, a hiány tudatos válasz.
5. Küszöbkérdés, résztételek, megjegyzés (nem kötelező).

A folytonos tételek pólusai konkrét mérési értékek (a kohorsz alsó és felső
ötödéből kerekítve, a vizsgálatvezető által módosítható):

| tétel | A pólus | közepes | B pólus | egység |
|---|---|---|---|---|
| F1 felső gerincmagasság | 10 | – | 5 | mm |
| F3 szájpadboltozat | 25 | – | 17 | mm |
| F4 gerincív szöge (tuber – locus caninus – papilla incisiva szög, két oldal átlaga) | 140 | – | 125 | ° |
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

1. Pólusonként béta-eloszlás a kvantilisekre illesztve: a „legvalószínűbb
   szám” a módusz, az alsó és felső határ a 2,5 % és 97,5 % kvantilis; két
   paraméter, három célérték, négyzetes eltérés minimalizálása. Diagnosztika
   szakértőnként: a megadott sávra jutó illesztett valószínűség (jelzés, ha
   < 0,85) és az illesztett módusz eltérése (jelzés, ha > 10 pont; erősen
   aszimmetrikus sávnál a három feltétel nem teljesíthető egyszerre, ilyenkor
   visszakérdezés). Monte-Carlo (4000 minta) → β.
2. Választípusonként: irányos + számok → β-minta; csak irány → előjel a
   hitfokból, |β| ~ félnormális(0; 1); nincs érdemi különbség → nulla
   középpontú normális, amelynek 90 %-os sávja ± 5 sikeres beteg / 100 a
   szakértő saját sikerszintjén (az űrlap tűrése 10 pont), a pólus-számok
   itt nem középpont; optimum → lineáris és görbület-komponens; nem tudom →
   kimarad.
3. Egyenlő súlyú lineáris pool tételenként; kimenet: β átlag és szórás
   (normális közelítés a modellhez), P(β > 0), OR-kvantilisek; a mérési
   pólusokkal egységnyi meredekség is.
4. Prior prediktív ellenőrzés: a pólusok 50/50 keverékéből következő sikerarány
   vs. a B1 alapráta.
5. Szerep szerinti bontás (fogorvos vs. fogtechnikus), PREDICT_EXPERT_ROLE
   választja a poolba kerülő csoportot (alap: fogorvos).
5b. Prior és adat azonos skálán, már a pilot-mintán: tételenként 2×2 tábla
   (pólus × siker), P(sikertelen | A) = plogis(α), P(sikertelen | B) =
   plogis(α + β), α széles priorral kiintegrálva, β-ra az egyesített szakértői
   prior normális közelítése; rácsos posterior, semleges priorral is
   (07_bayes_binaris_siker.csv, 6. ábra). Ez a szakértői prior elsődleges
   felhasználása; nagy mintánál ezt váltja a többváltozós modell.
6. Modell a betegadaton: a kétértékű sikerre Bayes-i logisztikus regresszió, a
   folytonos Δ-siker-indexre lineáris modell, mindkettő az összes tétellel,
   a koefficienseken az egyesített normális priorral (MCMCpack::MCMClogit;
   Stan-alapú változat a fordítási környezet javítása után), semleges és
   szakértői priorral egyaránt (érzékenység), posterior prediktív ellenőrzés.
   A szkript szimulált adaton mutatja be a folyamatot (abra_04).
6b. Betegadatok a szakértői ábrákon: a hat longitudinális beteg minden tételnél
   a mért érték alapján az A vagy B pólushoz (optimum tételnél a közepeshez)
   sorolódik a kérdés küszöbeivel; a protokoll szerinti siker (MCID-egység
   átlag ≥ 1) pólusonként sikeres/összes arányként kerül az 5. ábrára a
   szakértők pólusonkénti becslései mellé, és Haldane-korrigált esélyhányadosként
   az 1. ábrára a prior mellé (PREDICT_PATIENT_CSV; a kimenetben csak P01…
   sorszám, azonosító nem). Ugyanez a mechanizmus fogadja majd a teljes kohorszot.
7. A kis elemszámú feltáró elemzés (predict_bayes_feltaro.R) ugyanezekből a
   válaszokból korrelációs skálájú priort készít a folytonos Δ-siker-indexre;
   ez KÖZELÍTÉS (a sikerarány-különbség nem határozza meg a folytonos index
   együtthatóját), a kimenetekben így is jelölve, PREDICT_APPROX_PRIOR=0 esetén
   kikapcsolható (minden kimenet semleges priorral). A választípusok ott is
   külön kezeltek, a közelítő prior csak az elsődleges kimenetre kerül rá
   (PREDICT_PRIOR_ALL_OUTCOMES=1: érzékenységi futás).

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
