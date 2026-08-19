# PREDICT Bayes-i prior-specifikáció (v0.2)

**Státusz:** az első, változónkénti kvalitatív szakértői elicitation
összefoglalása (2026-08-19). A klinikai állításokat a PREDICT vezető
vizsgálója adta meg egyenként. A számszerű priorok és az outcome-hozzárendelés
még nem véglegesek.

## 1. Alapelv

A klasszikus protetikai tudás nem közvetlen betegszintű kimeneti
valószínűség, hanem a prognosztikai modell megfelelő paraméterének priorját
informálja. Megfigyeléses adatok mellett a posterior eredményt prognosztikai
összefüggésként, nem automatikusan okozati hatásként értelmezzük.

Minden kimenetet később egységesen úgy kódolunk, hogy a magasabb érték
rosszabbat jelentsen. Így kedvezőtlen anatómiai rizikókódolás mellett
`beta > 0` rosszabb prognózist jelent. A szakértő az iránybizonyosságot külön
kalibrálta a hatásnagyságtól:

- **gyenge iránybizonyosság:** `P(várt irány) = 0,80`;
- **mérsékelt iránybizonyosság:** `P(várt irány) = 0,90`;
- **erős iránybizonyosság:** `P(várt irány) = 0,98`.

Ezek nem a hatás százalékos nagyságai. A hatásnagyság-prior az outcome-skála
lezárásáig nyitott; a korábban felmerült 15–25%-os értéket a szakértő
kifejezetten visszavonta, ezért az nem része a specifikációnak.

## 2. Változónkénti szakértői elicitation

### F1 — felső állcsontgerinc profilja/magassága

- **Klasszikus állítás:** magasabb gerinc mellett nagyobb a fogsor által
  fedett felszín. A nagyobb felszín erősebb adhéziót és jobb retenciót ad.
- **Kedvezőtlen irány:** kisebb gerincmagasság.
- **Hatásalak:** lineárisnak feltételezett.
- **Meggyőződés:** erős.
- **Közvetlen mechanikai kimenet:** retenció.
- **MAI/QoL-hozzárendelés:** nyitott.
- **Javasolt prediktorkód:** `risk_F1 = -z(gerincmagasság)`.

### F2 — felső alámenős területek nagysága

- **Klasszikus állítás:** a növekvő alámenősség kezdetben javítja a
  retenciót, de túl nagy alámenősség fájdalmassá vagy nehézzé teszi a fogsor
  behelyezését és eltávolítását.
- **Hatásalak:** nem lineáris, optimális tartománnyal.
- **Meggyőződés:** erős.
- **Mechanizmusok:** retenciós előny és azzal ellentétes behelyezési/fájdalmi
  hátrány.
- **Az optimum helye:** nyitott.
- **MAI/QoL-hozzárendelés:** nyitott.
- **Modellkövetkezmény:** egyetlen lineáris beta helyett spline vagy
  előre definiált görbületi modell; az eltérő mechanikai és fájdalmi utak
  lehetőség szerint külön kimeneten.

### F3 — szájpadboltozat magassága

- **Klasszikus állítás:** magasabb boltozat kúposabb formát ad, amelyhez az
  alaplemez kúpos kötéshez hasonlóan illeszkedhet.
- **Kedvezőtlen irány:** kisebb boltozati magasság.
- **Hatásalak:** lineárisnak feltételezett.
- **Meggyőződés:** erős, de nem abszolút erős.
- **Közvetlen mechanikai kimenet:** retenció.
- **MAI/QoL-hozzárendelés:** nyitott.
- **Javasolt prediktorkód:** `risk_F3 = -z(boltozatmagasság)`.

### F4 — felső állcsontgerinc alakja/szöge

- **Klasszikus állítás:** nagyobb szög négyzetesebb állcsontformát jelez.
  Ilyenkor nagyobb eséllyel állíthatók a műszemfogak a gerincélvonalra.
  Kisebb szögnél a gerincen kívülre kerülő szemfogak külpontosan terhelhetik
  és elmozdíthatják a felső fogsort.
- **Kedvezőtlen irány:** kisebb szög.
- **Hatásalak:** küszöbös/platózó; a küszöb felett további lényeges előny nem
  feltételezett.
- **Meggyőződés:** mérsékelt.
- **Közvetlen mechanikai kimenet:** stabilitás.
- **MAI/QoL-hozzárendelés:** nyitott.

### F5 — lötyögő, csontmag nélküli gerinc

- **Klasszikus állítás:** a hiány kedvező; a jelenlét bármely lokalizációban
  kedvezőtlen.
- **Mechanizmus:** instabil alátámasztás és nyomási fájdalom.
- **Meggyőződés a jelenlét kedvezőtlen hatásában:** gyenge.
- **Lokalizáció:** a frontális forma talán rosszabb a tuberálisnál, de ez is
  gyenge feltételezés; a tuberális forma ritkább.
- **Elsődleges kód:** `nincs` vs `bárhol jelen van`.
- **Másodlagos kontraszt:** frontális vs tuberális, erős regularizálással.
- **MAI/QoL-hozzárendelés:** nyitott.

### F6 — interalveoláris vonal és rágósík szöge

- **Klasszikus állítás:** azt mutatja, hogy a gerincélvonalak vertikálisan
  egybeesnek-e. Ha az alsó szélesebb, a szabályos ollóharapás a felső
  műfogakat a felső gerincélvonalon kívülre kényszerítheti, külpontosan
  terhelve és destabilizálva a felső fogsort.
- **Kedvező érték:** körülbelül 90 fok.
- **Kedvezőtlen irány:** a 90 foktól való eltérés; a nyers előjel a mérési
  konvenciótól függ.
- **Hatásalak:** nem ismert.
- **Meggyőződés:** erős.
- **Javasolt prediktorkód:** `abs(F6 - 90)`; rugalmas hatásalak.
- **Közvetlen mechanikai kimenet:** stabilitás.

### F7 — torus palatinus

- **Klasszikus állítás:** bármilyen torus kedvezőtlen a hiányához képest.
- **Mechanizmus:** akadályozza a szívóhatást. Az orsó alak nyomásérzékenyebb,
  alámenősséget tartalmaz és nehezebb lenyomatozni. A plató alak technikai
  nehézsége tapasztalat szerint enyhe gipszminta-gravírozással kezelhető.
- **Meggyőződés, jelenlét vs hiány:** erős.
- **Meggyőződés, orsó vs plató:** gyenge az orsó kedvezőtlenebb voltában.
- **Kódolás:** két kontraszt: `bármilyen torus vs nincs`, illetve
  `orsó vs plató`.
- **Közvetlen mechanikai kimenet:** retenció, másodlagosan nyomási fájdalom.

### F8 — antagonista fogazat a felső fogpótlásnál

- **Klasszikus állítás:** az antagonista felől érkező erők a fogsorra ható
  legerősebb erők közé tartoznak. Rossz irányú erők elmozdíthatják a fogsort.
  Újonnan készülő antagonista esetén a fogérintkezések felett teljesebb a
  kontroll. Kivehető antagonista kisebb erőt képes kifejteni, mint a
  természetes/rögzített fogazat.
- **Fontos feltétel:** a 2-es és 3-as kategória önmagában nem kedvezőtlen;
  akkor válik azzá, ha az erők iránya biomechanikailag hibás, például sorvadt
  maxilla mellett erőltetett ollóharapáskor.
- **Meggyőződés:** erős.
- **Modellszerep:** nem egyszerű irányított főhatás, hanem okklúziós/anatómiai
  interakciós vagy effektusmódosító tényező, különösen F4/F6 mellett.
- **Közvetlen mechanikai kimenet:** stabilitás.

### F9 — garatreflex

- **Klasszikus állítás:** erős garatreflex mellett az alaplemez kiterjesztése
  elmaradhat az optimálistól; a lenyomatvétel gyorsabb/pontatlanabb lehet; a
  fogsor által kiváltott inger tartós betegterhelést jelenthet.
- **Meggyőződés:** gyenge.
- **Kódértelmezés:** a 2-es kategória előre jelzett, de tényleges fennakadást
  nem okozó garatreflex. Prognosztikailag az 1-eshez hasonló. Az elsődleges
  kontraszt `ténylegesen befolyásolta a kezelést: igen/nem`.
- **Időbeliség:** a tényleges kezelési fennakadás kezelés közben megismert
  folyamatváltozó, nem tiszta baseline prediktor.
- **Lehetséges utak:** pontatlanabb illeszkedés/retenció és közvetlen
  viselési megterhelés/QoL.
- **Prediktorblokk:** nem anatómiai, hanem beteg-/kezelési tényező.

### A1 — mandibularis gerincforma Kaán szerint

- **Klasszikus állítás:** az 1→5 kategóriák egyre rosszabbnak tartott
  gerincformák.
- **Mechanizmus:** a felső gerinchez hasonlóan romló alátámasztás és
  stabilitás.
- **Hatásalak:** monoton romló, de telítődő; a 4-es és 5-ös fok között már
  nincs nagy prognosztikai eltérés.
- **Meggyőződés a főhatásban:** erős.
- **Interakció:** valódi, szuperadditív interakciók várhatók más kedvezőtlen
  mandibularis képletekkel. A partnereket az adott változóknál jelöljük.
- **Kódolás:** nem egyszerű lineáris 1–5 pont és nem önkényes dichotómia;
  monoton, telítődő ordinális hatás.

### A2 — mandibularis gerincprofil/magasság modellanalízissel

- **Konstrukció:** ugyanazt a gerincállapotot méri, mint A1, de A1
  szemrevételezéses ordinális, A2 pedig modellanalízissel meghatározott
  folytonos mérés.
- **Irány, mechanizmus és meggyőződés:** az A1-ével azonos.
- **Modellkövetkezmény:** A1 és A2 nem két független anatómiai hatás.
  Alternatív mérések vagy közös látens mandibularis gerincállapot indikátorai.

### A3 — buccinator tasak

- **Szakértői állítás:** nincs előzetesen kedvező vagy kedvezőtlen forma.
- **Megfigyelés:** a lebenyezett felszín általában alacsonyabb gerinc mellett
  fordul elő.
- **Cél:** feltáró vizsgálat arra, van-e önálló prognosztikai hatása a
  gerincállapotra történő korrekció után.
- **Priorirány:** nincs; nulla körüli, szimmetrikus regularizáló prior.

### A4 — torus mandibularis

- **Klasszikus állítás:** a jelenlét kedvezőtlen; a nagy torus várhatóan
  rosszabb a kicsinél.
- **Mechanizmus:** nyomásérzékenység és a szívóhatás akadályozása.
- **Meggyőződés, jelenlét vs hiány:** erős.
- **Meggyőződés, nagy vs kis:** gyenge.
- **Kódolás:** `bármilyen torus vs nincs`, majd `nagy vs kis` kontraszt.
- **Aszimmetria:** irány nélküli, feltáró.
- **Közvetlen kimenetek:** retenció és fájdalom.

### A5 — lingualis tasak

- **Sorrend:** `mandibulához préseli az ujjat` (legjobb) → `nem szűkíti`
  (köztes) → `kifelé préseli` (legrosszabb).
- **Mechanizmus:** a fogsorra ható izomerő ne kifelé mutasson, mert akkor az
  izmok elmozdítják a fogsort az alapjáról.
- **Meggyőződés:** erős.
- **Interakció:** A5 × A11 szájfenék, kedvezőtlen irányban, gyenge
  meggyőződéssel.
- **Kódolási következmény:** a kedvező nyers referenciakategória **2**, nem 1.
- **Közvetlen mechanikai kimenet:** stabilitás.

### A6 — feszes ínyborítás a tuberculumon

- **Sorrend:** `az egészet borítja` (legjobb) → `elülső harmadát` →
  `egyáltalán nem` (legrosszabb).
- **Mechanizmus:** feszes íny alatt a tuberculum nem mozog és nem változtatja
  alakját; az alaplemez ráterjesztve hatásos megtámasztási felületet kap.
- **Meggyőződés:** erős.
- **Konstrukció:** az A7–A9-cel közös tuberculum-konstrukció indikátora.
- **Aszimmetria:** irány nélküli, feltáró.

### A7 — tuberculum alakja és mérete

- **Sorrend:** `fordított körte` (legjobb) → `kicsi, de elkülönülő` →
  `plicaszerű` (legrosszabb).
- **Mechanizmus:** a forma meghatározza az alaplemez kiterjeszthetőségét; a
  plicaszerű formára nem terjeszthető rá megfelelően az alaplemez.
- **Meggyőződés:** erős.
- **Konstrukció:** az A6 ugyanennek a kis képletnek egy másik jellemzése;
  együttjárás, nem külön mechanizmus vagy valódi interakció.
- **Terminológia:** az A7 plicaszerű kategóriája nem plica
  retromylohyoidea. A plica retromylohyoidea jelenleg nem PREDICT-változó.
- **Aszimmetria:** irány nélküli, feltáró.

### A8 — tuberculum–gerinc inklinációs szög

- **Klasszikus állítás:** minél kisebb a szögeltérés, annál kedvezőbb.
- **Mechanizmus:** jelentős eltérésnél a vertikális terhelésből oldalirányú
  elmozdító komponens, „csúszdaeffektus” keletkezhet.
- **Meggyőződés:** erős.
- **Kódértelmezés:** a 3-as `plicaszerű, nem vizsgálható` nem további
  súlyossági fok; jelentős eltérést és egyben az A7 válaszának validációját
  jelenti.
- **Kódolás:** `nincs eltérés` vs `jelentős eltérés`, külön
  konzisztenciajelzővel.
- **Konstrukció:** az A6–A9 közös tuberculum-konstrukció része.
- **Aszimmetria:** irány nélküli, feltáró.

### A9 — tuberculum alakváltozása nyitás-záráskor

- **Sorrend:** `nem változik` (legjobb) → `kissé változik` → `teljesen
  mozgékony` (legrosszabb).
- **Mechanizmus:** stabil tuberculum teljesen bevonható az alátámasztásba;
  részleges változásnál csak a stabil rész fedhető; teljes mobilitásnál nem
  ad stabil megtámasztást.
- **Meggyőződés:** erős.
- **Konstrukció:** az A6–A9 közös tuberculum-konstrukció része.
- **Aszimmetria:** irány nélküli, feltáró.

### A10 — állcsontreláció szöge

- **Definíció:** a felső és alsó gerincélvonal legelülső pontjait összekötő
  egyenes és a rágósík szöge; a mandibula maxillához viszonyított sagittalis
  helyzetét, gyakorlatilag az Angle-osztályozást reprezentálja.
- **Szakértői állítás:** ami nem Angle I, általában nehezebb eset, de nincs
  hozzá előre csatolt várt betegkimenet.
- **Priorirány:** nincs; feltáró változó.
- **Hatásalak:** nem nyers lineáris hatás; Angle I-től való eltérés vagy
  rugalmas kategóriák.

### A11 — szublingualis tájék/szájfenék

- **Sorrend:** `puhán elődomborodó` (legjobb) → `nem elődomborodó` →
  `tömött, elődomborodó` (legrosszabb).
- **Mechanizmus:** puhán elődomborodó szájfenék mellett alakítható ki
  szívóhatású alsó fogsor.
- **Önálló prognosztikai szerep:** erős meggyőződés; valószínűleg a
  legfontosabb nem-gerinc jellegű mandibularis tulajdonság.
- **Interakció A1/A2-vel:** rossz gerinc és tömött szájfenék együtt a külön
  hatások összegénél is rosszabb lehet; gyenge meggyőződés.
- **Interakció A5-tel:** gyenge meggyőződés.
- **Kódolási következmény:** a kedvező nyers referenciakategória **2**, nem 1.

### A12 — spinae mentales

- **Elsődleges kontraszt:** `nem tapintható` (kedvező) vs `tapintható vagy
  nyomásérzékeny` (kedvezőtlen).
- **Meggyőződés az elsődleges kontrasztban:** erős.
- **Mechanizmus:** a felszínessé váló spinae mentales a mandibularis gerinc
  sorvadásának jele; terheléskor fájdalmat is okozhat.
- **Másodlagos kontraszt:** `nyomásérzékeny` vs `csak tapintható` feltáró;
  bizonytalan, hogy valójában külön prognosztikai kategóriák-e.
- **Konstrukció:** elsősorban az A1/A2 mandibularis gerincsorvadási
  konstrukció további indikátora, nem biztosan független anatómiai hatás.

### A13 — TMI-funkció

- **Sorrend:** `panaszmentes` → `hangjelenség fájdalom/korlátozottság nélkül`
  → `fájdalom és/vagy mozgáskorlátozottság`.
- **Mechanizmus:** temporomandibularis diszfunkció mellett a fogsorkészítés a
  súlyossággal lineárisan nehezebbé válik.
- **Meggyőződés:** erős.
- **Modellszerep:** általános, additív klinikai főhatás; nem feltételezünk
  automatikusan interakciót minden anatómiai változóval.
- **Prediktorblokk:** nem anatómiai képlet, hanem klinikai súlyosbító tényező.

### A14 — antagonista fogazat az alsó fogpótlásnál

- **Szakértői állítás:** teljesen azonos az F8-cal.
- **Modellszerep:** nem egyszerű irányított főhatás; az erők iránya,
  nagysága, az okklúziós felállítás és az anatómia közötti interakció számít.
- **Közvetlen mechanikai kimenet:** stabilitás.

## 3. Közös konstrukciók

A kvalitatív elicitation alapján a változók nem kezelhetők 23 független
regressziós hatásként.

### Mandibularis gerincállapot

- A1: klinikai, ordinális gerincforma;
- A2: folytonos modellanalízis ugyanarról a konstrukcióról;
- A12: a sorvadás további klinikai jele.

Ezekhez közös látens konstrukció vagy előre kijelölt alternatív modellek
indokoltak. Egyidejű, független teljes hatásokkal azonos anatómiai információt
többször számolnánk.

### Tuberculum megtámasztási és kiterjeszthetőségi konstrukció

- A6: feszes ínyborítás;
- A7: alak és méret;
- A8: inklináció;
- A9: funkcionális alakstabilitás.

Ezek ugyanannak a kis képletnek egymással erősen összefüggő jellemzői. Közös
látens/kompozit prediktor vagy egymást váltó specifikációk indokoltak.

### Nem anatómiai prediktorblokk

- F8/A14: antagonista és okklúziós effektusmódosító;
- F9: beteg-/kezelési folyamatváltozó;
- A13: általános klinikai súlyosbító főhatás.

## 4. Előre jelölt interakciók

Csak klinikailag előre megnevezett interakciók kerülhetnek a modellbe:

- F8/A14 × kedvezőtlen okklúziós erőirány × releváns anatómia
  (különösen F4/F6): erős kvalitatív meggyőződés;
- A11 × A1/A2 mandibularis gerincállapot: kedvezőtlen, szuperadditív, gyenge
  meggyőződés;
- A5 × A11: kedvezőtlen, gyenge meggyőződés.

Az A6–A9 együttjárása és az A1/A2/A12 együttjárása **nem automatikusan
interakció**: ezek ugyanazon konstrukció több indikátorai.

## 5. Aszimmetria

Az összes kétoldali változónál az aszimmetria irány nélküli, feltáró
hipotézis. Nem feltételezzük előre, hogy az aszimmetria rosszabb vagy jobb.
Az oldalankénti adatokat ezért meg kell őrizni; a kizárólagos „rosszabbik
oldal” összevonás nem lehet az egyetlen specifikáció.

## 6. Nyitott outcome-hozzárendelés

A szakértő sok anatómiai változónál közvetlenül retenciós, stabilitási,
alátámasztási, fájdalmi vagy technikai mechanizmust adott meg, de azt még nem,
hogy ez milyen erősséggel informálja a PREDICT jelenlegi MAI-, OHIP- vagy
GOHAI-paraméterét.

Ezért a kvalitatív anatómiai tudást egyelőre nem vetítjük automatikusan
közvetlen QoL-priorra. A következő elicitation-kör feladata:

1. elsődleges prognosztikai outcome és időpont kiválasztása;
2. annak eldöntése, hogy a retenció/stabilitás közvetlenül mérhető-e;
3. az anatómia → mechanikai funkció → MAI/QoL utak elkülönítése;
4. változónként annak rögzítése, hogy mely regressziós paraméter kapja a
   szakmai priort.

## 7. Kódkönyv-korrekciók

Az elicitation alapján a jelenlegi elemzőkód alábbi feltételezéseit javítani
kell a modell implementálása előtt:

- **A1:** monoton, telítődő 1→5 hatás; 4 és 5 közel azonos. Az önkényes
  `1–2 kedvező vs 3–5 kedvezőtlen` vagy `csak 1 kedvező` dichotómiák nem az
  elsődleges szakértői specifikációk.
- **A3:** nincs kedvező referenciakategória és nincs irányított sorrend.
- **A5:** a legkedvezőbb nyers kategória 2, majd 1, majd 3.
- **A11:** a legkedvezőbb nyers kategória 2, majd 1, majd 3.
- **A12:** elsődlegesen 1 vs 2/3; a 2 vs 3 különbség feltáró.
- **A8:** a 3-as nem ordinális végpont, hanem az A7 plicaszerű válaszával
  összefüggő validációs kategória.
- **F9:** elsődlegesen 1/2 vs 3, de a 3-as nem baseline információ.
- **F8/A14:** nem egyszerű `1 kedvező, 2–3 kedvezőtlen` változók.

## 8. A számszerű priorok kialakításának hátralévő lépései

1. A hihető hatásnagyságok elicitalása klinikailag értelmezhető kimeneti
   skálán.
2. A standardizálási és kontrasztkódolási szabályok lezárása.
3. Prior prediktív ellenőrzés: a prior által generált betegkimenetek legyenek
   klinikailag hihetők.
4. Legalább szkeptikus, tankönyvi és szakértői prior-szenzitivitás.
5. Outcome-adatok előtti időbélyegzett, verziózott priorregiszter.

## 9. Első módszertani és klinikai horgonyok

- [Stability in Mandibular Denture](https://www.ncbi.nlm.nih.gov/books/NBK549861/)
- [Ii, 2016 — inhibitory factors for suction-effective mandibular dentures](https://doi.org/10.14399/jacd.36.184)
- [Fish, 1962 — mandibular molar-region anatomy and denture](https://doi.org/10.1016/0022-3913(62)90036-7)
- [Bayesian Analysis Reporting Guidelines](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526359/)
- [Prior elicitation in clinical research — review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7917693/)
- [TRIPOD Statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC4297220/)
