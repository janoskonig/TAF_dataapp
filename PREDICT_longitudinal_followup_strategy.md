# PREDICT longitudinális utánkövetési stratégia

**Státusz:** felülvizsgált pilot elemzési és adatgyűjtési stratégia, a valódi
végső eredményfuttatás előtt
**Rögzítés dátuma:** 2026-08-19
**Felülvizsgálat dátuma:** 2026-08-19 — a kis minta miatt 4+4 állcsonti
kategóriára egyszerűsítve
**Cél:** az alsó és felső állcsont nyolc előre összevont klinikai kategóriája és az utánkövetési orális
életminőség, rágóképesség, illetve színkeverési teljesítmény kapcsolatának
vizsgálata kizárólag kétállcsontos (`denture_type == "both"`) betegekben,
információvesztő dichotomizálás nélkül.

## 1. Vizsgálati populáció

- Elsődleges és egyetlen elemzési populáció: minden olyan meglévő beteg,
  akinek fogsortípusa `denture_type == "both"`.
- A jelenlegi elérhető kétállcsontos kohorsz nagysága: 31 beteg.
- Minden alkalmas beteget azonos eljárással kell megkeresni; anatómiai lelet,
  kiindulási életminőség vagy várható kimenet alapján nem történhet
  prioritásadás.
- A `lower` és `upper` fogsortípusú betegek nem részei ennek az elemzésnek, és
  nem vonhatók össze a `both` csoporttal. Esetleges külön vizsgálatuk új,
  elkülönített protokollt igényel.

## 2. Utánkövetési célértékek

- 20 teljes utánkövetés: leíró, hipotézisgeneráló pilot.
- 25 teljes utánkövetés: értékelhető, de bizonytalan pilot főelemzés.
- 30–31 teljes utánkövetés: a jelenlegi kétállcsontos kohorszból elérhető
  ideális cél.

Teljes esetnek kimenetenként az számít, akinél rendelkezésre áll:

1. a szükséges állcsonti kategória;
2. a kimenet kiindulási értéke;
3. ugyanazon kimenet utánkövetési értéke;
4. a megfelelő elhorgonyzó kérdés;
5. az alapvető korrekciós változók.

## 3. Betegmegkeresési protokoll

- Telefonon elsődlegesen időpont-egyeztetés történjen, ne a kérdőív felvétele.
- A meghívás legyen semleges; az anatómiai hipotézist és a várt változás
  irányát ne közöljük a beteggel.
- Rögzítendő:
  - megkeresés dátuma;
  - sikeres vagy sikertelen kapcsolatfelvétel;
  - vállalja-e a visszarendelést;
  - egyeztetett időpont;
  - visszautasítás vagy meghiúsulás oka, ha önként megadja;
  - megjelent-e a viziten.
- A nem visszatérő betegek kiindulási adatait nem töröljük. A visszatérők és
  nem visszatérők kiindulási jellemzőit aggregáltan össze kell hasonlítani a
  szelekciós torzítás felmérésére.
- Az adatfelvételnek meg kell felelnie az alkalmazandó etikai engedélynek,
  beleegyezési és adatvédelmi követelményeknek.

## 4. Utánkövetési vizit

Egységes protokoll szerint rögzítendő:

- OHIP-5 mind az öt tétele;
- GOHAI mind a tizenkét tétele;
- `responsiveness_today_situation_recall`;
- `responsiveness_change`;
- `chewing_today_situation_recall`;
- `chewing_change`;
- utánkövetési MAI hue-degree, ha elvégezhető;
- F9;
- dropout státusz;
- a fogsor átadása óta eltelt idő;
- a fogsoron végzett köztes korrekciók és lényeges események.

Az adatfelvételi mód minden betegnél azonos legyen. Telefonon és személyesen
felvett kimeneteket elsődleges elemzésben nem szabad automatikusan
összevonni.

## 5. Előre definiált állcsonti kategóriák

A sok, részben összetartozó tétel helyett nyolc kategória szerepel. Mindegyik
0–1 kedvezőtlenségi skálát kap, és külön modellbe kerül; összesített
teherpontszámot nem képzünk.

### 5.1. Alsó állcsont

1. **Gerincprofil (A1):** `1 -> 0`, `2 -> 1/3`, `3 -> 2/3`, `4/5 -> 1`.
   Az A2 folytonos modellanalízis ugyanennek alternatív mérése, ezért később
   csak külön érzékenységi elemzésben használható, nem második független
   prediktorként.
2. **Szájfenék (A11):** `2 -> 0` (puhán elődomborodó), `1 -> 0,5`,
   `3 -> 1` (tömött, elődomborodó).
3. **Torus mandibularis (A4):** `0`, ha egyik oldalon sincs; `1`, ha legalább
   az egyik oldalon 2-es vagy 3-as kód szerepel.
4. **Tuberculum (A6–A9):** A6, A7 és A9 kódolása `1 -> 0`, `2 -> 0,5`,
   `3 -> 1`; A8 kódolása `1 -> 0`, `2/3 -> 1`. A két oldal és a négy tétel
   közös átlaga adja a kategóriaértéket.

### 5.2. Felső állcsont

1. **Gerincprofil (F1):** a nagyobb magasság kedvezőbb; a mért
   mintatartományban fordított 0–1 skálára képezzük, ahol 1 a legalacsonyabb
   mért profil.
2. **Lötyögő gerinc (F5):** `0`, ha nincs; `1`, ha tuber- vagy frontális
   lokalizációban jelen van.
3. **Torus palatinus (F7):** `0`, ha nincs; `1`, ha plató- vagy orsó alakban
   jelen van.
4. **Garatreflex (F9):** `0`, ha nem befolyásolta érdemben a kezelést
   (1-es/2-es kód); `1`, ha jelentősen befolyásolta (3-as kód). Az F9
   beteg-/kezelési tényező és részben folyamatváltozó, nem tiszta baseline
   anatómiai prediktor, ezért külön óvatossággal értelmezendő.

A3, A5, A12 és A13 nem kap külön elsődleges modellt ebben a kis mintás
stratégiában. F8 és A14 protetikai státusz, ezért nem állcsonti anatómiai
kategória.

## 6. Anchor-leképezés

- OHIP-5: `responsiveness_change`.
- GOHAI: `responsiveness_change`.
- MAI: `chewing_change`.

Az anchor sorrendje:

1. Sokat romlott
2. Kicsit romlott
3. Változatlan maradt
4. Kicsit javult
5. Sokat javult

Az elemzésben az anchort nem dichotomizáljuk. A beteg szubjektív
változását ötkategóriás ordinális kimenetként tartjuk meg.

## 7. Elsődleges statisztikai elemzés

### 7.1. Folytonos kimenetek

A nyers változás dichotomizálása helyett kiindulási értékre korrigált
utánkövetési modellt használunk.

Minden `category` prediktorra külön, egyváltozós anatómiai modellt illesztünk:

```text
OHIP_followup  ~ OHIP_baseline  + category
GOHAI_followup ~ GOHAI_baseline + category
MAI_followup   ~ MAI_baseline   + category
```

A nyolc kategória nem kerül egyszerre ugyanabba a modellbe; a kohorsz ehhez
túl kicsi lenne.

A fő hatásmutató a regressziós együttható (`beta`) és annak 95%-os
konfidenciaintervalluma:

- OHIP: pozitív beta = rosszabb utánkövetési kimenet;
- GOHAI: negatív beta = rosszabb utánkövetési kimenet;
- MAI: a projekt jelenlegi iránykonvenciója szerint pozitív beta = rosszabb
  utánkövetési kimenet.

Az iránykonvenciókat a végső futtatás előtt adatfelvételi és mérési
dokumentáció alapján ismét ellenőrizni kell.

### 7.2. Anchor-elemzés

Az ötkategóriás anchorhoz proporcionális odds ordinális logisztikus modellt
illesztünk:

```text
anchor_ordinal ~ category
```

A fő hatásmutató az OR a kategória teljes 0→1 kedvezőtlenségi változására:

- `OR < 1`: kisebb esély kedvezőbb anchor-kategóriára;
- `OR > 1`: nagyobb esély kedvezőbb anchor-kategóriára;
- `OR = 1`: nincs észlelhető kapcsolat.

A proporcionális odds feltételt ellenőrizni kell. Súlyos sérülése esetén
részleges proporcionális odds modell vagy egyszerűbb ordinális/rang-alapú
elemzés szükséges.

## 8. Korrekciós változók és modellkomplexitás

- Kis mintás modell: kiindulási kimenet + egy állcsonti kategória.
- Életkorra korrigált modell: előre kijelölt érzékenységi elemzés.
- Nem csak akkor kerülhet a modellbe, ha a teljes esetszám és az
  eseményeloszlás ezt stabilan lehetővé teszi. Fogsortípus nem kerül a
  modellbe, mert az elemzési populációban minden beteg `both`.
- Kis mintában nem illesztjük egyszerre a nyolc kategóriát és
  interakciót.
- Automatikus változószelekció és p-érték alapján történő modellépítés nem
  használható.

## 9. Bizonytalanság és kis mintás eljárások

- Minden eredményhez hatásbecslést és 95%-os intervallumot közlünk.
- Folytonos kimenetekhez HC3-robusztus standard hiba és/vagy bootstrap
  konfidenciaintervallum használható.
- Rangkorrelációs érzékenységi elemzéshez permutációs p-érték és bootstrap
  intervallum használható.
- Ordinális modell instabilitása vagy szeparáció esetén penalizált vagy
  gyengén informatív Bayes-modell használható, az alkalmazott priorok teljes
  közlésével.
- Penalizálás nem pótolja a hiányzó információt; instabil eredményt
  bizonytalanként kell jelenteni.

## 10. Multiplicitás és elemzési hierarchia

1. Elsődleges elemzési keret: a nyolc előre rögzített kategória külön modellje.
2. Elsődleges kimenet: kiindulási értékre korrigált utánkövetési OHIP-5.
3. Másodlagos kimenetek: GOHAI és MAI.
4. Anchor-elemzések: külső, beteg által jelzett változásvizsgálat.

A nyolc kategória p-értékei kimenetenként Benjamini–Hochberg
FDR-korrekciót kapnak. A hangsúly a hatásbecslésen és a 95%-os intervallumon,
nem a szignifikancia szerinti rangsoroláson van.

## 11. Hiányzó adatok

- Kimenetenként közölni kell a teljes esetszámot és a hiányzás okait.
- Nem imputálunk mesterséges utánkövetési eredményeket.
- Nagyon kis mintában kategóriánként teljes eseteken történik az elemzés.
- A felület ellenőrzését segítő szimuláció nem imputál valódi kutatási
  kimenetet, nem ír adatbázist, és nem közölhető kutatási eredményként.
- Ha később a minta és a hiányzási mintázat lehetővé teszi, többszörös
  imputáció csak előre rögzített változókkal és külön érzékenységi elemzésként
  alkalmazható.
- A dropout nem kódolható automatikusan romlásként vagy változatlanságként.

## 12. Szelekciós torzítás ellenőrzése

A visszatérő és nem visszatérő betegek között aggregáltan összehasonlítandó:

- életkor;
- nem;
- kiindulási OHIP-5;
- kiindulási GOHAI;
- kiindulási MAI;
- a nyolc állcsonti kategória kiindulási értékei és hiányzási arányai.

A különbségeket elsősorban standardizált különbségekkel és intervallumokkal,
nem pusztán p-értékekkel kell bemutatni.

## 13. Értelmezési korlátok

- A vizsgálat pilot és asszociációs; oksági hatást nem igazol.
- A kis minta széles bizonytalansági intervallumokat eredményezhet.
- A visszatérők szelektált alcsoportot alkothatnak.
- Az anchor és az objektív/életminőségi score eltérő konstrukciókat mérhet.
- A null eredmény nem bizonyítja a hatás hiányát, ha az intervallum széles.
- A szignifikáns exploratív eredmény önmagában nem tekinthető validációnak.

## 14. Kötelező riportálás

Minden fő eredmény mellett szerepeljen:

- elemzett betegszám;
- hiányzó esetek száma;
- prediktor és kimenet pontos definíciója;
- hatásbecslés;
- 95%-os intervallum;
- modellkorrekciók;
- feltétel- és érzékenységi ellenőrzések;
- pilot/exploratív státusz.

## 15. Elemzés előtti ellenőrzőlista

- [ ] Az elemzési adatállomány kizárólag `denture_type == "both"` betegeket tartalmaz.
- [ ] Minden alkalmas beteget azonos módon kerestünk meg.
- [ ] A megkeresési és megjelenési státusz dokumentált.
- [ ] Az OHIP/GOHAI/MAI skálairányok ellenőrzöttek.
- [ ] Az anchor-leképezés változatlan: OHIP/GOHAI → orális egészség, MAI → rágás.
- [ ] A nyolc kategória 4+4-es definíciója a végső valódi eredményfuttatás előtt rögzített.
- [ ] Nem készült összesített anatómiai score, és a nyolc kategória nem került egyszerre egy modellbe.
- [ ] Az elsődleges kimenet a kiindulási értékre korrigált utánkövetési OHIP-5.
- [ ] Az elemzés nem használ adatvezérelt dichotomizálást.
- [ ] A modellkomplexitás megfelel a teljes esetszámnak.
- [ ] A visszatérők és nem visszatérők összehasonlítása elkészült.
- [ ] Minden eredmény mellett szerepel intervallum és korlátozás.

## 16. Gyorsított adatfelvételi munkafelület

Az utánkövetéshez külön, hozzáférési kóddal védett munkafelület készült a
`/followup` útvonalon. A modul kizárólag a legfrissebb betegrekord alapján
`denture_type == "both"` kohorszba tartozó betegeket mutatja.

Fő funkciói:

- kizárólag belépés után látható teljes név és TAJ, valamint kutatási
  azonosító szerinti beteglista;
- megkeresési státusz, időpont, visszautasítás és távolmaradás naplózása;
- telefonszám rögzítése és mobilról közvetlen hívás indítása;
- beleegyezés, eltelt idő, köztes korrekciók és események rögzítése;
- előre bejelölt válaszok nélküli, hiánytalan kitöltést megkövetelő
  OHIP–GOHAI–anchor kérdőív;
- F9 és – kiindulási MAI esetén – utánkövetési MAI rögzítése;
- automatikus hiányellenőrzés a vizit lezárása előtt;
- közvetlen azonosító nélküli, elemzésre előkészített CSV-export.

A közvetlen betegazonosítók csak a védett klinikai munkafelületen jelennek
meg. A rendszer ezeknek az oldalaknak a böngésző-cache-elését tiltja; a
kutatási CSV-export továbbra sem tartalmaz nevet, TAJ-t, telefonszámot vagy
születési dátumot.

Az új adatok a `followup_visits` és `followup_contact_attempts` táblákba
kerülnek. A felület nem frissíti és nem írja felül a `patients` tábla meglévő
sorait. Más környezetben a `migrate_followup_visits.sql` migrációt alkalmazni,
valamint a `SECRET_KEY` és `FOLLOWUP_ACCESS_CODE` környezeti változókat
biztonságosan beállítani kell.

A régi `/questionnaire3`, `/submit_questionnaire3`, `/upload_final_mai` és
`/submit_final_mai` útvonalak lezártak: minden GET- és POST-kérés tájékoztató
üzenetet és HTTP 410 választ kap, ezért ezeken keresztül új utánkövetési adat
már nem írható a `patients` táblába.
