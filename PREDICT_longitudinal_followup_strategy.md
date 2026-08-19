# PREDICT longitudinális utánkövetési stratégia

**Státusz:** előre rögzített pilot elemzési és adatgyűjtési stratégia
**Rögzítés dátuma:** 2026-08-19
**Cél:** a mandibularis anatómiai hátrányterhelés és az utánkövetési orális
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

1. a szükséges anatómiai konstrukció;
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

## 5. Előre definiált anatómiai konstrukciók

Minden konstrukció legfeljebb egy ponttal járul hozzá az összesített
hátrányterheléshez.

### 5.1. Gerincatrophia

- Kedvezőtlen, ha `A1_Kaan >= 3` vagy `A12 >= 2`.
- Az A1/A12 blokk együtt is legfeljebb egy pontot ad.

### 5.2. Torus mandibularis

- Kedvezőtlen, ha `A4` értéke 2 vagy 3 legalább az egyik oldalon.

### 5.3. Lingualis tasak

- Kedvezőtlen, ha `A5 = 3` legalább az egyik oldalon.

### 5.4. Tuberculum-konstruktum

- A6, A7 és A9 kódolása: `1 -> 0`, `2 -> 0,5`, `3 -> 1`.
- A8 kódolása: `1 -> 0`, `2/3 -> 1`.
- Elsődleges oldal-összevonás: a jobb és bal oldal átlaga.
- A konstrukció értéke az A6–A9 blokkok átlaga.
- Bináris érzékenységi definícióban kedvezőtlen, ha a blokkátlag `>= 0,5`.
- A8 nélküli változat csak előre megnevezett érzékenységi elemzés.

### 5.5. Szájfenék

- Kedvezőtlen, ha `A11 = 3`.

### 5.6. Összesített anatómiai hátrányterhelés

- Elsődleges anatómiai prediktor: a fenti öt konstrukció `0–5` közötti
  összege.
- Elsődleges elemzésben folytonos/ordinális pontszámként szerepel; nem osztjuk
  utólag alacsony és magas csoportokra.
- Az egyes konstrukciók külön elemzése másodlagos és exploratív.

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

Az elsődleges elemzésben az anchort nem dichotomizáljuk. A beteg szubjektív
változását ötkategóriás ordinális kimenetként tartjuk meg.

## 7. Elsődleges statisztikai elemzés

### 7.1. Folytonos kimenetek

A nyers változás dichotomizálása helyett kiindulási értékre korrigált
utánkövetési modellt használunk.

Elsődleges OHIP-modell:

```text
OHIP_followup ~ OHIP_baseline + anatomical_burden_0_5
```

Másodlagos modellek:

```text
GOHAI_followup ~ GOHAI_baseline + anatomical_burden_0_5
MAI_followup   ~ MAI_baseline   + anatomical_burden_0_5
```

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
anchor_ordinal ~ anatomical_burden_0_5
```

A fő hatásmutató az OR egy további anatómiai hátránypontonként:

- `OR < 1`: kisebb esély kedvezőbb anchor-kategóriára;
- `OR > 1`: nagyobb esély kedvezőbb anchor-kategóriára;
- `OR = 1`: nincs észlelhető kapcsolat.

A proporcionális odds feltételt ellenőrizni kell. Súlyos sérülése esetén
részleges proporcionális odds modell vagy egyszerűbb ordinális/rang-alapú
elemzés szükséges.

## 8. Korrekciós változók és modellkomplexitás

- Elsődleges kis mintás modell: kiindulási kimenet + anatómiai score.
- Életkorra korrigált modell: előre kijelölt érzékenységi elemzés.
- Nem csak akkor kerülhet a modellbe, ha a teljes esetszám és az
  eseményeloszlás ezt stabilan lehetővé teszi. Fogsortípus nem kerül a
  modellbe, mert az elemzési populációban minden beteg `both`.
- Kis mintában nem illesztünk egyszerre sok anatómiai komponenst és
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

1. Elsődleges prediktor: `anatomical_burden_0_5`.
2. Elsődleges kimenet: kiindulási értékre korrigált utánkövetési OHIP-5.
3. Másodlagos kimenetek: GOHAI és MAI.
4. Anchor-elemzések: külső, beteg által jelzett változásvizsgálat.
5. Az öt külön konstrukció eredménye exploratív.

Az elsődleges elemzés egyetlen előre kijelölt hipotézist tesztel. A
másodlagos és exploratív elemzéseket egyértelműen meg kell jelölni; szükség
esetén Benjamini–Hochberg FDR-korrekció alkalmazandó.

## 11. Hiányzó adatok

- Kimenetenként közölni kell a teljes esetszámot és a hiányzás okait.
- Nem imputálunk mesterséges utánkövetési eredményeket.
- Nagyon kis mintában az elsődleges elemzés teljes eseteken történik.
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
- anatómiai hátrányterhelési score.

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
- [ ] Az öt konstrukció definíciója nem változott a kimenetek megtekintése után.
- [ ] Az elsődleges prediktor a `0–5` anatómiai score.
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

- maszkolt TAJ és kutatási azonosító szerinti beteglista;
- megkeresési státusz, időpont, visszautasítás és távolmaradás naplózása;
- beleegyezés, eltelt idő, köztes korrekciók és események rögzítése;
- előre bejelölt válaszok nélküli, hiánytalan kitöltést megkövetelő
  OHIP–GOHAI–anchor kérdőív;
- F9 és – kiindulási MAI esetén – utánkövetési MAI rögzítése;
- automatikus hiányellenőrzés a vizit lezárása előtt;
- közvetlen azonosító nélküli, elemzésre előkészített CSV-export.

Az új adatok a `followup_visits` és `followup_contact_attempts` táblákba
kerülnek. A felület nem frissíti és nem írja felül a `patients` tábla meglévő
sorait. Más környezetben a `migrate_followup_visits.sql` migrációt alkalmazni,
valamint a `SECRET_KEY` és `FOLLOWUP_ACCESS_CODE` környezeti változókat
biztonságosan beállítani kell.
