# Mit mutatnak jelenleg a PREDICT adatai?

**Exploratív Bayes-i pillanatkép, 2026-08-19**

## Rövid válasz

A jelenlegi adatokban három fő mintázat látszik:

1. A rosszabb **A1 mandibularis gerincállapot** a kiindulási OHIP- és
   GOHAI-életminőséggel a szakmailag várt irányban jár együtt. Az életkorra,
   nemre és fogsortípusra korrigált, semleges prioros posteriorban
   `P(beta > 0)` OHIP esetén 95%, GOHAI esetén 82%, MAI esetén viszont csak
   58%.
2. Az **A6–A9 tuberculumjellemzők**, különösen az A9 mozgékonyság, szintén a
   rosszabb kiindulási QoL irányába mutatnak, de a credible intervalok szélesek
   és mind tartalmazzák a nullát. MAI-kapcsolat gyakorlatilag nem látszik.
3. Az **F7 torus palatinus** és részben az A4 torus mandibularis a jelenlegi
   mintában nem a várt irányt mutatja. Ez érdekes prior–adat feszültség, de a
   keresztmetszeti, kis és többszörösen elemzett mintából nem következik, hogy
   a klasszikus állítás téves.

Ez még nem prospektív PREDICT-validáció. A modellhez szükséges utánkövetési
kimenet jelenleg túl kevés és adatminőségileg problémás.

## Adatállomány

| Adat | Elérhető beteg |
|---|---:|
| Összes beteg | 48 |
| Kiindulási OHIP | 48 |
| Kiindulási GOHAI | 48 |
| Kiindulási MAI | 41 |
| Utánkövetési OHIP | 9 |
| Utánkövetési GOHAI | 9 |
| Végső MAI | 5 |
| F5/F7/F8 felső klinikai blokk | 39 |
| A1/A3–A14 alsó klinikai blokk | körülbelül 40–41 |
| F1–F4/F6 modellanalízis | 9 |
| A2 modellanalízis | 5 |
| A10 mérés | 6 |

A keresztmetszeti elemzés ezért a kiindulási OHIP-, GOHAI- és MAI-kimenetekre
korlátozódik. A magasabb modellezett outcome minden esetben rosszabbat jelent.

## Módszer

- Az elemzés betegazonosítókat nem használ és nem ír ki.
- A prediktorokat a 2026-08-19-i szakértői elicitation alapján kódoltuk.
- Minden modell egyetlen anatómiai prediktort vizsgál, így a kis mintában nem
  illesztünk túl nagy közös regressziót.
- Az elsődleges érzékenységi modell életkorra, nemre és `both` vs egyetlen
  állcsontot érintő fogsortípusra korrigál.
- A likelihoodot HC3-robusztus lineáris regresszió közelíti.
- A beta semleges/szkeptikus priorja `Normal(0, 0.5²)` a standardizált
  kimeneti skálán.
- A táblázatbeli beta egy teljes kedvező→kedvezőtlen kontrasztot jelent;
  folytonos változóknál egy prediktor-szórásegységet.
- Az elicitált 80/90/98%-os iránypriorokat **nem alkalmaztuk** a jelenlegi
  OHIP/GOHAI/MAI-modellekre, mert még nincs eldöntve, hogy a mechanikai
  szakmai állítás melyik outcome-paramétert informálja.
- Minden eredmény exploratív; nincs változószelekció vagy többszörös
  összevetés utáni megerősítő döntés.

## Fő korrigált eredmények

A posterior valószínűség azt jelzi, mennyire valószínű a szakmailag várt
pozitív irány **semleges prior mellett**. A 95%-os credible intervalok minden
felsorolt esetben szélesek.

| Prediktor | Outcome | n | Posterior beta | 95% CrI | P(várt irány) |
|---|---|---:|---:|---:|---:|
| A1 telítődő gerincrizikó | OHIP | 35 | +0,54 | −0,09; +1,17 | 95% |
| A1 telítődő gerincrizikó | GOHAI | 35 | +0,34 | −0,40; +1,08 | 82% |
| A1 telítődő gerincrizikó | MAI | 32 | +0,07 | −0,68; +0,83 | 58% |
| A6–A9 tuberculum pilot-kompozit | OHIP | 35 | +0,38 | −0,38; +1,14 | 84% |
| A6–A9 tuberculum pilot-kompozit | GOHAI | 35 | +0,45 | −0,27; +1,18 | 89% |
| A6–A9 tuberculum pilot-kompozit | MAI | 32 | +0,02 | −0,83; +0,87 | 52% |
| A9 mozgékonysági rizikó | OHIP | 35 | +0,43 | −0,29; +1,16 | 88% |
| A9 mozgékonysági rizikó | GOHAI | 35 | +0,46 | −0,25; +1,16 | 90% |
| F5 lötyögő gerinc jelenléte | OHIP | 33 | +0,19 | −0,49; +0,88 | 71% |
| F5 lötyögő gerinc jelenléte | GOHAI | 33 | +0,13 | −0,58; +0,83 | 64% |
| F5 lötyögő gerinc jelenléte | MAI | 31 | +0,17 | −0,49; +0,83 | 70% |
| F7 torus palatinus jelenléte | OHIP | 33 | −0,48 | −1,05; +0,09 | 5% |
| F7 torus palatinus jelenléte | GOHAI | 33 | −0,10 | −0,72; +0,52 | 37% |
| F7 torus palatinus jelenléte | MAI | 31 | −0,44 | −1,09; +0,22 | 10% |
| A4 torus mandibularis jelenléte | OHIP | 34 | −0,47 | −1,07; +0,14 | 7% |
| A4 torus mandibularis jelenléte | GOHAI | 34 | −0,23 | −0,87; +0,41 | 24% |
| A11 szájfenékrizikó | OHIP | 35 | +0,18 | −0,50; +0,86 | 70% |
| A11 szájfenékrizikó | GOHAI | 35 | +0,02 | −0,70; +0,75 | 53% |

### Értelmezés

- **A1:** jelenleg ez a legkoherensebb jel. Az 1-es Kaán-kategóriában az
  átlagos OHIP 4,31, a 4-esben 10,33 volt, de a 4-esben csak három beteg van,
  az 5-ös pedig nem követi egyszerűen ezt a trendet. A telítődő kódolás és a
  bizonytalanság megtartása ezért fontos.
- **Tuberculum:** a QoL-jel főként A9-ből, részben A6/A8-ból származik. A
  kompozit nem validált mérőskála; csak pilot-összefoglaló.
- **F5:** minden kimeneten enyhén a várt irányba mutat, de az adat még közel
  sem döntő.
- **F7:** a torusszal rendelkező csoportban a nyers kiindulási MAI átlagosan
  alacsonyabb (jobb) volt, mint torus nélkül. A fordított jel korrigálás és
  rangalapú érzékenység mellett is megmarad, ezért valódi prior–adat
  konfliktusként követendő, de nem tekinthető kész klinikai eredménynek.
- **A4:** QoL-on hasonlóan fordított pilotjel látszik, MAI-on nincs érdemi
  irány.
- **A11:** a feltételezett erős klinikai jelentőséggel szemben a jelenlegi
  keresztmetszeti outcome-ok csak gyenge jelet adnak. Ez lehet azért is, mert
  a releváns közvetlen outcome a retenció, amelyet a jelen adatok nem mérnek.

## Ritkán mért felső modellanalízis: csak pilotjel

Az F1/F3/F4/F6 változóknál csak 8–9 outcome-pár van, ezért korrigált modell
nem értelmezhető.

- **F4:** a kisebb szög mindhárom baseline outcome esetében a várt rosszabb
  irányba mutatott. Semleges prior mellett `P(várt irány)` OHIP 95%, GOHAI
  99%, MAI 85%. Ez érdekes, de n=8–9.
- **F3:** a kisebb boltozat szintén inkább a várt irányba mutatott
  (`P` körülbelül 72–86%).
- **F1:** a jel mindhárom outcome-on az elicitált iránnyal ellentétes volt.
- **F6:** a 90 foktól való eltérés különösen GOHAI-n az elicitált iránnyal
  ellentétes volt.

Ezekből jelenleg nem szabad rangsort vagy klinikai következtetést alkotni.

## Prospektív kimenetek: jelenleg nem elemezhetők megbízhatóan

A 9 teljes utánkövetési QoL-kérdőívből háromban mind az 5 OHIP-tétel és mind a
12 GOHAI-tétel pontosan a legrosszabb értéket kapta. A visszarendelési HTML-
űrlap ugyanezeket jelöli be alapértelmezetten. Emiatt nem dönthető el, hogy a
három beteg valóban minden tételben a legrosszabb választ adta-e, vagy az
űrlapot a default válaszok módosítása nélkül küldték el.

A végső MAI csak öt betegnél érhető el. Ebből sem anatómiai prognosztikai
modell, sem megbízható változáselemzés nem készíthető.

## További adatminőségi megállapítások

- **A7–A8 validációs ellentmondás:** kilenc oldalon volt A7=3
  (`plicaszerű`). A szakértői definíció szerint ilyenkor A8-nak is 3-nak
  kellene lennie, de nyolc oldalon nem ez szerepel.
- **Tuberculum-konstrukció:** A6–A9 Spearman-korrelációi −0,13 és 0,76 között
  vannak; a leíró Cronbach-alpha 0,65. Ez nem támaszt alá automatikusan
  egydimenziós reflektív skálát. A konstrukció lehet inkább formatív.
- **Ritka kategóriák:** A12=3 és A13=3 csak egy-egy betegnél fordul elő, ezért
  a súlyos kategória külön hatása nem becsülhető.
- **A10:** csak hat mérés van, 0,8–106 közötti tartománnyal; a mértékegységet
  és adatbevitelt ellenőrizni kell.
- **Aszimmetria:** irány nélküli, feltáró elemzésben nem látszik stabil,
  minden outcome-on azonos jel.

## Mit lehet most állítani?

Óvatosan ezt:

> A jelenlegi keresztmetszeti adatok előzetesen támogatják, hogy a súlyosabb
> mandibularis gerincállapot és bizonyos tuberculumjellemzők rosszabb
> betegjelzett életminőséggel járhatnak együtt. A mechanikai MAI-kimeneten ez
> a kapcsolat nem látszik. Néhány torus- és felső morfológiai változó az
> elicitált iránnyal ellentétes pilotjelet mutat. Az eredmények a kis minta,
> hiányzó prospektív kimenetek, adatminőségi problémák és párhuzamos
> összevetések miatt hipotézisgenerálók.

Nem állítható még, hogy bármely anatómiai változó előre jelzi az új fogsor
utáni QoL-t vagy rágóképességet.

## Validációs minősítés

**Megosztható a fenti korlátozások kötelező feltüntetésével; megerősítő
eredményként nem használható.**

- A 48 sor 48 egyedi betegrekordnak felel meg.
- A riport négy kiemelt posterior eredményét a gépi kimenetből függetlenül
  visszaellenőriztük; eltérés nem volt.
- A kimeneti fájlok nem tartalmaznak nevet, TAJ-t vagy betegazonosítót.
- A nyers-standardizált és rang-normalizált korrigált modellek előjele 48
  összevetésből 42-ben (87,5%) megegyezett. A hat eltérés mind nulla közeli,
  bizonytalan becslésnél jelentkezett.
- A fő korlát nem számítási hiba, hanem a prospektív kimenetek hiánya,
  lehetséges default-válasz torzítás, kis alcsoportok és párhuzamos exploráció.

## Reprodukálható kimenetek

- elemzőszkript: [`bayes_current_data_exploration.py`](bayes_current_data_exploration.py)
- teljes modellkimenet: [`bayes_neutral_item_models.csv`](stat_output/bayes_neutral_item_models.csv)
- prediktorregiszter: [`bayes_predictor_registry.csv`](stat_output/bayes_predictor_registry.csv)
- audit: [`bayes_current_data_audit.json`](stat_output/bayes_current_data_audit.json)
- tuberculum-korrelációk: [`bayes_tuberculum_spearman.csv`](stat_output/bayes_tuberculum_spearman.csv)
