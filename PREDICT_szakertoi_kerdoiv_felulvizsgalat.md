# PREDICT szakértői kérdőív – felülvizsgálat egy elutasító visszajelzés nyomán (2026-09-08)

**Kiváltó ok.** Egy felkért professzor a kérdőívet kitöltetlenül hagyta, és
ennyit írt rá: „Szakmaiatlan, tudományos feldolgozásra alkalmatlan kérdőív.”
A megjegyzés nem nevez meg konkrétumot, ezért az eszközt tételesen átnéztük:
a webes űrlapot (v2.0: `expert_priors.py`, `expert_texts.py`,
`templates/expert_*.html`), a Word-változatokat (`make_expert_form_docx.py`),
a felkérő levelet (`MAIL_TEXTS`), a felkészítőt, a protokollt
(`PREDICT_szakertoi_elicitacio_protokoll.md`) és az elemző szkriptet
(`predict_szakertoi_prior_logit.R`). Két kérdést tettünk fel: mit láthat benne
szakmaiatlannak egy tapasztalt protetikus, és mi az, ami a válaszok tudományos
feldolgozhatóságát ténylegesen veszélyezteti.

**Kiindulási állapot.** Külső szakértői válasz eddig nem érkezett: a
`predict_expert_priorok.csv` csak a vizsgálatvezető 2026-08-19-i kvalitatív
sorait tartalmazza, az eszközt a vizsgálatvezető saját végigpróbálásán kívül
senki nem pilotálta. Ez tehát az első külső visszajelzés, és a kérdőív első
külső próbája.

## 1. Összefoglaló ítélet

A kritika nem alaptalan, de nem a módszertan egészét találja el. A kérdőív
módszertani váza a szakirodalommal összhangban van (SHELF-elvek: a betegek
közötti változatosság és a tudás bizonytalanságának szétválasztása,
pólusonkénti kvantilisek, kalibrációs gyakorlat, egyéni, egymástól független
becslés, egyenlő súlyú keverés érzékenységi elemzéssel). Amit egy professzor
joggal kifogásolhat, az három rétegben van:

1. **Regiszter és forma.** Köznyelvi, helyenként bizalmaskodó megfogalmazás;
   kvíz-jellegű gyakorlókérdések; a tankönyvi tudás lekicsinylésének érzetét
   keltő mondatok; hiányzó formai elemek (felelős személy, elérhetőség, etikai
   hivatkozás, egységes verziószám, módszertani irodalom).
2. **Klinikai tartalom.** Több tétel pólusa a vizsgálat saját, kívülről nem
   értelmezhető mérésére hivatkozik (F1, F3, F4, F6), és ezek az értékek nem
   a kohorsz ötödeiből, hanem hat–kilenc beteg mérési tartományán *kívülről*
   származnak. Szakmailag különböző helyzetek vannak összevonva (A10, TUB, F7,
   A4, F8). A siker definíciója nem olyan kimenet, amelyet a klinikus a
   praxisában megfigyel.
3. **Feldolgozhatóság.** Horgonyzó példa valódi tétellel és konkrét számokkal;
   a „tankönyvi rossz” pólus mindig a B; a kérdezett feltételes hatás és a
   protokoll szerinti elsődleges (2×2-es, marginális) felhasználás nem ugyanaz
   a mennyiség; a kimenet MCID-küszöbei a protokoll szerint sem véglegesek;
   rosszul definiált B2/B3 mennyiségek; a 35–45 percnél valószínűleg jóval
   nagyobb terhelés.

**Javaslat.** A kérdőívet v2.1-ként átdolgozni az 5. fejezet 1. prioritású
pontjai szerint, *mielőtt* további meghívó megy ki; a professzort rövid
levélben megkérdezni, mit tart szakmaiatlannak (6. fejezet), és felajánlani a
facilitált, interjús elicitációt, amely a SHELF eredeti formájához közelebb
áll, mint az önkitöltős űrlap.

## 2. Mi válthatta ki a „szakmaiatlan” ítéletet

### 2.1 Regiszter és hangnem

A kérdőív szándékosan „köznyelvi” lett (commit: *expert questionnaire in plain
language*). Tapasztalt klinikusnak és különösen egyetemi oktatónak ez nem
olvashatóságot, hanem igénytelenséget jelez. Példák és javasolt szakkifejezés:

| hely | jelenlegi szöveg | javasolt |
|---|---|---|
| F5 cím (`ITEMS`) | „Lötyögő gerinc a felső állcsonton” | „Lebernyeges (mobilis, ún. flabby) gerinc a maxillán” |
| F8 cím | „Mi van a felső fogsorral szemben (antagonista)” | „Az antagonista fogazat / fogpótlás a felső teljes fogsorral szemben” |
| A3 rögzítés | „mit csinál a tasak szájnyitáskor” | „a buccinator-tasak viselkedése szájnyitáskor” |
| A5 rögzítés | „az izmok kifelé nyomják az ujjat” | „a szájfenéki izomzat nyeléskor az ujjat a lingualis tasakból kifelé tolja” |
| UI (`start_button`) | „Kezdjük” | „A kitöltés megkezdése” |
| UI (`a_h2`, eyebrow-ok) | „Néhány szó Önről”, „Először / Általában / Végül” | „A kitöltő szakmai háttere”, „A. / B. / C. / D. rész” |
| UI (`how_4_lead`) | „Hol lehet az igazság?” | „Bizonytalansági tartomány” |
| UI (`d4`) | „Mit hagytunk ki?” | „Nem kérdezett, de fontosnak tartott tényezők” |
| Felkészítő (`prep_h1`) | „hogyan adjunk számot arról, amiben nem vagyunk biztosak?” | „A bizonytalanság számszerű megadása” |

Két mondat külön is bánthatja a kitöltőt:

* „Nem a tankönyvre, hanem az Ön saját … tapasztalatára vagyunk kíváncsiak,
  akkor is, ha az eltér a tanultaktól” + „Elődeink és tanáraink ezt a
  tapasztalatukból tanították”: annak, aki a tankönyvet írta vagy tanítja, ez
  úgy hangzik, hogy a diszciplína tudását anekdotának tekintjük. Javasolt
  keret: *a tankönyvi tanítás számszerűsítése és mért adatokkal való
  ellenőrzése*, amelyhez a tanítás hordozóit kérdezzük.
* Felkérő levél: „Így kis betegszám mellett is értelmezhető eredményt
  kapunk.” Ez úgy olvasható, hogy a szakértői vélemény pótolja a hiányzó
  adatot. Javasolt: a prior és a mért adat külön is bemutatásra kerül, a
  frissítés semleges priorral is elkészül (érzékenység), és maga az elicitáció
  önálló, közölhető vizsgálat (a protokoll ezt így is írja, a levél nem).

### 2.2 A gyakorlókérdések

A felkészítő négy általános műveltségi kérdést ad (Budapest–Bécs távolság,
Semmelweis életkora, 65 év felettiek aránya, Duna–Rajna), „Ne keressen utána”
utasítással. Egy professzor számára ez kvíz, amely az ő tudását teszteli, nem
pedig módszertani gyakorlat. A SHELF-ben a „training” facilitátorral zajlik;
önkitöltős, szakmaidegen tudáskérdés a leggyengébb formája. A gyakorlat
válaszai ráadásul jelenleg nem kerülnek az exportba (`BACKGROUND_CSV_COLUMNS`
nem tartalmazza a `gyak_*` mezőket), így a protokoll 3. pontjában ígért
kalibrációs leírás sem készülhet el belőlük.

Lehetőségek: (a) szakmai „seed” kérdések ismert válasszal (publikált
fogatlansági arány adott korcsoportban, az átadás utáni korrekciós ülések
átlagos száma a szakirodalomban, az OHIP átlagos változása új teljes fogsor
után egy ismert vizsgálatban), egyértelműen *nem kötelező* gyakorlatként, a
SHELF-re hivatkozva; (b) a gyakorlat helyett tízperces, telefonos/videós
facilitált bevezetés; (c) a gyakorlat elhagyása és a kalibráció leírásának
kivétele a protokollból.

### 2.3 Formai hiányok

* A webes oldalakon sehol nem szerepel a vizsgálatvezető neve, beosztása és
  elérhetősége (csak a levél aláírásában, környezeti változóból); a lábléc
  egyetlen intézményi sor.
* Nincs etikai engedélyre vagy annak nem szükséges voltára utaló mondat.
* Nincs egyetlen módszertani hivatkozás sem (SHELF, EFSA-útmutató), pedig a
  protokoll ezekre épül; a kitöltő nem látja, hogy bevett módszerről van szó.
* A Word-űrlap fejléce „v1.2 (2026-09-06)”, az alkalmazás v2.0
  (`make_expert_form_docx.py`, 96. sor; `FORM_VERSION`).
* A felkérő levél tárgya „kb. 30 perc”, a törzs „35–45 perc”.
* Az OHIP a kérdőívben „OHIP”, a protokollban OHIP-5; a MAI a kérdőívben
  „rágásteszt”, a definícióban hue-fok. Ugyanannak a mérésnek több neve van.

### 2.4 Terhelés

Tizenhat tétel × (irány + irány-bizonyosság + legfeljebb kilenc szám + küszöb +
egy-két résztétel + megjegyzés), ehhez négy gyakorló-, tizenkét háttér- és
általános, nyolc záró kérdés: nagyságrendileg 250 mező. A Word-változat a
tételeknél tizenhat alkalommal ismétli ugyanazt a sűrű számrácsot a „mindkét
változatnál” sorral és a két lábjegyzettel együtt. A 35–45 perc gondos
kitöltőnél inkább 60–90 perc; ez önmagában is elutasítást okozhat.

## 3. Klinikai-tartalmi problémák tételenként

A folytonos tételek pólusértékei a protokoll szerint „a kohorsz alsó és felső
ötödéből kerekítve” származnak. Valójában F1, F3, F4 és F6 eddig 6–9 betegnél
van mérve (`stat_output/bayes_predictor_registry.csv`: n = 9;
`bayes_feltaro_R/03_betegszintu_tabla_anonim.csv`: n = 6), és a pólusok ezen
betegek tartományán kívül esnek:

| tétel | mért tartomány (n = 6) | A pólus | B pólus | megjegyzés |
|---|---|---|---|---|
| F1 | 7,3–10,0 mm | 10 | 5 | a B a mért tartomány alatt |
| F3 | 19,8–25,8 mm | 25 | 17 | a B a mért tartomány alatt |
| F4 | 130,5–139,0° | 140 | 125 | mindkét pólus a tartományon kívül |
| F6 | 79,1–87,9° (eltérés 2–11°) | ≤ 5 | ≥ 20 | a B a tartományon kívül |

Ezért a pólusokat nem lehet „a kohorsz ötödeiként” bemutatni; extrapolált
forgatókönyvek, és ezt vagy vállalni kell, vagy klinikai osztályozásra
cserélni.

| tétel | probléma | javaslat |
|---|---|---|
| **F1** | „a mintán kb. 10 / 5 mm” – a mérés (a gerincél és a bukkális áthajlás átlagos távolsága, Blender-addon) nincs megnevezve; kívülálló nem tudja a saját fogalmára vetíteni. | A mérés definíciója ábrával; vagy a Cawood–Howell-osztályozás (1988) fokozatai, amelyeket minden protetikus ismer. |
| **F3** | Az addon a szájpad legmagasabb pontját az *okklúziós síktól* méri (Z = 0), tehát az érték a gerincmagasságot is tartalmazza; „lapos szájpad (kb. 17 mm)” klinikailag értelmezhetetlen (17 mm nem lapos). | A boltozat mélységét a gerincélek síkjához viszonyítva megadni, vagy a klinikai U-alakú / V-alakú / lapos felosztást használni, és a mm-értéket csak az elemzésben. |
| **F4** | A „tuber – locus caninus – papilla incisiva” szög csúcsa a szemfogponton van; ott a *szögletes* ívnél a szög a 90°-hoz közelít, a V-alakúnál a 180°-hoz, tehát a szöveg („a nagyobb szög szögletesebb”) ellentmond a geometriának, hacsak a mérés csúcsa nem máshol van. A kohorsz 130–139°-os értékei nem döntik el. | A mérési SOP-pal egyeztetni, ábrát tenni a tételhez; vagy a klasszikus ívforma-osztályokat (négyzetes / ovális / V-alakú) kérdezni, és a szöget csak az elemzésben használni. **Ellenőrizendő a következő meghívás előtt.** |
| **F6** | Az interalveoláris vonal és a rágósík szöge klasszikus fogalom (a 80° alatti szög keresztharapásos felállítást indokol), de a tétel az eltérés irányát összevonja (alsó szélesebb vagy felső), majd résztételben kérdezi, számít-e; a „≥ 20°” B pólus a kohorsz tartományán kívül. | A klinikai irányt megnevezni (az alsó gerinc az felső előtt/kívül fut), a 80°-os klasszikus küszöböt pólusként használni, a „közepes” forgatókönyvet elhagyni. |
| **F5** | „lötyögő” | „lebernyeges (mobilis) gerinc”; a lokalizációs résztétel jó. |
| **F7** | „van torus (plató vagy orsó)” egy kontrasztban. | Fő kontraszt maradhat, de a résztétel kötelezővé tehető a nagyság/alak miatt. |
| **F8** | A = kivehető pótlás (teljes fogsor, overdenture, fémlemezes együtt), B = természetes fogazat vagy rögzített pótlás; a „nincs, most készül” kimarad. A professzor kombinációs szindrómára (Kelly) gondol. | A pólusokat klinikai helyzetként megnevezni: „teljes fogsor mindkét állcsontban” vs. „megtartott alsó frontfogak / rögzített pótlás az alsó állcsontban”; a résztétel (kategória vs. erőirány) jó. |
| **A1** | Kaán-fokozat (1–5) és mm-küszöb egy tételben; a mm-mérés (A2, modellanalízis) definiálatlan. | A mm-küszöb kérdését elhagyni vagy a mérést definiálni. |
| **A5, A11** | Háromfokozatú változó két pólusként + „az inkább az A-hoz áll közel” résztétel. | Három forgatókönyv (mint az optimum tételeknél), vagy sorrendezés. |
| **TUB** | Négy jellemző (A6–A9) egyetlen „csupa jó / csupa rossz” kompozit pólusban; ez nem egy változó, a szakértő nem tud rá arányt mondani. | A6, A7, A9 külön tétel (A8 a regiszter szerint validációs kategória), vagy a „legfontosabb” résztétel után csak arra kérni nagyságot. |
| **A10** | Angle II. és III. összevonva; ezek ellentétes állcsontviszonyok, más következménnyel. | Két külön kontraszt (I. vs. II., I. vs. III.), vagy a mért szög osztályai. |
| **A12** | rendben | – |

**Hiányzó klasszikus tényezők.** A professzor keresni fogja: vestibulum-
mélység, frenulumok tapadása, linea mylohyoidea, retromylohyoid tér, a nyelv
mérete és helyzete, nyál, a nyálkahártya reziliencia, House-féle garatforma,
temporomandibuláris státusz, garatreflex. Az utóbbi kettő (A13, F9) a
vizsgálatban mért változó, mégsem tétel. A D4 („mit hagytunk ki”) kérdés ezt
csak attól tudja meg, aki a kérdőívet végig kitölti. Javasolt egy mondat a
tájékoztatóban: *a vizsgálatban mért 16 tételről kérdezünk; a nem mért
klasszikus tényezőket (felsorolás) a záró kérdésben jelezheti.*

## 4. Módszertani problémák (a „tudományos feldolgozásra alkalmatlan” oldal)

1. **A kimenet nem megfigyelhető a klinikus számára.** A siker az OHIP-5-,
   GOHAI- és MAI-változásból képzett, MCID-egységben számolt kompozit három
   hónapnál, a régi fogsorhoz képest. A klinikus nem ezt látja, hanem azt,
   hogy a beteg hordja-e, panaszkodik-e, hányszor jön korrekcióra, milyen a
   retenció és a stabilitás. A „100 betegből hány sikeres” számok ezért egy
   *más* kimenetre vonatkoznak, mint amit az elemzés annak vesz. A protokoll
   szerint a MCID-küszöbök (OHIP-5 4,5 a 0–20-as skálán; GOHAI 16 a 12–60-as
   skálán; MAI 8,6 hue-fok) még véglegesítendők, tehát a kérdőív egy nem
   lezárt definícióra kér becslést. A régi fogsorhoz viszonyított javulás
   ráadásul erősen függ attól, volt-e egyáltalán korábbi fogsor (B5). Javaslat:
   klinikai sikerdefinícióra elicitálni („három hónappal az átadás után a beteg
   a fogsort rendszeresen hordja, lényeges panasz nélkül, a retenció és a
   stabilitás a kontrollon megfelelő”), és a mért kompozithoz való viszonyt az
   elemzésben, kimondott feltevésként kezelni (vagy külön híd-kérdéssel
   elicitálni).
2. **Horgonyzás.** A kidolgozott példa valódi tételt (F1) használ konkrét
   számokkal (85/60, sávok 75–92 és 45–72), és ugyanezek a számok ismétlődnek
   a tájékoztató „Száz beteg” bekezdésében (`how_3`), a Word-űrlapon és a
   felkészítőben. A SHELF és az EFSA-útmutató kifejezetten óv attól, hogy a
   példa az elicitált mennyiségre vonatkozzon. Javaslat: nem kérdezett,
   fiktív adottság a példában, más nagyságrendű számokkal.
3. **Vezető elrendezés.** A protokoll szerint „B pólus (az elődök szerint
   kedvezőtlen változat)”: mind a 16 tételnél a B a tankönyvi rossz. A
   tapasztalt kitöltő ezt azonnal látja, és a válasz a várt irányba tolódik.
   Javaslat: az A/B hozzárendelés tételenként (vagy kitöltőnként) véletlen,
   a hozzárendelés tárolva.
4. **Feltételes hatás vs. marginális felhasználás.** A tájékoztató két, „csak
   ebben az egy adottságban különböző” beteget kér (feltételes hatás), a
   protokoll 5b pontja szerinti elsődleges felhasználás viszont tételenkénti
   2×2-es tábla (marginális összefüggés). Ráadásul a „minden más azonos”
   feltevés az együtt járó képleteknél (A1/A2/A12, A6–A9, F1/F5) klinikailag
   irreális. Döntés kell: ha a 2×2-es összevetés az elsődleges, marginális
   kérdés kell („két tipikus beteg, ahogy a praxisban látja őket”); a
   feltételes megfogalmazás csak a többváltozós modellhez indokolt.
5. **Irány-bizonyosság.** A pólusonkénti eloszlásokból P(β > 0) már
   következik; külön kérdezve redundáns, és az 50 %-os opció az irány
   megjelölése után ellentmondás (akkor „nincs érdemi különbség” kellene).
   Javaslat: csak akkor kérdezni, ha a nagyságot nem adja meg.
6. **B2 és B3.** „Hány százalékban múlik a siker az adottságokon” és „hányszor
   annyit számít” operacionális definíció nélküli mennyiségek; feldolgozni
   nem lehet őket. Javaslat: elhagyni, vagy megszámlálható formába tenni
   („100 sikertelen fogsorból hánynál az anatómia a fő ok”).
7. **Kvantilis-illesztés.** A módusz + 2,5 % + 97,5 % három célérték két
   paraméterre; a szkript jelzi, ha a lefedettség < 0,85 vagy a módusz > 10
   ponttal eltér, de a protokoll nem mondja meg, mi történik a jelzett
   válaszokkal. A SHELF kvartilis-módszere (medián, alsó és felső kvartilis)
   könnyebben adható és pontosan illeszthető.
8. **Beküldést gátló konzisztencia-szabályok.** A program nem engedi
   beküldeni az ellentmondó választ, így a kitöltő a program kedvéért írja át
   a véleményét. A SHELF visszajelzést és újrakérdezést javasol, nem tiltást.
   Javaslat: csak az alsó ≤ pont ≤ felső szabály maradjon gátló, a többi
   figyelmeztetés, a válasz eredeti formában is tárolva.
9. **Kalibráció szerep nélkül.** A gyakorlókérdések válaszai nem szolgálnak
   sem képzésre (nincs facilitátor), sem súlyozásra (Cooke-módszer); az
   exportba sem kerülnek. El kell dönteni a szerepüket, vagy elhagyni őket.
10. **Nincs pilot, nincs tartalmi validálás.** Az egyetlen végigpróbálás a
    vizsgálatvezetőé. Két-három külső klinikus időmérős, hangosan gondolkodós
    pilotja a további meghívások előtt; a közléshez a tartalmi validálás
    leírása (az elicitációs riportálási ajánlások ezt kérik).

## 5. Javasolt teendők

**1. prioritás – a következő meghívó előtt**

* Terminológia és regiszter átírása (2.1), a tankönyvre és a kis betegszámra
  utaló mondatok átfogalmazása.
* A kidolgozott példa és a `how_3` bekezdés horgonyzásmentesítése.
* A/B véletlen hozzárendelése, tárolással; az export és az R-regiszter ennek
  megfelelő kezelése.
* Vizsgálatvezető neve, beosztása, elérhetősége, etikai hivatkozás, egységes
  verziószám és dátum minden oldalon és a Word-űrlapon; a levél tárgyának és
  törzsének egyeztetése; a mérések egységes elnevezése (OHIP-5, GOHAI, MAI).
* Klinikai sikerdefiníció a kérdőívben, a kompozithoz való viszony a
  protokollban.
* F1, F3, F4, F6: mérési definíció ábrával vagy klinikai osztályozás; az F4
  irányának ellenőrzése; a pólusok extrapolált voltának kimondása.
* A10 szétválasztása; TUB helyett külön tételek vagy csak a legfontosabbra
  kért nagyság.
* Gyakorlókérdések: szakmai seed-kérdések nem kötelezőként, vagy elhagyás.
* Konzisztencia-szabályok figyelmeztetéssé alakítása.

**2. prioritás – erősen ajánlott**

* Terhelés csökkentése: kvartilis-módszer; a sávokat csak a kitöltő által
  legfontosabbnak rangsorolt tételeknél kérni; őszinte időbecslés.
* A5 és A11 három forgatókönyvvel; F6 iránnyal és 80°-os küszöbbel.
* B2/B3 elhagyása vagy megszámlálható formába tétele; az irány-bizonyosság
  csak nagyság nélkül.
* A nem kérdezett klasszikus tényezők explicit felsorolása a tájékoztatóban.
* A `gyak_*` mezők exportja, ha a kalibráció marad.

**3. prioritás – a közléshez**

* Facilitált (interjús) elicitáció mint alternatív mód, rögzítve, a
  módot az exportban jelölve.
* Második kör az egyesített eloszlás visszajelzésével (SHELF-féle
  viselkedéses aggregáció), ha a szakértők száma megengedi.
* Pilot-jelentés (idő, félreértett tételek), a protokoll v2.1 verziójában.

## 6. A professzor megkeresése (levélvázlat)

> Tisztelt Professzor Úr / Asszony!
>
> Köszönöm, hogy időt szánt a PREDICT szakértői kérdőívre, és köszönöm az
> őszinte megjegyzését. Fontos nekünk, hogy az eszköz szakmailag megállja a
> helyét, ezért nagyon hálás lennék, ha két-három mondatban megírná, mely
> pontokat tartja szakmaiatlannak vagy feldolgozásra alkalmatlannak: a
> tételek megfogalmazását, a sikerdefiníciót, a számszerű becslés módját
> vagy magát a szakértői prior elvét. A kérdőívet a visszajelzések alapján
> átdolgozzuk.
>
> Ha a kérdőíves forma nem felel meg, szívesen felkeresném egy 30–40 perces
> személyes vagy telefonos beszélgetésre, amelyben ugyanezeket a kérdéseket
> strukturált interjú formájában, ábrákkal tenném fel; ez a módszertan
> (Sheffield Elicitation Framework) eredeti formájához is közelebb áll.
>
> Tisztelettel, …

A levél ne érveljen és ne oktasson; a módszertani hátteret (SHELF,
EFSA-útmutató, Johnson és mtsai 2010) csak akkor érdemes megemlíteni, ha a
válaszból kiderül, hogy az elv maga a kifogás tárgya.

## 7. Ami védhető, és amit nem érdemes kidobni

* A kétféle bizonytalanság szétválasztása, a pólusonkénti kvantilisek, a
  „nem tudom megítélni” és „a nagyságát nem tudom megbecsülni” válaszok, az
  egyéni, független kitöltés, az egyenlő súlyú keverés érzékenységi
  elemzéssel: mind az EFSA 2014-es útmutatójával, a SHELF-fel és O'Hagan
  (2019) ajánlásaival összhangban van.
* A szakértői prior mint módszer nem „szubjektív” a tudománytalanság
  értelmében: a hatóságok és a klinikai módszertani irodalom bevett eljárása,
  feltéve, hogy az elicitáció dokumentált, a prior és az adat külön is
  bemutatható, és semleges priorral is elkészül az elemzés. Ha a professzor
  kifogása ennek az elvnek szól, a válasz nem a kérdőív módosítása, hanem a
  prior-érzékenységi elemzés és az elicitáció önálló közlése.
* A protokoll elemzési lépései (béta-illesztés diagnosztikával, prior
  prediktív ellenőrzés, szerep szerinti bontás) rendben vannak; a 4.4 és
  4.7 pontok pontosítást, nem elvetést kívánnak.

## 8. Hivatkozások

* O'Hagan A, Buck CE, Daneshkhah A, et al. *Uncertain Judgements: Eliciting
  Experts' Probabilities.* Wiley, 2006.
* Gosling JP. SHELF: the Sheffield Elicitation Framework. In: Dias LC, Morton
  A, Quigley J (eds). *Elicitation: The Science and Art of Structuring
  Judgement.* Springer, 2018.
* O'Hagan A. Expert knowledge elicitation: subjective but scientific. *The
  American Statistician* 2019;73(sup1):69–81.
* EFSA. Guidance on Expert Knowledge Elicitation in Food and Feed Safety Risk
  Assessment. *EFSA Journal* 2014;12(6):3734.
* Johnson SR, Tomlinson GA, Hawker GA, Granton JT, Feldman BM. Methods to
  elicit beliefs for Bayesian priors: a systematic review. *J Clin Epidemiol*
  2010;63:355–369.
* Cooke RM. *Experts in Uncertainty: Opinion and Subjective Probability in
  Science.* Oxford University Press, 1991.
* Hemming V, Burgman MA, Hanea AM, McBride MF, Wintle BC. A practical guide
  to structured expert elicitation using the IDEA protocol. *Methods Ecol
  Evol* 2018;9:169–180.
* Tversky A, Kahneman D. Judgment under uncertainty: heuristics and biases.
  *Science* 1974;185:1124–1131.
* Cawood JI, Howell RA. A classification of the edentulous jaws. *Int J Oral
  Maxillofac Surg* 1988;17:232–236.
* Kruschke JK. Bayesian Analysis Reporting Guidelines. *Nat Hum Behav*
  2021;5:1282–1291.
