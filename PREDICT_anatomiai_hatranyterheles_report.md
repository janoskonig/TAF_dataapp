# Mandibularis anatómiai hátrányterhelési index

**PREDICT keresztmetszeti pilot elemzés — 2026-08-19**

## Rövid eredmény

Az előre rögzített, ötkonstrukciós egész pontos anatómiai score a mandibularis
mintában technikailag jól számítható és megfelelő szórást mutatott. A nagyobb
anatómiai teher a rosszabb OHIP-5 felé mutatott, de a bizonytalansági
intervallum a nullát is tartalmazta: `rho=+0,21`
(`n=40`, bootstrap 95%-os CI `-0,12;
0,50`, kétoldali permutációs `p=0,199`).

A fokozatos 0–5 score valamivel erősebb OHIP-jelet adott
(`rho=+0,29`, 95%-os CI `-0,04;
0,58`), de ez is exploratív. A GOHAI, a MAI és az
önbevallott rágóképesség nem mutatott koherens, azonos irányú összefüggést.

## Kutatási kérdés

> Együtt jár-e a nagyobb, szakértőileg előre meghatározott mandibularis
> anatómiai hátrányterhelés a rosszabb kiindulási orális életminőséggel?

Ez keresztmetszeti asszociációs kérdés. A jelen elemzés nem igazolja, hogy a
score előre jelzi az elkészülő új fogsor eredményét.

## Populáció és kimenetek

- Forrás: `patients.csv`, 48 egyedi betegrekord.
- Mandibularis elemzési populáció: `both` vagy `lower`, összesen
  **41 beteg**.
- Teljes ötkonstrukciós score: **40 beteg**.
- Elsődleges kimenet: OHIP-5 összeg, 0–20, magasabb = rosszabb.
- Másodlagos kimenetek: GOHAI fordított irányban, MAI-huedegree és
  önbevallott rágóképesség fordított irányban, így minden elemzett kimenetnél
  a magasabb érték rosszabbat jelent.
- A digitális mintával rendelkező mandibularis alcsoport külön pilotként
  szerepel; az A2 modellanalízis eredménye még nem része a score-nak.

## Score-definíció

Az index formatív: különböző anatómiai hátrányokat összegez, ezért az
összetevőknek nem kell magas belső konzisztenciát mutatniuk. Minden konstrukció
legfeljebb egy pontot ér.

| Konstrukció | Egész pontos kedvezőtlen definíció |
|---|---|
| Gerincatrophia | A1 >= 3 vagy A12 >= 2; A1/A2/A12 együtt is legfeljebb egy pont |
| Torus mandibularis | A4=2 vagy 3 legalább az egyik oldalon |
| Lingualis tasak | A5=3 legalább az egyik oldalon |
| Tuberculum | A6–A9 közös, 0–1-re skálázott blokkátlaga >= 0,5 |
| Szájfenék | A11=3 |

A fokozatos érzékenységi score minden konstrukciót 0–1 között kódol, és az
öt konstrukciót azonos súllyal összeadja. A kétoldali változóknál az elsődleges
fokozatos specifikáció a két oldal átlagát használja; a rosszabbik oldalas és
az A8 nélküli változat külön érzékenységi elemzés.

## Score-eloszlás

0 pont: 6, 1 pont: 7, 2 pont: 8, 3 pont: 15, 4 pont: 3, 5 pont: 1.

| Konstrukció | Kedvezőtlen / mérhető | Prevalencia |
|---|---:|---:|
| Gerincatrophia | 23/41 | 56,1% |
| Torus mandibularis | 8/40 | 20,0% |
| Lingualis tasak | 27/41 | 65,9% |
| Tuberculum-blokk | 20/41 | 48,8% |
| Szájfenék | 7/41 | 17,1% |

## Elsődleges eredmény: OHIP-5

- Spearman-korreláció: `rho=+0,21`.
- Bootstrap 95%-os intervallum: `-0,12; 0,50`.
- Kétoldali permutációs p-érték: `0,199`.
- Életkorra, nemre és egy- vs kétállcsontos fogsortípusra korrigált HC3-robusztus
  lineáris modell: egy további hátránypont mellett az OHIP becsült változása
  `+0,52` pont
  (95%-os CI `-1,07; 2,11`;
  `n=34`).

Az irány mindkét modellben a szakmailag várt, de a becslés pontatlan; a nulla
és egy klinikailag értelmezhető pozitív kapcsolat egyaránt összeegyeztethető
az adatokkal.

## Másodlagos kimenetek

| Kimenet | n | Spearman rho | Bootstrap 95%-os CI |
|---|---:|---:|---:|
| GOHAI, magasabb = rosszabb | 40 | -0,04 | -0,36; 0,27 |
| MAI, magasabb = rosszabb | 34 | -0,23 | -0,55; 0,13 |
| Önbevallott rágás, magasabb = rosszabb | 40 | -0,02 | -0,35; 0,31 |

A score tehát nem viselkedik univerzális kimeneti prediktorként. Ez
értelmezhető úgy, hogy az anatómiai mechanizmusok eltérően jelennek meg a
beteg által megélt életminőségben és a színkeverési tesztben, illetve hogy a
kis minta mellett a becslések instabilak.

## Score-specifikációs érzékenység OHIP-re

| Specifikáció | n | Spearman rho | Bootstrap 95%-os CI |
|---|---:|---:|---:|
| Egész pontos, elsődleges | 40 | +0,21 | -0,12; 0,50 |
| Fokozatos, oldalátlag | 40 | +0,29 | -0,04; 0,58 |
| Fokozatos, rosszabbik oldal | 40 | +0,22 | -0,12; 0,51 |
| Fokozatos, A8 nélkül | 40 | +0,28 | -0,04; 0,57 |
| Egész pontos, gerinc csak A1 | 40 | +0,25 | -0,07; 0,54 |
| Egész pontos, tuberculum A8 nélkül | 40 | +0,19 | -0,13; 0,48 |

Az eredmény iránya nem egyetlen oldal-összevonási szabálytól függ, de a
hatásnagyság érzékeny a pontos score-definícióra. Ezért a végleges új
adatgyűjtés előtt ugyanazt a szabályt kell protokollban rögzíteni.

A konstrukciónkénti elhagyásos ellenőrzésben az OHIP-korreláció
`+0,12` és `+0,31`
között mozgott. A torus-komponens elhagyásakor volt a legnagyobb
(`rho=+0,31`), összhangban a korábbi
ellenirányú A4 pilotjellel. Ez **nem indokolja a torus utólagos kihagyását**;
csak megmutatja, hogy melyik komponens csökkenti leginkább a globális score
OHIP-kapcsolatát.

## Digitális mintás alcsoport

A mandibularis digitális mintás alcsoportban 14
beteg volt, közülük 13 teljes score–OHIP pár. Az egész pontos score
és OHIP korrelációja `rho=+0,41` volt
(95%-os CI `-0,20; 0,85`).
Ez a kis, szelektált alcsoport önmagában nem alkalmas validációra. Az A2
modellmérés később a gerincatrophia-komponenst helyettesítheti vagy
finomíthatja, de nem kaphat külön hatodik pontot.

## TDK-ban védhető következtetés

> A szakértőileg előre meghatározott, ötkonstrukciós mandibularis anatómiai
> hátrányterhelési index a jelen mintában megvalósítható és megfelelő
> variabilitást mutatott. A nagyobb anatómiai teher a rosszabb OHIP-5 felé
> mutatott, de az összefüggés bizonytalan volt, és nem ismétlődött meg
> koherensen minden másodlagos kimeneten. Az eredmények hipotézisgenerálók;
> az index prognosztikai értékét új, prospektív adatokon kell vizsgálni.

## Kötelező korlátok

1. A score-definíció a meglévő adatok előzetes megtekintése után, de a jelen
   formális futtatás előtt került rögzítésre; ezért ez nem prospektíven
   preregisztrált validáció.
2. Keresztmetszeti baseline kimenetekből nem állítható, hogy az anatómia
   előre jelzi az új fogsor sikerét vagy a kezelés nehézségét.
3. A minta kicsi, a bootstrap intervallumok szélesek, és a másodlagos
   összevetések többszörös exploráció részei.
4. A tuberculum blokk formatív és az A7–A8 adatkonzisztencia ismert problémája
   miatt külön A8 nélküli érzékenységi specifikáció készült.
5. Az azonos súlyok tartalmi döntést jelentenek; a jelen adatokból becsült
   súlyok túlillesztést okoznának.
6. A digitális alcsoport kiválasztottsága és kis elemszáma miatt annak
   eredménye csak pilot.

## Reprodukálhatóság

- Elemzőszkript: `anatomical_burden_analysis.py`
- Aggregált gépi eredmény: `stat_output/anatomiai_hatranyterheles_summary.json`
- QA-audit: `stat_output/anatomiai_hatranyterheles_audit.json`
- Ábrák: `stat_output/anatomiai_hatranyterheles_*.png`
- A kimenetek nem tartalmaznak nevet, TAJ-számot vagy betegszintű adatot.
