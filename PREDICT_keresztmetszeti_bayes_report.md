# PREDICT mint keresztmetszeti Bayes-i vizsgálat

**Adatpillanat:** 2026-08-19
**Státusz:** exploratív, hipotézisgeneráló keresztmetszeti elemzés

## Javasolt tanulmánykérdés

> A vizsgálat időpontjában együtt járnak-e a klasszikusan kedvezőtlennek
> tekintett fogatlan állcsonti anatómiai tulajdonságok a rosszabb objektív
> rágóképességgel, önbevallott rágóképességgel és orális életminőséggel?

Ez a kérdés a jelen adatokkal vizsgálható. Nem állít időbeli előrejelzést vagy
okozati hatást, és nem igényli a hiányos utánkövetési adatokat.

Lehetséges cím:

> **Edentulous anatomy, masticatory performance and oral-health-related
> quality of life: an expert-informed Bayesian cross-sectional pilot study**

## Hipotézisek

1. **H1 — betegjelzett életminőség:** kedvezőtlenebb anatómia mellett rosszabb
   a kiindulási OHIP/GOHAI-kompozit.
2. **H2 — önbevallott rágóképesség:** kedvezőtlenebb anatómia mellett rosszabb
   a jelenlegi önbevallott rágóképesség.
3. **H3 — objektív rágóképesség:** kedvezőtlenebb anatómia mellett magasabb,
   azaz rosszabb a kiindulási MAI.
4. **H4 — szakmai tudás és adat:** az elicitált iránypriorok és a semleges
   priorból származó adatjel összevetése megmutatja, hol van
   prior–adat-konkordancia vagy konfliktus.

## Vizsgálati terv

- **Dizájn:** keresztmetszeti.
- **Populáció:** 48 kezelésre jelentkező beteg; a prediktor-specifikus
  elemzésekben 30–41 teljes eset.
- **Primer outcome:**
  `QoL_worse = z[(z(OHIP_sum) + z(60-GOHAI_sum))/2]`.
- **QoL-kompozit indoka:** az OHIP és a fordított GOHAI Pearson-korrelációja
  0,735; a kétkomponensű alpha 0,847.
- **Másodlagos outcome:** standardizált kiindulási MAI, magasabb = rosszabb.
- **További betegjelzett outcome:** önbevallott rágóképesség, magasabb
  modellezett érték = rosszabb.
- **Strukturális hiányzás:** felső és alsó állcsonti modellek külön.
- **Korrekció:** életkor, nem, `both` vs egyetlen állcsont fogatlansága.
- **Primer prior:** minden anatómiai beta `Normal(0, 0,5²)`.
- **Irányprior-szenzitivitás:** azonos half-Normal hatásnagyság, de
  `P(beta>0)` = 0,80 / 0,90 / 0,98 az elicitált gyenge / mérsékelt / erős
  bizonyosságnak megfelelően.
- **Becslés:** HC3-robusztus regressziós likelihood és Bayes-i
  regularizálás; minden eredmény standardizált skálán.

A primer következtetés a semleges priorból készül. Az irányinformált elemzés
csak érzékenység, mert a szakmai állításokat eredetileg retencióra és
stabilitásra, nem közvetlenül OHIP/GOHAI/MAI-paraméterekre elicitaltuk.

## Fő eredmények semleges prior mellett

### Primer outcome: OHIP/GOHAI QoL-kompozit

Az alábbi modellek életkorra, nemre és fogsortípusra korrigáltak. A pozitív
beta a kedvezőtlen anatómia és a rosszabb QoL együttjárását jelenti.

| Prediktor | n | Posterior beta | 95% CrI | P(beta>0) |
|---|---:|---:|---:|---:|
| A1 telítődő gerincrizikó | 35 | +0,47 | −0,21; +1,16 | 91% |
| A9 tuberculum-mozgékonyság | 35 | +0,47 | −0,25; +1,19 | 90% |
| A8 inklinációs rizikó | 35 | +0,29 | −0,35; +0,92 | 81% |
| A6 feszesíny-rizikó | 35 | +0,34 | −0,44; +1,11 | 80% |
| A6+A7+A9 primer tuberculum-kompozit | 35 | +0,27 | −0,51; +1,05 | 75% |
| A5 lingualis tasak | 35 | +0,19 | −0,57; +0,94 | 69% |
| F5 lötyögő gerinc | 33 | +0,17 | −0,53; +0,87 | 68% |
| A11 szájfenék | 35 | +0,11 | −0,60; +0,81 | 62% |
| F7 torus palatinus | 33 | −0,31 | −0,89; +0,28 | 15% |
| A4 torus mandibularis | 34 | −0,39 | −0,98; +0,21 | 10% |

Az A1 és A9 adja a legkoherensebb várt irányú QoL-jelet. Egyetlen 95%-os
credible interval sem zárja ki a nullát. Az A8-as jelet különösen óvatosan
kell kezelni az A7–A8 validációs inkonzisztencia miatt.

### Önbevallott rágóképesség

| Prediktor | n | Posterior beta | 95% CrI | P(beta>0) |
|---|---:|---:|---:|---:|
| F5 lötyögő gerinc | 33 | +0,41 | −0,21; +1,02 | 90% |
| A6 feszesíny-rizikó | 35 | +0,40 | −0,25; +1,05 | 88% |
| A9 tuberculum-mozgékonyság | 35 | +0,34 | −0,30; +0,99 | 85% |
| A8 inklinációs rizikó | 35 | +0,30 | −0,33; +0,93 | 83% |
| A1 telítődő gerincrizikó | 35 | +0,24 | −0,49; +0,97 | 74% |
| A5 lingualis tasak | 35 | +0,22 | −0,49; +0,92 | 73% |
| A4 torus mandibularis | 34 | −0,41 | −1,08; +0,26 | 11% |

Itt az anatómiai és betegjelzett állapot közötti jel koherensebb: F5, A6,
A8 és A9 mind a szakmailag várt irányba mutat. A mandibularis torus továbbra
is ellenirányú.

### Objektív MAI

Az objektív rágóképesség nem mutat koherens kapcsolatot az elicitált
anatómiai irányokkal:

- F5: `P(beta>0)=70%`;
- A6: 66%;
- A1: 58%;
- primer tuberculum-kompozit: 49%;
- A5: 21%;
- A12: 12%;
- F7: 10%.

A fő keresztmetszeti eredmény ezért nem az, hogy „az anatómia meghatározza az
objektív MAI-t”, hanem az, hogy bizonyos anatómiai jellemzők inkább a beteg
által megélt állapottal járnak együtt, miközben az objektív keverési teszt
ettől részben független dimenziót mérhet.

## Állcsont-specifikus többváltozós modellek

### Alsó állcsont

A kibővített alsó modell egyidejűleg tartalmazta az A1 gerincállapotot, a
tuberculum-kompozitot, A11-et, A5-öt, A4-et és A13-at, továbbá a három
korrekciós változót. Ez n=33 mellett 10 paraméter, tehát csak erősen
regularizált pilotmodell.

QoL esetén semleges priorral:

- A1 gerinc: `P(beta>0)=75%`;
- tuberculum: 59%;
- A11: 63%;
- A5: 64%;
- A4 torus: 31%;
- A13 TMI: 35%.

Önbevallott rágóképességnél:

- tuberculum: 87%;
- A5: 82%;
- A1: 71%;
- A4 torus: 3%, erősen ellenirányú pilotjel.

MAI esetén a többváltozós modell egyik anatómiai főhatásra sem adott stabil,
várt irányú adatjelet.

### Felső állcsont

Az F5, F7 és az F8 nominális kategóriáit tartalmazó felső modellben:

- F5 és rosszabb önbevallott rágás: `P(beta>0)=90%`;
- F5 és QoL: 64%;
- F5 és MAI: 67%;
- F7 és QoL: 24%;
- F7 és MAI: 8%;
- F7 és önbevallott rágás: 25%.

Az F8 elicitált hatása interakciófüggő, de a kedvezőtlen erőirány nincs külön
mérve; ezért itt csak nominális korrekciós változóként szerepelt.

## Interakciók

A gyenge iránypriorral előre jelölt interakciók semleges prioros adatjele:

| Interakció | QoL P(pozitív) | MAI P(pozitív) | önbevallott rágás P(pozitív) |
|---|---:|---:|---:|
| A1 × A11 | 44% | 34% | 55% |
| A5 × A11 | 62% | 50% | 77% |

Jelenleg nincs meggyőző keresztmetszeti bizonyíték szuperadditív
interakcióra. Az A5 × A11 és önbevallott rágás kapcsolata követésre érdemes,
de nagyon bizonytalan.

## Mit csinál az irányinformált prior?

Az irányprior ugyanazt a hatásnagyság-priort használja, csak az előjel
előzetes esélyét módosítja. A jelen minta mellett ez nagyon erős hatású:

| Prediktor/outcome | Semleges P(beta>0) | Elicitált prior | Irányinformált P(beta>0) |
|---|---:|---:|---:|
| A1 → QoL | 91% | 98% | 99,8% |
| F5 → QoL | 68% | 80% | 90% |
| F7 → QoL | 15% | 98% | 90% |
| A4 → QoL | 10% | 98% | 85% |
| F7 → MAI | 10% | 98% | 84% |
| A12 → MAI | 12% | 98% | 87% |

Az első két sorban prior és adat ugyanabba az irányba mutat. Az utolsó négy
sorban viszont az erős prior az ellenirányú adatjel ellenére pozitív
posteriort hoz létre. Ez matematikailag helyes, de tartalmilag azt jelenti,
hogy a posterior főként a priori szakmai állítást tükrözi.

Ezért a keresztmetszeti közleményben kötelező egymás mellett bemutatni:

1. semleges prioros posterior;
2. elicitált irányprioros posterior;
3. a kettő közötti elmozdulást;
4. a prior–adat konfliktust.

## Adatminőségi és módszertani korlátok

- A vizsgálat keresztmetszeti: reverz okság, adaptáció és nem mért zavaró
  tényezők nem zárhatók ki.
- A primer QoL-kompozit utólag definiált, ezért exploratív.
- Az önbevallott rágóképesség ordinális kategóriáit a pilotban standardizált
  folytonos skálaként modelleztük; ordinális regressziós érzékenységi elemzés
  szükséges egy kézirat véglegesítése előtt.
- A posteriorok HC3-robusztus normális likelihood-közelítésből származnak,
  nem teljes generatív Bayes-modellből vagy MCMC-becslésből.
- A korrigált modellek elemszáma többnyire 30–35.
- Sok anatómiai összevetés készült; a magas posterior valószínűségek nem
  megerősítő döntési küszöbök.
- F1/F3/F4/F6 csak 8–9 outcome-párral rendelkezik, így legfeljebb
  esettanulmány-szerű pilotjel.
- Az A7=3 és A8=3 logikai egyezés kilenc oldalból nyolcon sérül.
- A12=3 és A13=3 csak egy-egy beteg.
- A6–A9 ugyanazt a képletet írja le, de nem biztos, hogy reflektív látens
  skálát alkot; a primer kompozit ezért A8 nélkül, formatív pilotként készült.
- A jelenlegi résztvevők kezelésre jelentkező, szelektált populációt alkotnak;
  az eredmény nem általánosítható automatikusan minden fogatlan betegre.
- Az alsó többváltozós MAI-modell irányprioros importance sampling effektív
  mintanagysága alacsony volt; ebből irányinformált következtetés nem készült.

## Validációs minősítés

**Megosztható exploratív pilotként, a fenti korlátozások kötelező
feltüntetésével; megerősítő vagy prognosztikai eredményként nem.**

Az elemzés újrafuttatva reprodukálta a közölt táblázatokat. A 48 forrássorhoz
48 egyedi belső azonosító tartozik, és az exportált eredménytáblák nem
tartalmaznak betegazonosítót. A QoL-kompozit közölt korrelációja és alphája a
metaadatból visszaszámolva egyezik. A primer állítások a semleges prioros,
kovariánsokra korrigált eredményekből származnak.

## Védhető keresztmetszeti következtetés

> Ebben a kezelésre jelentkező keresztmetszeti mintában a súlyosabb
> mandibularis gerincforma és egyes tuberculumjellemzők — különösen a
> mozgékonyság — nagyobb valószínűséggel jártak együtt rosszabb
> betegjelzett orális életminőséggel. A lötyögő gerinc, a feszes
> ínyborítás hiánya és a tuberculum mozgékonysága rosszabb önbevallott
> rágóképesség irányába mutatott. Az objektív MAI nem mutatott hasonlóan
> koherens anatómiai gradienst. A torusváltozók több elemzésben a klasszikus
> várakozással ellentétes jelet adtak. A széles bizonytalansági tartományok és
> a priorérzékenység miatt az eredmények hipotézisgenerálók.

## Reprodukálható fájlok

- [bayes_cross_sectional_analysis.py](bayes_cross_sectional_analysis.py)
- [cross_sectional_item_models.csv](stat_output/cross_sectional_item_models.csv)
- [cross_sectional_multivariable_models.csv](stat_output/cross_sectional_multivariable_models.csv)
- [cross_sectional_interactions.csv](stat_output/cross_sectional_interactions.csv)
- [cross_sectional_metadata.json](stat_output/cross_sectional_metadata.json)
