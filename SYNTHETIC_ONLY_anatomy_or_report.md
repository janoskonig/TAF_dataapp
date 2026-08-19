# SZINTETIKUS szimuláció – anchor-alapú egyedi anatómiai OR-ok

> **Nem empirikus kutatási eredmény.** A 1000 rekord számítógéppel generált,
> valódi beteget nem reprezentál, és nem használható fel hiányzó utánkövetés
> pótlására vagy anatómiai hatás igazolására.

## Szimulációs cél

Az elemzési pipeline ellenőrzése null-hatás forgatókönyvben. Az OHIP-, GOHAI-
és MAI-változás generálása szándékosan nem függött az anatómiai képletektől.
Ezért a várt valódi OR minden képletnél és kimenetnél 1,00.

- N = 1000
- Seed = 20260819
- Anchorok: OHIP és GOHAI = `responsiveness_change`; MAI = `chewing_change`
- Javulás-anchor: `Kicsit javult` vagy `Sokat javult`
- Romlás-anchor: `Kicsit romlott` vagy `Sokat romlott`
- A score-ok iránya úgy lett megfordítva, hogy a nagyobb érték mindig az adott
  irányú nagyobb változást jelentse; OHIP és MAI esetén az alacsonyabb score jobb
- Modell: képletenként külön logisztikus regresszió, korrigálva életkorra,
  nemre, fogsortípusra és a megfelelő kiindulási kimenetre
- Többszörös összehasonlítás: Benjamini–Hochberg FDR, összesen 132 teszt

## Anchor-alapú ROC/Youden küszöbök

- Anchor-alapu szamottevo OHIP-5 javulas: score `OHIP_change_improvement`, Youden-küszöb `3.000`, AUC `0.697`, szenzitivitás `0.624`, specificitás `0.678`
- Anchor-alapu szamottevo OHIP-5 romlas: score `OHIP_change_deterioration`, Youden-küszöb `1.000`, AUC `0.719`, szenzitivitás `0.437`, specificitás `0.859`
- Anchor-alapu szamottevo GOHAI javulas: score `GOHAI_change_improvement`, Youden-küszöb `4.000`, AUC `0.729`, szenzitivitás `0.778`, specificitás `0.556`
- Anchor-alapu szamottevo GOHAI romlas: score `GOHAI_change_deterioration`, Youden-küszöb `1.000`, AUC `0.771`, szenzitivitás `0.395`, specificitás `0.915`
- Anchor-alapu szamottevo MAI javulas: score `MAI_change_improvement`, Youden-küszöb `7.316`, AUC `0.597`, szenzitivitás `0.466`, specificitás `0.674`
- Anchor-alapu szamottevo MAI romlas: score `MAI_change_deterioration`, Youden-küszöb `1.993`, AUC `0.584`, szenzitivitás `0.351`, specificitás `0.757`

## Korrigált OR (95%-os Wald CI)

| Képlet | OHIP jav. | OHIP roml. | GOHAI jav. | GOHAI roml. | MAI jav. | MAI roml. |
|---|---:|---:|---:|---:|---:|---:|
| F5 | 0.85 (0.65–1.12) | 1.10 (0.80–1.51) | 0.89 (0.68–1.17) | 1.02 (0.72–1.43) | 0.89 (0.69–1.16) | 1.03 (0.77–1.37) |
| F7 | 1.01 (0.76–1.33) | 0.74 (0.53–1.04) | 1.08 (0.81–1.43) | 0.93 (0.66–1.33) | 1.06 (0.82–1.39) | 0.86 (0.64–1.16) |
| F8 | 0.93 (0.71–1.21) | 0.83 (0.61–1.14) | 0.91 (0.70–1.19) | 0.89 (0.64–1.25) | 1.08 (0.84–1.39) | 0.92 (0.70–1.22) |
| A1_Kaan | 1.13 (0.87–1.48) | 0.99 (0.72–1.35) | 1.07 (0.82–1.41) | 0.87 (0.62–1.22) | 1.15 (0.89–1.48) | 0.81 (0.61–1.08) |
| A3_jobb | 0.85 (0.65–1.12) | 1.26 (0.92–1.72) | 1.06 (0.81–1.40) | 0.58 (0.41–0.83) | 0.94 (0.73–1.22) | 0.96 (0.72–1.28) |
| A3_bal | 0.94 (0.71–1.23) | 1.15 (0.83–1.57) | 0.96 (0.73–1.26) | 1.09 (0.77–1.53) | 0.82 (0.63–1.07) | 1.25 (0.94–1.66) |
| A4_jobb | 1.30 (0.96–1.75) | 0.83 (0.58–1.18) | 1.25 (0.92–1.70) | 0.63 (0.42–0.95) | 0.97 (0.73–1.29) | 0.95 (0.69–1.30) |
| A4_bal | 0.86 (0.63–1.15) | 1.10 (0.78–1.55) | 1.01 (0.75–1.36) | 1.21 (0.84–1.74) | 0.96 (0.73–1.28) | 1.11 (0.81–1.51) |
| A5_jobb | 1.05 (0.80–1.37) | 0.93 (0.68–1.27) | 0.90 (0.69–1.18) | 1.11 (0.80–1.55) | 0.98 (0.76–1.27) | 0.98 (0.74–1.30) |
| A5_bal | 0.99 (0.76–1.29) | 0.97 (0.71–1.32) | 1.16 (0.89–1.52) | 0.98 (0.70–1.36) | 1.07 (0.83–1.37) | 0.85 (0.64–1.13) |
| A6_jobb | 1.04 (0.79–1.36) | 0.90 (0.66–1.24) | 1.08 (0.82–1.41) | 0.75 (0.53–1.05) | 1.14 (0.88–1.48) | 0.89 (0.67–1.18) |
| A6_bal | 1.04 (0.79–1.36) | 0.94 (0.69–1.29) | 1.02 (0.78–1.34) | 0.89 (0.64–1.26) | 1.16 (0.90–1.50) | 0.85 (0.64–1.13) |
| A7_jobb | 0.87 (0.67–1.14) | 1.08 (0.79–1.47) | 1.08 (0.82–1.41) | 1.08 (0.77–1.51) | 1.00 (0.77–1.29) | 1.00 (0.76–1.33) |
| A7_bal | 1.02 (0.78–1.34) | 0.89 (0.65–1.22) | 1.17 (0.89–1.53) | 1.02 (0.73–1.42) | 1.04 (0.80–1.34) | 0.94 (0.71–1.25) |
| A8_jobb | 1.08 (0.81–1.43) | 0.84 (0.60–1.17) | 1.25 (0.94–1.66) | 1.04 (0.73–1.47) | 1.06 (0.81–1.39) | 0.94 (0.70–1.26) |
| A8_bal | 0.85 (0.65–1.12) | 1.11 (0.81–1.53) | 0.95 (0.72–1.25) | 1.05 (0.74–1.47) | 0.97 (0.74–1.26) | 1.08 (0.81–1.44) |
| A9_jobb | 1.15 (0.88–1.51) | 1.07 (0.78–1.48) | 1.03 (0.78–1.35) | 0.92 (0.65–1.29) | 0.72 (0.55–0.93) | 1.21 (0.91–1.61) |
| A9_bal | 1.16 (0.88–1.52) | 0.73 (0.53–1.01) | 1.27 (0.96–1.67) | 0.77 (0.54–1.09) | 1.07 (0.83–1.39) | 0.89 (0.67–1.19) |
| A11 | 1.04 (0.77–1.39) | 1.16 (0.83–1.62) | 1.04 (0.77–1.39) | 0.85 (0.59–1.23) | 0.86 (0.65–1.14) | 0.89 (0.65–1.21) |
| A12 | 1.00 (0.77–1.31) | 1.03 (0.75–1.40) | 0.97 (0.74–1.27) | 0.95 (0.68–1.32) | 1.14 (0.88–1.47) | 0.92 (0.69–1.22) |
| A13 | 0.94 (0.70–1.26) | 1.08 (0.77–1.52) | 0.91 (0.68–1.22) | 0.99 (0.69–1.42) | 0.88 (0.66–1.16) | 1.08 (0.80–1.47) |
| A14 | 0.72 (0.53–0.96) | 1.07 (0.76–1.50) | 0.95 (0.71–1.27) | 1.05 (0.73–1.51) | 0.96 (0.73–1.27) | 1.23 (0.90–1.68) |

## QA

- Eseményarányok: `{"ohip_meaningful_improvement": 0.496, "ohip_meaningful_deterioration": 0.226, "gohai_meaningful_improvement": 0.637, "gohai_meaningful_deterioration": 0.174, "mai_meaningful_improvement": 0.405, "mai_meaningful_deterioration": 0.274}`
- Globális FDR után szignifikáns tesztek száma: **0 / 132**
- A null-szimulációban kapott FDR-pozitív jel(ek): **nem volt**
- Becsült korrigált OR-tartomány: **0.58–1.30**
- Hiányzó cellák száma: **0**

## Értelmezés

Az egyes OR-ok 1 körüli véletlen eltérései Monte Carlo-ingadozások. Mivel az
adatgenerálásban az anatómia bizonyítottan nem hatott a kimenetekre, az esetleg
szignifikánsnak látszó eredmény is ismert álpozitív. Ezekből nem lehet
kijelenteni, hogy bármely anatómiai képlet növeli vagy csökkenti a valódi
betegek számottevő OHIP-, GOHAI- vagy MAI-változásának esélyét. Ehhez valódi,
prospektív utánkövetési adatokon előre rögzített modell szükséges.
