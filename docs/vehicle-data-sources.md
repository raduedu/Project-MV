# Vehicle data sources (Porsche library cars)

Public real-world data only. Confidence: **high** = manufacturer document, **medium** = reputable database quoting
manufacturer data or a single forum/database source, **estimate** = derived by the stated method.
Searches were run in Nov 2026; several Porsche PDFs have unreadable font encoding, so some "Porsche" numbers are
read through quoting sites (named in the URL column).

Common estimate methods: yaw inertia = m*(0.45*wheelbase)^2; rolling radius = 0.98 * (rim radius + sidewall);
brake torque = 1.4 g * m * g * r_avg; downforce ClA = F*9.81/(0.5*1.225*v^2); tyre mu 1.35-1.40 (Cup 2 R / Trofeo RS class)
at static corner load; driver 75 kg added to DIN mass; tyre load exponents 0.85 / 0.90.

## 1. Porsche Taycan Turbo GT with Weissach package (`porsche_taycan_turbo_gt_weissach.yaml`)

| Parameter | Value | Source URL | Confidence |
|---|---|---|---|
| mass_kg | 2295 (DIN 2,220 + 75) | Porsche TD sheet (2,290 kg DIN, non-Weissach): https://newsroom.porsche.com/dam/jcr:db7be217-7b2a-44a6-b2b3-3493b5baf1c1/pag-taycan-turbo-gt-td-en.pdf.PDF ; Weissach -70 kg: https://newsroom.porsche.com/christophorus/en/2024/413/taycan-turbo-gt.html ; EU 2,295: https://ev-database.org/car/2145/Porsche-Taycan-Turbo-GT-Weissach | high / medium (2,220 Weissach DIN from configurator via search) |
| wheelbase / length / width | 2.900 / 4.968 / 1.998 m | Porsche TD sheet (same URL) | high |
| track f/r | 1.690 / 1.655 m | Porsche TD sheet | high |
| front_weight_frac | 0.48 | none published | estimate |
| cg_height, roll split, yaw inertia | 0.45 m, 0.55, 3908 | method above | estimate |
| Cd x A | 0.31 x 2.33 = 0.722 m2 | Cd: Christophorus article; A=2.33: Porsche TD sheet (non-Weissach CdA 0.676 at Cd 0.29) | high / medium (Weissach Cd with non-W area) |
| downforce | 220 kg total, assumed at 300 km/h -> ClA 0.507; 40/60 | https://newsroom.porsche.com/christophorus/en/2024/413/taycan-turbo-gt.html (speed not stated) | medium (value) / estimate (speed, split) |
| tyres | 265/35 ZR21 front, 305/30 ZR21 rear | Porsche TD sheet | high |
| tyre mu / load exp / fz_ref | 1.35 / 0.85-0.90 / static | method above | estimate |
| brakes | PCCB 420x40 / 410x32; max torque 11,100 Nm | Porsche TD sheet (discs); torque estimate | high (discs) / estimate |
| steering ratio | 13.9:1 on-centre | Porsche TD sheet | high |
| max wheel angle, lock_deg | 34 deg, 470 deg | not published | estimate |
| battery usable | 97 kWh (105 gross) | Porsche TD sheet | high |
| max regen | 400 kW | https://newsroom.porsche.com/en_AU/2024/products/porsche-taycan-turbo-gt-product-highlights-37958.html | high |
| max discharge | 625 kW (= 580 kW/0.93) | power figures: https://newsroom.porsche.com/christophorus/en/2024/413/taycan-turbo-gt.html (580 sustained, 760 launch, 815 overboost 2 s, Attack Mode +120 kW 10 s) | estimate (derived from high power data) |
| system power / torque | 580 kW sustained (used), 760 kW launch, 1,240 Nm | Porsche TD sheet | high |
| per-axle motor split | rear 800 Nm / 355 kW, front 440 Nm / 225 kW | none published; sums match 1,240 Nm and 580 kW; rear ~12,000 Nm wheel torque in 1st (Porsche powertrain article) consistent | estimate |
| gear ratios | rear 15.0 / 7.1, front 7.1 | "~15:1, ~8:1, front ~8:1" https://newsroom.porsche.com/en/products/taycan/powertrain-18555.html ; Turbo GT rear ratio "longer" + front drive "overhauled" (Christophorus) | medium (std Taycan) / estimate (Turbo GT values) |
| rpm_max | 16,500 | 305 km/h x 7.1 | estimate |
| top speed | 305 km/h (Weissach, closed track) | Porsche newsroom | high |
| 0-100 / 0-200 | 2.2 s / 6.4 s (launch control) | Porsche newsroom | high |

Biggest uncertainties (Taycan):
1. Rear 2nd-gear and front ratios and the max motor rpm (only "~8:1" for the standard Taycan is public); sets power
   delivery above ~250 km/h and top speed.
2. Front/rear motor power and torque split (not published).
3. Downforce reference speed and front/rear balance; weight distribution and CG height.
4. Sustained (580 kW) vs launch (760-815 kW) power: the file models sustained only, so launch performance is understated.

## 2. Porsche 911 GT3 RS (992.1) (`porsche_911_gt3_rs_992.yaml`)

| Parameter | Value | Source URL | Confidence |
|---|---|---|---|
| mass_kg | 1525 (1,450 DIN + 75) | https://newsroom.porsche.com/en/2022/products/porsche-911-gt3-rs-world-premiere-29177.html | high |
| wheelbase / length / width | 2.457 / 4.572 / 1.900 m | https://www.auto-data.net/en/porsche-911-992-gt3-rs-4.0-525hp-pdk-46354 | medium |
| track f/r | 1.630 / 1.582 m | https://en.drivar.de/?p=34689 (quoting Porsche data) | medium |
| weight distribution | 38.8 / 61.2 % | https://en.drivar.de/?p=34689 | medium |
| cg_height, roll split, yaw inertia | 0.47 m, 0.56, 1864 | method above | estimate |
| Cd | 0.39 (wings at max) | auto-data.net (same) | medium |
| frontal area | 2.14 m2 | 0.85 x width x height | estimate |
| downforce | 409 kg @ 200 km/h, 860 kg @ 285 km/h -> ClA 2.16; 40/60 | Porsche newsroom (same as mass) | high (totals) / estimate (split) |
| tyres | 275/35 R20 front, 335/30 R21 rear | Porsche newsroom | high |
| tyre mu etc. | 1.40 lat | method above | estimate |
| brakes | 408 mm 6-piston front, 380 mm rear; max torque 7,400 Nm | Porsche newsroom (discs) | high / estimate |
| steering ratio | 14.1:1 | https://en.drivar.de/?p=34689 | medium |
| engine | 386 kW @ 8,500, 465 Nm @ 6,300, limit 9,000 | newsroom (power), auto-data.net (torque, limit) | high / medium |
| torque curve | 12 points through the published points | shaped estimate | estimate |
| gear ratios | 3.75 / 2.38 / 1.70 / 1.34 / 1.11 / 0.957 / 0.844 | 991.2 GT3 RS Porsche sheet https://newsroom.porsche.com/dam/jcr:e00c2789-42c7-4dab-a818-a732003194b0/2019_911_GT3_RS_Technical_Specifications.pdf.PDF ; Rennlist thread "992 GT3 RS gear and final drive ratios" (same gears as GT3) | medium-low |
| final drive | 4.27 | Rennlist thread above (forum only) | low (treat as estimate) |
| top speed / 0-100 | 296 km/h (7th, DRS) / 3.2 s | Porsche newsroom | high |

Biggest uncertainties (GT3 RS):
1. Gear ratios and final drive (992 RS "shorter overall ratio" is stated by Porsche, numbers are forum-level).
2. Aero: frontal area, front/rear split, and which wing/DRS state Cd 0.39 belongs to (sim top speed overshoots).
3. Torque curve shape below 8,500 rpm; tyre grip level.

## 3. Porsche 918 Spyder, Weissach package (`porsche_918_spyder.yaml`)

| Parameter | Value | Source URL | Confidence |
|---|---|---|---|
| mass_kg | 1709 (1,634 DIN Weissach + 75; base car 1,674) | https://www.ultimatespecs.com/car-specs/Porsche/105879/Porsche-918-Spyder-Weissach-Package.html ; https://www.auto-data.net/en/porsche-918-spyder-4.6-v8-887hp-hybrid-pdk-21380 | medium |
| wheelbase / length / width | 2.730 / 4.643 / 1.940 m | auto-data.net (same) | medium |
| track f/r | 1.664 / 1.612 m | auto-data.net, ultimatespecs | medium |
| weight distribution | 43 / 57 % | Porsche release via search (Autoblog/Winding Road) https://autoblog.com/2013/09/09/porsche-official-details-918-spyder-frankfurt | medium |
| cg, roll split, yaw inertia | 0.45 m, 0.52, 2588 | method above | estimate |
| Cd | 0.36 | ultimatespecs (Weissach) | medium |
| frontal area; downforce | 1.85 m2; ClA 0.85 split 40/60 | not published | estimate |
| tyres | 265/35 R20 front, 325/30 R21 rear | auto-data.net | medium |
| brakes | PCCB 410/390 mm; max torque 8,200 Nm | ultimatespecs; torque estimate | medium / estimate |
| steering ratio, lock | 14.0, 32 deg | not found | estimate |
| V8 | 447 kW (608 PS) @ 8,700, 540 Nm @ 6,700, limit 9,150 | auto-data.net; Porsche via stuttcars | medium-high |
| rear motor | 115 kW, 375 Nm, 9,150 rpm | Porsche (115 kW); auto-data.net (Nm) | medium |
| front motor | 95 kW, 210 Nm, 16,000 rpm | Porsche (95 kW); auto-data.net (Nm, rpm) | medium |
| PDK ratios / FD | 3.91 2.29 1.58 1.19 0.97 0.83 0.67 / 3.09 | spec text via search (ultimatespecs gives FD 3.09, reverse 3.55) | medium |
| front gearbox ratio | 7.74 | derived: 16,000 rpm at 265 km/h | estimate |
| battery | 6.8 kWh (nominal) | Porsche | high (nominal) / usable unknown |
| battery discharge / regen | 226 / 180 kW | not published; motor sum / efficiency | estimate |
| top speed / 0-100 | 345 km/h / 2.6 s | ultimatespecs, stuttcars (214 mph) | medium |

Biggest uncertainties (918):
1. Downforce and frontal area (no published figures); top speed in sim (381 km/h) overshoots 345 km/h, which is
   tyre-limited in reality.
2. Hybrid control: usable battery energy, peak discharge/regen, and how the rear motor torque is blended.
3. Front motor ratio (derived) and PDK ratios (secondary source); Weissach mass figures vary 1,634-1,640 kg.
