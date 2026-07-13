# KBR Variable Selection Pilot

Generated: 2026-06-25

This is a quick Python adaptation of the Volgograd / Andic-Ogunc pseudo-OOS idea.
Target is `mom - 100`; candidate variables are lagged only, so this pilot avoids direct future leakage.
It is a screening diagnostic, not a promoted model.

Gate used here: `RRMSE <= 0.95` and outperform ratio `>= 0.55` versus AR benchmark.

## Horizon h=1

| Candidate | RRMSE | RMAE | Outperform | Pass |
|---|---:|---:|---:|---|
| d_Ki | 0.960 | 0.977 | 0.52 | False |
| spread_Ruonia_Ki | 0.980 | 0.961 | 0.55 | False |
| fl_potrb_zad | 0.995 | 0.997 | 0.53 | False |
| Prod | 0.996 | 1.019 | 0.43 | False |
| Serv | 0.999 | 1.019 | 0.43 | False |
| fl_dep | 0.999 | 1.047 | 0.45 | False |
| usd_nom_i | 1.006 | 0.978 | 0.52 | False |
| Nonprod | 1.008 | 1.001 | 0.49 | False |

## Horizon h=2

| Candidate | RRMSE | RMAE | Outperform | Pass |
|---|---:|---:|---:|---|
| Ki | 0.965 | 1.013 | 0.57 | False |
| d_Ki | 0.993 | 1.014 | 0.39 | False |
| Prod | 0.999 | 1.007 | 0.47 | False |
| Serv | 0.999 | 1.005 | 0.49 | False |
| fl_potrb_zad | 1.005 | 1.012 | 0.43 | False |
| fl_dep | 1.006 | 1.006 | 0.52 | False |
| spread_Ruonia_Ki | 1.007 | 0.996 | 0.58 | False |
| usd_nom_i | 1.009 | 0.968 | 0.57 | False |

## Horizon h=12

| Candidate | RRMSE | RMAE | Outperform | Pass |
|---|---:|---:|---:|---|
| Prod | 0.983 | 0.973 | 0.52 | False |
| spread_Ruonia_Ki | 0.987 | 0.962 | 0.53 | False |
| Serv | 0.989 | 0.984 | 0.56 | False |
| Ruonia | 0.991 | 0.990 | 0.57 | False |
| all_real | 0.998 | 0.999 | 0.53 | False |
| usd_nom_i | 0.999 | 0.999 | 0.52 | False |
| fl_dep | 1.000 | 0.992 | 0.57 | False |
| Nonprod | 1.003 | 1.003 | 0.46 | False |
