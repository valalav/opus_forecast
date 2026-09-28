# Tariff calendar source audit

Audit date: 2026-09-28. Region code `7` is Kabardino-Balkarian Republic (KBR).

## Result

There is not yet a validated actual monthly tariff/CPI series suitable for
testing realized tariff effects. A bounded historical cap proxy can, however,
be derived from primary KBR schedules with explicit effective intervals. It is
saved separately in
[`data/tariffs/kbr_nalchik_monthly_payment_cap_proxy.json`](../data/tariffs/kbr_nalchik_monthly_payment_cap_proxy.json),
with raw published caps in
[`data/tariffs/kbr_nalchik_payment_caps.csv`](../data/tariffs/kbr_nalchik_payment_caps.csv)
and a reproducible builder in
[`data/tariffs/build_cap_proxy.py`](../data/tariffs/build_cap_proxy.py).

This proxy is the month-on-month change in the *published legal maximum
household communal-payment cap*, not a realized tariff or CPI change. It uses
the annual cumulative cap relative to the prior December payment level, resets
the cap baseline to zero each January as requested, and computes each monthly
proxy as `(1 + current_cap/100) / (1 + previous_cap/100) - 1`, in percent.
Months within an explicitly published constant cap interval become explicit
zeros for this cap-change proxy only; this does not claim actual prices were
flat. The January reset is likewise a proxy convention, not an assertion that
actual tariff growth across December/January is zero. The series carries no
weight and has no actual CPI interpretation.

Coverage is discontinuous: 2018-01 through 2020-12, then 2025-01 through
2026-12. Years 2021-2024 remain omitted because the primary schedules and
publication vintages needed to establish the monthly cap path were not
verified in this audit. The added 2020 schedule creates a continuous
36-month 2018-2020 block, but no continuous 24-month held-out sequence.
The proxy can support a limited cutoff-safe experiment if the caller respects
gaps and labels the regressor as a plan-cap proxy; it cannot establish actual
tariff-model improvement.

Primary sources:

- [KBR Head's official decree page](https://glava.kbr.ru/documents/ukazy/-ob-utverzhdenii-znacheniy-predelnykh-maksimalnykh-indeksov-izmeneniya-razmera-vnosimoy-grazhdanami-platy-za-kommunalnye-uslugi-v-munitsipalnykh-obrazovaniyakh-kabardino-balkarskoy-respubliki-na-2026-god.html) — decree title and text, No. 136-УГ.
- [Official KBR newspaper decree PDF](https://smikbr.ru/arhiv/2025/ap/12/uk136.pdf) — decree and appendix with dated cap columns.
- [Official KBR newspaper issue dated 20 December 2025](https://smikbr.ru/arhiv/2025/kbp/12/20d.pdf) — publication issue and tabulated values.
- [KBR Head's official document page for No. 136-УГ](https://glava.kbr.ru/documents/ukazy/-ob-utverzhdenii-znacheniy-predelnykh-maksimalnykh-indeksov-izmeneniya-razmera-vnosimoy-grazhdanami-platy-za-kommunalnye-uslugi-v-munitsipalnykh-obrazovaniyakh-kabardino-balkarskoy-respubliki-na-2026-god.html) — is a reference link; independent verification returned 403. The calendar conservatively uses the verified official Gazette issue date 2025-12-20, not the signing date.
- [Official Russian legal-publication portal subject listing](https://publication.pravo.gov.ru/search/subjects?index=1686&pageS=) — lists the decree; this search listing is not itself a dated monthly tariff series.

## Coverage and gaps

| Period | KBR evidence located | Can populate actual monthly tariff history? | Limitation |
| --- | --- | --- | --- |
| 2016–2017 | No primary monthly observed series or KBR cap schedule verified. | No | Missing actual rates and publication vintages. |
| 2018 | [Official KBR Gazette, 8 Dec 2017](https://smikbr.ru/arhiv/2017/kbp/12/08d.pdf), cap table page 1, lists Nalchik at 0% Jan-Jun and 5.3% Jul-Dec. | Cap proxy only | Legal payment ceilings, not realized tariff changes. Publication issue date is 2017-12-08. |
| 2019 | [Official KBR Gazette, 21 Dec 2018](https://smikbr.ru/arhiv/2018/kbp/12/21d.pdf), Decree 197-УГ, page 1, lists Nalchik at 1.7% Jan-Jun and 8.0% Jul-Dec. | Cap proxy only | Legal payment ceilings, not realized tariff changes. Publication issue date is 2018-12-21. Cap is relative to prior-December household payment. |
| 2020 | [Official KBR Gazette, 20 Dec 2019](https://smikbr.ru/arhiv/2019/kbp/12/20d.pdf), Decree 130-УГ, page 1, lists Nalchik at 0% Jan-Jun and 9.4% Jul-Dec. | Cap proxy only | Legal payment ceilings, not realized tariff changes. Publication issue date is 2019-12-20; PDF hash is recorded in `data/tariffs/source_manifest.json`. |
| 2021 | Decree 167-УГ for 2021 is identified by number/date (14 Dec 2020); no primary Gazette appendix with Nalchik numeric intervals and publication date verified. | No | Decree date alone does not establish publication date or monthly values. |
| 2022 | Decree 132-УГ for 2022 is identified (15 Dec 2021); secondary legal summaries report 0% Jan-Jun and 9.4% Jul-Dec, but primary appendix values/publication date were not verified. | No | Do not transcribe secondary-source values as primary verified data. |
| 2023 | Decree 125-УГ for Dec 2022 and 2023 is identified in a regional legal catalogue; no primary schedule/page or publication vintage recovered. | No | No verified Nalchik monthly numeric schedule. |
| 2024 | [Official KBR Gazette, 23 Dec 2023](https://smikbr.ru/arhiv/2023/kbp/12/23d.pdf), Decree 131-УГ, approves municipal maximum indices for 2024–2028 and says its detailed appendix is on glava.kbr.ru. | No | Schedule appendix values/effective intervals and publication vintage not recovered; do not infer them from 2025. |
| 2025 | [Official KBR Gazette, 21 Dec 2024](https://smikbr.ru/arhiv/2024/kbp/12/21d.pdf), Decree 146-УГ, page 1, lists Nalchik at 0% Jan-Jun and 15.8% Jul-Dec. | Cap proxy only | Legal payment ceilings, not realized tariff changes. Publication issue date is 2024-12-21. |
| 2026 | Decree 136-УГ, [official KBR page](https://glava.kbr.ru/documents/ukazy/-ob-utverzhdenii-znacheniy-predelnykh-maksimalnykh-indeksov-izmeneniya-razmera-vnosimoy-grazhdanami-platy-za-kommunalnye-uslugi-v-munitsipalnykh-obrazovaniyakh-kabardino-balkarskoy-respubliki-na-2026-god.html) and [Gazette issue](https://smikbr.ru/arhiv/2025/kbp/12/20d.pdf), page 3, list Nalchik at 1.7% from Jan 1 and 14.4% from Oct 1. | Cap proxy only | Legal payment ceilings, not realized tariff changes. Conservative verified publication date is 2025-12-20; Oct-Dec 2026 values are announced plan as of audit date. |

The Gazette decree for 2019 specifies the maximum-index calculation as a
comparison of a maximum household-payment amount in a month against the prior
December payment amount. The cap values are cumulative payment ceilings, not
month-on-month price changes. The derivation in the proxy builder is a
mathematical transformation of those published limits; it does not make the
limits into actual observed tariffs.

## Existing national forecast input

`data/external/med_forecast_2027_2029_20260924/tariff_calendar.csv` is a
transcription of the Russian Ministry of Economic Development's national
forecast PDF (the source README gives public release date 2026-09-24). It has
planned national anchors for 2026–2029, including a 9.9% total communal-payment
anchor for October 2026 and 11% for July 2027. The README explicitly says
these are not approved KBR tariffs. It is scenario/forecast data, not a KBR
regional observation or historical regional calendar. The local input hashes
at audit time are:

- `tariff_calendar.csv`: `70270436e0e303bd6742c13e4dff929de3a6784359eee4b50743bb8220904d75`
- `manifest.json`: `91557e4d2aa8a7bcafdd0571c40b54ec2e397b8686f7bc4b357c5c6b473a8007`

The OPR10.2026 calculations use an expert scenario weight of 5.958% for
communal services and use the national 9.9%/11% forecast anchors. This is
documented in `archive/results/opr102026_adjusted_20260928/REPORT.md` and its
calculation manifest; it is not a historical item-weight series. It should
not be attached as a time-varying historical weight to the caps above.

## Use in leakage-safe backtests

- Treat the JSON only as `monthly_change_of_nalchik_payment_cap_proxy_pct`, a
  discontinuous plan feature. It has 36 monthly records for 2018-2020 and 24
  for 2025-2026 (60 total), but no continuous 24-month held-out path. Do not use it as observed CPI/tariff inflation.
- The five known-at dates are the source-publication dates: 2017-12-08,
  2018-12-21, 2019-12-20, 2024-12-21 and 2025-12-20. Each annual schedule's future
  monthly entries are available only from its publication date onward.
- A future tariff plan may enter a forecast origin only if its public
  `known_at` date is on or before that origin. Keep each dated revision as a
  separate vintage; select the latest vintage available at the origin.
- The JSON now has 36 contiguous 2018-2020 monthly records plus 24 records for 2025-2026; 2021-2024 remain gaps. No zeros are implied for months omitted from the cap table. Lack of a
  scheduled cap entry is missing information, not a zero tariff change.
- When actual regulated-item CPI data become available, record item/measure,
  geography, unit, reference month, first-publication date, source URL and
  any later revisions separately. Do not silently substitute household cap
  decisions or national forecast anchors for that series.

No performance improvement from tariff features is established by this
source audit.
