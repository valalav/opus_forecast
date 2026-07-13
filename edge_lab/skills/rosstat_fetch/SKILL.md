---
name: rosstat-fetch
description: Use this skill when you need to download external data from Rosstat or CBR. It handles retries, headers, and ssl issues automatically.
---

# Rosstat Fetch Skill

**Goal**: Robustly download files from Russian government sites (Rosstat, CBR) that frequentLy fail due to geo-blocking or poor server performance.

## Capabilities
1. **Robust Download**: Uses `requests.Session` with proper User-Agent and retries.
2. **SSL bypass**: Handles legacy SSL certs often found on these sites.

## Instructions
1.  **Identify URL**: Get the direct link.
2.  **Execute**: Run `scripts/fetch.py --url <url> --output <path>`.

## Usage
```bash
python edge_lab/skills/rosstat_fetch/scripts/fetch.py --url "https://rosstat.gov.ru/storage/mediabank/cpi_2025.xlsx" --output "data/cpi.xlsx"
```
