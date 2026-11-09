"""Football film analysis: chop plays, chart them, and build a scouting report.

Pipeline packages:

- `ingest` — download the film and pull frames
- `plays` — find play boundaries and keep ordered clips
- `chart` — tag run/pass, offense/defense, special teams, formations
- `teams` — choose which team to scout and its player photos
- `scout` — build the scouting report
- `web` — local website and API
- `storage` — D1/R2/memory catalog
- `cutter` / `tasks` — Cloudflare job services
"""

__version__ = "0.1.0"
