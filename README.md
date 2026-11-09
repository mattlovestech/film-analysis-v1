# film-analysis

Upload football film, or paste a public YouTube link. Claude marks the start and end of each play.

```bash
export ANTHROPIC_API_KEY=...
uv run film-analysis
```

Open http://127.0.0.1:8000, drop in a video or paste a YouTube link. Each play is saved as its own clip in the `film-analysis` R2 bucket. Previous analyses stay on the page, so you can open one again and see how it was split.

```bash
docker build -t film-analysis .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY film-analysis
```
