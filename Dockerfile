FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" \
    FILM_ANALYSIS_HOST=0.0.0.0 \
    FILM_ANALYSIS_PORT=8000
EXPOSE 8000
# The image already has uv as a tool. Run the installed command so startup
# does not sync again and exit before the port is open.
ENTRYPOINT []
CMD ["film-analysis"]
