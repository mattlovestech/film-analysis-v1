import { Container, ContainerProxy } from "@cloudflare/containers";

export { ContainerProxy };

interface Env {
  WEBSITE: DurableObjectNamespace<Website>;
  CUTTER: DurableObjectNamespace<Cutter>;
  TASK: DurableObjectNamespace<Task>;
  ANTHROPIC_API_KEY: string;
  CLOUDFLARE_ACCOUNT_ID: string;
  CLOUDFLARE_API_TOKEN: string;
  D1_DATABASE_ID: string;
  CLOUDFLARE_ACCESS_KEY_ID: string;
  CLOUDFLARE_SECRET_ACCESS_KEY: string;
  CLOUDFLARE_S3_ENDPOINT: string;
  R2_BUCKET: string;
  FILM_INTERNAL_TOKEN: string;
}

function containerEnv(env: Env): Record<string, string> {
  return {
    ANTHROPIC_API_KEY: env.ANTHROPIC_API_KEY,
    CLOUDFLARE_ACCOUNT_ID: env.CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN: env.CLOUDFLARE_API_TOKEN,
    D1_DATABASE_ID: env.D1_DATABASE_ID,
    CLOUDFLARE_ACCESS_KEY_ID: env.CLOUDFLARE_ACCESS_KEY_ID,
    CLOUDFLARE_SECRET_ACCESS_KEY: env.CLOUDFLARE_SECRET_ACCESS_KEY,
    CLOUDFLARE_S3_ENDPOINT: env.CLOUDFLARE_S3_ENDPOINT,
    R2_BUCKET: env.R2_BUCKET,
    FILM_INTERNAL_TOKEN: env.FILM_INTERNAL_TOKEN,
    FILM_ANALYSIS_HOST: "0.0.0.0",
    FILM_ANALYSIS_PORT: "8000",
    FILM_ANALYSIS_DATA: "/tmp/film-data",
  };
}

const PORT_READY_MS = 120_000;

class FilmContainer extends Container<Env> {
  defaultPort = 8000;
  enableInternet = true;
  pingEndpoint = "container/ping";

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.envVars = containerEnv(env);
  }

  override async fetch(request: Request): Promise<Response> {
    try {
      await this.startAndWaitForPorts({
        startOptions: {
          envVars: this.envVars,
          entrypoint: this.entrypoint,
          enableInternet: true,
        },
        ports: this.defaultPort,
        cancellationOptions: {
          instanceGetTimeoutMS: 60_000,
          portReadyTimeoutMS: PORT_READY_MS,
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not start the container.";
      return new Response(`Failed to start container: ${message}`, { status: 500 });
    }
    return super.fetch(request);
  }
}

export class Website extends FilmContainer {
  sleepAfter = "10m";
  entrypoint = ["/app/.venv/bin/film-analysis"];
}

export class Cutter extends FilmContainer {
  sleepAfter = "10m";
  entrypoint = ["/app/.venv/bin/film-cutter"];
}

Cutter.outboundByHost = {
  "film.internal": (request, env) => routeTask(request, env as Env),
};

export class Task extends FilmContainer {
  sleepAfter = "30s";
  entrypoint = ["/app/.venv/bin/film-task"];
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.startsWith("/internal/")) {
      if (!authorized(request, env.FILM_INTERNAL_TOKEN)) {
        return new Response("Unauthorized", { status: 401 });
      }
      const film = url.pathname.match(/^\/internal\/films\/(\d+)(\/.*)?$/);
      if (film) {
        return env.CUTTER.getByName(`film-${film[1]}`).fetch(rewrite(request, film[2] || "/"));
      }
      const task = url.pathname.match(/^\/internal\/tasks\/([A-Za-z0-9_-]+)(\/.*)?$/);
      if (task) {
        return env.TASK.getByName(task[1]).fetch(rewrite(request, task[2] || "/"));
      }
      return new Response("Not found", { status: 404 });
    }

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request) });
    }
    const headers = new Headers(request.headers);
    headers.delete("x-film-worker-url");
    headers.set("x-film-worker-url", url.origin);
    headers.set("x-film-delegate", "1");
    const website = await env.WEBSITE.getByName("web").fetch(new Request(request, { headers }));
    return allowLocalPage(request, await startFilm(env, website));
  },
};

function corsHeaders(request: Request): Headers {
  const headers = new Headers();
  const origin = request.headers.get("Origin") || "";
  if (/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(origin)) {
    headers.set("Access-Control-Allow-Origin", origin);
    headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    headers.set("Access-Control-Allow-Headers", "Content-Type");
    headers.set("Vary", "Origin");
  }
  return headers;
}

function allowLocalPage(request: Request, response: Response): Response {
  const extra = corsHeaders(request);
  if (!extra.has("Access-Control-Allow-Origin")) return response;
  const headers = new Headers(response.headers);
  extra.forEach((value, key) => headers.set(key, value));
  return new Response(response.body, { status: response.status, headers });
}

async function startFilm(env: Env, website: Response): Promise<Response> {
  const projectId = website.headers.get("x-film-start");
  const headers = new Headers(website.headers);
  headers.delete("x-film-start");
  headers.delete("x-film-youtube");
  headers.delete("x-film-workers");
  if (!projectId) {
    return new Response(website.body, { status: website.status, headers });
  }
  const youtube = website.headers.get("x-film-youtube") ?? "";
  const requested = Number(website.headers.get("x-film-workers"));
  const workers = Number.isInteger(requested) && requested >= 1 && requested <= 16 ? requested : 4;
  let cutter: Response;
  try {
    cutter = await env.CUTTER.getByName(`film-${projectId}`).fetch(
      new Request("http://container/start", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${env.FILM_INTERNAL_TOKEN}`,
        },
        body: JSON.stringify({
          project_id: Number(projectId),
          youtube: youtube || null,
          worker_url: "http://film.internal",
          workers,
        }),
      }),
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not start that film.";
    return Response.json({ detail: message }, { status: 502 });
  }
  if (!cutter.ok) {
    const detail = (await cutter.text()).slice(0, 300) || "Could not start that film.";
    return Response.json({ detail }, { status: 502 });
  }
  return new Response(website.body, { status: website.status, headers });
}

function routeTask(request: Request, env: Env): Response | Promise<Response> {
  const task = new URL(request.url).pathname.match(/^\/internal\/tasks\/([A-Za-z0-9_-]+)(\/.*)?$/);
  if (!task) {
    return new Response("Not found", { status: 404 });
  }
  if (!authorized(request, env.FILM_INTERNAL_TOKEN)) {
    return new Response("Unauthorized", { status: 401 });
  }
  return env.TASK.getByName(task[1]).fetch(rewrite(request, task[2] || "/"));
}

function authorized(request: Request, token: string): boolean {
  if (!token) {
    return false;
  }
  return request.headers.get("Authorization") === `Bearer ${token}`;
}

function rewrite(request: Request, path: string): Request {
  const url = new URL(request.url);
  url.pathname = path.startsWith("/") ? path : `/${path}`;
  return new Request(url, request);
}
