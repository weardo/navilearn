# Deploying NaviLearn to Hetzner (navi.soupup.ai)

This is the production runbook for NaviLearn on a Hetzner box. The app runs as a
single Docker container that listens only on `127.0.0.1:8600`. A Cloudflare
tunnel is the only public entry point, so the container is never exposed to the
open internet. Secrets live on the host, never in the image or in git. The site
is served at `https://navi.soupup.ai`.

Architecture at a glance:

```text
browser -> Cloudflare edge (TLS) -> cloudflared tunnel -> 127.0.0.1:8600 -> navi container
```

## 1. Prerequisites on the Hetzner box

- Docker Engine and the Docker Compose plugin (`docker compose version`).
- `cloudflared` installed and authenticated to the Cloudflare account that owns
  the `soupup.ai` zone (`cloudflared tunnel login`).
- A non-root deploy user (referred to below as the deploy user) that is a member
  of the `docker` group.
- Outbound network access so `fastembed` can download the `bge-small` model on
  first use. The ONNX summarizer model ships inside the image at
  `./models/t5-small-onnx-int8/`, so it needs no download.

Verify the basics:

```bash
docker compose version
cloudflared --version
id   # confirm the deploy user is in the docker group
```

## 2. Secrets (host only, never committed)

Secrets stay on the host in `/opt/navi/.env`. This file is git-ignored and must
never be committed.

```bash
sudo mkdir -p /opt/navi
sudo chown "$USER":"$USER" /opt/navi
cp /opt/navi/.env.example /opt/navi/.env   # after the repo is in place (step 3)
chmod 600 /opt/navi/.env
```

Fill in the real values. The keys NaviLearn reads:

```ini
# LLM (Groq primary, cloud fallback chain)
LLM_PROVIDER=groq
LLM_MODEL=groq/llama-3.1-8b-instant
GROQ_API_KEY=<real-groq-key>
GROQ_MODEL=llama-3.1-8b-instant
LLM_FALLBACK_MODELS=<comma-separated fallback models>

# Speech / translation
SARVAM_API_KEY=<real-sarvam-key>

# Embeddings and vector store
EMBEDDING_MODEL=bge-small
VECTOR_BACKEND=supabase

# Supabase (Postgres backend + auth)
DB_BACKEND=supabase
SUPABASE_URL=<real-supabase-url>
SUPABASE_ANON_KEY=<real-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<real-service-role-key>
SUPABASE_DB_PASSWORD=<real-db-password>

# Security: the in-app code runner MUST stay off in production
NAVI_ENABLE_CODE_RUN=false

# pyarrow segfault guard (see runtime note below)
ARROW_DEFAULT_MEMORY_POOL=system
```

Runtime note: `ARROW_DEFAULT_MEMORY_POOL=system` is required. Without it this
build's `pyarrow` can segfault when Arrow serialization runs off the import
thread. Keep it set in the container environment.

Confirm the file is locked down:

```bash
ls -l /opt/navi/.env   # expect: -rw------- and the deploy user as owner
```

## 3. Deploy the container

Put the repo on the box at `/opt/navi`, either by cloning or by rsync from your
workstation.

Clone:

```bash
git clone <navilearn-repo-url> /opt/navi
cd /opt/navi
```

Or rsync from your workstation (excludes local state and secrets):

```bash
rsync -av --delete \
  --exclude '.venv' --exclude '.git' --exclude '.env' \
  --exclude '.chroma' --exclude 'outputs' \
  /mnt/data/astra/projects/jobprep/navilearn/ deploy@<hetzner-host>:/opt/navi/
```

With `/opt/navi/.env` in place (step 2), build and start:

```bash
cd /opt/navi
docker compose up -d --build
```

The container binds to `127.0.0.1:8600` on the host, so it is reachable only
from the box itself. Follow the logs until Streamlit reports it is serving:

```bash
docker compose logs -f
```

Health check the container directly on the loopback address:

```bash
curl -sf http://127.0.0.1:8600/_stcore/health && echo OK
```

`_stcore/health` returns `ok` when Streamlit is up. If `curl` fails, inspect the
logs before touching Cloudflare.

## 4. Cloudflare tunnel (dedicated)

Use a DEDICATED tunnel for NaviLearn. Do not add `navi.soupup.ai` as another
ingress rule on the existing shared `cptsd.in` tunnel: that tunnel already has
multiple connectors running, and mixing hostnames across connectors invites
split-routing where some requests land on a connector that has no route for
`navi`. A separate tunnel keeps NaviLearn's routing isolated and predictable.

Create the tunnel (writes credentials JSON under `~/.cloudflared/`):

```bash
cloudflared tunnel create navi
```

Write `~/.cloudflared/config.yml`. Replace `<TUNNEL-UUID>` with the id printed by
the create command:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /home/deploy/.cloudflared/<TUNNEL-UUID>.json

ingress:
  - hostname: navi.soupup.ai
    service: http://127.0.0.1:8600
  - service: http_status:404
```

The trailing `http_status:404` is the required catch-all fallback. Streamlit
relies on WebSocket upgrades for its live reruns, and `cloudflared` proxies
WebSockets transparently, so no extra flags are needed.

Create the proxied DNS record in the `soupup.ai` zone (an orange-cloud CNAME
pointing at the tunnel):

```bash
cloudflared tunnel route dns navi navi.soupup.ai
```

Run `cloudflared` as a managed service so it survives reboots:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

If you prefer an explicit unit instead of `service install`, a minimal
`/etc/systemd/system/cloudflared-navi.service` is:

```ini
[Unit]
Description=cloudflared tunnel for navi.soupup.ai
After=network.target

[Service]
User=deploy
ExecStart=/usr/bin/cloudflared tunnel run navi
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-navi
```

## 5. Verify end to end

1. Open `https://navi.soupup.ai` in a browser. TLS is terminated at the
   Cloudflare edge, so the padlock should be green with no warning.
2. Sign in with a demo account.
3. Confirm the dashboard loads and renders data (progress, recommendations, and
   the study widgets). A working dashboard means the container, the Supabase
   backend, and the tunnel are all healthy.

If the page loads but data is missing, check `docker compose logs -f` for
Supabase auth or connection errors, which almost always point at a wrong value
in `/opt/navi/.env`.

## 6. Operations

Update to a new version:

```bash
cd /opt/navi
git pull                       # or re-run the rsync from step 3
docker compose up -d --build   # rebuilds and recreates the container
```

Tail logs:

```bash
cd /opt/navi
docker compose logs -f
```

Restart without rebuilding:

```bash
cd /opt/navi
docker compose restart
```

Roll back to a known-good revision:

```bash
cd /opt/navi
git log --oneline -n 10        # find the last good commit
git checkout <good-commit-sha>
docker compose up -d --build
```

After rollback, re-run the health check from step 3 and the browser check from
step 5. To roll forward again, `git checkout` the branch tip and rebuild.

Restart the tunnel independently of the app if routing looks wrong:

```bash
sudo systemctl restart cloudflared        # or cloudflared-navi
```

## 7. Security posture

- The in-app code runner is disabled: `NAVI_ENABLE_CODE_RUN=false` in
  `/opt/navi/.env`. Leave it off in production. Do not enable arbitrary code
  execution on a public host.
- Auth is demo-grade and row-level security is intentionally relaxed for the
  hackathon build. This is documented, with the privacy tradeoffs and an
  optional hardening path, in [../docs/CH5_PRIVACY.md](./CH5_PRIVACY.md).
- The REST API (`api.py`) is OPTIONAL and is NOT exposed by this compose. Only
  the Streamlit container is built and only port `127.0.0.1:8600` is served
  through the tunnel. If you later publish the API, treat its surface and auth
  per [../docs/API_DOCS.md](./API_DOCS.md) and give it its own reviewed ingress
  rule; do not bolt it onto the NaviLearn hostname unreviewed.
- Secrets never enter the image or git. They live only in `/opt/navi/.env`
  (`chmod 600`, owned by the deploy user). Rotate the Groq, Sarvam, and Supabase
  keys if the host is ever compromised.
