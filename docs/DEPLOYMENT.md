# Deployment Guide

Deploy the JADT Topic Modeling website to `compare-topic-model-jadt2026.fr`.

The stack uses two containers:
- **web**: Dash/Gunicorn app (port 8050, internal only)
- **caddy**: Reverse proxy with automatic HTTPS (ports 80/443)

## Prerequisites

On the server (OVH VPS or any Debian/Ubuntu):

- Docker Engine >= 20.10
- Docker Compose v2

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# log out and back in
```

## 1. Copy files to the server

```bash
rsync -avz --progress website_output/ user@your-server:/opt/jadt-website/
```

## 2. DNS

Point both `compare-topic-model-jadt2026.fr` and `www.compare-topic-model-jadt2026.fr` to your server's IP (A record).

Caddy will automatically obtain Let's Encrypt certificates once DNS resolves.

## 3. Build and run

```bash
ssh user@your-server
cd /opt/jadt-website
docker compose up -d --build
```

The site will be live at `https://compare-topic-model-jadt2026.fr/`.

## 4. Verify

```bash
# Container status
docker compose ps

# Logs
docker compose logs -f --tail=50

# Health check
docker inspect --format='{{.State.Health.Status}}' $(docker compose ps -q web)
```

## 5. Firewall

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (redirects to HTTPS)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

Do **not** expose port 8050 publicly -- Caddy handles all external traffic.

## 6. Update workflow

```bash
# From your local machine
rsync -avz --progress --delete website_output/ user@your-server:/opt/jadt-website/

# On the server
cd /opt/jadt-website
docker compose up -d --build
```

## 7. Monitoring

```bash
# Live logs
docker compose logs -f

# Resource usage
docker stats --no-stream

# Caddy access logs
docker compose logs caddy
```

## Architecture

```
Internet
  |
  v
[Caddy :80/:443]  -- automatic HTTPS, security headers
  |
  v
[Dash/Gunicorn :8050]  -- read-only container, non-root user
```

### Key files

| File | Role |
|---|---|
| `Dockerfile` | Builds the Dash app image (non-root `appuser`, health check) |
| `docker-compose.yml` | Orchestrates web + Caddy, sets resource limits and security options |
| `Caddyfile` | Reverse proxy config with security headers for `compare-topic-model-jadt2026.fr` |

## Troubleshooting

| Issue | Fix |
|---|---|
| Caddy fails to get certificate | Check DNS is pointing to this server, ports 80/443 are open |
| Container exits immediately | `docker compose logs web` to see the error |
| Out of memory | Increase `memory` limit in `docker-compose.yml` or add swap |
| Permission denied | Ensure files are owned by the user running Docker |
