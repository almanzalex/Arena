# Separate OpenEnv 0.4.x pilot service for Arena R-05 qualification.
#
# The Arena client must NOT embed this process. Start the service, then connect
# with packaging.base_url / ARENA_OPENENV_BASE_URL.
#
# From repo root:
#   docker compose -f docker/openenv/docker-compose.yml up --build -d
#   export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000

FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

# Install Arena from the source tree with the openenv extra so the frozen pilot
# server (RPS + vector) is available as a separate process.
COPY pyproject.toml README.md LICENSE ./
COPY arena ./arena

RUN python -m pip install --no-cache-dir '.[openenv]' \
 && python -c "import arena, openenv; print(arena.__version__, openenv.__version__)"

EXPOSE 8000

ENV OPENENV_HOST=0.0.0.0
ENV OPENENV_PORT=8000
ENV OPENENV_ENV=rps

CMD ["sh", "-c", "python -m arena.adapters.task_openenv.server --host \"${OPENENV_HOST}\" --port \"${OPENENV_PORT}\" --env \"${OPENENV_ENV}\""]
