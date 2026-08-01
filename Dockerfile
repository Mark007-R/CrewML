# CrewML production image — Day 27.
#
# ONE image serves both the API (default CMD) and the Streamlit dashboard
# (docker-compose overrides the command): they share the code and ~1.5 GB of
# ML wheels, so separate images would double the build for zero isolation —
# the dashboard is already just an HTTP client of the API.
#
# Datasets and run artifacts are NOT baked in: data/ holds the sealed holdout
# splits and artifacts/ the run store, both mounted as volumes by compose so
# the honesty seals and run history survive image rebuilds.
FROM python:3.11-slim

# lightgbm/xgboost wheels link against OpenMP, which slim strips out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before code: a code edit must not re-pay the wheel download.
# Whole-install retry loop: Docker Desktop's build-container gateway drops the
# occasional index query under sustained load, and pip reports the blip as
# "(from versions: none)" and aborts the entire install — so retry the install
# itself, with the wheel cache kept alive inside the layer so each attempt
# resumes from what already downloaded (cache is purged once install succeeds).
COPY requirements.txt .
RUN ok=0; for i in 1 2 3 4 5; do \
      pip install --retries 10 --timeout 60 -r requirements.txt && ok=1 && break; \
      echo "pip attempt $i failed; retrying in 15s"; sleep 15; \
    done; [ "$ok" = "1" ] && rm -rf /root/.cache/pip

COPY crewml/ crewml/
COPY scripts/ scripts/
COPY results/ results/

# Unprivileged user: the Day-19 sandbox assumes the service itself isn't root,
# and nothing in the API needs privileges the volumes don't grant. The volume
# mountpoints must exist in the image owned by crew — a named volume copies
# its initial ownership from here; otherwise Docker roots them at start.
RUN useradd --create-home crew \
    && mkdir -p /app/artifacts /app/data \
    && chown -R crew:crew /app
USER crew

EXPOSE 8000

CMD ["uvicorn", "crewml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
