# CrewML — Hugging Face Space image (Day 30).
#
# One container, both processes: Spaces exposes exactly ONE port, so the
# docker-compose split (api + dashboard + redis) collapses here — start.sh
# boots the API privately on 127.0.0.1:8000 and the Streamlit dashboard on
# 0.0.0.0:7860 (the Space's app_port). No redis: crewml/cache.py fails open
# to its file cache whenever CREWML_REDIS_URL is unset, so the node cache
# still works, just per-container.
#
# Unlike the compose stack, data/ IS baked in: a Space has no host to bind-
# mount from, and the sealed benchmark splits are what the demo scores
# against. The byte-level seals (train/holdout_file_sha256) verify the copies
# match results/dataset_manifest.json exactly as they do on the host.
FROM python:3.11-slim

# lightgbm/xgboost wheels link against OpenMP, which slim strips out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before code: a code edit must not re-pay the wheel download.
# (No in-layer retry loop here — that guard was for Docker Desktop's flaky
# build-container gateway; HF's builders haven't shown the blip.)
COPY requirements.txt .
RUN pip install --retries 10 --timeout 60 -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY crewml/ crewml/
COPY scripts/ scripts/
COPY results/ results/
COPY data/ data/
COPY start.sh .

# UID 1000 is the Spaces convention for the runtime user. HOME must be real
# and writable: Streamlit writes ~/.streamlit at boot and the executor's
# sandbox children inherit it.
RUN useradd --create-home --uid 1000 crew \
    && mkdir -p /app/artifacts \
    && chown -R crew:crew /app
USER crew
ENV HOME=/home/crew

# Space runtime defaults, mirroring docker-compose where the reasoning
# carries over (Day 27): the free Space pod reports many vCPUs it won't
# grant, so bound joblib/loky and the OpenMP pools explicitly or the crew
# starves its own sandbox children into timeouts. The rlimit memory jail
# stays off (CREWML_EXECUTOR_MEM_MB=0) for the same reason as in-container
# compose runs: RLIMIT_AS trips on OpenMP thread stacks long before real
# memory pressure; the pod's own cgroup is the enforcement.
#
# CREWML_LLM_PROVIDER=groq + no baked key means the Space boots in MOCK mode
# (offline, clearly labelled by /healthz and the dashboard) until the
# GROQ_API_KEY Space secret is set — the key arrives as runtime env only and
# is never part of this image.
ENV CREWML_LLM_PROVIDER=groq \
    CREWML_API_URL=http://127.0.0.1:8000 \
    CREWML_EXECUTOR_MEM_MB=0 \
    LOKY_MAX_CPU_COUNT=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

EXPOSE 7860

CMD ["bash", "start.sh"]
