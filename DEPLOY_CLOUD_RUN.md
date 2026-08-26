# Deploy to Google Cloud Run

The Dockerfile listens on `$PORT` (Cloud Run sets 8080) and bakes the retrieval
models into the image, so cold starts do not download ~1.2 GB.

## One time

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Store the OpenRouter key in Secret Manager (better than a plain env var):

```bash
printf '%s' 'sk-or-v1-YOUR_KEY' | gcloud secrets create openrouter-api-key --data-file=-
```

## Deploy

Run from the repo root:

```bash
gcloud run deploy raglens \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars ENABLE_SCHEDULER=0,OPENROUTER_APP_NAME=RAGLens \
  --set-secrets OPENROUTER_API_KEY=openrouter-api-key:latest
```

First run asks to create an Artifact Registry repo — say yes. Build takes a few
minutes because the image includes the models (~4 GB).

The command prints a `https://raglens-xxxxx-uc.a.run.app` URL when done.

## Notes

- `--min-instances 0` scales to zero, so it is free when idle. The first request
  after a cold start waits ~15 s for the models to load into memory.
- `ENABLE_SCHEDULER=0` turns off the background evaluation job. With scale-to-zero
  it would re-run on every cold start against a fresh empty DB and waste tokens.
  The "Performance over time" charts stay empty on Cloud Run.
- To keep the background job and the charts, run one always-on instance instead:
  add `--min-instances 1 --no-cpu-throttling` and drop `ENABLE_SCHEDULER=0`. That
  is billed continuously (not free).
- The SQLite monitoring DB is per-instance and not persisted. Fine for a demo.

## Redeploy

```bash
gcloud run deploy raglens --source . --region us-central1
```

Flags set on the first deploy are remembered.
