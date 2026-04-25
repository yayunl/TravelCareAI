# Travel Care AI

Educational **decision-support** demo for **travelers**: rule-based triage hints, a small **local RAG** corpus (`data/corpus.jsonl`), optional **OpenAI** wording, and **Google Maps JavaScript + Places** for nearby hospitals when the traveler searches a place.

This is **not** a medical device and **not** a diagnosis.

## Quick start

```bash
cd travelcareAI
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GOOGLE_MAPS_API_KEY (and optionally OPENAI_API_KEY, DEFAULT_TRAVEL_LOCATION)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

**Download report** saves a **PDF** (generated in the browser via [jsPDF](https://github.com/parallax/jsPDF) from the CDN; allow network access to `cdnjs.cloudflare.com` if you use a strict blocker).

## Docker

The repo includes a `Dockerfile` and `.dockerignore`. The image runs **Uvicorn** on `0.0.0.0` and honors the **`PORT`** environment variable (Cloud Run sets this automatically; locally it defaults to **8080**). **`LOG_DISABLE_FILE=1`** is set in the image so logs go to **stdout** (better for platforms with ephemeral disks).

Build and run locally (pass the same secrets you would put in `.env`; do not bake secrets into the image):

```bash
docker build -t travelcare-ai .
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="sk-..." \
  -e GOOGLE_MAPS_API_KEY="..." \
  -e GOOGLE_MAPS_SERVER_KEY="..." \
  travelcare-ai
```

Then open `http://localhost:8080/`. See [.env.example](.env.example) for optional variables (`SERPER_API_KEY`, `OPENAI_JUDGE_MODEL`, `APP_ACCESS_CODE`, etc.).

### Google Cloud Run (short checklist)

1. Build for linux/amd64 (Cloud Run’s default) and push to Artifact Registry, e.g.  
   `docker build --platform linux/amd64 -t REGION-docker.pkg.dev/PROJECT/REPO/travelcare-ai:TAG .`  
   `docker push REGION-docker.pkg.dev/PROJECT/REPO/travelcare-ai:TAG`
2. Deploy: `gcloud run deploy SERVICE_NAME --image REGION-docker.pkg.dev/PROJECT/REPO/travelcare-ai:TAG --region REGION`
3. In the Cloud Run service, add **environment variables or Secret Manager references** for `OPENAI_API_KEY`, `GOOGLE_MAPS_API_KEY`, `GOOGLE_MAPS_SERVER_KEY`, and any other keys from `.env.example` you need.
4. After you have a stable URL, restrict the **browser** Maps key by **HTTP referrer** to that origin; configure the **server** Maps key for server-side Geocoding/Places (IP or API restrictions as appropriate for Cloud Run egress).

## Environment

| Variable | Purpose |
|----------|---------|
| `GOOGLE_MAPS_API_KEY` | Browser key with **Maps JavaScript API**, **Places API**, and **Geocoding API** enabled (Geocoding turns typed addresses into map coordinates). Restrict by HTTP referrer to your dev URL in Google Cloud Console. |
| `OPENAI_API_KEY` | Optional. If unset, the server returns rule + RAG citations only (no `llm` block). |
| `OPENAI_MODEL` | Optional, default `gpt-4o-mini`. |
| `DEFAULT_TRAVEL_LOCATION` | Optional default for trip context (e.g. `Paris, France`) when the client does not send `location` on `/api/assist`. |

## API

- `GET /api/public-config` — returns whether Maps / OpenAI are configured, optional `defaultTravelLocation`, and the Maps key for the browser when configured.
- `POST /api/assist` — JSON `{ "message": "...", "language": "en", "location": "City, Country" }` → care level, citations, optional LLM JSON. Field `location` is optional; omit or leave empty for location-agnostic behavior unless `DEFAULT_TRAVEL_LOCATION` is set.

## Architecture (request flow)

High-level flow for a single `POST /api/assist` call (see `app/main.py`, `app/triage.py`, `app/rag.py`, `app/research_agent.py`, `app/llm.py`, `app/severity_resolution.py`).

```mermaid
flowchart TB
  subgraph client["Client browser"]
    UI[Session and trip context]
    UI -->|POST api assist| API
  end

  subgraph server["FastAPI assist"]
    API[Receive JSON body]

    API --> COMBINE[Combine message and chat for signals]
    COMBINE --> TRIAGE[rule_triage keyword and regex]
    TRIAGE --> RULES[Rules care_level emergency rationale]

    COMBINE --> RAGQ[Build RAG query]
    RAGQ --> RAGBRANCH{merged retrieval mode}
    RAGBRANCH -->|single_turn_tools| PLAN[LLM plans corpus queries]
    PLAN --> RAGM[rag retrieve merged]
    RAGBRANCH -->|default| RAG1[rag retrieve]
    RAGM --> CITES[Citations from chunks]
    RAG1 --> CITES

    API --> GEO{Need geocode}
    GEO -->|yes| GEOCODE[Geocode trip text]
    GEO -->|no| MAPOK[Use client coordinates]
    GEOCODE --> MAPOK

    MAPOK --> LOCAL{Local time lookup}
    LOCAL -->|yes| DESTCTX[Destination civil time]
    LOCAL -->|no| SKIPCTX[Skip local context]
    DESTCTX --> RESQ
    SKIPCTX --> RESQ

    subgraph research["Optional research loop"]
      RESQ{Research enabled}
      RESQ -->|yes| RLOOP[OpenAI tool loop Places web]
      RLOOP --> RDIG[Digest and structured JSON]
      RESQ -->|no| NORES[No research fields]
    end

    RDIG --> BASE
    NORES --> BASE

    subgraph basebuild["Assemble response"]
      BASE[Base dict citations rules triage snapshot]
    end

    RULES --> BASE

    subgraph llm_main["Optional traveler LLM"]
      KEY{OpenAI key set}
      BASE --> KEY
      KEY -->|no| SKIP[No llm block]
      KEY -->|yes| AUG[augment_with_openai]
      AUG --> PARSE[Normalize traveler JSON]
      PARSE --> LLMOUT[llm object optional severity_assessment]
    end

    subgraph severity["Severity merge"]
      MERGE[merge_effective_severity]
      LLMOUT --> MERGE
      MERGE --> EFF[Effective care_level and emergency]
      MERGE --> SRC{severity_source}
      SRC -->|rules| SRULES[From keyword triage]
      SRC -->|llm_adjusted| SLLM[From model payload]
      MERGE --> REJFL[May set override_rejected]
    end

    SKIP --> RESP
    EFF --> RESP
    REJFL --> RESP

    RESP[JSON response]
  end

  RESP --> UI2[Report UI and PDF download]
```

### Severity merge (detail)

```mermaid
flowchart LR
  T[rule_triage] --> R[Rules snapshot]
  L[LLM JSON] --> SA{Valid disagreement}
  SA -->|no| E1[Effective equals rules]
  SA -->|yes| G{Server emergency flag}
  G -->|blocked downgrade| E2[Keep rules reject override]
  G -->|allowed| E3[Use suggested severity]
  R --> MERGE[merge_effective_severity]
  SA --> MERGE
  MERGE --> OUT[Response fields]
```

## Emergencies while traveling

Use the **official emergency and police numbers for your destination** (they differ by country). This app does not replace local emergency services.

## Next steps (paper / product)

- Replace overlap RAG with embeddings + eval scenario bank.  
- Add facility-type routing (clinic vs ER) from triage output.  
- Curate destination-specific official health URLs into the corpus with snapshots.
