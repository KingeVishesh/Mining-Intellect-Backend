# Mining Intellect Backend v2 — API Documentation

## Overview

This backend runs 4 LangGraph graphs on LangGraph Cloud.
The frontend interacts via **Supabase Edge Functions**, which wrap the LangGraph Cloud API.

---

## LangGraph Cloud — Raw API

### 1. Create a thread (required before each workflow run)

```
POST https://[deployment].us.langgraph.app/threads
Headers: { "x-api-key": "<LANGGRAPH_API_KEY>" }
Body: {}

Response: { "thread_id": "abc123..." }
```

### 2. Start a workflow run

```
POST https://[deployment].us.langgraph.app/threads/{thread_id}/runs
Headers: { "x-api-key": "<LANGGRAPH_API_KEY>" }
Body: {
  "assistant_id": "project_research",   // project_research | analog_finder | report_generator | project_discovery
  "input": {
    // project_research:
    "project_id": "uuid",
    "project_name": "Athabasca Basin Project",
    "material": "uranium",
    "company": "NexGen Energy"

    // analog_finder + report_generator:
    // "project_id": "uuid"

    // project_discovery:
    // "materials": ["uranium", "copper"]
  }
}

Response: { "run_id": "run_abc...", "status": "pending" }
```

### 3. Poll run status

```
GET https://[deployment].us.langgraph.app/threads/{thread_id}/runs/{run_id}
Headers: { "x-api-key": "<LANGGRAPH_API_KEY>" }

Response: {
  "status": "pending" | "running" | "interrupted" | "success" | "error"
}
```

### 4. Get full state (when interrupted — fetch what needs human review)

```
GET https://[deployment].us.langgraph.app/threads/{thread_id}/state
Headers: { "x-api-key": "<LANGGRAPH_API_KEY>" }

Response: {
  "values": {
    // project_research interrupted:
    "extracted_fields": { "country": "Canada", "tonnage_mt": 12.5, ... },
    "field_statuses": { "country": "found", "npv_usd_millions": "search_miss", ... },
    "validation_errors": ["Missing lat/lng"],
    "exa_sources": ["https://..."],
    "message": "Please review extracted data for 'Athabasca Basin'..."

    // analog_finder interrupted:
    // "scored_analogs": [{ "name": "...", "similarity_score": 87, ... }]

    // report_generator interrupted:
    // "model_1": { "total_tonnage_kt": 1500, "total_grade_pct": 2.1, "conviction_pct": 72 }
    // "model_2": { ... }
  },
  "next": ["human_review"]
}
```

### 5. Resume after human review

```
POST https://[deployment].us.langgraph.app/threads/{thread_id}/state
Headers: { "x-api-key": "<LANGGRAPH_API_KEY>" }

// project_research resume:
Body: {
  "values": {
    "human_approved": true,
    "human_edits": {           // optional field corrections
      "latitude": 54.2,
      "longitude": -108.7
    }
  }
}

// analog_finder resume:
Body: {
  "values": {
    "human_approved": true,
    "approved_analogs": [      // filtered list of analogs to keep
      { "name": "Arrow Deposit", "similarity_score": 92, ... }
    ]
  }
}

// report_generator resume:
Body: {
  "values": {
    "human_approved": true,
    "human_model_edits": {     // optional model number overrides
      "model_1": { "conviction_pct": 65 }
    }
  }
}
```

---

## Supabase Edge Functions (Frontend Interface)

These functions simplify the LangGraph API for the frontend.

### `POST /functions/v1/research-project`

Starts a `project_research` graph run.

**Request:**
```json
{
  "project_id": "uuid",
  "project_name": "Athabasca Basin Project",
  "material": "uranium",
  "company": "NexGen Energy"
}
```

**Response:**
```json
{
  "thread_id": "thread_abc...",
  "run_id": "run_xyz..."
}
```

---

### `POST /functions/v1/find-analogs`

Starts an `analog_finder` graph run.

**Request:**
```json
{
  "project_id": "uuid"
}
```

**Response:**
```json
{
  "thread_id": "thread_abc...",
  "run_id": "run_xyz..."
}
```

---

### `POST /functions/v1/generate-report`

Starts a `report_generator` graph run.

**Request:**
```json
{
  "project_id": "uuid"
}
```

**Response:**
```json
{
  "thread_id": "thread_abc...",
  "run_id": "run_xyz..."
}
```

---

### `POST /functions/v1/check-workflow-status`

Poll a running workflow.

**Request:**
```json
{
  "thread_id": "thread_abc...",
  "run_id": "run_xyz..."
}
```

**Response:**
```json
{
  "status": "running" | "interrupted" | "success" | "error",
  "pending_review": {    // only present when status == "interrupted"
    "extracted_fields": { ... },
    "message": "Please review..."
  }
}
```

---

### `POST /functions/v1/submit-human-review`

Resume a workflow after human review.

**Request:**
```json
{
  "thread_id": "thread_abc...",
  "approved": true,
  "edits": {
    "human_approved": true,
    "human_edits": { "latitude": 54.2 }
  }
}
```

**Response:**
```json
{
  "status": "resumed"
}
```

---

## Human-in-the-Loop Flow (Frontend)

```
User clicks "Research Project"
  → POST /functions/v1/research-project
  → returns { thread_id, run_id }

Frontend polls every 5 seconds:
  → POST /functions/v1/check-workflow-status { thread_id, run_id }
  → status = "interrupted"
  → pending_review = { extracted_fields: {...}, message: "..." }

Frontend shows review modal:
  → User edits/approves extracted data

User clicks "Approve":
  → POST /functions/v1/submit-human-review { thread_id, approved: true, edits: {...} }

Frontend continues polling:
  → status = "success"
  → Frontend reloads project data from Supabase
```

---

## Report JSON Schema

The `content_json` field in the `reports` table always follows this structure:

```json
{
  "metadata": {
    "project_name": "...",
    "material": "uranium",
    "generated_at": "2026-04-23T10:00:00Z",
    "report_type": "full"
  },
  "executive_summary": {
    "summary_text": "...",
    "overall_assessment": "Positive | Cautious | Negative",
    "key_takeaway": "..."
  },
  "project_overview": {
    "project_summary": "...",
    "key_characteristics": ["..."],
    "official_mre_summary": "...",
    "drilling_data_summary": "..."
  },
  "resource_estimates": {
    "comparison_table": [
      {
        "model": "Model 1 (Independent)",
        "mi_tonnage_kt": 975.0,
        "mi_grade_pct": 2.1,
        "mi_contained_mlb": 45.1,
        "inferred_tonnage_kt": 418.0,
        "inferred_grade_pct": 1.99,
        "inferred_contained_mlb": 18.4,
        "total_tonnage_kt": 1393.0,
        "total_grade_pct": 2.1,
        "total_contained_mlb": 63.5,
        "description": "Independent estimate using 4 analog projects and 12 rules."
      },
      {
        "model": "Model 2 (Updated)",
        "..."
      },
      {
        "model": "Official MRE",
        "..."
      }
    ],
    "independent_analysis": {
      "confidence_pct": 72.5,
      "key_factors": ["Arrow Deposit", "Patterson Lake South"]
    },
    "updated_analysis": {
      "confidence_pct": 84.0,
      "key_factors": ["Reconciled with NI 43-101 estimate"]
    },
    "compliance_summary": [
      "Estimates are internal MI models — NOT NI 43-101 or JORC compliant.",
      "For investment decisions, rely only on officially filed technical reports."
    ]
  },
  "actionable_recommendations": [
    {
      "recommendation": "...",
      "priority": "High",
      "rationale": "..."
    }
  ],
  "lessons_summary": {
    "total_lessons_applied": 14,
    "high_confidence_lessons": 8
  },
  "key_uncertainties_and_strengths": {
    "strengths": ["..."],
    "uncertainties": ["..."]
  }
}
```

---

## Deployment

See the main plan document for LangGraph Cloud deployment steps.

**Required environment variables on LangGraph Cloud:**
- `GROK_API_KEY`
- `ANTHROPIC_API_KEY` (optional fallback)
- `EXA_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=mining-intellect-v2`
