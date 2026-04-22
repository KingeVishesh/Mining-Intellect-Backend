# Mining Intellect Backend v2

Modular LangGraph backend for [MiningIntellect.com](https://miningintellect.com).

## Architecture

4 independent LangGraph graphs, each deployable and callable separately:

| Graph | Purpose | Human Checkpoint |
|---|---|---|
| `project_research` | Exa search → extract all project fields → geocode → validate | Review extracted data |
| `analog_finder` | Find 5-10 comparable projects (DB + Exa) → score them | Approve/remove analogs |
| `report_generator` | Build resource models (analogs + rules) → LLM narrative | Review model numbers |
| `project_discovery` | Scheduled: discover new projects via Exa → save as drafts | None (background) |

## Setup

```bash
cp .env.example .env
# Fill in .env with your API keys

pip install -r requirements.txt
```

## Running Locally (LangGraph Studio)

```bash
pip install langgraph-cli
langgraph dev
```

Then open `http://localhost:8123` in LangGraph Studio.

## Testing

```bash
# Test project_research graph
python scripts/test_workflow.py \
  --graph project_research \
  --project-id "your-project-uuid" \
  --name "Arrow Deposit" \
  --material "uranium" \
  --company "NexGen Energy"

# Test analog_finder (project must already be in Supabase)
python scripts/test_workflow.py --graph analog_finder --project-id "your-project-uuid"

# Test report_generator (analogs must be saved first)
python scripts/test_workflow.py --graph report_generator --project-id "your-project-uuid"

# Test project_discovery
python scripts/test_workflow.py --graph project_discovery --materials uranium copper
```

## Compile Rules

```bash
# Compile lesson files from ../backend/storage/lesson_library into Supabase
python scripts/compile_rules.py --material uranium
python scripts/compile_rules.py --material all
```

## Deployment (LangGraph Cloud)

1. Push this repo to GitHub
2. Go to [smith.langchain.com](https://smith.langchain.com) → Deployments → New Deployment
3. Connect this GitHub repo
4. Set environment variables (see `.env.example`)
5. Deploy → get your deployment URL

## Report JSON Schema

See `docs/api.md` for the full `MiningReport` schema that the frontend reads.
Frontend field mapping:
- `resource_estimates.comparison_table` — the comparison table
- `resource_estimates.independent_analysis.confidence_pct` — Model 1 conviction
- `resource_estimates.updated_analysis.confidence_pct` — Model 2 conviction
