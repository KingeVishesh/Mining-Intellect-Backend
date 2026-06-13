-- Gold Resource Predictor v2 schema.
--
-- This is a fresh gold-only data surface. It intentionally does not depend on
-- legacy projects / analogs / model_runs tables.
--
-- Apply with Supabase migration tooling when ready:
--   supabase migration new gold_resource_predictor_v2
--   # paste this SQL into the generated migration
--   supabase db push

create extension if not exists pgcrypto;

create table if not exists public.gold_projects (
  id uuid primary key default gen_random_uuid(),
  external_key text unique,
  company_name text,
  project_name text not null,
  material text not null default 'gold' check (material = 'gold'),
  country text,
  region text,
  district text,
  latitude double precision,
  longitude double precision,
  deposit_family text,
  deposit_subtype text,
  tectonic_belt text,
  mineralization_mode text,
  mineralization_pattern text,
  host_rock_class text,
  mining_method_class text,
  project_stage_class text,
  recovery_method text,
  data_status text not null default 'candidate'
    check (data_status in ('candidate', 'truth_validated', 'evidence_ready', 'prediction_ready', 'excluded')),
  exclusion_reason text,
  source_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.gold_mre_truths (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.gold_projects(id) on delete cascade,
  truth_status text not null default 'validated'
    check (truth_status in ('validated', 'uncertain', 'rejected')),
  effective_date date,
  publication_date date not null,
  cutoff_date date generated always as (coalesce(effective_date, publication_date)) stored,
  source_url text not null,
  source_title text,
  source_publisher text,
  source_document_type text,
  resource_standard text,
  measured_tonnage_mt numeric,
  measured_grade_gpt numeric,
  indicated_tonnage_mt numeric,
  indicated_grade_gpt numeric,
  mi_tonnage_mt numeric,
  mi_grade_gpt numeric,
  inferred_tonnage_mt numeric,
  inferred_grade_gpt numeric,
  total_tonnage_mt numeric,
  total_grade_gpt numeric,
  mi_contained_oz numeric,
  inferred_contained_oz numeric,
  total_contained_oz numeric,
  validation_notes text,
  parallel_task_id text,
  raw_parallel_output jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint gold_mre_truths_positive_resource check (
    coalesce(mi_tonnage_mt, 0) >= 0
    and coalesce(inferred_tonnage_mt, 0) >= 0
    and coalesce(total_tonnage_mt, 0) >= 0
    and coalesce(mi_grade_gpt, 0) >= 0
    and coalesce(inferred_grade_gpt, 0) >= 0
    and coalesce(total_grade_gpt, 0) >= 0
  )
);

create unique index if not exists gold_mre_truths_validated_one_per_project_idx
  on public.gold_mre_truths(project_id)
  where truth_status = 'validated';

create table if not exists public.gold_pre_mre_evidence (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.gold_projects(id) on delete cascade,
  mre_truth_id uuid references public.gold_mre_truths(id) on delete set null,
  cutoff_date date not null,
  source_url text not null,
  source_title text,
  source_publisher text,
  source_date date,
  source_document_type text,
  evidence_status text not null default 'accepted'
    check (evidence_status in ('accepted', 'rejected')),
  rejection_reason text,
  fact_type text not null,
  value_num numeric,
  value_text text,
  unit text,
  confidence text not null default 'medium'
    check (confidence in ('high', 'medium', 'low')),
  is_mre_tainted boolean not null default false,
  fact_payload jsonb not null default '{}'::jsonb,
  parallel_task_id text,
  raw_parallel_output jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint gold_pre_mre_evidence_pre_cutoff check (
    source_date is null or source_date < cutoff_date
  ),
  constraint gold_pre_mre_evidence_rejection_reason check (
    evidence_status = 'accepted'
    or rejection_reason is not null
  )
);

create table if not exists public.gold_analog_candidates (
  id uuid primary key default gen_random_uuid(),
  target_project_id uuid not null references public.gold_projects(id) on delete cascade,
  candidate_project_name text not null,
  candidate_company_name text,
  candidate_country text,
  candidate_region text,
  candidate_district text,
  candidate_deposit_family text,
  candidate_deposit_subtype text,
  candidate_tectonic_belt text,
  candidate_mineralization_mode text,
  candidate_mineralization_pattern text,
  candidate_host_rock_class text,
  candidate_mining_method_class text,
  candidate_project_stage_class text,
  candidate_recovery_method text,
  source_url text not null,
  source_date date,
  source_title text,
  resource_standard text,
  total_tonnage_mt numeric,
  total_grade_gpt numeric,
  total_contained_oz numeric,
  mi_tonnage_mt numeric,
  mi_grade_gpt numeric,
  inferred_tonnage_mt numeric,
  inferred_grade_gpt numeric,
  drill_meters numeric,
  drill_holes integer,
  best_intercepts jsonb not null default '[]'::jsonb,
  geometry_payload jsonb not null default '{}'::jsonb,
  raw_parallel_output jsonb not null default '{}'::jsonb,
  parallel_task_id text,
  created_at timestamptz not null default now(),
  constraint gold_analog_candidates_positive_resource check (
    coalesce(total_tonnage_mt, 0) >= 0
    and coalesce(total_grade_gpt, 0) >= 0
    and coalesce(total_contained_oz, 0) >= 0
  )
);

create table if not exists public.gold_analog_decisions (
  id uuid primary key default gen_random_uuid(),
  target_project_id uuid not null references public.gold_projects(id) on delete cascade,
  analog_candidate_id uuid not null references public.gold_analog_candidates(id) on delete cascade,
  decision text not null check (decision in ('accepted', 'rejected')),
  decision_rules jsonb not null default '[]'::jsonb,
  rejection_reasons text[] not null default array[]::text[],
  accepted_at timestamptz,
  created_at timestamptz not null default now(),
  constraint gold_analog_decisions_reason_required check (
    decision = 'accepted'
    or cardinality(rejection_reasons) > 0
  )
);

create unique index if not exists gold_analog_decisions_candidate_once_idx
  on public.gold_analog_decisions(analog_candidate_id);

create table if not exists public.gold_backtest_batches (
  id uuid primary key default gen_random_uuid(),
  run_label text not null unique,
  batch_status text not null default 'created'
    check (batch_status in ('created', 'running', 'complete', 'blocked', 'failed')),
  requested_count integer not null default 0,
  evaluated_count integer not null default 0,
  pass_count integer not null default 0,
  no_prediction_count integer not null default 0,
  failure_count integer not null default 0,
  blocked_reason text,
  input_selector jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  completed_at timestamptz,
  constraint gold_backtest_batches_counts_nonnegative check (
    requested_count >= 0
    and evaluated_count >= 0
    and pass_count >= 0
    and no_prediction_count >= 0
    and failure_count >= 0
  )
);

create table if not exists public.gold_prediction_runs (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.gold_projects(id) on delete cascade,
  mre_truth_id uuid references public.gold_mre_truths(id) on delete set null,
  backtest_batch_id uuid references public.gold_backtest_batches(id) on delete set null,
  run_mode text not null default 'blind_no_mre'
    check (run_mode in ('blind_no_mre', 'cached_replay', 'production')),
  run_status text not null
    check (run_status in ('predicted', 'no_prediction', 'failed')),
  input_hash text not null,
  cutoff_date date not null,
  evidence_fact_ids uuid[] not null default array[]::uuid[],
  analog_candidate_ids uuid[] not null default array[]::uuid[],
  analog_decision_ids uuid[] not null default array[]::uuid[],
  no_prediction_reasons text[] not null default array[]::text[],
  predicted_total_tonnage_mt numeric,
  predicted_total_grade_gpt numeric,
  predicted_total_contained_oz numeric,
  predicted_mi_tonnage_mt numeric,
  predicted_mi_grade_gpt numeric,
  predicted_inferred_tonnage_mt numeric,
  predicted_inferred_grade_gpt numeric,
  predictor_version text not null,
  calculator_trace jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint gold_prediction_runs_no_prediction_reason check (
    run_status <> 'no_prediction'
    or cardinality(no_prediction_reasons) > 0
  ),
  constraint gold_prediction_runs_prediction_values check (
    run_status <> 'predicted'
    or (
      predicted_total_tonnage_mt is not null
      and predicted_total_grade_gpt is not null
      and predicted_total_contained_oz is not null
    )
  )
);

create unique index if not exists gold_prediction_runs_input_hash_idx
  on public.gold_prediction_runs(project_id, run_mode, input_hash);

create table if not exists public.gold_prediction_scores (
  id uuid primary key default gen_random_uuid(),
  prediction_run_id uuid not null references public.gold_prediction_runs(id) on delete cascade,
  mre_truth_id uuid not null references public.gold_mre_truths(id) on delete cascade,
  threshold_pct numeric not null default 5.0,
  core_pass boolean not null default false,
  split_pass boolean not null default false,
  production_like_pass boolean not null default false,
  tonnage_error_pct numeric,
  grade_error_pct numeric,
  contained_error_pct numeric,
  mi_tonnage_error_pct numeric,
  mi_grade_error_pct numeric,
  inferred_tonnage_error_pct numeric,
  inferred_grade_error_pct numeric,
  failure_class text,
  failure_reason text,
  score_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create unique index if not exists gold_prediction_scores_run_once_idx
  on public.gold_prediction_scores(prediction_run_id);

create table if not exists public.gold_parallel_cache (
  id uuid primary key default gen_random_uuid(),
  task_kind text not null
    check (task_kind in ('mre_truth', 'pre_mre_evidence', 'analog_research')),
  cache_key text not null unique,
  project_id uuid references public.gold_projects(id) on delete cascade,
  cutoff_date date,
  request_payload jsonb not null,
  response_payload jsonb not null,
  response_status text not null default 'complete'
    check (response_status in ('complete', 'blocked', 'failed')),
  provider_task_id text,
  provider_error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists gold_projects_lookup_idx
  on public.gold_projects(material, deposit_subtype, tectonic_belt, mining_method_class);

create index if not exists gold_mre_truths_project_cutoff_idx
  on public.gold_mre_truths(project_id, cutoff_date);

create index if not exists gold_pre_mre_evidence_project_fact_idx
  on public.gold_pre_mre_evidence(project_id, fact_type, evidence_status);

create index if not exists gold_analog_candidates_target_idx
  on public.gold_analog_candidates(target_project_id, candidate_deposit_subtype, candidate_tectonic_belt);

create index if not exists gold_analog_decisions_target_decision_idx
  on public.gold_analog_decisions(target_project_id, decision);

create index if not exists gold_prediction_runs_project_status_idx
  on public.gold_prediction_runs(project_id, run_status, created_at desc);

create index if not exists gold_parallel_cache_project_kind_idx
  on public.gold_parallel_cache(project_id, task_kind, cutoff_date);

alter table public.gold_projects enable row level security;
alter table public.gold_mre_truths enable row level security;
alter table public.gold_pre_mre_evidence enable row level security;
alter table public.gold_analog_candidates enable row level security;
alter table public.gold_analog_decisions enable row level security;
alter table public.gold_prediction_runs enable row level security;
alter table public.gold_prediction_scores enable row level security;
alter table public.gold_backtest_batches enable row level security;
alter table public.gold_parallel_cache enable row level security;

revoke all on table public.gold_projects from anon, authenticated;
revoke all on table public.gold_mre_truths from anon, authenticated;
revoke all on table public.gold_pre_mre_evidence from anon, authenticated;
revoke all on table public.gold_analog_candidates from anon, authenticated;
revoke all on table public.gold_analog_decisions from anon, authenticated;
revoke all on table public.gold_prediction_runs from anon, authenticated;
revoke all on table public.gold_prediction_scores from anon, authenticated;
revoke all on table public.gold_backtest_batches from anon, authenticated;
revoke all on table public.gold_parallel_cache from anon, authenticated;

grant select, insert, update, delete on table public.gold_projects to service_role;
grant select, insert, update, delete on table public.gold_mre_truths to service_role;
grant select, insert, update, delete on table public.gold_pre_mre_evidence to service_role;
grant select, insert, update, delete on table public.gold_analog_candidates to service_role;
grant select, insert, update, delete on table public.gold_analog_decisions to service_role;
grant select, insert, update, delete on table public.gold_prediction_runs to service_role;
grant select, insert, update, delete on table public.gold_prediction_scores to service_role;
grant select, insert, update, delete on table public.gold_backtest_batches to service_role;
grant select, insert, update, delete on table public.gold_parallel_cache to service_role;
