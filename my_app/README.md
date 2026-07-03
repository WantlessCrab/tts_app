# TTS App integration into larger local_llm project

## Authority boundary

```text
tts_app:
  owns acquire, convert, canonical PDF, extraction, semantic artifacts,
  UI sentence artifacts, audio chunks, audio timing, audiobook manifests,
  artifact catalogs, and document/audio job state.

local_llm:
  owns model turns, RAG execution, TurnPackets, eval/training/tuning evidence,
  projections, and packet groups.

data_stack:
  owns PostgreSQL schema authority. No TTS schema is introduced here.

local_llm_router:
  owns manual browser/session/provider routing. TTS acquire does not own browser routing.
```

## Implemented source spine

```text
TraceContext / trace_id:
  lineage across gateway, acquire, convert, processor, artifacts, and jobs.

DocumentAsset / document_id:
  content-hash-based canonical document identity.

AudiobookJob / job_id:
  file-backed job state and append-only event ledger.

ArtifactRef / artifact_id:
  file-backed artifact identity with role, schema version, hash, size, path, and metadata.

Audiobook / audiobook_id:
  generated audio artifact family for one document/profile.
```

## Implemented API surfaces

```text
GET  /health
POST /api/v1/ingest/url
POST /api/v1/ingest/file
POST /api/v1/ingest
GET  /api/v1/jobs
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/events
GET  /api/v1/traces/{trace_id}/artifacts
GET  /api/v1/traces/{trace_id}/jobs
GET  /api/v1/documents/{document_id}
GET  /api/v1/documents/{document_id}/pdf
GET  /api/v1/artifacts/{artifact_id}
GET  /api/v1/audiobooks
GET  /api/v1/audiobooks/{book_id}/status
GET  /api/v1/audiobooks/{book_id}/manifest
GET  /api/v1/audiobooks/{book_id}/artifacts
GET  /api/v1/audiobooks/{book_id}/ui-sentences
GET  /api/v1/audiobooks/{book_id}/ui-sentences/pages/index
GET  /api/v1/audiobooks/{book_id}/ui-sentences/pages/{page_number}
GET  /api/v1/audiobooks/{book_id}/semantic
GET  /api/v1/audiobooks/{book_id}/semantic/pages/index
GET  /api/v1/audiobooks/{book_id}/semantic/pages/{page_number}
GET  /api/v1/audiobooks/{book_id}/audio-timing
GET  /api/v1/audiobooks/{book_id}/stage3/checkpoint
GET  /api/v1/audiobooks/{book_id}/stage3/chunks/{chunk_id}
```

Legacy compatibility routes are retained where the current frontend or existing operator habits may still call them.

## Implemented artifact roles

```text
canonical_pdf
raw_extraction
citation_ready
semantic
semantic_shard_index
ui_sentences
ui_sentences_shard_index
audio_chunk
audio_timing
stage3_checkpoint
audiobook_manifest
```

## Large-document behavior now represented in source

```text
page-sharded semantic artifacts
page-sharded UI sentence artifacts
bounded Stage 3 audio worker queue
Stage 3 checkpoint JSON
per-chunk Stage 3 checkpoint JSON
audio timing artifact generated after Stage 3
orphaned audio registered as degraded/orphan artifacts rather than fake ready chunks
```

## Remaining source-side limitations

```text
audio_timing.v1 is measured chunk duration plus proportional sentence windows;
it is not word-level forced alignment.

ui_sentences.v2 carries backend coverage where provable;
it does not guarantee perfect character coverage for every irregular PDF geometry case.

BackgroundTasks remain transitional execution machinery;
job_store is durable state, not a full worker lease/queue system.
```