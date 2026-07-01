# Multi-GPU Offline Inference Framework - Architecture Specification

## Design Goal

Build a lightweight, maintainable offline multi-GPU inference engine with the following guarantees:

- Multiple input JSON files
- Multiple output JSON files
- One-to-one mapping between input/output files
- Output record count equals input record count
- Output record order equals input record order
- Dynamic GPU scheduling
- Resume support
- Clear module boundaries
- Future extensibility (vLLM / Ray / OpenAI API)

This project targets **single-machine multi-GPU offline inference**. Do **not** over-engineer for distributed execution.

---

## Core Principles

1. Single Responsibility
2. Stateless Worker
3. Dynamic Task Scheduling
4. Explicit Interfaces
5. Storage Independent
6. Backend Independent
7. Resume First
8. Testability First

---

## Module Layout

```text
src/uav_iqa/inference/

├── config.py
├── types.py
├── queue.py
├── storage.py
├── checkpoint.py
├── repository.py

├── executor.py
├── worker.py

├── scheduler.py
├── collector.py
├── writer.py

├── validator.py
├── metrics.py
├── engine.py
```

---

## Module Responsibilities

### Engine

Responsible only for lifecycle management.

Responsibilities:

- initialize modules
- start workers
- start scheduler
- start collector
- wait until completion
- graceful shutdown

Engine must NOT contain business logic.

### Scheduler

Responsible only for task scheduling.

Responsibilities:

- scan input files
- generate tasks
- enqueue pending tasks
- resume unfinished tasks

Scheduler does NOT perform inference.

Scheduler does NOT write output.

### TaskRepository

TaskRepository is the only component accessing checkpoint storage.

Responsibilities:

- create tasks
- query pending tasks
- mark running
- mark success
- mark failure
- resume tasks

Implementation: Phase 1 uses SQLite.

### TaskQueue

Abstract queue interface:

```python
put(task)
get()
close()
```

Phase 1 implementation: `multiprocessing.Queue`.

### Worker

Worker is completely stateless.

```
Receive Task
    ↓
Load data via Storage
    ↓
Executor.infer()
    ↓
Return Result
```

Worker must NOT:

- parse file paths
- write output files
- maintain file state
- merge chunks

One Worker owns exactly one GPU. Model loaded exactly once.

### Executor

```python
prepare()
infer(batch)
finalize()
```

Avoid NLP-specific names. Implementations: HuggingFace Transformers, vLLM, OpenAI API, future backends.

### Storage

```python
scan_files()
count_records()
load_chunk()
write_output()
```

Phase 1: `JsonStorage`. Future: JsonlStorage, ParquetStorage, ArrowStorage.

### Collector

Receive Result, track completed chunks, determine file completion, notify Writer.

Collector must NOT dump JSON, perform merge, or write files.

### Writer

Merge chunks, restore original order, atomic file write. Only Writer may create output JSON.

### Validator

Validate: file count, record count, chunk completeness, output order, record identity.

### Metrics

Runtime statistics: queue size, tasks completed, samples/sec, average latency, ETA.

---

## Task Design

```python
@dataclass(slots=True)
class Task:
    task_id: str
    file_path: Path
    chunk_id: int
    start: int
    end: int
```

Use continuous ranges, not `indices: list[int]`. YAGNI.

## Result Design

```python
@dataclass(slots=True)
class Result:
    task_id: str
    file_path: Path
    chunk_id: int
    outputs: list
```

Result lifetime is short. Collector releases memory after Writer finishes.

---

## Chunk vs Batch

- **Chunk**: scheduling unit (e.g., 256 samples)
- **Batch**: GPU forward unit (e.g., 16 samples)

```
Chunk(256) → 16 forward passes → Result
```

---

## Scheduling Strategy

Dynamic shared task queue. Workers repeatedly:

```
TaskQueue.get() → Infer → ResultQueue.put()
```

No static file-to-GPU assignment.

---

## Checkpoint

SQLite schema:

```text
tasks
--------------------------------
task_id
file_path
chunk_id
start
end
status
retry_count
updated_at
```

Status: PENDING, RUNNING, SUCCESS, FAILED.

Resume at Task granularity.

---

## Memory Strategy

Generator must NOT load all JSON files.

```
Scan files → Generate Task → Worker loads Chunk on demand
```

Collector releases memory immediately after file written.

---

## Output Requirements

```
input/          output/
a.json (100)    a.json (100)
b.json (5200)   b.json (5200)
```

Guarantees: identical file count, filenames, record count, record order.

---

## Development Phases

### Phase 1: Foundation

- types
- config
- queue
- storage
- checkpoint (SQLite)
- repository

### Phase 2: Inference

- executor
- worker

Goal: A Worker can execute one Task successfully.

### Phase 3: Scheduling

- scheduler
- collector
- writer

Goal: Dynamic multi-GPU execution with correct output generation.

### Phase 4: Runtime

- engine
- validator
- metrics

Goal: Complete lifecycle management with resume support.

### Phase 5: CLI

Create `scripts/inference.py` as the single entry point for multi-GPU inference.

---

## Explicit Non-Goals (MVP)

- Distributed execution
- Ray
- Redis Queue
- RabbitMQ
- JSONL
- Parquet
- Sampling
- Shuffle
- Priority scheduling
- Automatic retry policy

Keep the first version simple, deterministic, and fully testable.

---

## Primary Design Objective

Build a robust, testable, resumable, single-machine multi-GPU offline inference engine with clean architecture and clear separation of responsibilities.
