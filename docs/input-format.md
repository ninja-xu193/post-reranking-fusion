# Input format

One JSON object per line, containing one query. `query_id` is unique within a `domain`. Candidate IDs are nonempty, unique strings. Both score arrays are finite and aligned by document ID.

```json
{
  "query_id": "toy-example",
  "domain": "toy-domain",
  "candidates": [
    {"doc_id": "doc-a", "bm25": 4.0, "reranker": 0.5},
    {"doc_id": "doc-b", "bm25": 2.0, "reranker": 0.5}
  ],
  "qrels": {"doc-a": 1, "doc-outside-the-pool": 2}
}
```

The example above is formatted for reading; each actual JSONL record occupies one line. A full query judgment map is required, including judged documents outside the candidate pool. Missing judgments are treated as zero. All grades must be nonnegative and finite. Training converts positive grades to binary labels; nDCG preserves grades.

The library supports variable candidate depths. It neither pads short lists nor assumes every list has 100 documents. All methods in an actual comparison must still receive the same candidate IDs. Input does not require passage text, server paths, raw model responses or authentication details.
