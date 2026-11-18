# reljoin — closed-loop queries without the two-way blow-up

`reljoin` answers cyclic (closed-loop) conjunctive queries over bag relations
using **layered ordered index seeks**. It never materialises a two-relation
intermediate, never hands the multi-table join to SQL, and streams results in
global lexicographic order.

The motivating situation: `A(x,y)`, `B(y,z)`, `C(z,x)`. Computing `A ⋈ B`
first creates a huge intermediate set that `C` then almost entirely discards.
`reljoin` instead walks one variable at a time, and at each layer keeps only
the values on which **every** relation that mentions the current variable
agrees.

```
$ reljoin run --decl query.json --index idx
[1,2,3,24]      # x=1, y=2, z=3, bag multiplicity 24
```

---

## Quick start (from an empty directory)

```bash
git clone <this repo> reljoin && cd reljoin      # or copy the tree
mkdir -p demo/data

cat > demo/query.json <<'JSON'
{"version": 1,
 "order": ["x", "y", "z"],
 "relations": [
   {"name": "A", "vars": ["x", "y"]},
   {"name": "B", "vars": ["y", "z"]},
   {"name": "C", "vars": ["z", "x"]}]}
JSON

printf '[1,2]\n[1,2]\n'            > demo/data/A.jsonl
printf '[2,3]\n[2,3]\n[2,3]\n'     > demo/data/B.jsonl
printf '[3,1]\n[3,1]\n[3,1]\n[3,1]\n' > demo/data/C.jsonl

python3 -m reljoin prepare --decl demo/query.json --data demo/data \
                           --index demo/idx --memory 32M
python3 -m reljoin explain --decl demo/query.json --index demo/idx
python3 -m reljoin run     --decl demo/query.json --index demo/idx
```

Output:

```
[1,2,3,24]
```

Removing `C.jsonl`'s rows (or deleting the relation from the declaration) makes
the result **empty**, and the process still exits 0 — an empty result is a
successful query.

The full end-to-end reproduction, including the oracle, the hotspot and the
large-output observations, is:

```bash
python3 tests/reproduce.py --workdir /tmp/reljoin-demo
```

It generates all data with `tests/gen_data.py`; no large data is stored in the
repository.

---

# Valid-time (temporal) queries

The static commands answer *whether* sources form a closed loop. The temporal
commands additionally answer **when** the loop holds: the same query
declaration describes the relations, aliases, variables and the global order,
while the time axis lives in the data files of a separate directory — time is
never an extra join variable.

## Temporal data format

Each line of a temporal `<relation>.jsonl` is one JSON object:

```json
{"values":[1,2],"since":0,"until":8,"copies":2}
```

* `values` correspond to the declaration's columns, signed `int64`, booleans
  rejected;
* `since`/`until` are signed `int64` ticks of the half-open interval
  `[since, until)`, so `since < until` is required; there are no time zones
  and no infinite endpoints;
* `copies` is optional (default `1`), an arbitrary-precision **positive**
  integer (`bool` is not accepted as an integer); the rows themselves are
  never expanded into copies.

At time *t* the multiplicity of one tuple is the sum of `copies` over every
input row covering *t* (`since <= t < until`); overlapping and fully
duplicate rows accumulate. A complete binding's multiplicity at *t* is the
product of the per-atom tuple multiplicities — aliases each contribute their
own factor, even when two aliases read the same physical tuple. Records whose
interval never intersects the query window are valid data, not errors.

## Temporal commands

```bash
python3 -m reljoin temporal-prepare --decl query.json --data tdata/ \
                                    --index tidx --memory 32M
python3 -m reljoin temporal-explain --decl query.json --index tidx
python3 -m reljoin temporal-run     --decl query.json --index tidx \
                                    --from 0 --to 10
```

`temporal-run` requires `--from L --to U` with `L < U`; the query covers only
`[L, U)` and emits one object per positive-multiplicity, non-empty segment,
ordered first by the global variable order and then by the segment start:

```json
{"values":[1,2,3],"since":2,"until":3,"multiplicity":24}
```

Adjacent segments of the **same binding** with the same multiplicity are
merged into one maximal interval, even if several factors change on the same
tick; holes and different bindings are never merged. Output is streamed, so a
result larger than the memory budget never accumulates in the process.

### Worked example (also asserted literally in the tests)

```
A(1,2): [0,8) copies 2, plus [3,6) copies 1
B(2,3): [1,7) copies 3
C(3,1): [2,5) copies 4, and [5,9) copies 4
```

Window `[0,10)` yields exactly `[2,3)·24`, `[3,6)·36`, `[6,7)·24` — there is
no split at tick 5, because the two C rows meet there with equal copies and
the multiplicity never changes. The exact integer weighted duration
`sum((until-since)*multiplicity)` is `1·24 + 3·36 + 1·24 = 156`. Window
`[4,8)` leaves `[4,6)·36` and `[6,7)·24`.

### How the temporal engine works (and what it does not do)

It does **not** re-run the static query once per integer tick, and it does
**not** form the Cartesian product of the atoms' intervals.

1. **Candidate phase** — the same layered leapfrog over ordered primary-key
   seeks as the static engine finds each complete tuple binding exactly once.
   Time is not part of the candidate intersection.
2. **Time phase** — once a binding is complete, every alias opens a bounded
   range cursor over its bound tuple's event rows (`tick < to`), and the
   cursors are heap-merged in tick order. Each interval was stored at prepare
   time as two events (`+copies` at `since`, `-copies` at `until`); the sweep
   keeps one running multiplicity per alias, multiplies them, and emits a
   segment only when the product changes. Same-tick events (including several
   factors changing together) make one boundary.

Both build and query accept `--memory`; the event sort spills to disk through
a multi-pass external merge, and per-tuple trajectories larger than the budget
are read in fixed-size fetch buffers. With input endpoints unchanged, growing
the time span from 10 to 10¹⁵ does not change the number of events, boundary
ticks or index accesses — the span itself is never enumerated.

### Temporal index on disk

```
tidx/
  manifest.json                  # reljoin-temporal-index-v1; written last
  tables/<relation>.<digest>.db  # event table per distinct atom signature
```

Each table is a `WITHOUT ROWID` table keyed by the tuple columns in global
order plus `tick`, carrying a variable-length signed delta BLOB (arbitrary
precision, so multiplicities above 2⁶⁴ are exact). Builds are staged and
renamed atomically: a failed build leaves no openable directory, and a failed
`--force` rebuild keeps the previous index valid. After a successful build the
original relation files can be removed; a fresh process answers from the
index alone. Static and temporal indexes recognise each other's format and
refuse to open with the wrong command (exit code 6) instead of returning empty
results.

### Temporal Python interface

```python
import reljoin

decl = reljoin.load_declaration("query.json")
idx  = reljoin.open_temporal_index("tidx", expected_declaration=decl)
with reljoin.temporal_query(idx, decl, window=(0, 10)) as q:
    for values, (since, until, multiplicity) in q:   # streamed segments
        ...
print(q.report())      # exact counters incl. weighted_duration (big integer)
```

`reljoin.temporal_query_segments(...)` yields `TemporalSegment(values, since,
until, multiplicity)` objects, and `reljoin.temporal_explain(decl, idx)` /
`reljoin.render_temporal_explain(doc)` give the structured and text plans.
Closing early, cancellation and write failures mark the run `incomplete` and
release every cursor and connection; only full exhaustion plus a successful
consumer finish is reported `complete`.

### Temporal statistics and exit codes

The `--stats` document adds `window_from`/`window_to`, `event_rows_read`,
`event_batches`, `sweep_boundaries`, `raw_segments_before_merge`,
`segments_merged`, `emitted_segments`, `bindings_with_segments` and the exact
integer `weighted_duration`. Exit codes are the same contract as the static
commands (0 complete, 2 usage, 3 declaration, 4 data, 5 missing, 6 damaged or
wrong-kind index, 7 incomplete, 8 resource).

```bash
python3 tests/reproduce_temporal.py --workdir /tmp/reljoin-tdemo
```

The temporal oracle (`tests/temporal_oracle.py`) reads only the raw interval
files, collects endpoints, enumerates bag records in each adjacent gap and
checks shared-variable equality — it never opens the index or the production
executor.

---

# Incremental maintenance (correction batches)

Daily full rebuilds and full before/after query diffs do not scale once the
maintained history is large.  The maintenance product applies a *correction
batch* incrementally: only the tuples the batch names are read, only the
closed-loop bindings those tuples can reach are recomputed, and everything
else in the stored view is left byte-for-byte untouched.

The new user operation is **apply a correction to maintained data**, not
another kind of full query.  The old static and temporal entry points keep
working unchanged.

## Correction files

A correction batch is a UTF-8 JSONL file; every non-blank line is exactly:

```json
{"relation": "A", "values": [1, 2], "since": 2, "until": 8, "delta": 1}
```

* `relation` must appear in the declaration and `values` must match its arity;
  values and endpoints are signed `int64` (booleans are not integers);
* `since < until` — intervals are half-open `[since, until)`;
* `delta` is a **non-zero**, arbitrary-precision signed integer (the only
  integer allowed to exceed 64 bits); it changes the relation's bag by that
  many copies over the interval;
* several lines may correct the same tuple on overlapping intervals; they are
  summed first, so a legal batch is never rejected because an early line
  temporarily dips a running total.  A batch is rejected as a whole if the
  *final* multiplicity of any tuple is negative on any period, even for a tuple
  that never participates in a closed loop.

## Commands (from an empty directory)

First build a temporal index exactly as above, then create the independent
maintenance library:

```bash
python3 -m reljoin temporal-prepare --decl query.json --data data \
    --index tidx --memory 32M
python3 -m reljoin maintain-init --decl query.json --index tidx \
    --maintain mlib --memory 32M
# The temporal index and the original files can now be deleted.
```

`maintain-init` copies the relation state into a self-contained library
(`manifest.json` + `state.db`), builds the initial closed-loop view and records
**revision 0**.  Reopening the library never touches the source again.

Apply a batch (optimistic revision check, one atomic commit):

```bash
cat > batch1.jsonl <<'JSONL'
{"relation": "A", "values": [1, 2], "since": 2, "until": 8, "delta": 1}
{"relation": "B", "values": [2, 3], "since": 4, "until": 6, "delta": -1}
JSONL
python3 -m reljoin maintain-apply --decl query.json --maintain mlib \
    --batch-id b1 --expect-revision 0 --corrections batch1.jsonl \
    --memory 32M --out b1.delta.jsonl --stats b1.stats.json
```

On success the relation state, the closed-loop view, the signed delta and the
batch receipt commit together and revision increments by one (a net-zero but
legal batch still increments).  The stdout/file delta is one object per row,
ordered by the global variable binding and then the segment start:

```json
{"values":[1,2,3],"since":2,"until":4,"delta_multiplicity":12}
{"values":[1,2,3],"since":6,"until":8,"delta_multiplicity":12}
```

`delta_multiplicity` is the signed new-minus-old multiplicity; zero rows are
not emitted, and adjacent pieces with the same delta are merged into maximal
segments rather than mechanically pairing old and new segments.  The receipt
records the exact signed weighted change
`sum((until-since)*delta_multiplicity)`.

Read the currently committed view (no join recomputation — it reads the stored
view table only) and the revision it was read at:

```bash
python3 -m reljoin maintain-run --decl query.json --maintain mlib \
    --from 0 --to 10 --stats-stdout
# stderr: maintain-run read revision 1 (N segment(s)) from the committed view
```

Re-fetch a committed batch's delta without re-applying it (used to recover a
delivery whose export failed after the commit):

```bash
python3 -m reljoin maintain-export --decl query.json --maintain mlib \
    --batch-id b1
python3 -m reljoin maintain-explain --decl query.json --maintain mlib
```

Exit codes add `10` for a batch conflict; a correction data problem is `4`;
an export that fails *after* a durable commit returns `7` and the stderr message
names the committed revision and the `maintain-export --batch-id` recovery
command — the batch is never re-applied and the revision is not claimed failed.

## Idempotency and conflicts

* **Re-send** — the same batch id with identical file bytes and the same
  original `--expect-revision` returns the stored receipt and delta and does
  **not** increment the revision, even if the current revision has moved.  The
  idempotency check runs before the revision conflict check.
* **Reused id, different content or expect-revision** — rejected with exit `10`.
* **Stale expect-revision on a new batch id** — rejected with exit `10`; two
  processes that submit the same expect-revision cannot both commit.

## How the increment stays incremental

`maintain-explain` prints the full plan; in summary:

1. the JSONL batch is streamed into a disk-backed scratch database (a batch
   larger than memory is never held in RAM, and the scratch lives in the temp
   directory, never inside the library);
2. per touched tuple the batch events are merge-joined with the tuple's current
   events, proving the final per-period multiplicity is non-negative;
3. each touched tuple seeds the layered leapfrog walk, clamped to its values;
   bindings reachable through several changed atoms are deduplicated on disk,
   so no binding is swept twice and no unrelated binding is visited;
4. for each affected binding the old and post-batch event streams of every atom
   are heap-merged — same-tick changes across atoms form one boundary, so the
   product difference includes every cross term automatically, with each alias
   contributing its own factor;
5. relation state, view, delta and receipt change in one SQLite transaction.

Readers open a read-only snapshot transaction and pin it for the whole stream,
so a reader that has started (even one still draining its first rows) keeps
seeing one revision while another process commits; a fresh reader sees the new
revision.  A SIGKILL or rollback leaves relations, view and receipts together
at the previous revision.

## Python interface

```python
import reljoin

store = reljoin.open_maintain_store("mlib")                 # verifies, deep
result = reljoin.apply_corrections(
    store, batch_id="b1", expect_revision=0,
    correction_path="batch1.jsonl", memory_bytes=32 << 20,
)
print(result.receipt.revision, result.receipt.weighted_change)

with reljoin.MaintainQuery(store, window=(0, 10)) as q:
    revision = q.revision
    for values, (since, until, multiplicity) in q:
        ...
for row in reljoin.stream_batch_delta(store, "b1"):         # item-by-item
    ...
print(reljoin.fetch_receipt(store, "b1"))
```

## Maintenance tests and data

```bash
python3 -m unittest tests.test_maintain_acceptance \
    tests.test_maintain_lifecycle tests.test_maintain_scale
python3 tests/reproduce_maintain.py --workdir /tmp/reljoin-mdemo
```

`tests/maintain_oracle.py` rebuilds the relation bags from the raw interval and
correction files and enumerates every binding from scratch between endpoints;
it never opens the maintenance database or a production cursor.  The lifecycle
tests use real subprocesses (a pinned-revision reader vs. a concurrent writer,
same-revision writer races with exit `10`, a SIGKILL with exit `-9` and an
export write failure that keeps the committed receipt).  The scale tests
generate many unrelated loops and confirm a one-tuple correction visits one
loop, plus a fragmented batch that spills under a tight memory budget while RSS
stays bounded.


---

## Commands

### `prepare`

Reads `<relation>.jsonl` for every relation named in the declaration and
writes a read-only index directory.

```
python3 -m reljoin prepare --decl query.json --data data/ --index idx/ --memory 64M
```

* `--memory SIZE` is required (`32M`, `512M`, `1G`, or plain bytes). It bounds
  the external-sort buffer; input that does not fit spills to
  `--sort-dir` (default: system temp) and is merged in multiple passes, so
  build memory is bounded independently of input size.
* `--index` is published **atomically**: everything is built in a hidden
  staging directory next to it, fsynced, then renamed. If anything fails
  (missing file, malformed line, out-of-range integer, Ctrl-C), the staging
  directory is removed and no index appears. A pre-existing index is only
  replaced with `--force`, and even then it stays intact if the rebuild fails.
* Duplicate rows are aggregated at build time into the 16-byte multiplicity
  column `m`, so the index is deduplicated and exact.
* Progress and a summary go to stderr; `--json` prints the manifest.

The declaration's **column order** defines each relation's JSON layout. The
index primary key is those columns **projected onto the global variable
order**, so every prefix of the global order is an index prefix. Reordering
columns in the declaration is therefore never confused with reordering the
JSON array, and a file whose arity disagrees with the declaration is a hard
error rather than a silent mis-join. The file's SHA-256, record count and key
mapping are recorded in `manifest.json`.

### `explain`

Prints, without running the query:

* the global variable order and what it means (both the solving order and the
  emission order);
* each atom (relation occurrence / alias), its bytes-per-column layout, the
  index table that serves it, its **access order** (key columns in global
  order), and how many source rows were collapsed into each multiplicity;
* every layer: which atoms are **active** (mention the variable), which are
  **completing** (their key becomes fully bound there), the bound key prefix
  each atom seeks on, and where its multiplicity factor comes from;
* an explicit statement that no SQL join and no two-relation intermediate is
  used.

`--json` emits the same document machine-readably.

### `run`

```
python3 -m reljoin run --decl query.json --index idx/ \
    [--out FILE] [--limit N] [--max-output-bytes N] \
    [--stats FILE] [--memory SIZE]
```

Emits one line per complete binding:

```
[<v0>,<v1>,...,<vk-1>,<multiplicity>]
```

in ascending lexicographic order of the global variable order. Each distinct
binding appears exactly once with its **exact positive integer** multiplicity:
the product of the multiplicities of the corresponding tuple in every atom
(aliases contribute independently). Multiplicities are Python arbitrary
precision integers — a product of many 16-byte factors cannot wrap at 2⁶⁴.

Exit codes are a contract:

| code | meaning |
|------|---------|
| 0 | enumeration completed (including an empty result) |
| 2 | usage error |
| 3 | declaration problem |
| 4 | source-data problem |
| 5 | index missing / never completed |
| 6 | index damaged, truncated or from another declaration |
| 7 | **run incomplete** (consumer closed, cancelled, write failed, `--limit` or `--max-output-bytes` reached) |
| 8 | resource error |

`--stats FILE` writes exact counters (never estimates):

```json
{"status": "complete",
 "index_lookups": 9914,
 "index_seeks_first": 3004, "index_seeks_ge": 2348, "index_seeks_gt": 3032,
 "point_probes": 1530,
 "candidate_values": [1, 1500, 15], "candidate_values_total": 1516,
 "candidates_accepted": 633, "candidates_rejected": 0,
 "complete_bindings": 15, "emitted": 15,
 "max_live_atoms": 2, "peak_rss_kib": 19620,
 "peak_prefix_len": 3, "elapsed_seconds": 0.051, ...}
```

`peak_rss_kib` is the process's real peak resident set. `index_lookups` is the
actual number of cursor executions, split into first-value reads,
`>=` seeks, `>` seeks and exact-key probes. `candidate_values` counts the
values produced by the layer's intersection, i.e. how many values actually got
pushed down.

---

## Python interface

```python
import reljoin

decl = reljoin.load_declaration("query.json")
idx  = reljoin.open_index("idx", expected_declaration=decl)   # verifies tables
q    = reljoin.query(idx, decl, memory_bytes=16 << 20)

try:
    for values, multiplicity in q:      # (tuple, int)
        ...                             # values in global order
finally:
    q.close()                           # releases cursors; marks incomplete if
                                        # the loop did not exhaust
print(q.report())                       # exact statistics
```

`reljoin.query(...)` also works without the declaration (the index carries
one), and `reljoin.query_bindings(...)` yields `Binding(values, multiplicity)`
objects. Closing early is safe and truthful: the run is reported as
`incomplete`, never as a completed empty result.

`reljoin.explain(decl, idx)` returns the structured explain document;
`reljoin.render_explain(doc)` renders the text form.

---

## How the solver works

Layers are the global variable order, `0 … m-1`.

* **Active atom** at layer `L`: an atom whose variable set contains the
  variable of `L`.
* **Completing atom** at layer `L`: an atom whose largest key level is `L`
  (after binding `L`, its entire key is known).

At layer `L`, each active atom exposes an ascending stream of distinct values
of the layer variable, restricted to rows matching the already-bound key
prefix. That stream is read with index seeks:

```sql
SELECT c_k FROM t
WHERE c_0=? AND … AND c_{k-1}=?           -- bound prefix
  AND c_k > ?                             -- or >= for leapfrog
ORDER BY c_0,…,c_n LIMIT 1                -- an index range seek
```

The solver intersects those streams by **leapfrog**: take the maximum current
value, seek every lagging atom to `>= that`, repeat. Only when all streams
agree is the value pushed down to the next layer. An atom with millions of
rows but few distinct values at a layer therefore costs a few seeks, not one
step per row. At the level where an atom completes, its key is fully bound and
a point lookup returns its bag count; that count multiplies into the running
product.

Bound state per recursion level: the current prefix (fixed depth), one
ascending stream per active atom (a cursor plus one value), and the current
output binding. There is **no per-layer value set**, no list of rows, and no
materialised join — independent of fan-out. Output is a generator: the
consumer receives bindings as they are found, so a result with millions of
rows runs in flat memory.

### Why there is no two-way intermediate

There is never a moment where two relations are joined into a new intermediate
relation. The only state carried between layers is the current binding prefix;
each atom is consulted incrementally through its own index. The run
statistics demonstrate this: in the hotspot scenario below, `|A|·|B| =
9 000 000`, yet the run performs 9 914 index lookups and produces
1 516 candidate values.

### Correctness sketch

Every complete binding has, for each atom, a tuple matching it in every
column. At layer `L` that tuple witnesses membership of the binding's value in
the atom's stream, so the value survives the intersection; by induction every
complete binding is reached exactly once. Conversely, a value is pushed down
at `L` only if every atom mentioning `L` has a matching row under the current
prefix; completing atoms are additionally point-probed. Ascending streams and
ascending recursion over the global order make emission lexicographic.
Scanner output is verified in the tests against an independent Cartesian
oracle.

### Multiplicity

`m` is stored as a 16-byte big-endian unsigned integer (so a single tuple may
occur up to 2¹²⁸−1 times; the builder refuses to go beyond). Counts are
converted to Python ints on read and multiplied in Python. Nothing is ever
added or multiplied inside SQLite's 64-bit integer arithmetic. A single
duplicate group above 2⁶⁴ is tested (multiplicity 1000⁹ ≈ 10²⁷).

---

## Verification

```
python3 -m unittest discover -s tests -v
```

| test file | what it proves |
|---|---|
| `tests/test_oracle_acceptance.py` | results and multiplicities equal an independent Cartesian-enumeration oracle (`tests/oracle.py`), which reads only raw JSONL and never touches the index, SQLite, or the solver |
| `tests/test_lifecycle.py` | missing vs. damaged index; truncation, bit-flip, stray WAL; atomic build, failed rebuild keeping old index; cancellation, early close, write failure all reported as incomplete; declaration validation |
| `tests/test_query_plans.py` | every SQL statement the atom executors issue is an index seek on the primary key (no scans, no temp B-trees) |
| `tests/test_anti_explosion.py` | hotspot and large-output scenarios: candidate counts and index lookups far below the two-way join size; flat peak RSS; real statistics reported |

### The oracle

`tests/oracle.py` enumerates `A × B × C`, keeps assignments that agree on
shared variables, and counts bags — the most literal possible definition of
the query. It is deliberately exponential and is used only on small cases.
Acceptance covers: the hand triangle (`24`), removal of the closing relation
(empty), self-aliases with permuted columns, a self-join of one relation in
both directions, shuffled global/decaration orders, empty relations, `int64`
extremes including `-2⁶³` and `2⁶³−1`, many repeated aliases exceeding 64-bit
totals, 4-cycles, chorded 4-cycles, and random instances.

### Observations from the generated scenarios

Generate and reproduce (all data generated, none committed):

```bash
python3 tests/reproduce.py --workdir /tmp/reljoin-demo
```

**Hotspot** (`gen_data.py --scenario hotspot`, 1 hot `x`, 1 500 `y`, 1 200
`z`, closure 4):

```
status: complete, 15 bindings
candidate_values: [1, 1500, 15]        (x, y, z layers)
index_lookups: 9914                     (vs |A|·|B| = 2 250 000 combos)
max_live_atoms: 2,  peak_rss_kib ≈ 19 300
```

**Large output** (`gen_data.py --scenario bigout`, 20 000 bindings):

```
status: complete, 20 000 bindings, all emitted
index_lookups: 280 001, candidate_values: [20000, 20000, 20000]
peak_rss_kib ≈ 20 000 in a 4 MiB budget
```

Peak RSS stays essentially constant as the number of emitted bindings grows,
because nothing is collected before emission.

**Missing vs. damaged** (same script):

```
$ reljoin run --index does-not-exist   → exit 5, "index-missing: … run 'prepare' first"
$ reljoin run --index damaged-copy     → exit 6, "index-corrupt: … fails its SHA-256 check"
```

---

## Design notes and limits

* **Index layout.** One SQLite `WITHOUT ROWID` table per distinct atom
  (relation plus variable assignment), key columns are the atom's columns in
  global-order level order, and `m BLOB NOT NULL` holds the multiplicity.
  Aliases that share a signature share a table; aliases with different
  column assignments get separate tables, so the alias multiplicity is
  independent. The database is opened `immutable=1, query_only=1` after
  verification.
* **Verification** on `run`/`explain`: manifest parse and shape checks, then
  per file size, SHA-256, and SQLite `quick_check` plus row-count and
  multiplicity-shape checks, plus rejection of stray `-wal`/`-journal` files.
  Missing → exit 5, damaged → exit 6. `--no-deep-verify` skips only the SQLite
  level (size and hash are always checked).
* **Bounded memory.** Build: external multi-pass merge sort with a buffer
  derived from `--memory`. Query: fixed-depth prefix, one cursor per atom, one
  binding at a time. `--limit` and `--max-output-bytes` stop deliberately and
  report incomplete.
* **Cancellation.** SIGINT/SIGTERM/SIGHUP during `run` closes cursors and
  exits 7 with the statistics written.
* **No online update, no checkpoint/resume** — rebuild with `prepare --force`.
* **Limits.** At most 256 variables (layers); a single tuple's multiplicity
  must fit in 16 bytes; the declaration must be connected through shared
  variables (e.g. two variables appearing in separate components would be
  rejected as unconstrained — the oracle asserts this too).
