# Reference dataset

Working directory for the OL reference benchmark. **Nothing in `clips/`,
`annotations/`, `analyses/`, or `manifests/` is committed** — only this README,
`annotation_template.json`, and the empty directory markers are tracked.

## Layout

```
clips/        rep_001.mp4                 raw film
annotations/  rep_001.json                expert football label
analyses/     rep_001_analysis.json       analyze_video() output
manifests/    reference_manifest.json     what belongs to what
              lt_pass_vertical_set_v1.json  a benchmark selection
```

`rep_id` is the join key. The `<rep_id>_analysis.json` naming matters: the
benchmark's `infer_rep_id()` strips the `_analysis` suffix, so reps keep their
identity all the way through to the built benchmark.

## Two kinds of truth, kept separate

| | source | lives in |
|---|---|---|
| Football label (`quality`, `notes`) | a human who knows OL play | `annotations/` |
| Movement measurements + `trust` | `analyze_video` | `analyses/` |

The quality verdict is never inferred from the CV numbers, and the CV numbers
are never adjusted to match the verdict. Keeping them apart is what makes it
possible to later ask whether the measurements actually track expert judgement.

`quality` is one of `good`, `not_good`, `uncertain`, and it describes **the
rep**, not the player. A future NFL lineman still has bad reps in high school,
so do not mark a rep good on reputation.

Leave `position`, `technique`, or `side` as `unknown` when the film genuinely
does not show it. Unknown is a usable value; a guess is not. An unknown field
simply means the rep will not match a filter on that field.

## Workflow

```bash
# 1. annotate (repeat --note for multiple observations)
python scripts/create_annotation.py \
    --video rep_001.mp4 --rep-id rep_001 \
    --position LT --play-type pass --technique vertical_set --side left \
    --quality good --note "good initial set" --note "stays square"

# 2. check annotations before spending GPU time
python scripts/validate_reference_dataset.py --dataset data/reference \
    --stage annotations

# 3. analyze
python scripts/analyze_reference_dataset.py --dataset data/reference

# 4. check the analyses landed
python scripts/validate_reference_dataset.py --dataset data/reference \
    --stage analyzed

# 5. select eligible reps
python scripts/prepare_reference_benchmark.py --dataset data/reference \
    --position LT --play-type pass --technique vertical_set --quality good \
    --output data/reference/manifests/lt_pass_vertical_set_v1.json

# 6. build
python scripts/build_benchmark.py \
    --manifest data/reference/manifests/lt_pass_vertical_set_v1.json \
    --name lt_pass_vertical_set_v1 \
    --output benchmarks/lt_pass_vertical_set_v1.json
```
