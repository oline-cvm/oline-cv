"""Reference-dataset tooling: expert annotations, manifests, and selection.

This package owns the *football* side of the reference dataset — the expert's
judgement about a rep — and keeps it strictly separate from the *measured*
side produced by ``oline_cv.pipeline.analyze_video``.

Nothing in here computes, infers, or rewrites a football label. The expert
annotation is the label; the analysis JSON is the measurement. They are joined
only by ``rep_id``.
"""

from oline_cv.reference.schema import (
    ANNOTATION_SCHEMA_VERSION,
    QUALITY_VALUES,
    annotation_context,
    annotation_quality,
    make_annotation,
    validate_annotation,
)

__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "QUALITY_VALUES",
    "annotation_context",
    "annotation_quality",
    "make_annotation",
    "validate_annotation",
]
