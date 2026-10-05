# Engineering change inventory — 2026-10-05

The pairwise verifier and separate aggregator architecture is preserved. This
working tree implements formal S1 collection and fail-closed gold/prediction/
matching freezes, real provider generation, complete extraction metrics and
response bootstrap, plus service provenance, API isolation and browser repairs.
No benchmark response text, human gold or real model output was manufactured.

`M` means a tracked file changed; `??` means a new/untracked file. This inventory
includes pre-existing edits retained during the repair. Small build-script/CLI
changes remove unused imports and satisfy static checks. The previous extension
ZIP is preserved verbatim under its original-content hash; use the newly generated
package for this source state. The inventory records the pre-publication working tree; the verified repair is published in Git history. Superseded ZIP backups remain local and are excluded from publication.

| State | Exact file/artifact |
|---|---|
| ` M` | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| ` M` | [.gitignore](../.gitignore) |
| ` M` | [README.md](../README.md) |
| ` M` | [data/eval/s1/ANNOTATION_GUIDELINES.md](../data/eval/s1/ANNOTATION_GUIDELINES.md) |
| ` M` | [data/eval/s1/README.md](../data/eval/s1/README.md) |
| ` M` | [data/eval/s1/metadata.json](../data/eval/s1/metadata.json) |
| ` M` | [requirements.lock.txt](../requirements.lock.txt) |
| ` M` | [requirements.txt](../requirements.txt) |
| ` M` | [scripts/build_debug_corpus.py](../scripts/build_debug_corpus.py) |
| ` M` | [scripts/build_mini_corpus.py](../scripts/build_mini_corpus.py) |
| ` M` | [scripts/generate_s1_llm_cache.py](../scripts/generate_s1_llm_cache.py) |
| ` M` | [scripts/validate_s1_annotations.py](../scripts/validate_s1_annotations.py) |
| ` M` | [src/api/app.py](../src/api/app.py) |
| ` M` | [src/api/routes.py](../src/api/routes.py) |
| ` M` | [src/api/schemas.py](../src/api/schemas.py) |
| ` M` | [src/cli.py](../src/cli.py) |
| ` M` | [src/components/extractors.py](../src/components/extractors.py) |
| ` M` | [src/components/retrievers.py](../src/components/retrievers.py) |
| ` M` | [src/components/verifiers.py](../src/components/verifiers.py) |
| ` M` | [src/core/interfaces.py](../src/core/interfaces.py) |
| ` M` | [src/core/registry.py](../src/core/registry.py) |
| ` M` | [src/core/types.py](../src/core/types.py) |
| ` M` | [src/eval/agreement.py](../src/eval/agreement.py) |
| ` M` | [src/eval/extraction_eval.py](../src/eval/extraction_eval.py) |
| ` M` | [src/eval/s1_annotations.py](../src/eval/s1_annotations.py) |
| ` M` | [src/extension/content.js](../src/extension/content.js) |
| ` M` | [src/extension/manifest.json](../src/extension/manifest.json) |
| ` M` | [src/extension/popup.js](../src/extension/popup.js) |
| ` M` | [src/pipeline.py](../src/pipeline.py) |
| ` M` | [src/service.py](../src/service.py) |
| ` M` | [src/ui/app.py](../src/ui/app.py) |
| ` M` | [src/ui/s1_annotation.py](../src/ui/s1_annotation.py) |
| ` M` | [tests/test_api.py](../tests/test_api.py) |
| ` M` | [tests/test_extractors.py](../tests/test_extractors.py) |
| ` M` | [tests/test_pipeline.py](../tests/test_pipeline.py) |
| ` M` | [tests/test_registry.py](../tests/test_registry.py) |
| ` M` | [tests/test_s1_annotations.py](../tests/test_s1_annotations.py) |
| ` M` | [tests/test_types.py](../tests/test_types.py) |
| ` M` | [tests/test_ui.py](../tests/test_ui.py) |
| `??` | [data/eval/s1/EVALUATION_PROTOCOL.md](../data/eval/s1/EVALUATION_PROTOCOL.md) |
| `??` | [data/eval/s1/calibration_examples.json](../data/eval/s1/calibration_examples.json) |
| `??` | [data/eval/s1/guidelines.sha256](../data/eval/s1/guidelines.sha256) |
| `??` | [docs/ENGINEERING_CHANGES.md](../docs/ENGINEERING_CHANGES.md) |
| `??` | [docs/IMPLEMENTATION_MAP.md](../docs/IMPLEMENTATION_MAP.md) |
| `??` | [docs/S1_READINESS.md](../docs/S1_READINESS.md) |
| `??` | [docs/extension-package.json](../docs/extension-package.json) |
| `??` | [nullius-extension.sha256](../nullius-extension.sha256) |
| `??` | [nullius-extension.zip](../nullius-extension.zip) |
| `??` | [scripts/package_extension.py](../scripts/package_extension.py) |
| `??` | [src/eval/provenance.py](../src/eval/provenance.py) |
| `??` | [src/eval/s1_predictions.py](../src/eval/s1_predictions.py) |
| `??` | [src/eval/s1_workflow.py](../src/eval/s1_workflow.py) |
| `??` | [tests/conftest.py](../tests/conftest.py) |
| `??` | [tests/test_extension.py](../tests/test_extension.py) |
| `??` | [tests/test_research_hardening.py](../tests/test_research_hardening.py) |
| `??` | [tests/test_s1_workflow.py](../tests/test_s1_workflow.py) |

Additional local verification artifacts (ignored by Git):

- `.test-artifacts/full.xml`: earlier full run showing the three UI failures.
- `.test-artifacts/full-before-message-fix.xml`: subsequent run showing one outdated assertion.
- `.test-artifacts/full-final.xml`: final full-suite results.
- `.test-artifacts/gates.json`: actual benchmark and missing-human-artifact gate checks.

See [S1_READINESS](S1_READINESS.md) for exact executed counts and remaining evidence
requirements. The source package manifest includes all extension file hashes.
