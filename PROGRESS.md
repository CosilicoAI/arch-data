# PR #227 — main-merge lane (24 commits incl. #258/#260; PR-event CI reconciliation)

## State

- Complete at `3f9d7a1` (plus this journal commit). Merge commit `97eb36d` ("Merge remote-tracking branch
  'origin/main' into HEAD") is on the detached head that extends the pushed
  PR head `7a694be`; nothing is pushed. `git merge origin/main` produced **no
  conflicts**.
- The merged tree's full suite reproduced the pull_request-event CI exactly
  (82 failed / 25 errors); after the fix it is **1,955 passed, 1 skipped**,
  direct exit 0, with every other CI job step green.

## Done

- Read `AGENTS.md` and the four prior lane journals below.
- Merged `origin/main` (24 commits past the PR base `6fb700e2`: #249 append
  workflow, #250/#251 UC family-type and composition crosses, #255 UK
  transport/energy facts, #258 transport/energy follow-ups, #260 the issue-259
  UC element / employment / Housing Benefit / childcare packages).
- Audited the auto-merge on all four files both sides touched
  (`chronicle/source_package.py`, `chronicle/suite.py`,
  `docs/agent-source-package-harness.md`,
  `tests/test_chronicle_source_package.py`): each file's merge-vs-branch diff
  is exactly main's additions and its merge-vs-main diff exactly the branch's,
  and the overlapping regions are orthogonal (main adds package aliases,
  `source_row_dimensions` plumbing and a longer command list; #227 adds the
  kind/identity rules, `CHRONICLE_DB_FILENAME` and the parseability preflight).
  Nothing from either side was dropped.
- Root-caused the PR-event failures by direct execution, not inference:
  `inventory_source_artifacts(db/data)` on the merged tree reports **34 errors,
  all `manifest_kind_missing`** and no other code. 33 are manifests main created
  after #227's explicit-kind rule; 1 is a frozen kindless manifest
  (`db/data/dwp/uc_households_family_type_april_december_2025/manifest.yaml`)
  whose bytes main replaced, so it left the freeze.
- Confirmed all 34 are `publisher_table` and not `microdata_release`: each
  package directory holds exactly one committed data file beside its manifest,
  each manifest has one vintage with a mapping (never a list) spec, and each has
  a `packages/<source>/<package>/source_package.yaml` that parses it — the
  definition `docs/agent-source-package-harness.md` gives for
  `publisher_table`. A release package holds only `manifest.yaml`.
- Confirmed nothing else in main's data violates #227: for all 34 entries the
  recorded R2 route is one of the exact keys `_raw_r2_route_keys` allows
  (main's new `raw/uk/<source>/...` keys are the current country-aware
  canonical shape; the older `raw/<source>/...` spelling is the accepted
  pre-country legacy) and every declared `sha256` matches the committed bytes.
  No filename or digest collides with the two `microdata_release` manifests.
- Confirmed the frozen list needs five removals: four keys whose manifests main
  deleted and the one main modified.

## Done (baseline)

- Full suite on the merged tree, clean tree, nothing edited during the run:
  **82 failed, 1848 passed, 1 skipped, 42 warnings, 25 errors in 1301.92s
  (21:41)**, direct exit **1** -- exactly the pull_request-event CI counts the
  brief reports (82 failed, 25 errors). Log:
  `evidence/merge-baseline-pytest.log`.
- All 107 FAILED/ERROR lines, by module: `test_issue_254_transport_energy_sources`
  37F, `test_issue_257_transport_energy_followups` 26F,
  `test_issue_259_uc_elements_housing_benefit` 9F+18E,
  `test_dwp_uc_composition_source_packages` 2F+7E,
  `test_chronicle_manifest_kind` 3F, `test_chronicle_source_package` 2F,
  `test_chronicle_artifacts` 1F, `test_chronicle_bundle` 1F,
  `test_chronicle_microdata_registration` 1F.
- Every one is classification (a), one rule: the explicit-kind rule
  (`chronicle/registration.py:manifest_kind` /
  `ManifestKindError` / `manifest_kind_missing`). 303 `ManifestKindError`
  mentions in the log; the twelve `AssertionError`s are the reporting commands'
  `assert report.valid` / frozen-list assertions over the same 34 manifests.
  No classification (b) and no classification (c) failure exists: nothing on
  main asserts a behaviour #227 changed, and the merge itself broke nothing.
- Non-pytest CI steps already pass on the merged tree, pre-fix:
  `chronicle init` + `load all` exit 0, `uv build` exit 0, wheel install +
  import/CLI smoke exit 0, `uv run ruff check .` exit 0.

## Done (fixes)

- `fc020c3`: `kind: publisher_table` on the 34 manifests, inserted immediately
  after `package_id` exactly where `_with_declared_kind` writes it and where
  `test_a_fetch_into_a_frozen_manifest_declares_its_kind` pins it; five entries
  dropped from `GRANDFATHERED_KINDLESS_MANIFESTS` (168 -> 163) -- the four whose
  manifests main deleted plus `uc_households_family_type_april_december_2025`,
  which main modified and which now declares its kind; the freeze docstring now
  names deletion as a reason an entry leaves, which
  `test_the_frozen_list_only_shrinks` already enforced. `git diff --unified=0 --
  db/data` is exactly `34 +kind: publisher_table` and nothing else.
- `3f9d7a1`: `FREEZE_SIZE` 168 -> 163. The drop had left five slack slots under a
  `<=` bound, so five new kindless manifests could have landed with freeze
  entries of their own and both guards would have stayed green -- what the
  module docstring's "never added" forbids. The comment records that 168 was the
  `ba8147a7` snapshot size, which the module docstring still states.
- No production refusal was weakened, and no `chronicle/` behaviour changed: the
  only non-data, non-test edit is the freeze list's five removed entries and its
  docstring.

## Verification

- `inventory_source_artifacts(db/data)`: **valid, 0 errors** (199 manifests, 245
  artifacts, 0 missing, 0 checksum mismatches, 15 hash-only, 230 R2 links);
  34 errors before.
- Merge faithfulness proven mechanically: `git merge-tree --write-tree 7a694be
  9f8b77a` returns `17b4d386282a29b009e91f71e76d94bb336a21d3`, which is exactly
  `git rev-parse 97eb36d^{tree}` -- the merge commit is git's unedited 3-way
  merge, nothing hand-resolved.
- Every CI job step, direct exit 0: `ruff check chronicle policyengine_chronicle
  db scripts tests`, `ruff check scripts/ots_anchor.py tests/test_ots_anchor.py`,
  `pytest tests/test_ots_anchor.py` (30 passed), `chronicle init` / `load all` /
  `stats` (815 strata, 6,678 targets), `uv build`, wheel install + import/CLI
  smoke. `ruff format --check` on both changed Python files: already formatted.
- Full suite on the final committed tree `3f9d7a1`, clean, nothing edited during
  the run: **1,955 passed, 1 skipped, 42 warnings in 1432.55s (23:52)**, direct
  exit **0**. Zero `FAILED`/`ERROR` lines. 1,848 + 107 = 1,955: every one of the
  107 baseline failures now passes and nothing regressed.
- `uv lock --check` exit 0 (`Resolved 76 packages`), so CI's `uv sync --locked`
  is satisfied.

## Next

- None once the full suite is recorded. Two items for Max are in the report:
  main's in-place revision of the family-type manifest dropped the superseded R2
  object (no `previous_r2` recorded anywhere in the tree, and I will not assert
  an R2 object I cannot verify), and `docs/adr-chronicle-raw-microdata-
  identity.md` -- which the kind rule cites from ten places -- exists on neither
  side of the merge.

# PR #227 — round 4 fix lane (publish-raw / inventory microdata-identity residual)

## State

- Complete at `3f593f6` on branch `microdata-registration`, from pushed head
  `8416660` (nothing pushed from this lane). The full suite, Ruff lint and
  Ruff format checks all pass with direct exit 0.
- Six code/doc commits, each preceded by its failing regressions. Changed
  files: `chronicle/artifacts.py`, `tests/test_chronicle_microdata_staging.py`,
  `docs/agent-source-package-harness.md`, this journal. No `db/data/**`
  changes; no pin or pin-manifest changes.

## Done

- Read `AGENTS.md`, the prior round-3 journal, and the cited implementation
  sites in `chronicle/artifacts.py` / `chronicle/registration.py`.
- Confirmed the residual by direct execution before writing tests: publish
  uploaded once, rewrote `manifest_tables.yaml` and reported valid, and
  inventory reported valid, for alias in {sha256, archived-filename,
  archived-sha256} across declared/observed/r2-only table identities. The
  current-filename alias was already refused by
  `bytes_present_for_microdata_release_entry`.
- `de85517` red: 18 publish/inventory cases, plus three controls that already
  passed (distinct table bytes beside a release; two non-release manifests
  sharing one public table). `20a2cef` fix: `MICRODATA_IDENTITY_CODE`
  (`bytes_identified_by_microdata_release`), the reporting wrapper
  `_microdata_identity_errors`, the package sweep
  `_assert_no_package_microdata_identities` called from both directory-check
  blocks, the declared/recorded refusal in `_publish_raw_manifest_entry`
  before `read_bytes`, the observed-digest accumulation before the errors
  gate, the recorded-object skip, the preflight return and the upload, and
  `package_manifests` threaded into `_inventory_entry`.
- `adef369`: report the package code as vintage/filename, code and the manifest
  that carries the alias, matching the surrounding package-error shape.
- `e28d332` (adjacent gap, red-first): a release fetch for bytes a public table
  sibling already registers was still accepted, creating exactly the directory
  publish and inventory now refuse. `_assert_no_table_claims_release_identity`
  mirrors the classifier in both fetch passes.
- `6edcd77`/`016ffe2` red, `9093db3` fix (adjacent gap): the table's own
  archived `storage.previous_r2` identities were never compared, while the
  release's were. One shared `_recorded_identity_aliases` reader now feeds the
  package sweep and the table fetch. The source-package reader is left alone on
  purpose: it passes `require_digest=True`, and folding archived digests into
  its wanted set would satisfy the digest requirement it exists to enforce.
- `7935f75` plus the `9093db3` revision document the package-wide rule per
  command in `docs/agent-source-package-harness.md`.
- Hash-only entries and release entries are skipped throughout, so
  `validate_package_directory`'s `sha256_collision_across_manifests` and
  `archived_sha256_collision` codes are preserved, never replaced. The
  observed-digest publish check appends to the accumulated `errors`, which
  `refuse()` would also have preserved: that closure appends its reason to the
  same list rather than replacing it.
- Two entry-level codes are substituted on entries that stay refused either
  way, and both are reported by the package sweep instead: a declared digest
  aliasing a release with different local bytes reported `checksum_mismatch`
  before, and a declared digest matching a hash-only entry inside a release
  manifest reported the entry-level `sha256_collision_across_manifests` (the
  package-level code is unchanged). Nothing that was refused is now accepted.
- Affected-module runs green at `9093db3`: 1,054 tests across the eleven
  manifest/artifact/fetch modules, direct exit 0.
- Full suite at `3f593f6`, clean tree, no edits during the run:
  **1,833 passed, 1 skipped, 42 warnings in 1,460.12 seconds (24:20)**, direct
  exit 0. `uv run ruff check .` exit 0, `All checks passed!`.
  `uv run ruff format --check chronicle/artifacts.py
  tests/test_chronicle_microdata_staging.py` exit 0, `2 files already
  formatted`. `git diff --stat 8416660..HEAD`: four files, none under
  `db/data/**`; no pin or pin-manifest change.

## Next

- None. Fixes, verification and the lane report are complete. Commits remain on
  `microdata-registration`; nothing is pushed.

# PR #227 Astra gate bbd833a9 — round 3 fix lane

## State

- Complete. Started detached at `89737a1`; final code commit `31df1b7` fixes
  both findings. Only journal commits follow the tested code.
- Full final pytest passes with direct exit 0: **1,778 passed, 1 skipped,
  42 warnings in 1,625.15 seconds (27:05)**. Repository Ruff lint and formatting
  checks for all five changed Python files pass with direct exit 0.
- Report: `/tmp/chronicle-227-fix/out.md`; evidence:
  `/tmp/chronicle-227-fix/bbd833a9-round3/`. Prior report preserved there as
  `prior-out.md`; prior journals below remain unchanged.

## Done

- Committed the initial state/done/next journal in `25d5225`, then committed
  each finding's failing regressions before its production fix.
- Finding 1: 32 red cases changed the package byte snapshot; current-filename
  aliases raised after persistence, while archived-filename/checksum aliases
  also rewrote the table manifest and returned successfully. Test commit
  `cc6c644`; fix `31df1b7` shares the reader's kind/current/history classifier
  with fetch. Tables check names and all known digests before lock/publisher
  access, then observed in-memory digests before persistence. Reader unpinned
  refusals, hash-only/owner checks, and external release staging remain intact.
- Finding 1 regressions plus inherited reader controls pass: 66 tests. Four
  affected test modules pass: 540 tests, including distinct tables, relative
  paths, legitimate microdata staging, and under-lock identity invalidation.
- Finding 2: 4 red populated/empty publisher-table list cases reached preflight
  mkdir or under-lock replacement sentinels. Test commit `fb65e7a`; fix
  `abc459f` validates vintages through kind-aware `iter_file_specs` in both
  preparation passes. All 232 registration tests pass, direct exit 0.
- Exact reproduction commands, observed failures, test names, fix commit SHAs,
  focused checks, and final commands/results are in the external report.
- Final `uv run ruff check .`: direct exit 0, `All checks passed!`.
  Final changed-file `uv run ruff format --check`: direct exit 0,
  `5 files already formatted`. Logs: `final-ruff.log`, `final-format.log`.
- Full `uv run pytest -q -p no:cacheprovider`: direct exit 0, counts above;
  log `final-pytest.log`. Used permitted UV cache, offline resolution, cached
  real OTS executable, and removed live Supabase credentials. No pytest output
  pipeline and no code/test changes during the full run.
- Independent implementation, report, and scope audits pass. Exactly six
  allowed files changed; 135 inherited test function ASTs preserved; five new
  functions add 41 cases. All 15 UK pins, both complete pin manifests, and the
  prior journal suffix are unchanged. Evidence: `scope-audit.json` and `.md`.
- No protected/data/OTS edits, source packages, schemas, publisher facts, or
  consumer computations changed. No GitHub/network operations, pushes,
  branches, stash, deployment, or real publication. Worktree clean before this
  final journal update; final scope/diff checks pass.

## Next

- None. Both fixes, required verification, and final report are complete.
  Commits remain on detached HEAD.

# PR #227 Astra gate bbd833a9 — round 2 fix lane

## State

- Complete. Started detached at `46560bc`; final code commit `dfd3b99` fixes
  both findings. Only journal commits follow the tested code.
- Full final pytest passes with direct exit 0: **1,737 passed, 1 skipped,
  42 warnings in 1,461.69 seconds (24:21)**. Repository Ruff lint and formatting
  checks for all four changed Python files pass with direct exit 0.
- Report: `/tmp/chronicle-227-fix/out.md`; evidence:
  `/tmp/chronicle-227-fix/bbd833a9-round2/`. Prior report preserved there as
  `prior-out.md`; prior journals below remain unchanged.

## Done

- Committed the initial state/done/next journal in `79ae427`, then committed
  each finding's failing regressions before production implementation.
- Finding 1: initial 32 red cases returned bytes or rows despite valid public
  microdata sibling aliases; expanded run had 33 failures and 1 distinct-table
  control pass. Test commits `9c8738e`, `e543cea`; fix `dfd3b99` classifies
  sibling current/archived filenames and digests, including R2-only pins, before
  artifact I/O and rechecks observed digests before return/cache mutation.
  Unpinned table reads beside microdata require recorded identity before I/O;
  known-distinct tables pass. All 106 focused reader/artifact cases pass.
- Finding 2: 10 red cases reached mocked filesystem/replacement sentinels;
  1 agreeing-owner/history control passed. Test commit `883c487`; fix `651571b`
  explicitly validates every current/history locator and current identity,
  then uses the same effective-owner validator as publication/inventory in both
  preparation passes. Registration remains metadata-only; a read sentinel
  protects that contract. Final standalone registration suite: 228 passed.
- Exact reproduction commands, observed failures, test names, full commit SHAs,
  supplemental runs, and implementation details are in the external report.
- Final `uv run ruff check .`: direct exit 0, `All checks passed!`.
  Final `uv run ruff format --check` for four changed Python files: direct
  exit 0, `4 files already formatted`. Logs: `final-ruff.log`, `final-format.log`.
- Full `uv run pytest -q -p no:cacheprovider`: direct exit 0, counts above;
  log `final-pytest.log`. Used permitted UV cache, offline resolution, cached
  real OTS 0.7.2 executable, and removed live Supabase credentials. No pytest
  output pipeline and no code or test changes while the full suite ran.
- Independent implementation and report audits pass. Scope/AST audit confirms
  exactly the two implementation modules, two test modules, and this journal
  changed; all 255 inherited test functions are unchanged. Six new functions
  add 46 parametrized cases. Evidence: `scope-audit.json`.
- No protected/data/OTS edits; all 15 UK checksum/size pins unchanged. No
  source packages, contract schemas, or publisher facts changed. No GitHub
  calls, pushes, branches, stash, or real publication.

## Next

- None. Both fixes, required verification, and final report are complete.
  Commits remain on detached HEAD.

# PR #227 Astra gate bbd833a9 — round 1 fix lane

## State

- Complete. Started detached at `cb4c7ef`; final code commit `adb030c` fixes
  all three findings. Only journal commits follow the tested code.
- Full final pytest passes with direct exit 0: **1,691 passed, 1 skipped,
  42 warnings in 1,280.60 seconds (21:20)**. Repository Ruff lint and formatting
  checks for all six changed Python files pass.
- Report: `/tmp/chronicle-227-fix/out.md`. Evidence and prior report:
  `/tmp/chronicle-227-fix/bbd833a9-round1/`. Prior journals below are unchanged.

## Done

- Committed this lane's initial state/done/next journal in `7c8e0a7`, then
  committed each failing-first reproduction before its implementation.
- Finding 1: six red source-reader cases returned contradictory bytes through
  local, cache, fetch, and ZIP paths. Reproduction `4f7aea5`; fix `adb030c`
  reuses effective R2 directory validation and the shared owner-agreement
  helper with observed digests before return/cache mutation. Four positive
  resource controls added; final focused coverage passes (72 tests).
- Finding 2: 13 unsafe publish staging cases reached mocked side effects;
  the existing final-file symlink control passed. Reproduction `a164a2b`;
  fix `63c6bb9` applies the shared complete staging validator before reads,
  locks, or uploads and repeats it in locked publication. All 33 focused
  staging/publication/locking tests pass. No unsafe fixture bytes were written.
- Finding 3: four unrelated sibling collisions reached mutation sentinels;
  a conflict added during lock acquisition reached replacement. Reproduction
  `2040886`; fix `b820b18` validates the complete proposal with effective R2
  digests and archived collision checks in both preparation passes. All 263
  registration/package-directory tests pass.
- Exact failing commands, observations, fix/test names, commit map, and green
  commands are in the external report and per-finding evidence files.
- Final `uv run ruff check .`: direct exit 0 (`All checks passed!`). Final
  `uv run ruff format --check` on the six changed Python files: direct exit 0
  (`6 files already formatted`). Logs: `final-ruff.log`, `final-format.log`.
- Full `uv run pytest -q -p no:cacheprovider`: direct exit 0; complete counts
  above. Log: `final-pytest.log`. Used permitted UV cache, offline resolution,
  cached OTS 0.7.2 executable, and removed live Supabase credentials. No pytest
  pipeline and no code changes while the suite ran.
- Independent read-only review finds no remaining correctness gap. Scope audit
  verifies only the three implementation modules, three test modules, and this
  journal changed. AST audit preserves all 250 inherited source/registration
  test functions and the existing staging refusal assertion body.
- No protected/data/OTS edits; all 15 UK checksum/size pins unchanged. No
  source packages, schemas, publisher facts, or consumer computations changed.
  No GitHub/network operations, pushes, branches, stash, or real publication.

## Next

- None. All requested fixes and verification are complete. Commits remain on
  detached HEAD; final report is in the external output file.

# PR #227 integration lane — merge current PR #226 head

## State

- Integration committed once as `2c71acee53c4904f94c34977e31216e372fc88cb`.
  Its second parent is the exact requested local PR #226 head
  `d544ffe77ffd0d2cb3551fb6f2ec378979e5fa09`. Both histories are preserved.
- Started detached at `f778161f527c7976421e2096737a1988f93a498a`; initial lane
  journal committed as `b6f5f15e515da1f260f8f3f360b326219ee15b53`.
- Required verification finished on unchanged code commit `2c71ace`. Ruff
  passes; both requested pytest suites exit 1 on the same nine imported tests.
  Their fixtures omit `kind`, conflicting with #227's explicit-kind contract.
- Production kind enforcement and all imported test bodies remain unchanged.
  The instruction to pass those tests unchanged cannot also hold for these
  kindless fixtures; a concrete fixture-only patch awaits clarification.
- Final report: `/tmp/chronicle-227-fix/out.md`. Evidence:
  `/tmp/chronicle-227-integration/`; prior report preserved as `prior-out.md`.

## Done

- Verified `git cat-file -t d544ffe77ffd0d2cb3551fb6f2ec378979e5fa09` returns
  `commit`, then ran the requested `git merge --no-ff` against that object.
- Resolved seven artifact hunks, two registration hunks, and the manifest-test
  add/add. The report records both sides, each resolution, and semantics 1–6.
- Retained #226's non-mutating strict YAML loader and five unchanged loader
  tests, effective digest resolver, unidentified-owner byte agreement,
  initialization exemption, selected-only overrides, legacy derived routes,
  and path/vintage matching for YAML-aliased owners.
- Kept #227's kind/access/hash-only/licence contracts, consumer pins, locks,
  complete proposal validation, external staging, archived alias refusals, and
  repository-origin provenance. Cross-vintage refusal diagnostics still pass.
- Fixed an integration hazard in publish preflight deduplication: all selected
  manifests use their eventual override identities even when first encountered
  as siblings. Both new regressions reproduced upload-before-refusal, then pass.
- Independent audit verifies all 1,138 starting #227 test functions and all 14
  new #226 test functions unchanged. Only the six expected paths changed; all
  protected/data files, 15 UK pins, and the prior journal suffix are preserved.
- Resolved focused integration run: direct exit 1, 620 passed, 9 failed,
  16 warnings in 11.23s (`integration-resolved.log`). Remaining failures all
  come from the incoming kindless fixtures.
- Temporary compatibility preview adds exactly six `kind: publisher_table`
  declarations and changes no assertion or production code. All 10 imported
  artifact cases plus two sweep regressions pass: direct exit 0, 12 passed,
  179 deselected, 12 warnings in 1.22s (`fixture-kinds-focused.log`). The patch
  `/tmp/chronicle-227-integration/fixture-kinds.patch` is NOT applied to the repo.
- Final required gates (root commands use `UV_CACHE_DIR=/tmp/chronicle-uv-cache`
  and `UV_OFFLINE=1`; no pytest output pipeline):
  - `uv run ruff check .`: direct exit 0, all checks passed.
  - `uv run ruff format --check chronicle/artifacts.py chronicle/registration.py tests/test_chronicle_artifacts.py tests/test_chronicle_consumer_contract.py tests/test_chronicle_manifest_reading.py`:
    direct exit 0, 5 files already formatted.
  - `uv run pytest -q -p no:cacheprovider`: direct exit 1, 1,644 passed,
    9 failed, 7 skipped, 42 warnings in 1,224.25s (20:24).
  - Then separately: `uv run pytest -q -p no:cacheprovider tests/test_chronicle_artifacts.py tests/test_chronicle_manifest_reading.py tests/test_chronicle_source_package.py tests/test_chronicle_consumer_contract.py tests/test_chronicle_env.py`:
    direct exit 1, 541 passed, 9 failed, 37 warnings in 199.97s (3:19).
- Final logs: `final-ruff.log`, `final-format.log`, `final-full-pytest.log`,
  `final-requested-focused.log` under the evidence directory. No code changed
  between the tested merge commit and this final verification journal.
- No push, branches, stash, rebase, GitHub network, real publication, protected
  file edits, tracked `db/data/**` edits, or UK pin changes.

## Next

- Resolve the contradictory unchanged-test / required-kind instructions. The
  six-declaration fixture patch is concrete and separately proven; applying it
  requires allowing that narrow change to the imported fixture data. A green
  tracked-suite result is not claimed.
- No further implementation defect was found by the independent audit. If the
  fixture exception is accepted, apply the prepared patch as a small named
  follow-up commit and repeat the final gates on that commit.

# PR #227 Astra gate round 3

## State

- Detached lane rebased from `28647088` onto `origin/ops-rename-slice1`
  (`ba8147a7`). Integration and the full baseline pass on code commit `426651e`.
  All four findings and the scalar-entry integration correction are complete.
  Final Ruff and full pytest gates pass on code commit `838ab2c`.
- Evidence report: `/tmp/chronicle-227-fix/out.md`.
- Prior PR #227 journal preserved beside the report as `pr227-prior-progress.md`.
  The #226 journal below is retained verbatim.

## Done

- Replayed the PR #227 commits onto the new base without branches, stash, push,
  or GitHub network. Reconciled final module versions against both branch heads.
- Restored shared vintage validation, nonregular manifest refusals, canonical
  identities, exact resource spelling, environment aliases, and both test sets.

- Rebase integration checks passed: 537 artifact/peer/registration/package tests,
  27 source-path tests, and 160 consumer/env/kind/vintage tests. Ruff passes.
- Rebased the exact kindless freeze onto `ba8147a7` (168 publisher manifests),
  without editing data files. Both public staged releases and shared table-file
  revisions retain their distinct storage behavior.

- Read-only rebase scope audit passed: all 1,130 inherited test functions,
  protected files, 22 timestamp proofs, and 15 UK microdata pins are preserved.
- Source audit found a possible scalar-entry error-handling regression from
  integration. Reproduction and any correction are deferred until the baseline
  has completed, keeping the tested code fixed throughout that run.

- Required post-rebase full baseline completed before touching the findings:
  `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p no:cacheprovider`
  exited 0 with **1,587 passed, 7 skipped, 42 warnings** in **1,430.28 seconds**
  (23:50). Log: `/tmp/chronicle-227-fix/baseline-full.log`.

- Reproduced the rebase scalar-entry concern: six cases fail with uncaught
  `AttributeError` before the complete-manifest refusal. Red test committed
  before the integration correction; evidence is in `source-entry-red.log`.

- Finding 1 reproduced red with and without upload: fetch attempted to write
  the 2023 table and manifest over the 2022 filename. Both cases failed the
  no-write assertion (`finding1-red.log`); fix not yet applied.

- Finding 3 reproduced red in automatic and explicit commit modes: an unrelated
  repository with matching consumer blobs emitted Microcosm provenance. Both
  variants incorrectly returned success (`finding3-red.log`).

- Finding 2 reproduced red: four archived filename/digest variants reached a
  filesystem mutation; four corresponding package-validation cases accepted
  the conflict. Eight failures recorded in `finding2-red.log`.

- Fixed the scalar-entry integration regression: filename resolution now only
  runs on mapping entries; complete-manifest validation rejects other shapes.
  All 33 scalar and inherited source-path cases pass (`source-entry-green.log`).

- Finding 4 reproduced red in seven containment/symlink variants; fetch reached
  the publisher and returned success instead of refusing unsafe staging.
  Writes, uploads, and locks were mocked (`finding4-red.log`).

- Finding 1 fixed: fetch renders and validates the complete proposed package
  before any artifact write or upload, including every vintage and locator.
  Both regressions and 336 artifact/peer/package tests pass.

- Finding 2 fixed: current and historical R2 key/URI identities are compared
  independently of current entry metadata before registration mutation; package
  validation shares that check. Eight regressions and 258 package/registration
  cases pass; scoped Ruff lint and formatting pass.

- Finding 3 fixed: emission verifies the checkout origin as the expected GitHub
  repository and writes that verified identity into provenance. Unrelated,
  missing, and wrong-host origins refuse; HTTPS/SSH variants pass. All 40
  catalogue tests and scoped Ruff checks pass (`finding3-green.log`).

- Finding 4 fixed: the complete staging destination is checked for symlinks
  and repository/package containment before locking or reading publisher bytes,
  then checked again before persistence. Sixteen staging tests and 606 focused
  integration tests pass. Independent audits found no remaining gap in findings
  1--3; all changes are ready for the final required gates.

- Final repository Ruff check passes; format check passes for all 14 Python
  files changed in this lane. The exact final full pytest gate is running on
  `838ab2c`; `out.md` has the complete evidence and awaits only final counts.

- Final exact full gate completed with direct exit 0: **1,633 passed, 7 skipped,
  42 warnings** in **1,319.40 seconds (21:59)**. Log:
  `/tmp/chronicle-227-fix/final-pytest.log`. Repository Ruff lint and all 14
  changed Python format checks pass. No tested code changed after the run began.
- Final scope checks confirm upstream ancestry, a clean worktree before this
  journal update, no lane edits to protected/data/proof files, and unchanged
  15 UK pins. Final report: `/tmp/chronicle-227-fix/out.md`.

## Next

- None. Work and verification are complete; commits remain on detached HEAD.
  No push, branch creation, stash, or GitHub network was performed.

# Operational rename, slice 1 (chronicle#143, mechanism 3)

Lane C5's handoff notes previously lived here; its durable record is
`LANE_C5_REPORT.md`. This file now tracks the active lane on this branch.

## State

- Branch: `ops-rename-slice1`, cut from `origin/main` at `ff3efd3`.
- Scope: env names, R2 bucket configurability, `ledger.db` -> `chronicle.db`,
  and the docs for all three. Code and docs only; no infrastructure changes.
- Out of scope and deliberately untouched: the `ledger` console-script alias,
  the Supabase `"ledger"` schema and mirror table names, governance role ids and
  concept authorities, hash domains and schema ids, anything under `releases/`.
- This PR does not touch the source-data boundary. No package spec, parser,
  selector, manifest, or fact value changes.

## Done

- Read `AGENTS.md`, `docs/storage-architecture.md`,
  `docs/agent-source-package-harness.md`, and the mechanism-3 migration spec in
  the first comment of PolicyEngine/chronicle#143.
- Enumerated every ledger-named env read in tracked Python: the four real
  variables (`LEDGER_SOURCE_ARTIFACT_CACHE_DIR`, `LEDGER_SOURCE_ARTIFACT_FETCH`,
  `LEDGER_PE_US_DATA_ROOT`, `LEDGER_PE_UK_DATA_ROOT`) plus
  `POLICYENGINE_LEDGER_SCHEMA`. `LEDGER_MIRROR_TABLES`,
  `LEDGER_MIRROR_PRIMARY_KEYS`, and `LEDGER_DB_SCHEMA_VERSION` are module
  constants, not env reads, and name out-of-scope surfaces.
- Added `chronicle/env.py`: one shared `env_value`/`env_flag`/`env_names`
  helper reading `CHRONICLE_<X>` first, then `POLICYENGINE_LEDGER_<X>` and
  `LEDGER_<X>` with a once-per-process `ChronicleEnvDeprecationWarning` naming
  the preferred variable.
- Replaced all three ad-hoc helpers (`db/supabase_client._env`,
  `chronicle/source_package._env_value`/`_truthy_env`,
  `db/pe_source_inventory._env_value`) with the shared helper.
- Made the R2 bucket names configurable via `CHRONICLE_R2_RAW_BUCKET` and
  `CHRONICLE_R2_DERIVED_BUCKET`, plumbed through fetch-artifact, publish-raw,
  publish-derived and bootstrap-r2. Defaults unchanged at `ledger-raw` and
  `ledger-derived`. Both manifest write paths now preserve a recorded
  `storage.r2` block instead of restating it under a renamed bucket.
- Emitted `chronicle.db` for new suite outputs, with `ledger.db` still accepted
  on read and on derived-artifact kind inference.
- Added `tests/test_chronicle_env.py` plus artifact tests: 75 hermetic tests
  covering the lookup ladder, precedence, the once-per-process warning, and every
  real call site.
- Swept the docs. `docs/storage-architecture.md` gained an "Environment Variable
  Rename Window" section (the old text stated the fallback direction backwards)
  and a "Bucket Cutover" section; `docs/agent-source-package-harness.md` and
  `README.md` follow. Verified 186 distinct `ledger-raw` objects across 154
  tracked manifest files, every key content-addressed by sha256.

## Review fixes (gate round 1)

The Fable+Sol gate requested changes; both findings are applied on this branch.

- **[high] `fetch-artifact` could attach a recorded R2 URI to new bytes.** The
  preserve rule keyed on the bucket, so a repeated fetch that did not re-upload
  into the same bucket kept the recorded `storage.r2` block while rewriting the
  entry's `sha256`/`size_bytes`. Reproduced against this branch's parent by
  serving two different bodies from one URL: the entry ends up declaring the
  fetched bytes' `sha256` under a key addressed by the superseded bytes' one,
  both when the fetch only registers the bytes and when the bucket default has
  moved.
  The rule now keys on identity — the recorded key's `{sha256}/{filename}` tail
  against the fetched bytes. Identical preserves the block exactly; different
  raises `SourceArtifactRevisionError` before the cached artifact or its
  manifest entry is touched, naming recorded and fetched `sha256`/`size_bytes`
  and the ADR rule that same vintage plus new bytes is a new release revision.
  `--record-revision` opts in: the new bytes get their own content-addressed key
  under the configured bucket, never the old key, and the superseded block moves
  to `storage.previous_r2`. `publish-raw` applies the same check before treating
  a recorded block as history (`recorded_r2_identity_mismatch`, nothing
  uploaded).
- **[low] Env isolation was scoped to one module.** The autouse fixture moved to
  `tests/conftest.py` and now clears all three prefixes for every test.
  `db.supabase_client` resolves `LEDGER_SCHEMA` at import — during collection,
  before any fixture — so `tests/test_chronicle_namespace.py` re-imports it
  under the cleared environment instead of asserting the constant it bound at
  collection time.

`storage.previous_r2` is a sibling key, chosen because every reader
(`inventory-artifacts`, `publish-raw`, `source_package._artifact_content`, the
suite's raw-R2-link acceptance check) reads `storage.r2` alone, and
`publish-raw` already spreads the rest of the `storage` block when it writes
back, so a revision survives publication untouched. All 180 tracked manifest
entries that carry a `storage.r2` block are content-addressed and agree with
their declared `sha256` and `filename`, so the identity check never fires on
tracked data.

## Review fixes (gate round 2)

The second Fable+Sol gate requested changes again. Seven findings, each fixed
with a regression test on this branch. Plan, in dependency order:

1. **[high] `CHRONICLE_SCHEMA` does not reach the Supabase mirror writer.**
   `chronicle/harness.py` and `chronicle/mirror.py` default the schema to the
   literal `"ledger"`; only `db.supabase_client` reads the renamed variable.
   Resolve through the shared helper whenever no explicit `--schema` is given.
2. **[high] The derived-fact boundary check is not rename-safe.**
   `chronicle/consumer_contract.py` matches the `.ledger_derived` suffix only.
3. **[high] `fetch-artifact` cannot address a package's non-default manifest.**
   Seven tracked packages keep a `manifest_*_source_package.yaml`; three
   directories keep two. A fetch into one of them writes a third manifest and
   never sees the recorded block.
4. **[high] Revision protection vanishes when the entry has no `storage.r2`.**
5. **[medium] Recorded-R2 locator fields must be cross-checked**, not read as
   key-or-URI, before a block is preserved or published.
6. **[medium] `_read_manifest` must reject a malformed document**, not treat a
   non-mapping YAML payload as an absent manifest.
7. **[low] Schema resolution must be lazy** so no legacy variable is read at
   collection, before the autouse isolation fixture runs.

## State (round 2)

- Read both gate rounds on PolicyEngine/chronicle#226 and the code each finding
  names.
- Scanned all 154 tracked manifest files (187 `files` entries, every one
  carrying `storage.r2`): every recorded block supplies provider, bucket, key
  and uri; every key is content-addressed; every declared `sha256`/`filename`
  agrees with its key tail; no `uri` contradicts its `key`. Strict locator
  validation therefore refuses nothing that is tracked today.
- All seven findings are applied, each with a regression test, and each
  reproduced against this branch's previous head (`34d1d0f`) first.

### What each fix does

1. `chronicle/env.py` gains `default_chronicle_schema()`: one home for the
   `CHRONICLE_SCHEMA` -> `POLICYENGINE_LEDGER_SCHEMA` -> `LEDGER_SCHEMA` ->
   `"ledger"` ladder. `load_supabase_mirror`, its harness wrapper and the
   `--schema` CLI default all resolve through it when no schema is supplied;
   an explicit `--schema` still wins. Defaults unchanged.
2. `chronicle/consumer_contract.py` matches the whole final dot-segment of a
   `source_record_id` against both `ledger_derived` and `chronicle_derived`.
3. `fetch-artifact --manifest <filename>` selects which of a package's
   manifests the entry belongs to (default `manifest.yaml`); the name must be
   a filename inside `--out-dir`.
4. Revision protection now compares against the entry's recorded identity --
   the recorded key's `{sha256}/{filename}` once published, the declared
   `sha256` before that -- so a registered-but-unpublished entry, or one whose
   upload failed, is protected exactly like a published one.
5. `_validated_recorded_r2` cross-checks every supplied locator field against
   every other and against the content-addressed key shape. A contradiction is
   `RecordedR2LocatorError` at fetch time and `recorded_r2_locator_invalid` at
   publish time, never a preserved block.
6. `_read_manifest` refuses a non-mapping or unparseable document
   (`MalformedManifestError`) before the publisher is read at all;
   `inventory-artifacts` and `publish-raw` report it instead of crashing.
7. `db.supabase_client` resolves both schemas per call rather than at import,
   and `tests/conftest.py` strips the rename window in `pytest_configure`, so
   no module can read or warn from an operator's shell during collection.

All four refusals share a `SourceArtifactManifestError` base, so the
`fetch-artifact` CLI reports every one as exit 1 with nothing written.

### Reproduced against `34d1d0f` (the round-1 head)

Running the same operations against a checkout of the previous head:

1. `load_supabase_mirror` default `schema='ledger'`; with
   `CHRONICLE_SCHEMA=chronicle_probe` the load still reports `schema='ledger'`.
2. `'.chronicle_derived'.endswith('.ledger_derived')` is False: the boundary
   never fired for the chronicle spelling.
3. `fetch_source_artifact()` rejects `manifest_filename` as an unexpected
   keyword; a fetch into `ira_contributions/` writes `manifest.yaml`.
4. A fetch of different bytes over a registered (unpublished) entry was
   accepted silently: the entry's `sha256` was rewritten with no refusal.
5. A block whose `key` and `uri` named different objects was preserved
   verbatim, key sha `c63744a4...` beside uri sha `1e9b3fdb...`.
6. A list-valued `manifest.yaml` was overwritten by the fetch.
7. Importing `db.supabase_client` under `LEDGER_SCHEMA=zzz` bound
   `LEDGER_SCHEMA='zzz'` and emitted a `FutureWarning` at collection.

## Verification

- `uv run pytest -q`: green.
- `uv run ruff check .`: clean.
- `uv run ruff format --check .`: clean for every file this branch touches. 13
  files are unformatted on `main` already and are byte-identical here; CI runs
  `ruff check` only, so they are pre-existing and out of scope.
- CI's db CLI gate (`chronicle init` / `load all` / `stats`): passes.
- `CHRONICLE_R2_RAW_BUCKET=zzz CHRONICLE_SCHEMA=zzz uv run pytest -q`: green.
  Before the shared fixture it failed five tests — four bucket-default
  assertions in `tests/test_chronicle_artifacts.py` and the collection-time
  schema constant in `tests/test_chronicle_namespace.py`.

## Next

- Push and open the PR against `main`. Do not merge.
- Follow-up PR, after Max creates and backfills the new buckets: flip
  `DEFAULT_R2_RAW_BUCKET` / `DEFAULT_R2_DERIVED_BUCKET` to `chronicle-raw` /
  `chronicle-derived`.

## Review fixes (Sol gate round 3)

### State

- Detached HEAD: `fb1bc1df`, the PR #226 head supplied for the ten-finding Sol
  gate round.
- Scope: ten operational-rename findings in artifact fetch/publish/inventory,
  Supabase compatibility aliases, and the README cutover procedure. No tracked
  `db/data/**` manifest will be changed.
- `CLAUDE.md`, one of the requested initial reads, is absent from both this
  worktree and `/Users/maxghenis/PolicyEngine/chronicle`; `AGENTS.md` and the
  remaining requested guidance/code/tests are present.
- PR #227 is available read-only at
  `/Users/maxghenis/PolicyEngine/_worktrees/chronicle-227-fix`. Applicable
  non-microdata hunks from `daafac0` and `c0d9d74` will be ported with the same
  function names and shapes.

### Done

- Re-established the lane state in this committed progress log before making
  gate-round code or test changes.
- Read `AGENTS.md`, the Bucket Cutover and Publisher Revisions contracts in
  `docs/storage-architecture.md`, the README cutover instructions, and the
  named implementation/test surfaces. Confirmed Chronicle must preserve
  publisher bytes and provenance, refuse unsafe fetches before I/O, and leave
  schema/bucket value cutovers explicit.
- Reproduced findings 1, 4, 5, and 8 with 11 failing cases: mismatched manifest
  identity reached the publisher read, quoted year keys bypassed revision
  protection, duplicate year spellings and malformed entries reached I/O, and
  both refetch and revision discarded entry metadata.
- Ported the non-microdata parts of #227's `_assert_manifest_identifies`,
  `_select_vintage_entry`, `_FETCH_OWNED_FIELDS`, and in-place
  `_upsert_manifest` flow. Fetch now validates manifest identity and entry
  shape before I/O, resolves either year-key spelling while refusing both,
  preserves the recorded key spelling, and carries forward every field it does
  not own. The 11 focused cases now pass.
- Reproduced finding 2 with both default sweeps reporting only one of four
  supported manifest names, and finding 7 with all four falsy non-mapping
  `files` values producing `(inventory.valid, publish.valid) == (True, True)`.
- Ported #227's `is_manifest_filename` / `package_manifest_paths` shapes and
  made default root sweeps discover `manifest.yaml`, `manifest.yml`, and both
  `manifest_<package>` extensions. Both sweeps now share `_manifest_files`, so
  only `None` means absent and every other non-mapping value is reported. All
  106 artifact tests pass.
- Reproduced finding 3 as a green preserved-bucket skip for a key routed to the
  wrong package/year, and finding 6 as both a fetch reaching I/O and a green
  publish skip for a self-consistent `s3://` block under `storage.r2`.
- Raw publish now compares the recorded key with the canonical
  source/package/year key before any bucket-change skip. Recorded R2 validation
  also requires `provider='r2'` (and therefore an `r2://` effective URI). The
  focused canonical-key/provider tests, including the existing URI-only and
  stale-country cases, pass without uploads or manifest rewrites on refusal.
- Reproduced finding 9 with the shipped import raising `ImportError` for
  `LEDGER_SCHEMA`, then restored `LEDGER_SCHEMA` and `TARGETS_SCHEMA` as
  deprecated aliases of the stable defaults. Runtime queries remain on the
  lazy functions and still honor post-import environment changes; 55 focused
  namespace/env/client cases pass (1 skipped for absent real credentials).
- Reproduced both parts of finding 10: the README did not state that an
  unqualified mirror load writes to `ledger`, and it named the absent
  `supabase/migrations/20260504_chronicle_bronze.sql` file.
- README now instructs operators to create and apply the deployment migration,
  states the `ledger` runtime default, and gives both supported ways to target
  `chronicle`. The storage architecture and source-package harness use the same
  truthful procedure; no documentation names the absent SQL file. Both README
  regression tests pass.
- Adversarial review tightened the same contracts before final verification.
  Mixed-case package manifests were reproduced as omissions from both root
  sweeps and from the stray-default guard; default discovery now filters every
  recursive filename through the case-insensitive #227 helper. Missing
  `provider` and missing `uri` locators were each reproduced reaching publisher
  I/O; `storage.r2` now requires explicit `provider: r2` and an `r2://` URI.
  The artifact file's 112 tests pass.
- Clarified that *all* schema environment overrides, including deprecated
  spellings, precede the `ledger` default. The README test now also requires
  the create/apply-migration instruction, so deleting the guidance cannot pass
  vacuously. All 14 mirror tests pass.
- Completed the remaining non-microdata preflight port from #227. Reproduced
  four invalid explicit/inferred artifact filenames reaching `_read_artifact`,
  including `filename=manifest.yaml`, which could overwrite the selected
  manifest, and reproduced an undiscoverable `custom.yaml` manifest reaching
  publisher I/O. Ported `is_bare_filename`, `bare_filename`,
  `_infer_artifact_filename` from `daafac0`, and the manifest-like artifact
  refusal plus discoverable `_manifest_path` restriction from `c0d9d74`. All
  117 artifact tests pass after the pre-I/O fix.

### Next

- None in this lane. All ten Sol findings, the non-microdata #227 port
  completeness pass, and adversarial follow-ups are committed and verified.
  The final evidence and per-finding commit map are in the runner's external
  `-o out.md` report, not a repository-root file.

### Final verification

- `uv run ruff check .`: exit 0 (`All checks passed!`).
- `uv run ruff format --check` on all six changed Python files: exit 0
  (`6 files already formatted`).
- Full `uv run pytest -q -p no:cacheprovider`: direct exit 0, 995 passed,
  1 skipped, 18 warnings in 1539.05 seconds.
- A fresh `/tmp` copy of tracked USDA `fy69_to_current` reports two manifests
  and two artifacts in inventory. Publish includes both: FY2024 is one safe
  preserved-bucket skip and the known misrouted FY2025 package/year key is one
  explicit failure, with zero uploads. No tracked manifest was changed.
- The external `-o out.md` report records every failing-first command and
  observation, regression test, fix commit, and exact #227 port provenance.

## Review fixes (Sol gate round 2, eight findings)

### State

- Detached HEAD began at `c36f3fc8`, the supplied head of PR #226; base is
  `main` at `9da02431`. The worktree was clean at intake.
- Scope is the eight supplied findings: cutover compatibility, shared-file
  revision ownership, artifact/manifest path safety, duplicate YAML keys,
  malformed revision history, and import-time Supabase alias compatibility.
- No tracked `db/data/**` manifest will be modified. The whole-tree cutover
  regression will operate on a temporary copy and use a non-writing uploader.
- PR #227 is available read-only at
  `/Users/maxghenis/PolicyEngine/_worktrees/chronicle-227-fix` at `5557cb9e`.
  Applicable non-microdata hunks will be ported with the same function names
  and shapes so its later rebase stays straightforward.

### Done

- Read the existing `PROGRESS.md`, the Bucket Cutover and Publisher Revisions
  sections of `docs/storage-architecture.md`, and all six requested code/test
  files: `chronicle/artifacts.py`, `chronicle/harness.py`,
  `chronicle/source_package.py`, `db/supabase_client.py`,
  `tests/test_chronicle_artifacts.py`, and `tests/test_chronicle_env.py`.
- Read the GitNexus debugging/refactoring workflow guidance. No GitNexus MCP
  graph tools are exposed in this session, so dependency tracing will use
  repository search, focused tests, and the stacked PR's committed diffs.
- Confirmed the baseline gaps in the named code: `_read_manifest` still uses
  `yaml.safe_load`; `_root_manifest_paths` passes a non-default value to
  `rglob`; manifest-declared filenames reach direct path joins and reads;
  fetch updates only its selected manifest; `_superseding_storage` converts a
  non-list `previous_r2` to an empty history; and the Supabase compatibility
  constants are hard-coded defaults.
- **Finding 6 reproduced and fixed.** Red command:
  `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p
  no:cacheprovider tests/test_chronicle_artifacts.py::test_fetch_refuses_duplicate_manifest_keys_before_publisher_io`
  exited 1 with four failures; every duplicate (`source_id`, `package_id`,
  `files`, vintage) reached the publisher-read sentinel. Test-only commit:
  `56f9ce1`. Ported #227 commit `7f9bfe6`'s `StrictManifestLoader` and
  `load_manifest_document` verbatim into `chronicle/registration.py`, switched
  artifact manifest reads to it, and shared it with source-package artifact
  manifest reads. Fix commit: `77e6fda`. The same focused command now exits 0
  with 4 passed.
- **Findings 3, 4, and 5 reproduced and fixed.** Test-only commit: `264e46e`.
  The finding 4 command covering creation beside `manifest.yml`,
  `Manifest.yaml`, a mistyped named manifest, and a symlinked manifest exited
  1 with 4 failures, all reaching the publisher-I/O sentinel. The finding 5
  command covering parent traversal plus `*`, `?`, and character-class glob
  selectors exited 1 with 4 failures because none raised `ManifestNameError`.
  The finding 3 command covering absolute, parent-traversing, and symlinked
  artifact paths exited 1 with 3 failures because inventory considered each
  path valid (and the local, non-network publisher stub could read it).
- Artifact and manifest inputs are now resolved only as literal package-local
  directory entries: unsafe declared filenames return the named
  `non_canonical_filename` error, symlinks are refused before reads, sweep
  selectors cannot contain separators or glob syntax, and no new manifest
  spelling can be created beside an existing registry. Fix commit: `0017520`.
  Ported the #227-shaped `is_bare_filename`, `bare_filename`, `filename_key`,
  `matching_directory_entry`, and package-manifest helpers from `44e1f8d`,
  `7f9bfe6`, and `c2b7722`'s corresponding safety changes; the generalized
  registry-creation and exact sweep-selector guards are slice-1 additions.
  The focused post-fix command exits 0 with 13 passed, and the complete
  artifact module exits 0 with 132 passed.
- **Finding 2 reproduced and fixed.** Red command:
  `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p
  no:cacheprovider tests/test_chronicle_artifacts.py::test_shared_archive_revision_is_refused_through_an_unregistered_owner
  tests/test_chronicle_artifacts.py::test_record_revision_updates_every_owner_of_usda_shared_archive
  tests/test_chronicle_artifacts.py::test_record_revision_updates_every_same_manifest_owner`
  exited 1 with 3 failures: the empty selected vintage bypassed a sibling's
  recorded identity, the USDA second manifest retained its old checksum, and
  the SSA-style second key retained its old checksum. Test-only commit:
  `3636395`.
- Fetch now strictly loads every manifest in the package directory before
  publisher I/O, establishes one normalized byte identity for every entry
  naming the physical file, and applies the revision guard to all owners. An
  explicit revision rewrites every owner from payloads rendered before the
  first manifest write, preserves owner-specific metadata, and archives each
  owner's own R2 provenance. Ported #227's `_package_manifests` and
  `_assert_siblings_record_these_bytes` names/shapes from `c0d9d74`, including
  the normalized manifest-alias exclusion from `235c616`; the coordinated
  all-owner rewrite is slice-1-specific. Fix/docs commit: `7da26a9`. The red
  command now exits 0 with 3 passed; the artifact module exits 0 with 135
  passed.
- **Finding 1 reproduced and fixed.** The whole-tree regression copies tracked
  `db/data/**` to `tmp_path`, configures `CHRONICLE_R2_RAW_BUCKET=chronicle-raw`,
  and installs a successful non-writing uploader. Red command:
  `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p
  no:cacheprovider tests/test_chronicle_artifacts.py::test_documented_bucket_cutover_sweep_accepts_the_tracked_registry`
  exited 1 with the observed tuple `exit=1`, `valid=false`, 156 manifests, 187
  artifacts, 94 skipped, 93 failed, 94 R2-linked, zero uploaded, and five
  report errors. Test-only commit: `7f0f473`.
- A syntactically valid recorded R2 locator whose content-addressed tail
  matches the package-local checksum and filename is now preserved and skipped
  before reconstructing today's country/source/package/year route. This
  accepts 84 pre-country-prefix UK keys, eight explicit Statbel 2023 routes,
  the shared USDA route, and five fully recorded Eurostat manifests that omit
  package IDs, while locator contradictions and byte mismatches remain errors.
  Fix commit: `25deb29`; no #227 hunk applies. The same whole-tree command now
  exits 0 with 161 manifests, 194 artifacts, 194 skipped/R2-linked, zero
  uploaded or failed, and no errors. The artifact module exits 0 with 136
  passed.
- **Finding 7 reproduced and fixed.** Red command:
  `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p
  no:cacheprovider tests/test_chronicle_artifacts.py::test_fetch_refuses_non_list_previous_r2_before_publisher_io`
  exited 1 with 3 failures: mapping, scalar, and null `previous_r2` values all
  reached the publisher-I/O sentinel. Test-only commit: `e418813`.
  `_validated_recorded_storage` now rejects every present non-list history
  before `_read_artifact`, so `_superseding_storage` cannot replace malformed
  provenance. Fix commit: `8b7ab22`; no #227 hunk applies. The focused command
  now exits 0 with 3 passed (5 passed including ordinary and shared revision
  history controls).
- **Finding 8 reproduced and fixed.** A fresh-process regression sets each of
  `CHRONICLE_SCHEMA`, `POLICYENGINE_LEDGER_SCHEMA`, and `LEDGER_SCHEMA` before
  importing the compatibility constants, alongside
  `POLICYENGINE_TARGETS_SCHEMA`. Red command:
  `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p
  no:cacheprovider tests/test_chronicle_env.py::test_supabase_schema_compatibility_aliases_honor_import_time_environment`
  exited 1 with 3 failures; every subprocess returned `ledger` / `targets`.
  Test-only commit: `e2ceedf`.
- `LEDGER_SCHEMA` and `TARGETS_SCHEMA` are now import-time snapshots of the
  same lazy resolver functions runtime queries use, restoring their original
  environment-backed behavior while later environment mutations remain lazy
  through `chronicle_schema()` / `targets_schema()`. Fix commit: `f41ad0f`; no
  #227 hunk applies. The focused regression now exits 0 with 3 passed; it and
  the runtime/default namespace controls exit 0 with 11 passed.
- **Adversarial path-boundary follow-up reproduced and fixed.** Test-only
  commit `465a34c` showed normalized manifest aliases and a package-local
  artifact symlink reaching the publisher-read sentinel; invalid sweep names
  were accepted when the root was absent and escaped the CLI as tracebacks;
  manifest-named artifacts were not refused; and publish could upload a valid
  first entry before discovering an invalid later entry. The same audit also
  reproduced contradictory owners across sibling manifests being ignored by
  both sweeps.
- Fix commit `8e766ad` ports #227 commit `235c616`'s corrected
  `_package_manifests` normalized-alias guard, validates fetch destinations
  before publisher I/O, rejects manifest-named artifacts in publish and
  inventory, validates sweep selectors before checking the root, and reports
  those refusals cleanly from both CLIs. It also corrects the object-key docs
  to describe compatible recorded history. The six focused path/CLI cases now
  pass. Complete-package publish preflight and sibling-owner validation remain
  the next coherent fix.
- **Source-package reader follow-up reproduced and fixed.** Test-only commits
  `92af48c`, `9a51d99`, `3228238`, and `578c8b2` showed that a package spec
  could read an absolute or parent-traversing artifact, follow artifact and
  manifest symlinks, accept a normalized filename alias or manifest-named
  artifact, and select an unsafe/unsupported manifest path before artifact
  I/O. `SourceArtifactSpec` now resolves both manifest and artifact resources
  through the shared #227 filename-identity helpers before opening either.
  The ten focused cases pass, and the full source-package module passes with
  135 tests (13 warnings). Fix/journal commit: `64fe94b`.
- **Residual side-effect ordering reproduced.** The focused command covering
  `test_fetch_refuses_invalid_r2_identity_before_publisher_io` and
  `test_publish_preflights_entire_root_before_any_upload` exited 1 with three
  failures. Both slash-only R2 identity fields reached the publisher-read
  sentinel; a root sweep uploaded and rewrote the valid `a_good` package before
  reporting the unsafe filename in `z_bad`.
- Fetch now validates the source/package object-key components before reading
  publisher bytes. Raw publish now completes one root-wide read/identity/entry
  preflight and returns every refusal before its first uploader call or
  manifest rewrite. The three red cases plus the existing entry-, sibling-,
  and tracked-cutover preflight controls pass (9 tests). Fix commit:
  `4991e8f`. The complete artifact module passes with 154 tests (12 warnings).

### Next

None. The eight findings, required #227 ports, adversarial preflight follow-ups,
and final verification are complete. The external `-o out.md` report contains
the per-finding reproduction/fix/test/commit/port map.

### Final verification (eight-finding round)

- The first bare `uv run` lint invocations exited 2 before Ruff started because
  the sandbox denied access to `/Users/maxghenis/.cache/uv`. Re-running through
  the permitted existing cache (`UV_CACHE_DIR=/tmp/chronicle-uv-cache`) gave:
  `uv run ruff check .` exit 0 (`All checks passed!`) and `uv run ruff format
  --check` on the eight changed Python files exit 0 (`8 files already
  formatted`).
- `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p no:cacheprovider`:
  direct exit 0, 1,039 passed, 7 skipped, 18 warnings in 1,323.73 seconds.
- Focused artifact module: direct exit 0, 154 passed, 12 warnings in 7.15
  seconds. Focused source-package module: direct exit 0, 135 passed, 13 warnings
  in 193.89 seconds.
- `git diff c36f3fc8..HEAD -- db/data` is empty: no tracked source manifest was
  modified.

### Deliberate boundaries

- Did not add crash-atomic multi-file transactions for an unexpected write or
  process failure midway through a coordinated revision. Every deterministic
  refusal is preflighted before writes/uploads and the successful path updates
  all owners; true cross-file crash atomicity needs a separate staging/rollback
  design and is not one of the eight findings.
- Did not port #227's `iter_directory_entries`: its exact implementation depends
  on #227's microdata list-entry machinery, which this slice explicitly
  excludes. Also retained the current artifact-local `bare_filename` exception
  wrapper so filename refusals remain `SourceArtifactManifestError` and the
  existing CLI reports them cleanly; #227's later rebase can adopt its broader
  exception hierarchy together.
- Did not add a new rule for two physical artifact files whose names normalize
  to the same key when the exact requested spelling sorts first. That is
  defense-in-depth outside findings 3/4; there are no such collisions in the
  tracked tree. Manifest-name collisions are already refused independent of
  sort order.

## Peer round 4 (nine findings)

### State

- Started from detached HEAD `fa98993a` with a clean worktree; no branches,
  pushes, stashes, GitHub access, or tracked `db/data/**` changes are authorized.
- Scope: the nine supplied provenance, publication preflight, canonical
  identity, manifest-vintage, filename, and regular-file findings.
- Preserve the shared helper names used by stacked PR #227; its worktree is
  read-only and will not be modified.
- Report path: `/tmp/chronicle-226-round4/out.md` (no explicit runner `-o`
  path was provided in the visible request).

### Done

- Read the existing journal, storage architecture, role rules, and named code
  surfaces. Began reviewing the named test modules and shared registration
  helpers. GitNexus debugging guidance is available, but no graph tools are
  exposed; use repository search and hermetic regression tests.
- Established this committed state/done/next journal before implementation.

### Next

- Add and run failing regressions for each finding before its implementation.
- Fix and commit coherent steps, recording exact red commands and observations
  in the external report.
- Run full Ruff lint, formatting checks for changed Python files, and the full
  pytest suite with direct exit codes; record counts and final commit map.

### Round 4 reproduction checkpoint

- Added publication/tree/identity/alias/sibling regression coverage in
  `tests/test_chronicle_artifact_peer4.py` before any associated fix.
- Red command: `UV_CACHE_DIR=/tmp/chronicle-uv-cache uv run pytest -q -p
  no:cacheprovider tests/test_chronicle_artifact_peer4.py` (stdout/stderr saved
  to `/tmp/chronicle-226-round4/main-red.log`) exited 1: 60 failed, 12 warnings.
- Finding 3: all 4 non-regular tree cases reached build-ID inference before
  refusal. Finding 4: all 24 invalid publication identity cases reached an
  artifact read or build-ID inference. Finding 5: all 10 noncanonical manifest
  declarations reached the publisher read. Finding 6: all 4 single case/Unicode
  aliases reached artifact reads. Finding 8: 12 explicit-selector sibling cases escaped as plain
  `ValueError` through the functions and harness; 6 top-level CLI tests had
  an invocation error (corrected and verified below).
- Consumer guard, source resolver, and shared vintage regressions are being
  developed independently; all implementations remain gated on observed red.

### Finding 2: source provenance reader

- Reproduced malformed/contradictory/non-R2 locators and checksum/filename
  mismatches with 10 failures and 1 passing control before implementation;
  exact command and output are in the external report's source evidence.
- `SourceArtifactSpec` now reuses `_validated_recorded_r2`, checks the manifest
  identity, and uses the immutable object's digest to validate local/fetched
  bytes even without a separately declared checksum. Invalid metadata is
  refused before artifact/cache I/O; bad fetched bytes before cache writes.
- The same focused command now exits 0: 11 passed, 12 warnings. Inventory's
  half of finding 2 remains next; no source package or source data was changed.

### Finding 5: exact manifest declarations

- Present `source_id` and `package_id` now pass `_require_identity_segment`
  without stripping or stringification and must equal the fetch argument.
  Absent fields remain eligible for first registration; null/empty fields do
  not impersonate absence.
- Focused regressions and existing other-package refusal controls passed:
  direct exit 0, 12 passed. The red checkpoint above recorded all 10 new
  declaration cases reaching publisher I/O before this fix.

### Finding 7: shared logical vintage validation

- Before implementation, the new manifest-vintage tests exited 1: 5 failed,
  1 passed. Duplicate keys `2024` and `"2024"` reached artifact I/O through
  publish, inventory, source loading, and fetch of another selected vintage.
- `load_manifest_document` now invokes shared `validate_manifest_vintages`
  over the entire manifest, so every consumer refuses a logical duplicate
  through its existing controlled YAML error path. Shared names are unchanged.
- Six regressions plus existing quoted-year compatibility tests exit 0:
  8 passed, 12 warnings. Scoped Ruff lint/format checks both pass.

### Finding 1: configured derived provenance

- Consumer guard regressions ran red before implementation: direct exit 1,
  35 failed and 1 passed; configured bucket/prefix provenance did not produce
  `derived_fact_provenance`. Exact evidence is in the external report.
- The guard resolves derived bucket/prefix configuration lazily, retains exact
  archived routes, and checks raw locator fields, source-file locators, and
  source URL. Added `default_r2_derived_prefix` with the standard environment
  lookup ladder so publication and the guard can share it.
- Full consumer-contract module passes: direct exit 0, 95 passed, 32 warnings;
  scoped Ruff checks pass. Wiring configured prefixes into publication and
  refusing unrecognizable explicit routes remains a publication follow-up.

### Finding 9: regular source resources

- Directory and FIFO manifest/artifact regressions ran red: direct exit 1,
  4 failed, 2 passed. The resolver returned non-regular manifests and reached
  artifact-read sentinels; no test opened a FIFO.
- The source resolver now requires an existing selected entry to be a regular
  file after symlink and spelling checks. Missing artifact entries still use
  the existing checksum-validated cache/fetch path; zip-backed resources work.
- Same focused command exits 0: 6 passed, 12 warnings. The full source-package
  module is running. Exact red/green commands are in the external evidence.

### Finding 8: controlled sibling-manifest refusals

- `_package_manifests` now translates discovery's `ValueError` into
  `MalformedManifestError`, so publish/inventory and both CLI entry points
  use the shared controlled refusal path even with an explicit selector.
- The first post-fix command actually exited 1: 12 passed and 6 top-level
  CLI cases failed because the test passed argv to a zero-argument entry point.
  Corrected the test to set sys.argv and assert SystemExit. The corrected
  command exits 0: 18 passed, 12 warnings; no uploads or rewrites occur.
  The earlier premature passing-count journal entry is corrected here.

### Additional round 4 red checkpoint

- Added inventory locator/identity regressions, explicit raw identity overrides,
  root/nested/excluded-registry derived symlinks, and derived prefix publication
  integration before fixing those paths. Focused `additional-red.log` command
  exited 1: 24 failed, 1 passed, 60 deselected, 14 warnings.
- Finding 2 inventory: six malformed/contradictory identity cases reached
  artifact reads; absent separate checksum allowed wrong local bytes as a valid
  R2 link. Valid URI-only locator control passed.
- Finding 3 additional tree cases reached upload or silently skipped the link.
  Finding 4 explicit overrides reached reads. Finding 1 both-custom explicit
  route reached inference; all three prefix environment names were ignored by
  publication. Exact command and observations are in the external report.
- Full source-package module completed: direct exit 0, 157 passed, 13 warnings
  in 192.07 seconds, including both source-reader fixes.

### Finding 3: complete derived tree preflight

- Derived publication now walks the root and every descendant with `lstat`,
  allowing only directories and regular files. It refuses symlinks/FIFOs,
  including an excluded registry filename, before build-ID inference, file
  reads, uploads, or registry writes.
- Both failing-first tree groups now pass: direct exit 0, 7 passed,
  78 deselected, 12 warnings. Safe nested directories remain publishable.

### Finding 6: exact physical artifact spelling

- Publish and inventory now turn a single normalized alias with different
  physical spelling into `artifact_spelling_mismatch`, and neither reads
  bytes after alias errors. Duplicate-alias refusal also stops before reads.
- Case/Unicode regressions plus duplicate/symlink controls exit 0: 6 passed,
  12 warnings. Missing filenames retain their separate missing-name error.

### Finding 4: canonical identities on new publication paths

- Derived publication validates source/package identifiers before inference or
  artifact reads. Raw publication preserves original argument/declaration types
  until new-key validation, rejects noncanonical identifiers and contradictory
  declarations before reading artifacts, and retains root-wide preflight.
- Identity regressions plus declaration controls exit 0: 44 passed,
  41 deselected, 12 warnings. Six historical-identity controls and the full
  tracked-registry cutover test also pass: 7 passed, 12 warnings. Recorded
  objects keep their original routes even when old manifests omit package_id.

### Finding 2: inventory provenance

- Inventory now reuses `_validated_recorded_r2` before artifact reads, rejects
  contradictory/incomplete/non-R2 locators and mismatched checksum/filename
  declarations, and checks actual bytes against the recorded digest even when
  a separate checksum is absent. Invalid entries expose no R2 link.
- Seven failing-first defect cases plus the valid URI-only control now pass:
  direct exit 0, 8 passed, 83 deselected, 12 warnings. Together with `d5e9952`,
  both consumers named in finding 2 now enforce immutable provenance.

### Finding 1: publication route enforcement

- Publication and the consumer guard now share `is_derived_r2_route` and the
  lazy prefix resolver. Derived key generation honors all three environment
  spellings for the configured prefix.
- A both-custom explicit bucket/prefix combination must identify a configured
  or archived derived route, otherwise publication refuses it before build
  reads. The storage architecture documents the shared route configuration.
- Four failing-first route integration cases now pass: direct exit 0,
  4 passed, 87 deselected, 16 warnings. All nine findings are implemented;
  focused integration verification and full final checks remain.

### Final review follow-ups

- Focused integration passed: direct exit 0, 409 passed, 40 warnings in 15.10s.
  Ruff lint passed and all 8 changed Python files passed formatting checks.
- Interrupted the first full pytest run (direct exit 130) after peer review
  found that date/set YAML identity refusals could fail CLI JSON serialization.
  It is not a completed full-suite verification and will be restarted.
- Reproduced a remaining configured-route bypass with uppercase `R2://` in
  raw URI, source_file, or source URL: 3 failed, 3 lowercase controls passed.
  URI scheme matching is now case-insensitive. Full consumer module now passes:
  direct exit 0, 101 passed, 32 warnings; scoped Ruff checks pass.
- Additional failing-first checks cover date/set report serialization (8 failed),
  vintage namespace escapes and build-ID separators (8 failed), and fetch
  vintage namespace escapes. These publication fixes precede the full restart.

- Committed the publication follow-up regressions before their fixes. Red
  commands/logs: `report-identity-red.log` (8 failed),
  `namespace-followup-red.log` (8 failed, 99 deselected), and
  `fetch-vintage-red.log` (3 failed); all direct exits 1. CLI refusals hit
  non-JSON date/set values; malformed vintage/build segments reached reads or
  upload sentinels. Exact commands will be included in the final report.

- Raw publication now keeps original identity values for validation and uses
  separate string fields only in reports. Date/set refusals serialize cleanly
  through both CLIs; historical skips also retain serializable identity fields.
  The regression and history controls exit 0: 14 passed, 12 warnings.

- New publication vintage segments and derived build-ID segments now pass the
  same canonical segment validator, preventing API-provided slashes or '..'
  from moving an otherwise recognized route. Fetch applies the vintage check
  before publisher I/O; historical raw objects retain their existing routes.
  All 11 failing-first namespace cases pass (99 deselected, 12 warnings).

### Final verification in progress

- Final focused integration exits 0: 434 passed, 40 warnings in 11.11s.
- Final `uv run ruff check .` exits 0 (`All checks passed!`). Final
  `uv run ruff format --check` on the 8 changed Python files exits 0
  (`8 files already formatted`). Both use the permitted UV cache.
- `git diff --check fa98993a..HEAD` and the tracked `db/data` no-change check
  both exit 0. Worktree clean before this journal update.
- Restart the full suite at code commit `19d037f` with UV offline, offline
  OpenTimestamps client selection, and live Supabase credentials removed from
  the child environment. This keeps the required full run free of network.

### Final state, done, and next (peer round 4)

**State:** complete. All nine findings and the peer-review follow-ups are fixed
in small detached-HEAD commits. Final code commit: `19d037f`; the completed
full run included the subsequent journal commit `9fa6a1d`.

**Done:**

- Full `uv run pytest -q -p no:cacheprovider` returned direct exit 0:
  **1,238 passed, 7 skipped, 42 warnings in 1,468.40s (24m 28s)**.
  The child environment used the permitted UV cache, `UV_OFFLINE=1`, an offline
  OpenTimestamps client command, and no live Supabase URL/service/secret keys.
  Exact invocation and output are in `/tmp/chronicle-226-round4/out.md` and
  `/tmp/chronicle-226-round4/full-pytest.log`.
- Final whole-repository Ruff lint exited 0. Formatting checks on every changed
  Python file exited 0: 8 files already formatted. Final focused integration
  also exited 0: 434 passed, 40 warnings.
- No tracked `db/data/**` changes, GitHub/network access, pushes, branches,
  stashes, or edits to PR #227's worktree. Shared helper names remain intact.
- The external report contains each finding's red command and observed failure,
  fix, regression names, and commit SHA, plus the final verification and
  deliberate boundaries. It records the corrected CLI-test invocation and the
  interrupted first full run without counting either as passing verification.

**Next:** none in this fix lane. No push or branch operation is authorized.

## Peer round 5 fix lane: the two low residuals on head 83de2e8

**State:** opening. Worktree `_worktrees/chronicle-227-fix`, detached HEAD at
`83de2e8` (PR #227, branch `microdata-registration`). No push, no branch, no
stash, no `db/data` changes.

**The two findings (peer, verbatim summaries):**

1. [low] `fetch-artifact` and `register-artifact` never run the package-wide
   sweep `_assert_no_package_microdata_identities` that `publish-raw`
   (artifacts.py:1973) and `inventory-artifacts` (artifacts.py:4753) run. The
   fetch preflight validates siblings entry-by-entry
   (`_assert_manifest_valid_for_fetch`, 1242-1273); `_upsert_manifest`
   (3317-3339) repeats that plus `_assert_package_file_owner_identities_agree`;
   registration stops at owner agreement (registration.py:1600). A fetch or
   registration therefore carries forward a package that inventory reports as
   `bytes_identified_by_microdata_release`.
2. [low] `_validated_recorded_storage` (artifacts.py:2715-2736) refuses a
   non-list `storage.previous_r2` but not its elements.
   `_recorded_object_identities` (registration.py:1138-1171) silently skips a
   non-mapping element, blinding the round-4 archived-alias checks, while
   `_prepare_registration_payload` (registration.py:1580-1593) already refuses
   the same element. The four commands disagree.

**Reproduced by execution (both findings, before any fix):**

- Finding 1, own reproduction: a package whose *other* vintage carries the
  release's filename, checksum or archived R2 identity is refused by
  `inventory_source_artifacts` (`bytes_identified_by_microdata_release`) and
  accepted by `fetch_source_artifact`, which rewrites the table manifest and
  writes the bytes. Three alias variants, all three reproduce.
- Finding 2, own reproduction: `_recorded_object_identities` yields
  `[('microdata.csv', <sha>, ...)]` for a mapping element and `[]` for the
  equivalent bare `r2://` string; `_recorded_identity_aliases` returns
  `(set(), set())`; `_validated_recorded_storage` returns the storage block
  unchanged.
- An independent lane agent pinned the pre-fix tree read-only
  (`git archive 68bb76f`) and reproduced the fetch gap there for six alias
  variants in both the table and the release fetch direction (23 passed,
  exit 0), then saw every one of those gap assertions flip against the fixed
  tree (16 failed / 7 passed, exit 1).

**Red-first regressions (`tests/test_chronicle_artifact_peer5.py`):**

- `618f287` fetch/registration package sweep: 24 failed, 8 passed, exit 1.
  Fetch failures were `ORDERING VIOLATION: the publisher was read before the
  refusal`; registration failures were `DID NOT RAISE`. The 8 passing are the
  inventory controls that already report the code.
- `55636de` archived provenance: 19 failed, 7 passed, exit 1. Registration
  refused all six unreadable element shapes; fetch, publish-raw and inventory
  accepted every one, and the blinding case reported `valid=True errors=()`.
- `0169303` identity binding: 2 failed, exit 1,
  `AttributeError: 'NoneType' object has no attribute 'strip'`.

**Fixes:**

- `4d2d8fe` the package-wide sweep now runs in the fetch preflight (after every
  manifest is individually valid, before publisher I/O and before the lock, and
  again under the lock through the existing recursion), over the proposed tree
  in `_upsert_manifest`, and in `_prepare_registration_payload` inside the
  package validation that already refuses a directory it will not persist into.
- `f394205` `_validated_recorded_storage` validates every `previous_r2` element
  with the locator rules the current object gets, forwarding
  `source_id`/`package_id`/`bind_registration_identity`, exactly as
  `_prepare_registration_payload` already did.
- `f180178` `_clean_key_part` refuses a non-string key part as a `ValueError`,
  so binding a locator to a manifest that omits `source_id` is the controlled
  `RecordedR2LocatorError` rather than an `AttributeError` escaping every
  handler. That crash pre-existed on `storage.r2`; validating archived objects
  reaches the same binding, so the fix belongs with it.

**Verification so far:** `tests/test_chronicle_artifact_peer5.py` 60 passed.
Focused ten-module run with fix 1 only: 1038 passed, exit 0 (226s). Focused
twelve-module run with fixes 1 and 2: 1109 passed, exit 0 (238s). No tracked
manifest under `db/`, `data/` or `packages/` carries `previous_r2` at all, and
`git status --porcelain db data` is empty.

**Adversarial verification (five attack lanes, all by execution):**

- ~25 metadata-shape attacks on the sweep were all refused: relative and
  dotted output paths, unclassifiable manifest kinds on either side, list vs
  mapping entries, hash-only and unreadable access classes, case and Unicode
  filename aliases, non-bare filenames, YAML anchors, an alias arriving
  between the two preflight passes (fetch and registration), registration into
  a not-yet-existing manifest, `--allow-reissue`, and `--record-revision`
  sibling-owner rewrites.
- Both CLIs serialize every new refusal: `fetch-artifact` and
  `register-artifact` print the message and exit 1; `publish-raw` and
  `inventory-artifacts` emit JSON that parses, carrying
  `recorded_r2_locator_invalid:… storage.previous_r2[0]: …`. No round-4-style
  serialization failure, including for YAML date vintages and set identities.
- Two further defects surfaced and were fixed red-first:
  `8c99ac8` one locator parser for both readers. `_recorded_object_identities`
  read a `uri` with `urlsplit`, whose path stops at a `?` or `#`, while
  `_validated_recorded_r2` splits `provider://bucket/key` by partition and
  keeps reading. A stray `?` in the authority made a block naming a release's
  exact object read as naming nothing, so every archived-alias check went
  blind on it. Both now use the shared `split_r2_uri`.
- Measured, not assumed: validating archived objects drops an entry's
  effective recorded digest, but the sweep still sees the identity the current
  object records, and the refusal still lands before the publisher is read
  (`test_unreadable_history_does_not_blind_the_package_sweep`).

**Open, needs a decision (reported, not changed):**

- The sweep is a metadata check. `publish-raw` and `inventory-artifacts` also
  classify each entry's *observed* bytes, so a package-local file that IS a
  release's bytes under a table record with no (or a wrong) declared digest is
  still accepted by `fetch-artifact` and `register-artifact`. Closing it for
  fetch means hashing package-local artifacts in the preflight; closing it for
  registration would contradict its documented "checks recorded identities
  without opening artifacts" contract.
- `docs/adr-chronicle-raw-microdata-identity.md` is cited 11 times across
  `chronicle/` and `docs/` and does not exist (pre-existing).

- The weakening audit found one real regression from fix 2 and it is fixed:
  `59869e9`. Validating archived objects made `_effective_recorded_digest`
  return `None` for an entry whose own locator is valid, which dropped the
  entry out of `validate_package_directory` and stopped inventory and
  publish-raw naming `filename_collision_across_manifests` /
  `sha256_collision_across_manifests` between two manifests (the tree stayed
  invalid through the new locator code, so nothing became acceptable). The
  digest now comes from the current object alone; unreadable history keeps its
  own refusal.
- Measured on real data: the sweep fires on 0 of the 195 tracked package
  directories under `db/`, on both the pre-fix and the fixed tree.

**Next:** the final full suite, Ruff lint and format checks, and the report.
