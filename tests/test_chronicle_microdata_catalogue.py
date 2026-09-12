"""Hermetic tests for ``scripts/register_microdata_releases.py``.

The catalogue is resolved against ``tests/fixtures/microcosm``: synthetic
consumer manifests in Microcosm's two shapes (staged ``source_stages.json``
files and the flat ACS runtime manifest) that carry the reviewed pins
verbatim beside decoy artifacts the selectors must ignore. Nothing here reads
a checkout outside the repository, and nothing skips: a resolution failure is
a failure.

``emit`` must reproduce the two committed hash-only manifests byte for byte
from the fixture, and ``plan`` must print exactly the golden commands in
``tests/fixtures/microcosm/golden_plan.json`` -- every reviewed identity as an
argument, every unknown as a ``TODO`` that the real CLI refuses to run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from chronicle.harness import main as harness_main


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "microcosm"
GOLDEN_PLAN = FIXTURE_ROOT / "golden_plan.json"
FRS_MANIFEST = REPO_ROOT / "db" / "data" / "dwp" / "frs_2023_24" / "manifest.yaml"
SPI_MANIFEST = (
    REPO_ROOT / "db" / "data" / "hmrc" / "spi_public_use_tape_2022_23" / "manifest.yaml"
)
UK_STAGES = "packages/microcosm-build/src/microcosm/build/uk/source_stages.json"
UK_HMRC_STAGES = (
    "packages/microcosm-build/src/microcosm/build/uk/hmrc_income_source_stages.json"
)
#: The commits the committed registrations were transcribed from: the last
#: commit that changed each consumer manifest.
PIN_COMMITS = {
    UK_STAGES: "2fb2e2f8a99c37725bd6e7a15ff4c2595c912b77",
    UK_HMRC_STAGES: "de7451bd19ca46d2967e73cdf393908d29e72542",
}
PIN_COMMIT_ARGS = [
    arg
    for path, commit in PIN_COMMITS.items()
    for arg in ("--microcosm-commit", f"{path}={commit}")
]

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import register_microdata_releases as script  # noqa: E402


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    exit_code = script.main(argv)
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


def _harness_parser(monkeypatch) -> argparse.ArgumentParser:
    """Capture the real ``chronicle`` harness parser."""
    captured: dict[str, argparse.ArgumentParser] = {}

    def capture(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with contextlib.suppress(SystemExit):
        harness_main(["--help"])
    monkeypatch.undo()
    return captured["parser"]


# --------------------------------------------------------------------------
# The fixture checkout
# --------------------------------------------------------------------------


def test_fixture_checkout_holds_every_catalogued_manifest():
    catalogued = sorted({release.manifest for release in script.CATALOGUE})
    assert catalogued
    for relative in catalogued:
        path = FIXTURE_ROOT / relative
        assert path.exists(), relative
        payload = json.loads(path.read_text())
        assert payload["snapshot_of"]["repository"] == "PolicyEngine/microcosm"


def test_selection_ignores_decoys_and_resolves_every_catalogue_entry():
    resolved = script.resolve(FIXTURE_ROOT, script.CATALOGUE)
    by_id = {item.release.release_id: item for item in resolved}
    frs = yaml.safe_load(FRS_MANIFEST.read_text())["files"][2023]
    committed = {entry["filename"]: entry for entry in frs}

    assert len(by_id) == len(script.CATALOGUE)
    for tab in script.FRS_TABS:
        item = by_id[f"dwp-frs-2023-24:{tab}"]
        expected = committed[f"{tab}.tab"]
        assert item.stage["stage"] == "frs_spine"
        assert item.artifact["kind"] == "licensed_microdata"
        assert (item.filename, item.sha256, item.size_bytes, item.vintage) == (
            f"{tab}.tab",
            expected["sha256"],
            expected["size_bytes"],
            expected["vintage"],
        )
    spi = by_id["hmrc-spi-public-use-tape-2022-23:put2223uk"]
    spi_committed = yaml.safe_load(SPI_MANIFEST.read_text())["files"][2022][0]
    assert (spi.filename, spi.sha256, spi.size_bytes, spi.vintage) == (
        "put2223uk.tab",
        spi_committed["sha256"],
        spi_committed["size_bytes"],
        spi_committed["vintage"],
    )
    assert spi.url is None
    silc = by_id["statbel-be-silc-2023"]
    assert (silc.sha256, silc.size_bytes, silc.filename) == (None, None, None)
    assert by_id["census-cps-basic-monthly-2024"].url is None
    assert by_id["census-sipp-2023"].url is None
    assert by_id["federal-reserve-scf-2022-full"].sha256 is None
    assert by_id["census-acs-pums-2024-household"].filename == "csv_hus.zip"


def test_selection_accepts_agreeing_duplicates_and_refuses_conflicts():
    payload = json.loads((FIXTURE_ROOT / UK_STAGES).read_text())
    selector = script.ArtifactSelector(
        kind="licensed_microdata", match={"table": "adult"}
    )
    stage, artifact = script.select_artifact(payload, selector, release_id="x")
    # adult.tab is listed by two stages with identical bytes; the first wins.
    assert stage["stage"] == "frs_spine"

    conflicting = json.loads(json.dumps(payload))
    conflicting["stages"][1]["artifacts"][0]["sha256"] = "f" * 64
    with pytest.raises(script.CatalogueError, match="conflicting values"):
        script.select_artifact(conflicting, selector, release_id="x")

    with pytest.raises(script.CatalogueError, match="no Microcosm artifact matches"):
        script.select_artifact(
            payload,
            script.ArtifactSelector(kind="licensed_microdata", match={"table": "nope"}),
            release_id="x",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("sha256", "f" * 64), ("kind", "restricted_microdata")],
    ids=("checksum", "access-kind"),
)
def test_frs_catalogue_selector_refuses_cross_stage_pin_drift(field, value):
    payload = json.loads((FIXTURE_ROOT / UK_STAGES).read_text())
    release = next(
        release
        for release in script.CATALOGUE
        if release.release_id == "dwp-frs-2023-24:adult"
    )
    employment = next(
        stage for stage in payload["stages"] if stage["stage"] == "frs_employment"
    )
    adult = next(
        artifact
        for artifact in employment["artifacts"]
        if artifact.get("table") == "adult"
    )
    adult[field] = value

    with pytest.raises(script.CatalogueError, match="conflicting values"):
        script.select_artifact(
            payload,
            release.selector,
            release_id=release.release_id,
        )


def test_frs_catalogue_selector_refuses_relabelled_cross_stage_byte_identity():
    payload = json.loads((FIXTURE_ROOT / UK_STAGES).read_text())
    release = next(
        release
        for release in script.CATALOGUE
        if release.release_id == "dwp-frs-2023-24:adult"
    )
    employment = next(
        stage for stage in payload["stages"] if stage["stage"] == "frs_employment"
    )
    adult = next(
        artifact
        for artifact in employment["artifacts"]
        if artifact.get("locator") == "adult.tab"
    )
    adult["table"] = "adult_relabelled"
    adult["kind"] = "restricted_microdata"

    with pytest.raises(script.CatalogueError, match="conflicting values"):
        script.select_artifact(
            payload,
            release.selector,
            release_id=release.release_id,
        )


def test_resolve_refuses_a_missing_consumer_manifest(tmp_path):
    with pytest.raises(script.CatalogueError, match="manifest not found"):
        script.resolve(tmp_path / "no-such-checkout", script.CATALOGUE[:1])


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------


def test_emit_from_the_fixture_reproduces_the_committed_manifests_byte_for_byte(
    tmp_path,
):
    root = tmp_path / "data"
    registrations, blockers = script.emit(
        script.resolve(FIXTURE_ROOT, script.CATALOGUE),
        root=root,
        pin_commits=PIN_COMMITS,
        consumer_repository=script.CONSUMER_REPOSITORY,
    )

    # The pure emitter receives verified commits and repository identity; CLI tests
    # below exercise the mandatory commit/blob verification itself.
    assert len(registrations) == 15
    assert [blocker["release"] for blocker in blockers] == ["statbel-be-silc-2023"]
    assert "No hash is invented" in blockers[0]["reason"]
    assert all(r["hash_source"] == "consumer_pin" for r in registrations)
    assert all(r["r2_location"] is None for r in registrations)
    written = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    assert written == [
        "dwp/frs_2023_24/manifest.yaml",
        "hmrc/spi_public_use_tape_2022_23/manifest.yaml",
    ]
    assert (root / "dwp/frs_2023_24/manifest.yaml").read_bytes() == (
        FRS_MANIFEST.read_bytes()
    )
    assert (root / "hmrc/spi_public_use_tape_2022_23/manifest.yaml").read_bytes() == (
        SPI_MANIFEST.read_bytes()
    )


def test_emit_is_idempotent_over_the_committed_manifests(tmp_path):
    root = tmp_path / "data"
    for manifest in (FRS_MANIFEST, SPI_MANIFEST):
        target = root / manifest.relative_to(REPO_ROOT / "db" / "data")
        target.parent.mkdir(parents=True)
        target.write_bytes(manifest.read_bytes())

    registrations, _blockers = script.emit(
        script.resolve(FIXTURE_ROOT, script.CATALOGUE),
        root=root,
        pin_commits=PIN_COMMITS,
        consumer_repository=script.CONSUMER_REPOSITORY,
    )

    assert all(r["replaced"] for r in registrations)
    assert (
        root / "dwp/frs_2023_24/manifest.yaml"
    ).read_bytes() == FRS_MANIFEST.read_bytes()


def test_emit_refuses_a_pin_that_drifted_from_the_committed_one(tmp_path):
    root = tmp_path / "data"
    target = root / "dwp" / "frs_2023_24" / "manifest.yaml"
    target.parent.mkdir(parents=True)
    target.write_bytes(FRS_MANIFEST.read_bytes())
    checkout = tmp_path / "consumer"
    for path in FIXTURE_ROOT.rglob("*.json"):
        copy = checkout / path.relative_to(FIXTURE_ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    stages = json.loads((checkout / UK_STAGES).read_text())
    for stage in stages["stages"]:
        for artifact in stage["artifacts"]:
            if artifact.get("table") == "adult":
                artifact["sha256"] = "f" * 64
    (checkout / UK_STAGES).write_text(json.dumps(stages))

    resolved = script.resolve(checkout, script.CATALOGUE)

    with pytest.raises(script.HashOnlyRegistrationError, match="--allow-reissue"):
        script.emit(
            resolved,
            root=root,
            pin_commits=PIN_COMMITS,
            consumer_repository=script.CONSUMER_REPOSITORY,
        )
    assert target.read_bytes() == FRS_MANIFEST.read_bytes()


def _fixture_copy(destination: Path) -> Path:
    for path in FIXTURE_ROOT.rglob("*.json"):
        copy = destination / path.relative_to(FIXTURE_ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    return destination


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _committed_fixture_checkout(
    destination: Path,
    *,
    origin: str = "https://github.com/PolicyEngine/microcosm.git",
) -> tuple[Path, str]:
    checkout = _fixture_copy(destination)
    _git(checkout, "init", "-q")
    _git(checkout, "remote", "add", "origin", origin)
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "t")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-q", "-m", "consumer pins")
    return checkout, _git(checkout, "rev-parse", "HEAD")


@pytest.mark.parametrize("explicit", [False, True], ids=("automatic", "explicit"))
def test_emit_refuses_an_unrelated_repository_with_matching_consumer_blobs(
    tmp_path, capsys, explicit
):
    origin = "https://github.com/unrelated/lookalike-consumer.git"
    checkout, commit = _committed_fixture_checkout(
        tmp_path / "unrelated", origin=origin
    )
    assert (
        _git(checkout, "cat-file", "blob", f"{commit}:{UK_STAGES}")
        == (checkout / UK_STAGES).read_text().strip()
    )
    root = tmp_path / "data"
    argv = [
        "--microcosm-root",
        str(checkout),
        "--root",
        str(root),
        "--release",
        "dwp-frs-2023-24:adult",
        "--json",
        "emit",
    ]
    if explicit:
        argv += ["--microcosm-commit", commit]

    exit_code, out, err = _run(argv, capsys)

    assert exit_code == 1, out
    assert "repository identity" in err
    assert "PolicyEngine/microcosm" in err
    assert "unrelated/lookalike-consumer" in err
    assert not root.exists()


@pytest.mark.parametrize("explicit", [False, True], ids=("automatic", "explicit"))
@pytest.mark.parametrize(
    "origin",
    [None, "https://gitlab.com/PolicyEngine/microcosm.git"],
    ids=("missing-origin", "different-host"),
)
def test_emit_refuses_a_missing_or_wrong_host_consumer_origin(
    tmp_path, capsys, explicit, origin
):
    checkout, commit = _committed_fixture_checkout(tmp_path / "consumer")
    if origin is None:
        _git(checkout, "remote", "remove", "origin")
    else:
        _git(checkout, "remote", "set-url", "origin", origin)
    root = tmp_path / "data"
    argv = [
        "--microcosm-root",
        str(checkout),
        "--root",
        str(root),
        "--release",
        "dwp-frs-2023-24:adult",
        "emit",
    ]
    if explicit:
        argv += ["--microcosm-commit", commit]

    exit_code, _out, err = _run(argv, capsys)

    assert exit_code == 1
    assert "repository identity" in err
    assert "PolicyEngine/microcosm" in err
    assert not root.exists()


@pytest.mark.parametrize("staged", [False, True], ids=("dirty", "staged"))
def test_emit_refuses_dirty_or_staged_consumer_manifest_bytes(tmp_path, capsys, staged):
    checkout, pinned = _committed_fixture_checkout(tmp_path / "consumer")
    manifest_path = checkout / UK_STAGES
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    if staged:
        _git(checkout, "add", UK_STAGES)
    root = tmp_path / "data"

    exit_code, _out, err = _run(
        [
            "--microcosm-root",
            str(checkout),
            "--root",
            str(root),
            "--release",
            "dwp-frs-2023-24:adult",
            "emit",
        ],
        capsys,
    )

    assert exit_code == 1
    assert "do not match" in err
    assert UK_STAGES in err
    assert pinned in err
    assert not root.exists()


def test_emit_refuses_an_explicit_commit_whose_blob_differs_from_loaded_manifest(
    tmp_path, capsys
):
    checkout, old_commit = _committed_fixture_checkout(tmp_path / "consumer")
    manifest_path = checkout / UK_STAGES
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    _git(checkout, "add", UK_STAGES)
    _git(checkout, "commit", "-q", "-m", "new consumer pin blob")
    assert _git(checkout, "rev-parse", "HEAD") != old_commit
    root = tmp_path / "data"

    exit_code, _out, err = _run(
        [
            "--microcosm-root",
            str(checkout),
            "--root",
            str(root),
            "--release",
            "dwp-frs-2023-24:adult",
            "emit",
            "--microcosm-commit",
            old_commit,
        ],
        capsys,
    )

    assert exit_code == 1
    assert "do not match" in err
    assert UK_STAGES in err
    assert old_commit in err
    assert not root.exists()


def test_emit_refuses_a_tree_object_as_an_explicit_commit(tmp_path, capsys):
    checkout, _commit = _committed_fixture_checkout(tmp_path / "consumer")
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    root = tmp_path / "data"

    exit_code, _out, err = _run(
        [
            "--microcosm-root",
            str(checkout),
            "--root",
            str(root),
            "--release",
            "dwp-frs-2023-24:adult",
            "emit",
            "--microcosm-commit",
            tree,
        ],
        capsys,
    )

    assert exit_code == 1
    assert "not a commit" in err
    assert tree in err
    assert not root.exists()


@pytest.mark.parametrize("explicit", [False, True], ids=("automatic", "explicit"))
@pytest.mark.parametrize(
    ("origin", "repository"),
    [
        ("https://github.com/PolicyEngine/microcosm.git", "PolicyEngine/microcosm"),
        ("https://github.com/PolicyEngine/microcosm/", "PolicyEngine/microcosm"),
        ("git@github.com:PolicyEngine/microcosm.git", "PolicyEngine/microcosm"),
        ("ssh://git@github.com/PolicyEngine/microcosm.git", "PolicyEngine/microcosm"),
        ("https://GITHUB.COM/policyengine/microcosm.git", "policyengine/microcosm"),
    ],
    ids=("https", "https-trailing-slash", "ssh-scp", "ssh-url", "case-normalized"),
)
def test_emit_accepts_a_commit_whose_blob_matches_the_loaded_manifest(
    tmp_path, capsys, explicit, origin, repository
):
    checkout, commit = _committed_fixture_checkout(tmp_path / "consumer", origin=origin)
    root = tmp_path / "data"
    argv = [
        "--microcosm-root",
        str(checkout),
        "--root",
        str(root),
        "--release",
        "dwp-frs-2023-24:adult",
        "--json",
        "emit",
    ]
    if explicit:
        argv += ["--microcosm-commit", commit]

    exit_code, out, err = _run(argv, capsys)

    assert exit_code == 0, err
    assert len(json.loads(out)["registrations"]) == 1
    manifest = yaml.safe_load((root / "dwp/frs_2023_24/manifest.yaml").read_text())
    entry = manifest["files"][2023][0]
    assert entry["pinned_from"] == {
        "repository": repository,
        "path": UK_STAGES,
        "commit": commit,
    }
    assert entry["attested_by"] == repository


def test_commit_validation_uses_the_snapshot_resolve_actually_parsed(tmp_path):
    checkout, commit = _committed_fixture_checkout(tmp_path / "consumer")
    manifest_path = checkout / UK_STAGES
    committed_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(committed_bytes + b"\n")
    release = next(
        release
        for release in script.CATALOGUE
        if release.release_id == "dwp-frs-2023-24:adult"
    )
    item = script.resolve(checkout, (release,))[0]
    # Restoring the worktree after resolution must not change which bytes are
    # verified: registration values came from the earlier in-memory snapshot.
    manifest_path.write_bytes(committed_bytes)

    with pytest.raises(script.CatalogueError, match="do not match"):
        script.assert_manifest_matches_commit(
            checkout,
            UK_STAGES,
            commit,
            loaded_bytes=item.manifest_bytes,
        )


@pytest.mark.parametrize("explicit", [False, True], ids=("automatic", "explicit"))
def test_emit_refuses_a_microcosm_root_nested_in_an_enclosing_repository(
    tmp_path, capsys, explicit
):
    repository = tmp_path / "repository"
    checkout = _fixture_copy(repository / "vendor" / "microcosm")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "t@example.com")
    _git(repository, "config", "user.name", "t")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "nested consumer checkout")
    commit = _git(repository, "rev-parse", "HEAD")
    root = tmp_path / "data"
    argv = [
        "--microcosm-root",
        str(checkout),
        "--root",
        str(root),
        "--release",
        "dwp-frs-2023-24:adult",
        "emit",
    ]
    if explicit:
        argv += ["--microcosm-commit", commit]

    exit_code, _out, err = _run(argv, capsys)

    assert exit_code == 1
    assert "Git repository root" in err
    assert str(repository) in err
    assert str(checkout) in err
    assert not root.exists()


def test_emit_needs_a_commit_it_can_read_or_be_told(tmp_path, capsys):
    # The fixture inside this repository is committed, so a run against it
    # would read a Chronicle commit as if it were the consumer's. Outside any
    # repository there is no commit to read; inside one whose manifests are
    # untracked there is none either. Both refuse before writing.
    outside = _fixture_copy(tmp_path / "outside-git")
    assert not (outside / ".git").exists()
    untracked = _fixture_copy(tmp_path / "untracked")
    _git(untracked, "init", "-q")
    _git(untracked, "config", "user.email", "t@example.com")
    _git(untracked, "config", "user.name", "t")
    (untracked / "README").write_text("nothing pinned here")
    _git(untracked, "add", "README")
    _git(untracked, "commit", "-q", "-m", "unrelated")

    for checkout in (outside, untracked):
        exit_code, _out, err = _run(
            [
                "--microcosm-root",
                str(checkout),
                "--root",
                str(tmp_path / "data"),
                "emit",
            ],
            capsys,
        )
        assert exit_code == 1
        assert "--microcosm-commit" in err
        assert not (tmp_path / "data").exists()

    exit_code, _out, err = _run(
        [
            "--microcosm-root",
            str(outside),
            "--root",
            str(tmp_path / "data"),
            "emit",
            "--microcosm-commit",
            "not-a-commit",
        ],
        capsys,
    )
    assert exit_code == 2
    assert "40-hex commit" in err
    assert not (tmp_path / "data").exists()


def test_emit_refuses_an_unreadable_explicit_commit_before_writing(tmp_path, capsys):
    outside = _fixture_copy(tmp_path / "outside-git")

    exit_code, _out, err = _run(
        [
            "--microcosm-root",
            str(outside),
            "--root",
            str(tmp_path / "data"),
            "--json",
            "emit",
            *PIN_COMMIT_ARGS,
        ],
        capsys,
    )

    assert exit_code == 1
    assert "Cannot read consumer manifest" in err
    assert not (tmp_path / "data").exists()


def test_pin_commit_reads_the_last_commit_that_changed_the_file(tmp_path):
    repo = tmp_path / "consumer"
    (repo / "build").mkdir(parents=True)
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "config", "user.email", "t@example.com"], check=True)
    subprocess.run([*git, "config", "user.name", "t"], check=True)
    (repo / "build" / "stages.json").write_text("{}")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "pin"], check=True)
    pinned = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "other.txt").write_text("unrelated")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "unrelated"], check=True)

    assert script.pin_commit(repo, "build/stages.json") == pinned
    with pytest.raises(script.CatalogueError, match="records no commit"):
        script.pin_commit(repo, "build/missing.json")


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


def test_plan_matches_the_golden_commands_exactly(capsys):
    exit_code, out, _err = _run(
        ["--microcosm-root", str(FIXTURE_ROOT), "--root", "db/data", "--json", "plan"],
        capsys,
    )
    golden = json.loads(GOLDEN_PLAN.read_text())

    assert exit_code == 0
    assert json.loads(out) == golden
    assert len(golden["commands"]) == 9


def test_every_planned_command_parses_with_the_reviewed_identity_as_arguments(
    monkeypatch,
):
    parser = _harness_parser(monkeypatch)
    golden = json.loads(GOLDEN_PLAN.read_text())

    for command in golden["commands"]:
        argv = shlex.split(command)
        assert argv[:3] == ["uv", "run", "chronicle"]
        namespace = parser.parse_args(argv[3:])
        assert namespace.command == "fetch-artifact"
        assert namespace.access == "public"
        assert namespace.kind == "microdata_release"
        assert namespace.licence == "US-Government-Work"
        assert namespace.publisher
        assert namespace.vintage
        assert namespace.expected_sha256
        assert namespace.licence_evidence_issuer
        assert namespace.licence_evidence_scope
        assert namespace.licence_evidence_url
        assert namespace.upload_r2 is True


def test_plan_prints_a_todo_and_never_a_guess(capsys):
    golden = json.loads(GOLDEN_PLAN.read_text())
    by_release = {
        shlex.split(command)[shlex.split(command).index("--package-id") + 1]: command
        for command in golden["commands"]
    }

    # Every reviewed publisher checksum travels as an argument.
    assert (
        "--expected-sha256 d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11"
        in by_release["census-cps-asec-2023"]
    )
    assert "--expected-size-bytes 150165063" in by_release["census-cps-asec-2023"]
    # Unknowns are TODOs the CLI refuses: no URL, no publisher-bytes checksum,
    # no catalogued evidence URL.
    assert "--url TODO_PUBLISHER_URL" in by_release["census-cps-basic-monthly-2024"]
    assert "--url TODO_PUBLISHER_URL" in by_release["census-sipp-2023"]
    for package_id in (
        "census-cps-basic-monthly-2024",
        "census-acs-pums-2022-1yr",
        "federal-reserve-scf-2022",
        "census-sipp-2023",
    ):
        assert any(
            "--expected-sha256 TODO_REVIEWED_SHA256" in command
            for package, command in by_release.items()
            if package == package_id
        )
    assert all("TODO_EVIDENCE_URL" in command for command in golden["commands"])
    assert not any("TODO_VINTAGE" in command for command in golden["commands"])
    # The derived-h5 hash Microcosm pins for ACS 2022 never masquerades as
    # the publisher checksum.
    assert (
        "0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4"
        not in "\n".join(golden["commands"])
    )


def test_a_planned_command_with_a_todo_is_refused_before_anything_is_read(
    tmp_path, monkeypatch, capsys
):
    golden = json.loads(GOLDEN_PLAN.read_text())
    reads: list[str] = []

    def unexpected_read(source_url):
        reads.append(source_url)
        raise AssertionError("a planned command with a TODO must not read")

    monkeypatch.setattr("chronicle.artifacts._read_artifact", unexpected_read)
    monkeypatch.setattr(
        "chronicle.artifacts._upload_r2_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no upload")),
    )

    for command in golden["commands"]:
        argv = shlex.split(command)[3:]
        out_index = argv.index("--out-dir") + 1
        argv[out_index] = str(tmp_path / argv[out_index])
        exit_code = harness_main([*argv, "--staging-dir", str(tmp_path / "staging")])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert captured.out == ""
        assert "error:" in captured.err

    assert reads == []
    assert not (tmp_path / "db").exists()
    assert not (tmp_path / "staging").exists()
