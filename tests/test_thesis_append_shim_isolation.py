"""Exercise receipt's released commit-addressed gate through the Chronicle shim.

The shim supplies a full candidate OID, freezes the caller's git environment,
and prints the OIDs returned by the package. The installed wheel reads the
named commit's objects, so HEAD, the index and workspace may disagree without
changing its verdict. These subprocess tests cover all three workflow paths,
committed refusals, environment/configuration boundaries and scratch cleanup.

The former exact-checkout tests are removed where their subject was a private
worktree, its index, byte comparison or registration: the 0.6 shim creates none.
Cases for committed unsafe entries and caller configuration remain because the
package now enforces those refusals itself.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
from importlib import metadata

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIM_SCRIPTS = ROOT / "scripts"
SHIM = SHIM_SCRIPTS / "check_thesis_facts_append.py"

RELEASE_FILE_SUFFIXES = (
    ".json",
    ".producer.sig",
    ".freetsa.tsr",
    ".digicert.tsr",
)
LEDGER_RELATIVE = "ledger/official_observations.jsonl"

REFUSED = "thesis-facts append check refused: "
FAILED = "thesis-facts append check failed: "

CANDIDATE_LINE = re.compile(
    r"(?m)^candidate commit (?P<commit>[0-9a-f]{40,64}) "
    r"tree (?P<tree>[0-9a-f]{40,64})"
    r"(?: base commit (?P<base>[0-9a-f]{40,64}) tree (?P<base_tree>[0-9a-f]{40,64}))?$"
)


def _subject_line(clone: pathlib.Path, candidate: str, base: str | None) -> str:
    """The shim's second line: the candidate pair, and the base pair when given."""

    line = f"candidate commit {candidate} tree {_git(clone, 'rev-parse', candidate + '^{tree}')}"
    if base is not None:
        line += f" base commit {base} tree {_git(clone, 'rev-parse', base + '^{tree}')}"
    return line

# A commit id of the right shape that no repository holds.
ABSENT_OBJECT_ID = "0" * 40


@pytest.fixture(scope="module", autouse=True)
def _released_receipt_wheel():
    """Every subprocess uses the released installed distribution, not a checkout."""

    import receipt.append_gate

    distribution = metadata.distribution("receipt")
    assert distribution.version == "0.6.1"
    assert distribution.read_text("WHEEL") is not None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        source = json.loads(direct_url)
        assert "archive_info" in source
        assert source["url"].endswith("receipt-0.6.1-py3-none-any.whl")
    installed = pathlib.Path(distribution.locate_file("")).resolve()
    assert (
        pathlib.Path(receipt.append_gate.__file__).resolve().is_relative_to(installed)
    )


def _release_manifests() -> list[pathlib.Path]:
    return sorted((ROOT / "releases" / "manifests").glob("[0-9]" * 4 + "-*.json"))


_HEAD_RELEASE = json.loads(_release_manifests()[-1].read_text(encoding="utf-8"))
NEW_RELEASE_STEM = _release_manifests()[-1].stem
RELEASE_INDEX = int(_HEAD_RELEASE["releaseIndex"])
CANDIDATE_LINE_COUNT = int(_HEAD_RELEASE["state"]["lineCount"])
BASE_LINE_COUNT = int(_HEAD_RELEASE["append"]["previousLineCount"])
APPENDED_ROW_COUNT = int(_HEAD_RELEASE["append"]["appendedRowCount"])
PREFIX_LINE_COUNT = int(
    json.loads((ROOT / "ledger" / "immutable_prefix.json").read_text(encoding="utf-8"))[
        "prefixLineCount"
    ]
)
APPEND_GATE_OK = (
    f"thesis-facts append check OK: {CANDIDATE_LINE_COUNT} rows, "
    f"immutable prefix {PREFIX_LINE_COUNT}, "
    f"+{APPENDED_ROW_COUNT} appended vs base, release {RELEASE_INDEX}"
)


def _git(root: pathlib.Path, *arguments: str) -> str:
    """Run git in a fixture repository, ignoring the machine's own settings."""

    # Every GIT_-prefixed name goes, as in the shim: GIT_CONFIG_COUNT and its
    # numbered channels outrank every file, and GIT_DIR or GIT_WORK_TREE would
    # point the fixture at another repository (peer review, round 2).
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _copy_custody_tree(destination: pathlib.Path) -> pathlib.Path:
    root = destination / "root"
    shutil.copytree(ROOT / "ledger", root / "ledger")
    shutil.copytree(ROOT / "releases", root / "releases")
    return root


def _initialize(root: pathlib.Path) -> None:
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "shim-isolation@example.invalid")
    _git(root, "config", "user.name", "Shim Isolation")


def _commit(root: pathlib.Path, message: str, index_mutation=None) -> str:
    """Commit the working tree, and any entry only the index can carry.

    ``index_mutation`` runs between the add and the commit, because an entry
    with no file behind it -- a gitlink, say -- would be staged for deletion by
    the add if it were written first.
    """

    _git(root, "add", "-A")
    if index_mutation is not None:
        index_mutation(root)
    _git(root, "commit", "--quiet", "--allow-empty", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _release_file(root: pathlib.Path, suffix: str) -> pathlib.Path:
    return root / "releases" / "manifests" / f"{NEW_RELEASE_STEM}{suffix}"


def _replay_latest_release(
    destination: pathlib.Path,
    *,
    prepare=None,
    mutate=None,
    mutate_index=None,
) -> tuple[pathlib.Path, str, str]:
    """Build the prior release as base and the witnessed append as candidate.

    ``prepare`` runs before the base commit, so whatever it writes is in both
    commits and is therefore not part of the diff the gate judges. ``mutate``
    runs after the release files are restored and before the candidate commit,
    so whatever it writes is part of the candidate and of nothing else.
    ``mutate_index`` does the same for an entry that exists only in the index.
    """

    root = _copy_custody_tree(destination)
    ledger = root / LEDGER_RELATIVE
    full_ledger = ledger.read_bytes()
    rows = full_ledger.splitlines(keepends=True)
    assert len(rows) == CANDIDATE_LINE_COUNT
    ledger.write_bytes(b"".join(rows[:BASE_LINE_COUNT]))
    for suffix in RELEASE_FILE_SUFFIXES:
        _release_file(root, suffix).unlink()

    _initialize(root)
    if prepare is not None:
        prepare(root)
    base = _commit(root, "release base")

    ledger.write_bytes(full_ledger)
    for suffix in RELEASE_FILE_SUFFIXES:
        shutil.copyfile(_release_file(ROOT, suffix), _release_file(root, suffix))
    if mutate is not None:
        mutate(root)
    return root, base, _commit(root, "witnessed append", mutate_index)


def _replay_current_state(destination: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Commit the repository's present custody tree as the base to diff against.

    Used by the cases that ask what the gate says about a change made on top of
    the state as it stands, rather than about the last witnessed release.
    """

    root = _copy_custody_tree(destination)
    _initialize(root)
    return root, _commit(root, "current state")


DRIVER = """\
import subprocess
import sys

sys.path.insert(0, {scripts!r})

import check_thesis_facts_append as shim  # noqa: E402

{injection}

raise SystemExit(shim.main())
"""


def _run_shim(
    clone: pathlib.Path,
    *,
    commit: str,
    base_ref: str | None = None,
    temporary_root: pathlib.Path,
    injection: str | None = None,
    workspace: pathlib.Path | None = None,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the shim over ``clone`` with its scratch directories under our eye.

    ``temporary_root`` becomes the child's ``TMPDIR``, so the private directory
    the shim creates lands somewhere a test can look at afterwards and assert
    that nothing was left behind.
    """

    temporary_root.mkdir(parents=True, exist_ok=True)
    if injection is None:
        script = SHIM
    else:
        assert workspace is not None
        workspace.mkdir(parents=True, exist_ok=True)
        script = workspace / "driver.py"
        script.write_text(
            DRIVER.format(scripts=str(SHIM_SCRIPTS), injection=injection),
            encoding="utf-8",
        )

    child = os.environ.copy() if environment is None else dict(environment)
    child["TMPDIR"] = str(temporary_root)
    command = [
        sys.executable,
        str(script),
        "--root",
        str(clone),
        "--commit",
        commit,
    ]
    if base_ref is not None:
        command.extend(["--base-ref", base_ref])
    return subprocess.run(
        command,
        cwd=clone,
        env=child,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_nothing_left_behind(
    clone: pathlib.Path, temporary_root: pathlib.Path
) -> None:
    """No scratch directory survives, and no registration names one."""

    leftovers = sorted(temporary_root.iterdir())
    assert leftovers == [], leftovers
    listing = _git(clone, "worktree", "list", "--porcelain")
    assert "thesis-facts-gate-" not in listing, listing


def test_checkout_line_endings_do_not_change_the_committed_verdict(tmp_path):
    """Raw committed blobs stay authoritative when eol rewrites the workspace."""

    def _convert_the_ledger_to_crlf(root: pathlib.Path) -> None:
        (root / ".gitattributes").write_text(
            f"{LEDGER_RELATIVE} text eol=crlf\n", encoding="utf-8"
        )

    clone, base, candidate = _replay_latest_release(
        tmp_path, prepare=_convert_the_ledger_to_crlf
    )
    tree = _git(clone, "rev-parse", f"{candidate}^{{tree}}")
    ledger = clone / LEDGER_RELATIVE
    ledger.write_bytes(ledger.read_bytes().replace(b"\n", b"\r\n"))
    assert _git(clone, "hash-object", "--no-filters", LEDGER_RELATIVE) != _git(
        clone, "rev-parse", f"{candidate}:{LEDGER_RELATIVE}"
    )

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == (
        f"{APPEND_GATE_OK}\n{_subject_line(clone, candidate, base)}\n"
    )
    assert tree in completed.stdout
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def _commit_a_symlink_under_releases(root: pathlib.Path) -> None:
    link = root / "releases" / "manifests" / "0000-shortcut.json"
    link.symlink_to(pathlib.Path("..") / ".." / "ledger" / "immutable_prefix.json")


def _stage_a_gitlink_under_releases(root: pathlib.Path) -> None:
    """Record a submodule boundary under the release root, in the index only.

    A gitlink names a commit in another repository. Whatever a checkout puts at
    that path belongs to that repository and is no part of this commit, so
    nothing under it can be compared against anything this commit fixes.
    """

    head = _git(root, "rev-parse", "HEAD")
    _git(
        root,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{head},releases/manifests/0000-elsewhere",
    )


@pytest.mark.parametrize(
    ("case", "mutate", "mutate_index", "path", "mode"),
    [
        (
            "symlink",
            _commit_a_symlink_under_releases,
            None,
            "releases/manifests/0000-shortcut.json",
            "120000",
        ),
        (
            "gitlink",
            None,
            _stage_a_gitlink_under_releases,
            "releases/manifests/0000-elsewhere",
            "160000",
        ),
    ],
)
def test_a_protected_path_that_is_not_a_file_is_refused(
    tmp_path, case, mutate, mutate_index, path, mode
):
    """The package refuses committed links and gitlinks without following them."""

    clone, base, candidate = _replay_latest_release(
        tmp_path / case, mutate=mutate, mutate_index=mutate_index
    )

    assert _git(clone, "ls-tree", candidate, "--", path).startswith(mode)

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert FAILED in completed.stderr
    assert path in completed.stderr
    assert ("symlink" if mode == "120000" else "not regular") in completed.stderr
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def test_a_symlink_materialised_as_a_file_is_still_refused(tmp_path):
    """A repaired regular workspace file cannot hide a committed symlink."""

    clone, base, candidate = _replay_latest_release(
        tmp_path, mutate=_commit_a_symlink_under_releases
    )
    _git(clone, "config", "core.symlinks", "false")
    link = clone / "releases" / "manifests" / "0000-shortcut.json"
    target = os.readlink(link)
    link.unlink()
    link.write_text(target, encoding="utf-8")
    assert link.is_file() and not link.is_symlink()

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert FAILED in completed.stderr
    assert "symlink" in completed.stderr
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def test_an_absent_candidate_object_is_refused_without_a_verdict(tmp_path):
    """An unavailable named object fails closed and leaves no scratch state."""

    clone, _base, _candidate = _replay_latest_release(tmp_path)
    before = _git(clone, "worktree", "list", "--porcelain")

    completed = _run_shim(
        clone,
        commit=ABSENT_OBJECT_ID,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert FAILED in completed.stderr
    assert ABSENT_OBJECT_ID in completed.stderr
    assert _git(clone, "worktree", "list", "--porcelain") == before
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


# Braces are everywhere in this one, so the recording path is spliced in by
# name rather than by str.format.
ENVIRONMENT_SPY = """\
import json
import os
import pathlib
import sys as _sys

RECORDING = __RECORDING__

_real_run = subprocess.run
_real_popen = subprocess.Popen
_records = []


def _record(args, kwargs, caller):
    argv = args[0] if args else kwargs.get("args")
    given = kwargs.get("env")
    effective = dict(given) if given is not None else dict(os.environ)
    _records.append(
        {
            "argv": [str(item) for item in argv],
            "caller": caller,
            "explicit": given is not None,
            "git": {
                name: value
                for name, value in effective.items()
                if name.startswith("GIT_")
            },
        }
    )


def _spy_run(*args, **kwargs):
    _record(args, kwargs, _sys._getframe(1).f_globals.get("__name__", "?"))
    return _real_run(*args, **kwargs)


def _spy_popen(*args, **kwargs):
    _record(args, kwargs, _sys._getframe(1).f_globals.get("__name__", "?"))
    return _real_popen(*args, **kwargs)


subprocess.run = _spy_run
subprocess.Popen = _spy_popen


def _record_and_exit(code):
    pathlib.Path(RECORDING).write_text(json.dumps(_records), encoding="utf-8")
    return code


_real_main = shim.main
shim.main = lambda: _record_and_exit(_real_main())
"""


def test_no_inherited_git_variable_reaches_any_child(tmp_path):
    """No inherited GIT_ value reaches shim calls or the package's batch child.

    The shim drops all names by prefix; receipt then freezes its own object
    reader environment. Recording both run and Popen includes the long-lived
    cat-file child, not just discovery/configuration calls.
    """

    clone, base, candidate = _replay_latest_release(tmp_path)
    recording = tmp_path / "subprocess-environments.json"

    module = SHIM.read_text(encoding="utf-8")
    documented = re.search(r"There are (\d+) of them\.", module)
    assert documented is not None, "the tuple's comment no longer gives a count"

    hostile = os.environ.copy()
    for name in _documented_git_variables():
        hostile[name] = f"hostile-{name}"
    hostile["GIT_CONFIG_COUNT"] = "1"
    hostile["GIT_CONFIG_KEY_0"] = "core.hooksPath"
    hostile["GIT_CONFIG_VALUE_0"] = str(tmp_path / "hostile-hooks")

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
        injection=ENVIRONMENT_SPY.replace("__RECORDING__", repr(str(recording))),
        workspace=tmp_path / "driver",
        environment=hostile,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = json.loads(recording.read_text(encoding="utf-8"))
    git_calls = [record for record in records if record["argv"][0] == "git"]
    assert git_calls, records

    by_shim = [
        record
        for record in git_calls
        if record["caller"] == "check_thesis_facts_append"
    ]
    by_gate = [
        record for record in git_calls if record["caller"].startswith("receipt.")
    ]
    assert by_shim, [record["caller"] for record in git_calls]
    assert by_gate, [record["caller"] for record in git_calls]
    assert any("cat-file" in record["argv"] for record in by_gate)
    assert all("worktree" not in record["argv"] for record in git_calls)
    shim_globals = {record["git"]["GIT_CONFIG_GLOBAL"] for record in by_shim}
    assert all(
        record["git"]["GIT_CONFIG_GLOBAL"] not in shim_globals for record in by_gate
    )

    for record in git_calls:
        assert set(record["git"]) == {
            "GIT_NO_REPLACE_OBJECTS",
            "GIT_CONFIG_NOSYSTEM",
            "GIT_CONFIG_GLOBAL",
        }, record
        assert record["git"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert record["git"]["GIT_CONFIG_NOSYSTEM"] == "1"
        assert not record["git"]["GIT_CONFIG_GLOBAL"].startswith("hostile-")


def _documented_git_variables() -> tuple[str, ...]:
    sys.path.insert(0, str(SHIM_SCRIPTS))
    from check_thesis_facts_append import DOCUMENTED_GIT_VARIABLES

    return DOCUMENTED_GIT_VARIABLES


def test_the_documented_variable_list_is_not_narrower_than_the_drop():
    """The prefix rule covers every name the tuple records, and the count holds.

    The shim drops by prefix, so this is a check on the documentation rather
    than on the code: if a future git documents a GIT_ variable the prefix rule
    would miss, there is no such name and this says so; if the tuple and the
    comment beside it disagree about how many names it holds, that is a stale
    comment and this says that too.
    """

    documented = _documented_git_variables()
    assert len(set(documented)) == len(documented)
    assert sorted(documented) == list(documented)
    for name in documented:
        assert name.startswith("GIT_"), name

    module = SHIM.read_text(encoding="utf-8")
    stated = re.search(r"There are (\d+) of them\.", module)
    assert stated is not None
    assert int(stated.group(1)) == len(documented)


def test_a_post_checkout_hook_in_the_clone_does_not_run(tmp_path):
    """Object reads never invoke the clone's executable post-checkout hook."""

    clone, base, candidate = _replay_latest_release(tmp_path)
    marker = tmp_path / "the-hook-ran"
    hook = clone / ".git" / "hooks" / "post-checkout"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        f"#!/bin/sh\nprintf 'ran' > {marker}\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not marker.exists()
    assert APPEND_GATE_OK in completed.stdout
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


@pytest.mark.parametrize(
    ("key", "value", "reported"),
    [
        ("filter.smuggle.clean", "cat", "filter.smuggle.clean"),
        ("core.hooksPath", "hooks", "core.hookspath"),
        ("core.fsmonitor", "true", "core.fsmonitor"),
        ("include.path", "extra-config", "include.path"),
    ],
)
def test_repository_configuration_redirects_are_refused(tmp_path, key, value, reported):
    """The package audits hooks, fsmonitor and includes; the shim keeps filters."""

    clone, _base, candidate = _replay_latest_release(tmp_path)
    _git(clone, "config", key, value)
    before = _git(clone, "worktree", "list", "--porcelain")

    completed = _run_shim(
        clone,
        commit=candidate,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert (REFUSED if key.startswith("filter.") else FAILED) in completed.stderr
    assert reported in completed.stderr.lower()
    assert _git(clone, "worktree", "list", "--porcelain") == before
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


@pytest.mark.parametrize(
    "spelling",
    ["HEAD", "main", "codex/thesis-ledger-facts", "3a5ef7e", "A" * 40, ""],
)
def test_a_commit_that_is_not_a_full_object_id_is_refused_by_the_parser(
    tmp_path, spelling
):
    """A name the candidate can move, or an id short enough to be ambiguous.

    The parser refuses these, so the refusal costs nothing -- no directory is
    made, no git command runs -- and it is argparse's own exit status 2 rather
    than the gate's 1, because it is a usage error and not a verdict.
    """

    completed = subprocess.run(
        [
            sys.executable,
            str(SHIM),
            "--root",
            str(tmp_path),
            "--commit",
            spelling,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert "argument --commit" in completed.stderr
    assert "a full object id is required" in completed.stderr


def test_a_commit_argument_is_required(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SHIM), "--root", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "--commit" in completed.stderr
    assert "required" in completed.stderr


def test_an_exception_during_verification_removes_the_frozen_configuration(tmp_path):
    """The cleanup is in a `finally`, so it does not need the run to succeed.

    The gate is replaced by something that raises, which stands for every way
    the run can end that is neither a verdict nor a refusal.
    """

    clone, base, candidate = _replay_latest_release(tmp_path)
    injection = """\
def _explode(*args, **kwargs):
    raise RuntimeError("the gate did not get to finish")


shim.verify_append_gate = _explode
"""
    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
        injection=injection,
        workspace=tmp_path / "driver",
    )

    assert completed.returncode != 0
    assert "RuntimeError: the gate did not get to finish" in completed.stderr
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


WORKFLOW_PATHS = ("pull_request_target", "candidate", "push")


def _workflow_arguments(clone, workflow, base, candidate):
    """Select the full OID exactly where each workflow obtains its argument."""

    if workflow == "pull_request_target":
        # GitHub's test merge has the accepted candidate tree and two parents.
        merge = _git(
            clone,
            "commit-tree",
            f"{candidate}^{{tree}}",
            "-p",
            base,
            "-p",
            candidate,
            "-m",
            "synthetic pull request merge",
        )
        environment = {"MERGE_SHA": merge, "BASE_SHA": base}
        return environment["MERGE_SHA"], environment["BASE_SHA"]
    if workflow == "candidate":
        return _git(clone, "rev-parse", "HEAD"), base
    assert workflow == "push"
    environment = {"GITHUB_SHA": candidate}
    return environment["GITHUB_SHA"], None


@pytest.mark.parametrize("workflow", WORKFLOW_PATHS)
def test_every_workflow_verifies_the_named_commit_when_local_state_disagrees(
    tmp_path, workflow
):
    """Each installed-wheel path prints its selected OIDs despite other state."""

    clone, base, candidate = _replay_latest_release(tmp_path)
    selected, base_ref = _workflow_arguments(clone, workflow, base, candidate)
    tree = _git(clone, "rev-parse", f"{selected}^{{tree}}")
    ledger = clone / LEDGER_RELATIVE
    committed_blob = _git(clone, "rev-parse", f"{selected}:{LEDGER_RELATIVE}")

    # HEAD, the index and the workspace each disagree with the selected tree.
    _git(clone, "update-ref", "HEAD", base)
    ledger.write_bytes(b"{}\n")
    _git(clone, "add", LEDGER_RELATIVE)
    ledger.write_bytes(b"[]\n")
    ledger.chmod(0o755)
    (clone / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    (clone / ".git" / "info" / "exclude").write_text("ignored.txt\n", encoding="utf-8")
    (clone / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    _git(clone, "config", "status.showUntrackedFiles", "no")
    assert _git(clone, "rev-parse", "HEAD") != selected
    assert _git(clone, "rev-parse", f":{LEDGER_RELATIVE}") != committed_blob
    assert _git(clone, "hash-object", "--no-filters", LEDGER_RELATIVE) != committed_blob

    completed = _run_shim(
        clone, commit=selected, base_ref=base_ref, temporary_root=tmp_path / "tmp"
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    lines = completed.stdout.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("thesis-facts append check OK:")
    assert lines[1] == _subject_line(clone, selected, base_ref)
    assert tree in lines[1]
    if base_ref is not None:
        assert lines[0] == APPEND_GATE_OK
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


@pytest.mark.parametrize("workflow", WORKFLOW_PATHS)
def test_every_workflow_refuses_bad_committed_data_after_the_workspace_is_repaired(
    tmp_path, workflow
):
    """Repairing HEAD, index and disk cannot turn a refused candidate into PASS."""

    clone, base = _replay_current_state(tmp_path)
    ledger = clone / LEDGER_RELATIVE
    trusted = ledger.read_bytes()
    _drop_the_last_row(clone)
    candidate = _commit(clone, "truncate the committed ledger")
    selected, base_ref = _workflow_arguments(clone, workflow, base, candidate)
    before = _run_shim(
        clone, commit=selected, base_ref=base_ref, temporary_root=tmp_path / "tmp"
    )
    assert before.returncode == 1, before.stdout + before.stderr
    assert before.stdout == ""
    assert FAILED in before.stderr

    _git(clone, "update-ref", "HEAD", base)
    ledger.write_bytes(trusted)
    _git(clone, "add", LEDGER_RELATIVE)
    assert _git(clone, "rev-parse", "HEAD") != selected
    assert _git(clone, "rev-parse", f":{LEDGER_RELATIVE}") == _git(
        clone, "rev-parse", f"{base}:{LEDGER_RELATIVE}"
    )
    completed = _run_shim(
        clone, commit=selected, base_ref=base_ref, temporary_root=tmp_path / "tmp"
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert FAILED in completed.stderr
    if base_ref is not None:
        assert "change truncates the ledger" in completed.stderr
        assert completed.stderr == before.stderr
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def test_the_witnessed_release_passes_and_names_its_commit(tmp_path):
    """The accepting case, end to end, over a committed replay of the last
    release.

    stdout is the gate's own summary line and then one line the shim adds,
    naming the candidate commit and tree returned by the installed package.
    """

    clone, base, candidate = _replay_latest_release(tmp_path)
    tree = _git(clone, "rev-parse", f"{candidate}^{{tree}}")

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    assert completed.stdout == (
        f"{APPEND_GATE_OK}\n{_subject_line(clone, candidate, base)}\n"
    )
    match = CANDIDATE_LINE.search(completed.stdout)
    assert match is not None
    assert match.group("commit") == candidate
    assert match.group("tree") == tree
    assert match.group("base") == base
    assert match.group("base_tree") == _git(clone, "rev-parse", f"{base}^{{tree}}")
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def _rewrite_the_first_unfrozen_row(root: pathlib.Path) -> None:
    ledger = root / LEDGER_RELATIVE
    rows = ledger.read_bytes().splitlines(keepends=True)
    rows[PREFIX_LINE_COUNT] = b" " + rows[PREFIX_LINE_COUNT]
    ledger.write_bytes(b"".join(rows))


def _drop_the_last_row(root: pathlib.Path) -> None:
    ledger = root / LEDGER_RELATIVE
    rows = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(rows[:-1]))


@pytest.mark.parametrize(
    ("case", "mutation", "marker"),
    [
        (
            "rewrite",
            _rewrite_the_first_unfrozen_row,
            f"change rewrites existing line {PREFIX_LINE_COUNT + 1}",
        ),
        (
            "truncate",
            _drop_the_last_row,
            (
                f"change truncates the ledger: {CANDIDATE_LINE_COUNT} -> "
                f"{CANDIDATE_LINE_COUNT - 1} rows"
            ),
        ),
    ],
)
def test_the_append_only_diff_is_still_enforced_end_to_end(
    tmp_path, case, mutation, marker
):
    """A rewritten historical line and a truncated ledger, through the shim.

    These two outcomes used to be asserted by calling the gate's append-only
    check directly with a list of lines. They are asserted here the way the
    gate is actually used: two commits in a repository, the second judged
    against the first, with the refusal read off the process's stderr. The
    refusal is the gate's, spelled "failed", which is also how these cases say
    that the shim handed the gate a tree rather than refusing it one.
    """

    clone, base = _replay_current_state(tmp_path / case)
    mutation(clone)
    candidate = _commit(clone, f"candidate: {case}")

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert f"{FAILED}" in completed.stderr
    assert marker in completed.stderr
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def test_a_true_append_is_accepted_end_to_end(tmp_path):
    """And the third outcome: a real append is accepted.

    The append the old unit test accepted was any whole line added to the end.
    End to end an append also has to carry its witnessed release, so what is
    accepted here is the last real release of this ledger, replayed against the
    one before it.
    """

    clone, base, candidate = _replay_latest_release(tmp_path)

    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"+{APPENDED_ROW_COUNT} appended vs base" in completed.stdout
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def test_the_scratch_directory_is_private_to_the_run(tmp_path):
    """The private directory is created 0700, and the shim asserts that it was.

    The shim keeps its frozen global safe.directory configuration under this
    private directory and removes it when verification ends.
    """

    clone, base, candidate = _replay_latest_release(tmp_path)
    observed = tmp_path / "observed-mode.txt"
    injection = f"""\
import pathlib
import stat

_real_scratch = shim._scratch_root


def _record_the_mode():
    root = _real_scratch()
    pathlib.Path({str(observed)!r}).write_text(
        oct(stat.S_IMODE(root.stat().st_mode)), encoding="utf-8"
    )
    return root


shim._scratch_root = _record_the_mode
"""
    completed = _run_shim(
        clone,
        commit=candidate,
        base_ref=base,
        temporary_root=tmp_path / "tmp",
        injection=injection,
        workspace=tmp_path / "driver",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert observed.read_text(encoding="utf-8") == oct(stat.S_IRWXU)
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def _sparse_patterns(root: pathlib.Path, *patterns: str) -> None:
    info = root / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "sparse-checkout").write_text(
        "".join(f"{p}\n" for p in patterns), encoding="utf-8"
    )


@pytest.mark.parametrize("scope", ["local", "worktree"])
def test_the_shim_retains_its_sparse_checkout_policy(tmp_path, scope):
    """The caller retains sparse-configuration refusals receipt does not audit."""

    def prepare(root: pathlib.Path) -> None:
        (root / "notes.txt").write_text("unprotected, tracked\n", encoding="utf-8")

    clone, base, candidate = _replay_latest_release(tmp_path, prepare=prepare)
    _sparse_patterns(clone, "/ledger/", "/releases/")
    if scope == "worktree":
        _git(clone, "config", "extensions.worktreeConfig", "true")
        _git(clone, "config", "--worktree", "core.sparseCheckout", "true")
    else:
        _git(clone, "config", "core.sparseCheckout", "true")

    completed = _run_shim(clone, commit=candidate, temporary_root=tmp_path / "tmp")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith(REFUSED)
    assert "core.sparsecheckout" in completed.stderr.lower()
    _assert_nothing_left_behind(clone, tmp_path / "tmp")


def test_the_documented_variable_list_covers_the_installed_manual(tmp_path):
    """The tuple is held to git's own documentation where it is installed.

    Every ``GIT_`` name the installed ``git(1)`` manual page mentions must be
    in the tuple; the prefix rule drops them all regardless, so this is a
    check that the list has not fallen behind the git on the machine. Skips,
    never passes, where the manual page cannot be found.
    """

    import gzip

    completed = subprocess.run(
        ["git", "--man-path"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        pytest.skip("git --man-path is unavailable")
    man = pathlib.Path(completed.stdout.strip()) / "man1"
    candidates = [man / "git.1", man / "git.1.gz"]
    page = next((c for c in candidates if c.exists()), None)
    if page is None:
        pytest.skip(f"git(1) manual page not installed under {man}")
    raw = (
        gzip.decompress(page.read_bytes())
        if page.suffix == ".gz"
        else page.read_bytes()
    )
    text = raw.decode("utf-8", "replace").replace("\\-", "-").replace("\\_", "_")
    # roff writes most names as \\fBGIT_DIR\\fR, so a leading word boundary
    # would miss them (peer review, round 2); the class itself ends the match.
    mentioned = set(re.findall(r"GIT_[A-Z0-9_]+\b", text))
    mentioned.discard("GIT_")
    documented = set(_documented_git_variables())
    missing = sorted(
        name for name in mentioned if name not in documented and not name.endswith("_")
    )
    assert missing == [], f"documented in git(1) but not in the tuple: {missing}"
