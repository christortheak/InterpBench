"""Authoring parity means preserved input bytes and the same admission gates."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from steerlab_server import client_cli, cli_envelope
from steerlab_server.client import authoring_files, study_inputs, study_packs
from steerlab_server.experiment import experiment_store as store, manifest_files, task_inputs
from steerlab_server.experiment.manifest_errors import ExperimentStoreError

FIXTURE = Path(__file__).parent / "fixtures/study-assembly/pack.json"


def pack(name="shared-study"):
    document = json.loads(FIXTURE.read_bytes())
    document["study"]["name"] = name
    return json.dumps(document).encode()


def apply(root, data=None):
    data = data or pack()
    read = study_packs.preview(data, root=root)
    return study_packs.apply(data, root=root, expected=read["reviewSHA256"])


def test_shared_pack_preserves_fields_and_actual_pins(tmp_path):
    raw = FIXTURE.read_bytes()
    review = study_packs.preview(raw, root=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert review["files"][0]["disposition"] == "create"
    result = study_packs.apply(raw, root=tmp_path, expected=review["reviewSHA256"])
    study = result["study"]["document"]
    assert study["status"] == "draft"
    assert not set(study_packs.FREEZE_FIELDS) & study.keys()
    assert study["conditions"] == json.loads(raw)["study"]["conditions"]
    path = tmp_path / study["taskPromptsFile"]
    assert path.read_text() == json.loads(raw)["files"][study["taskPromptsFile"]]
    assert study["taskPromptsHash"] == manifest_files.digest_bytes(path.read_bytes())
    assert "revision" not in study and "manifestFileSHA256" not in study
    assert result["study"]["manifestFileSHA256"] == manifest_files.file_digest(
        str(tmp_path / "experiments/shared-study/experiment.json"))
    exported = study_packs.export("shared-study", root=tmp_path)
    assert exported["pack"]["files"][study["taskPromptsFile"]] == path.read_text()
    exported["pack"]["study"]["name"] = "copy"
    copy = apply(tmp_path, json.dumps(exported["pack"]).encode())
    assert copy["filesWritten"] == []
    assert copy["study"]["document"]["taskPromptsHash"] == study["taskPromptsHash"]


@pytest.mark.parametrize("change", ["pack", "root", "existing", "referenced", "identical"])
def test_stale_pack_review_cannot_write(tmp_path, change):
    raw = pack()
    if change == "referenced":
        doc = json.loads(raw)
        doc["study"]["judgeRubricFile"] = "prompts/rubric.txt"
        raw = json.dumps(doc).encode()
    review = study_packs.preview(raw, root=tmp_path)
    if change == "pack":
        raw += b" "
    elif change == "root":
        tmp_path = tmp_path / "other"
        tmp_path.mkdir()
    elif change == "existing":
        (tmp_path / "experiments/shared-study").mkdir(parents=True)
    else:
        path = tmp_path / ("prompts/rubric.txt" if change == "referenced" else "prompts/tasks/shared.jsonl")
        path.parent.mkdir(parents=True)
        path.write_text("rubric" if change == "referenced" else json.loads(raw)["files"]["prompts/tasks/shared.jsonl"])
    with pytest.raises(ExperimentStoreError):
        study_packs.apply(raw, root=tmp_path, expected=review["reviewSHA256"])
    assert not (tmp_path / "experiments/shared-study/experiment.json").exists()


@pytest.mark.parametrize("relative", ["../outside", "/outside", "runs/file", "experiments/other.json", "prompts/../../outside"])
def test_pack_files_are_contained(tmp_path, relative):
    doc = json.loads(pack())
    doc["files"] = {relative: "text"}
    with pytest.raises(ExperimentStoreError):
        study_packs.preview(json.dumps(doc).encode(), root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("destination", ["outside", "runs"])
def test_symlinked_prompt_tree_is_refused(tmp_path, destination):
    root = tmp_path / "workspace"
    root.mkdir()
    target = tmp_path / "outside" if destination == "outside" else root / "runs"
    target.mkdir()
    (root / "prompts").symlink_to(target, target_is_directory=True)
    with pytest.raises(ExperimentStoreError):
        study_packs.preview(pack(), root=root)
    assert list(target.iterdir()) == []


def test_failed_save_rolls_back_only_new_inputs(tmp_path, monkeypatch):
    doc = json.loads(pack())
    doc["files"]["prompts/reused.txt"] = "keep"
    reused = tmp_path / "prompts/reused.txt"
    reused.parent.mkdir()
    reused.write_text("keep")
    def fail(*args, **kwargs):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(store, "save_raw", fail)
    with pytest.raises(OSError):
        apply(tmp_path, json.dumps(doc).encode())
    assert reused.read_text() == "keep"
    assert not (tmp_path / "prompts/tasks/shared.jsonl").exists()
    assert not (tmp_path / "experiments/shared-study/experiment.json").exists()


def test_failed_atomic_publication_leaves_no_partial_input(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated link failure")
    monkeypatch.setattr(authoring_files.os, "link", fail)
    with pytest.raises(OSError, match="link failure"):
        authoring_files.publish_new(tmp_path / "input.jsonl", b"complete bytes")
    assert list(tmp_path.iterdir()) == []


def test_existing_input_is_never_replaced_on_publication(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"original")
    assert not authoring_files.publish_new(path, b"original")
    with pytest.raises(ExperimentStoreError):
        authoring_files.publish_new(path, b"replacement")
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_named_input_reports_pin_failure_and_saves_repairable_draft(tmp_path):
    doc = json.loads(pack())
    doc["files"]["prompts/tasks/shared.jsonl"] = '{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n'
    result = apply(tmp_path, json.dumps(doc).encode())
    assert "taskPromptsHash" not in result["study"]["document"]
    assert any("duplicate item id" in issue for issue in result["verificationIssues"])


def test_prompt_import_retains_metadata_and_old_input_and_is_idempotent(tmp_path):
    original = apply(tmp_path)["study"]
    old = tmp_path / original["document"]["taskPromptsFile"]
    old_bytes = old.read_bytes()
    text = '{"id":"p","transcript":[{"role":"user","content":"Answer"}],"factors":{"group":"first"},"unknown":[1,2]}\n'
    result = study_inputs.import_prompts("shared-study", text, root=tmp_path, expected=original["manifestFileSHA256"])
    output = tmp_path / result["prompts"]["path"]
    assert output.read_text() == text
    assert old.read_bytes() == old_bytes
    second = study_inputs.import_prompts("shared-study", text, root=tmp_path, expected=result["study"]["manifestFileSHA256"])
    assert not second["changed"]
    assert second["study"]["manifestFileSHA256"] == result["study"]["manifestFileSHA256"]
    with pytest.raises(ExperimentStoreError, match="changed"):
        study_inputs.import_prompts("shared-study", text + "\n", root=tmp_path, expected=original["manifestFileSHA256"])


@pytest.mark.parametrize("text", ["", "[]", '{"id":"a"}', '{"text":"a","id":2}', '{"text":"a","factors":[]}', '{"transcript":[]}'])
def test_invalid_prompt_import_leaves_workspace_unchanged(tmp_path, text):
    before = apply(tmp_path)["study"]
    with pytest.raises(ExperimentStoreError):
        study_inputs.import_prompts("shared-study", text, root=tmp_path, expected=before["manifestFileSHA256"])
    assert authoring_files.snapshot("shared-study", tmp_path) == before
    assert not (tmp_path / "prompts/tasks/versions").exists()


def test_frozen_import_refuses_before_new_input(tmp_path):
    before = apply(tmp_path)["study"]
    path = tmp_path / "experiments/shared-study/experiment.json"
    doc = before["document"]
    doc["status"] = "frozen"
    path.write_text(json.dumps(doc))
    with pytest.raises(ExperimentStoreError) as error:
        study_inputs.import_prompts("shared-study", '{"text":"new"}', root=tmp_path,
                                    expected=manifest_files.file_digest(str(path)))
    assert error.value.gate == "statusImmutable"
    assert not (tmp_path / "prompts/tasks/versions").exists()


def test_reviewed_vector_calls_scientific_gate_and_never_modifies_artifact(tmp_path):
    before = apply(tmp_path)["study"]
    directory = tmp_path / "runs/extraction"
    directory.mkdir(parents=True)
    tensor = directory / "vector.safetensors"
    sidecar = directory / "vector.json"
    tensor.write_bytes(b"tensor fixture; no tensor computation")
    sidecar.write_text(json.dumps({"modelID": "wrong/model"}))
    read = study_inputs.inspect_artifact("runs/extraction/vector", root=tmp_path)
    kwargs = dict(root=tmp_path, expected=before["manifestFileSHA256"],
                  artifact_sha256=read["artifactSHA256"], sidecar_sha256=read["sidecarSHA256"])
    # Scientific owner refuses the model mismatch even though both file reviews match.
    with pytest.raises(ExperimentStoreError, match="model"):
        study_inputs.attach_artifact("shared-study", "direction", read["reference"], **kwargs)
    tensor.write_bytes(b"changed")
    with pytest.raises(ExperimentStoreError) as error:
        study_inputs.attach_artifact("shared-study", "direction", read["reference"], **kwargs)
    assert error.value.gate == "staleManifest"
    assert authoring_files.snapshot("shared-study", tmp_path) == before
    assert tensor.read_bytes() == b"changed"


def test_reviewed_vector_success_pins_both_bytes_without_editing_the_run(tmp_path):
    from test_pinned_artifact_concepts import _workspace
    _, reference = _workspace(tmp_path)
    before = authoring_files.snapshot("gm-study", tmp_path)
    read = study_inputs.inspect_artifact(reference, root=tmp_path)
    contents = {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*") if p.is_file()}
    after = study_inputs.attach_artifact("gm-study", "crit-gm", reference, root=tmp_path,
        expected=before["manifestFileSHA256"], artifact_sha256=read["artifactSHA256"],
        sidecar_sha256=read["sidecarSHA256"], source_concept="crit")
    pin = after["document"]["concepts"][0]["vectorArtifact"]
    assert pin["sha256TensorHash"] == read["artifactSHA256"]
    assert pin["sha256SidecarHash"] == read["sidecarSHA256"]
    assert contents == {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*") if p.is_file()}


def test_review_and_reference_gates_are_reproducible():
    root = Path(__file__).resolve().parents[2]
    for script in ("check-client-assembly-reference.py",):
        result = subprocess.run([sys.executable, str(root / "scripts/ci" / script)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_cli_roundtrip_and_missing_or_extra_arguments(tmp_path, capsys):
    def run(*args):
        code = client_cli.main([*args, "--root", str(tmp_path), "--json"])
        document = json.loads(capsys.readouterr().out)
        for entry in document.get("advisories", []):
            assert isinstance(entry, dict)
            assert entry["code"] in cli_envelope.ADVISORY_CODES
            assert isinstance(entry["detail"], str)
        return code, document
    code, review = run("pack", "preview", str(FIXTURE))
    assert code == 0 and not review["changed"]
    assert run("pack", "apply", str(FIXTURE))[0] == 64
    assert run("pack", "preview", str(FIXTURE), "extra")[0] == 64
    code, applied = run("pack", "apply", str(FIXTURE), "--review-sha256", review["result"]["reviewSHA256"])
    assert code == 0 and applied["changed"]
    assert applied["result"]["verificationIssues"]
    assert applied["state"] == "ready"
    assert not applied.get("advisories")
    assert run("experiment", "inspect", "shared-study")[1]["result"]["manifestFileSHA256"]
    code, refusal = run("experiment", "import-prompts", "shared-study", "--file", str(FIXTURE), "--manifest-sha256", "0" * 64)
    assert code == 65 and refusal["error"]["repairAction"]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", "\r\n\n\r"])
def test_prompt_import_normalizes_only_record_separators(tmp_path, newline):
    before = apply(tmp_path)["study"]
    old_file = tmp_path / before["document"]["taskPromptsFile"]
    old_bytes = old_file.read_bytes()
    records = ['{"id":"a","text":"escaped\\r\\ntext","extra":{"keep":true}}',
               '{"id":"b","transcript":[{"role":"user","content":"next"}]}']
    text = newline.join(["", "  " + records[0] + "\t", " ", records[1], ""])
    expected = ("\n".join(records) + "\n").encode()
    result = study_inputs.import_prompts("shared-study", text, root=tmp_path,
                                        expected=before["manifestFileSHA256"])
    assert (tmp_path / result["prompts"]["path"]).read_bytes() == expected
    assert result["prompts"]["sha256"] == manifest_files.digest_bytes(expected)
    assert old_file.read_bytes() == old_bytes
    repeated = study_inputs.import_prompts("shared-study", text, root=tmp_path,
                                           expected=result["study"]["manifestFileSHA256"])
    assert not repeated["changed"]


def test_new_client_journey_imports_no_science_or_service_packages(tmp_path):
    script = '''
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        assert fullname.split('.')[0] not in {'torch','transformers','fastapi','uvicorn','peft','sae_lens'}, fullname
sys.meta_path.insert(0, Block())
from pathlib import Path
from steerlab_server.client import study_packs, study_inputs
root = Path(sys.argv[1])
data = Path(sys.argv[2]).read_bytes()
read = study_packs.preview(data, root=root)
result = study_packs.apply(data, root=root, expected=read['reviewSHA256'])
study_inputs.import_prompts('shared-study', '{"text":"next"}', root=root, expected=result['study']['manifestFileSHA256'])
study_packs.export('shared-study', root=root)
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(FIXTURE)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_pure_parser_and_run_loader_share_admission(tmp_path, newline):
    text = newline.join(['{"text":"a"}', '', '{"text":"b"}'])
    path = tmp_path / "input.jsonl"
    path.write_bytes(text.encode())
    from steerlab_server.experiment.manifest import Manifest
    manifest = Manifest.from_dict(json.loads(pack())["study"] | {"status":"draft", "taskPromptsHash": None})
    assert task_inputs.parse_prompts(text) == task_inputs.load_prompts(manifest, str(path), str(tmp_path))
