import hashlib
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.public_config_trace import trace_configuration


REPOSITORY = "PaddlePaddle/PaddleDetection"
COMMIT = "a" * 40


@pytest.fixture
def corpus_factory():
    roots = []

    def create(documents, current="v2.9.0"):
        root = Path(__file__).resolve().parents[1] / (".tmp-config-trace-" + uuid.uuid4().hex)
        root.mkdir()
        roots.append(root)
        sources, chunks = [], []
        for number, (version, path, text) in enumerate(documents):
            local = root / "sources" / version / path
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(text, encoding="utf-8")
            key = str(Path(path).with_suffix("")).replace("\\", "/")
            source = {
                "source_id": f"source-{number}", "version": version, "language": "zh",
                "document_key": key, "document_path": path, "path": path,
                "local_path": str(local.relative_to(root)).replace("\\", "/"),
                "repository": REPOSITORY, "commit": COMMIT,
                "source_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/{path}",
                "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            }
            sources.append(source)
            chunks.append({
                "source_id": source["source_id"], "document_id": f"{version}:zh:{key}",
                "version": version, "repository": REPOSITORY, "document_path": path,
            })
        return SimpleNamespace(root=root, manifest={
            "workspace_id": "pphuman", "repository": REPOSITORY,
            "current_version": current,
            "available_versions": list(dict.fromkeys(row[0] for row in documents)),
            "versions": {version: {"commit": COMMIT, "tag": version} for version, _, _ in documents},
            "sources": sources,
        }, chunks=chunks)

    yield create
    for root in roots:
        shutil.rmtree(root)


def test_reports_unindexed_base_without_inventing_parent_contents(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/pphuman/model.yml", "_BASE_: ['../runtime.yml']\n!UnsafeTag\nnum_classes: 1\n")])
    result = trace_configuration(index, ["v2.9.0:zh:configs/pphuman/model"], "latest")
    assert result["version"] == "v2.9.0"
    assert result["status"] == "partial"
    relation = result["relations"][0]
    assert relation["target_path"] == "configs/runtime.yml"
    assert relation["target_status"] == "not_indexed"
    assert relation["relation_type"] == "config_inherits"
    assert relation["evidence"]["line_start"] == 1
    assert "../runtime.yml" in relation["evidence"]["text"]
    assert result["gaps"][0]["kind"] == "not_indexed"


def test_finds_reverse_document_and_config_references(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "deploy/pipeline/config/tracker.yml", "type: BOTSORT\n"),
        ("v2.9.0", "deploy/pipeline/config/infer.yml", "MOT:\n  tracker_config: deploy/pipeline/config/tracker.yml\n"),
        ("v2.9.0", "docs/tracking.md", "使用 `deploy/pipeline/config/tracker.yml`。\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:deploy/pipeline/config/tracker"], "v2.9.0")
    assert {row["path"] for row in result["relations"]} == {"docs/tracking.md", "deploy/pipeline/config/infer.yml"}
    assert {row["direction"] for row in result["relations"]} == {"incoming"}
    assert {row["target_status"] for row in result["relations"]} == {"indexed"}
    assert result["gaps"] == []


def test_does_not_resolve_missing_parent_from_another_version(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "configs/pphuman/model.yml", "_BASE_: ['../runtime.yml']\n"),
        ("v2.8.0", "configs/runtime.yml", "learning_rate: 0.3\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:configs/pphuman/model"], "v2.9.0")
    assert result["relations"][0]["target_status"] == "not_indexed"
    assert result["relations"][0]["target_document_id"] is None
    with pytest.raises(ValueError, match="root"):
        trace_configuration(index, ["v2.8.0:zh:configs/runtime"], "v2.9.0")


def test_yaml_custom_tags_are_never_executed(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "configs/model.yml", "_BASE_:\n  - './base.yml'\nDanger: !!python/object/apply:os.system ['exit 9']\n"),
        ("v2.9.0", "configs/base.yml", "value: 1\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")
    assert result["relations"][0]["target_document_id"] == "v2.9.0:zh:configs/base"
    assert result["status"] == "ok"


def test_external_repository_link_cannot_bind_to_same_path_local_source(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "docs/reid.md", "见[配置](https://github.com/PaddlePaddle/PaddleClas/blob/develop/configs/base.yml)。\n"),
        ("v2.9.0", "configs/base.yml", "value: 1\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:docs/reid"], "v2.9.0")
    assert len(result["relations"]) == 1
    assert result["relations"][0]["target_status"] == "external_repository"
    assert result["relations"][0]["target_document_id"] is None


def test_relative_markdown_links_are_normalized_from_document_parent(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "docs/tutorials/start.md", "见[配置](../../configs/runtime.yml#section)。\n"),
        ("v2.9.0", "configs/runtime.yml", "value: 1\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:docs/tutorials/start"], "v2.9.0")
    assert result["relations"][0]["target_path"] == "configs/runtime.yml"
    assert result["relations"][0]["target_status"] == "indexed"


def test_repository_root_escape_is_a_gap_not_a_filesystem_read(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "_BASE_: ['../../secret.yml', 'C:/secret.yml']\n")])
    result = trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")
    assert {row["target_status"] for row in result["relations"]} == {"invalid_path"}
    assert len(result["gaps"]) == 2


@pytest.mark.parametrize("field,value", [("local_path", "../secret.txt"), ("repository", "Other/Project"), ("namespace", "dependency_reference")])
def test_rejects_source_provenance_drift_before_scanning(corpus_factory, field, value):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "value: 1\n")])
    index.manifest["sources"][0][field] = value
    with pytest.raises(ValueError):
        trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")


def test_rejects_source_hash_drift(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "value: 1\n")])
    (index.root / index.manifest["sources"][0]["local_path"]).write_text("value: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")


def test_refuses_unsupported_dynamic_base_reference(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "_BASE_: *dynamic_alias\n")])
    result = trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")
    assert result["relations"] == []
    assert result["gaps"][0]["kind"] == "unsupported_base_syntax"
    assert result["status"] == "partial"


def test_bounds_reject_overlong_root_list_and_unknown_version(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "value: 1\n")])
    with pytest.raises(ValueError, match="1.*8"):
        trace_configuration(index, ["v2.9.0:zh:configs/model"] * 9, "v2.9.0")
    with pytest.raises(ValueError, match="version"):
        trace_configuration(index, ["v2.9.0:zh:configs/model"], "v9.0.0")


def test_relation_budget_reports_truncation_without_unbounded_gaps(corpus_factory):
    text = "\n".join(f"参考 `configs/other-{number}.yml`。" for number in range(95))
    index = corpus_factory([("v2.9.0", "docs/references.md", text)])
    result = trace_configuration(index, ["v2.9.0:zh:docs/references"], "v2.9.0")
    assert len(result["relations"]) == 80
    assert len(result["gaps"]) == 80
    assert result["bounds"]["truncated"] is True
    assert result["status"] == "partial"


def test_unsupported_base_syntax_gaps_are_bounded(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "_BASE_: *dynamic\n" * 95)])
    result = trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")
    assert len(result["gaps"]) == 80
    assert result["bounds"]["truncated"] is True


def test_same_repository_unpinned_branch_remains_version_gap(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "docs/start.md", f"见[配置](https://github.com/{REPOSITORY}/blob/develop/configs/base.yml)。\n"),
        ("v2.9.0", "configs/base.yml", "value: 1\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:docs/start"], "v2.9.0")
    assert result["relations"][0]["target_status"] == "different_version"
    assert result["relations"][0]["target_document_id"] is None


def test_only_one_hop_is_returned_without_promoting_transitive_impact(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "configs/first.yml", "_BASE_: ['second.yml']\n"),
        ("v2.9.0", "configs/second.yml", "_BASE_: ['third.yml']\n"),
        ("v2.9.0", "configs/third.yml", "value: 1\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:configs/first"], "v2.9.0")
    assert len(result["relations"]) == 1
    assert result["relations"][0]["target_path"] == "configs/second.yml"
    assert result["bounds"]["depth"] == 1


def test_rejects_chunk_provenance_drift(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "value: 1\n")])
    index.chunks[0]["repository"] = "Other/Project"
    with pytest.raises(ValueError, match="chunk"):
        trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")


def test_source_cannot_relabel_a_different_commit_as_selected_release(corpus_factory):
    index = corpus_factory([("v2.9.0", "configs/model.yml", "value: 1\n")])
    source = index.manifest["sources"][0]
    source["commit"] = "b" * 40
    source["source_url"] = f"https://github.com/{REPOSITORY}/blob/{'b' * 40}/configs/model.yml"
    with pytest.raises(ValueError, match="commit"):
        trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")


def test_long_reference_line_keeps_actual_target_inside_bounded_quote(corpus_factory):
    index = corpus_factory([("v2.9.0", "docs/start.md", "说明" * 500 + " `configs/runtime.yml`。\n")])
    result = trace_configuration(index, ["v2.9.0:zh:docs/start"], "v2.9.0")
    assert "configs/runtime.yml" in result["relations"][0]["evidence"]["text"]
    assert len(result["relations"][0]["evidence"]["text"]) <= 600


def test_long_base_list_keeps_each_target_inside_its_original_quote(corpus_factory):
    values = [f"../long-parent-directory-{number}/runtime.yml" for number in range(30)]
    text = "_BASE_: " + repr(values) + "\n"
    index = corpus_factory([("v2.9.0", "configs/model.yml", text)])
    result = trace_configuration(index, ["v2.9.0:zh:configs/model"], "v2.9.0")
    assert len(result["relations"]) == 30
    assert "long-parent-directory-29/runtime.yml" in result["relations"][-1]["evidence"]["text"]
    assert all(len(row["evidence"]["text"]) <= 600 for row in result["relations"])


def test_repeated_equivalent_paths_are_one_explicit_relationship(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "docs/start.md", "使用 `deploy/config.yml`。\n也可使用 `./deploy/config.yml`。\n"),
        ("v2.9.0", "deploy/config.yml", "value: 1\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:docs/start"], "v2.9.0")
    assert len(result["relations"]) == 1
    assert result["relations"][0]["target_path"] == "deploy/config.yml"
    assert result["gaps"] == []


def test_markdown_link_is_document_relative_while_command_path_is_repo_relative(corpus_factory):
    index = corpus_factory([
        ("v2.9.0", "docs/start.md", "见[示例](configs/local.yml)。\n命令 `./configs/root.yml`。\n"),
        ("v2.9.0", "docs/configs/local.yml", "value: 1\n"),
        ("v2.9.0", "configs/root.yml", "value: 2\n"),
    ])
    result = trace_configuration(index, ["v2.9.0:zh:docs/start"], "v2.9.0")
    assert {row["target_path"] for row in result["relations"]} == {"docs/configs/local.yml", "configs/root.yml"}
    assert result["gaps"] == []
