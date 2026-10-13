"""
Tests for traceability.cli.

The CLI builds its config from the environment, which points at the
development database; every test patches it to the isolated test config.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from traceability import cli
from traceability.tests.conftest import make_cfg, requires_pg, reset_tables, write_json, requirement

pytestmark = requires_pg


@pytest.fixture()
def run(cfg):
    reset_tables(cfg)

    def _run(*argv):
        with patch.object(cli, "TraceabilityConfig", return_value=cfg):
            return cli.main(list(argv))
    return _run


def test_ingest_requirements_prints_report(run, real_srs, capsys):
    assert run("ingest-requirements", str(real_srs), "--no-embed", "--document-id", "doc-x") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["document_id"] == "doc-x" and report["requirements"] == 62


def test_ingest_graph_prints_report(run, real_v1_graph, repo_copy, capsys):
    assert run("ingest-graph", str(real_v1_graph), "--no-embed", "--repo-id", "simple",
               "--source-root", str(repo_copy)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["repo_id"] == "simple" and report["relationships"] == 22
    assert report["post_run_components_located"] == 20


def test_invalid_input_exits_2(run, tmp_path):
    bad = write_json(tmp_path / "dup.json", [requirement("FR-1"), requirement("FR-1")])
    assert run("ingest-requirements", str(bad), "--no-embed") == 2


def test_missing_file_exits_2(run, tmp_path):
    assert run("ingest-requirements", str(tmp_path / "nope.json"), "--no-embed") == 2


def test_dimension_mismatch_exits_2(run, real_srs, cfg):
    run("ingest-requirements", str(real_srs), "--no-embed")
    cfg.embed_dim = cfg.embed_dim + 1
    assert run("ingest-requirements", str(real_srs), "--no-embed") == 2


def test_build_vector_indexes(run, real_srs, capsys):
    run("ingest-requirements", str(real_srs), "--no-embed")
    capsys.readouterr()
    assert run("build-vector-indexes") == 0
    assert json.loads(capsys.readouterr().out) == {"vector_indexes": "hnsw"}


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])
