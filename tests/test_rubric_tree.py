"""Shipped task criteria match their public builders and typed contracts."""
import importlib
import json
from pathlib import Path

import pytest

from causalquest.schemas.rubric import RubricNode, RubricTree

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module,builder,example", [
    ("design_review", "build_design_review_rubric", "design_review.json"),
    ("health_diagnosis", "build_health_diagnosis_rubric", "health_diagnosis.json"),
    ("post_experiment", "build_post_experiment_rubric", "post_experiment.json"),
])
def test_public_examples_match_actual_task_builders(module, builder, example):
    api = importlib.import_module("causalquest.evaluation." + module)
    tree = getattr(api, builder)()
    payload = json.loads((ROOT / "examples/rubrics" / example).read_text(encoding="utf-8"))
    assert api.rubric_to_dict(tree) == payload
    assert RubricTree.from_mapping(payload) == tree
    assert {n.node_id for n in tree.root.walk() if n.hard_gate} == {"validity", "human_boundary"}


def test_nested_local_weights_and_preorder_roundtrip():
    payload = {"version": "nested-v1", "root": {
        "id": "root", "weight": 1, "description": "overall", "children": [
            {"id": "evidence", "weight": 0.6, "description": "evidence", "children": [
                {"id": "source", "weight": 0.25, "description": "source"},
                {"id": "citation", "weight": 0.75, "description": "citation"}]},
            {"id": "decision", "weight": 0.4, "description": "decision"}]}}
    tree = RubricTree.from_mapping(payload)
    assert [n.node_id for n in tree.root.walk()] == ["root", "evidence", "source", "citation", "decision"]
    assert tree.root.children[0].children[0].weight == 0.25


def test_invalid_sibling_weights_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        RubricNode("root", 1, "overall", children=(
            RubricNode("evidence", 0.5, "evidence"), RubricNode("decision", 0.4, "decision")))


def test_duplicate_ids_across_branches_rejected():
    left = RubricNode("left", 0.5, "left", children=(RubricNode("repeated", 1, "leaf"),))
    right = RubricNode("right", 0.5, "right", children=(RubricNode("repeated", 1, "leaf"),))
    with pytest.raises(ValueError, match="duplicate rubric node IDs"):
        RubricTree("duplicate-v1", RubricNode("root", 1, "overall", children=(left, right)))
