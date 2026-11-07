"""Rubric and verifier for running-experiment health diagnosis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from causalquest.schemas import ReviewReport, RubricNode, RubricTree


def build_health_diagnosis_rubric() -> RubricTree:
    """Return the versioned T2 rubric for evidence-driven health investigations."""

    return RubricTree(
        version="health-diagnosis-rubric-v1",
        root=RubricNode(
            node_id="health_diagnosis",
            weight=1.0,
            description="实验运行健康诊断总分",
            children=(
                RubricNode("validity", 0.25, "识别阻断因果解释的数据问题", hard_gate=True),
                RubricNode("localization", 0.15, "定位故障机制与受影响切片"),
                RubricNode("trajectory", 0.15, "按证据更新调查状态并控制调用"),
                RubricNode("evidence", 0.20, "关键结论绑定诊断工具证据"),
                RubricNode("decision", 0.20, "给出与严重度一致的运行建议"),
                RubricNode("human_boundary", 0.05, "保留人工处置边界", hard_gate=True),
            ),
        ),
    )


def rubric_to_dict(tree: RubricTree) -> dict[str, Any]:
    def node_to_dict(node: RubricNode) -> dict[str, Any]:
        return {
            "id": node.node_id,
            "weight": node.weight,
            "description": node.description,
            "hard_gate": node.hard_gate,
            "children": [node_to_dict(child) for child in node.children],
        }

    return {"version": tree.version, "root": node_to_dict(tree.root)}


def verify_health_diagnosis(
    response: Mapping[str, Any],
    oracle: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score diagnosis, trajectory, evidence provenance, and operational decision."""

    try:
        report = ReviewReport.from_mapping(response)
    except (TypeError, ValueError) as exc:
        return {
            "passed": False,
            "score": 0.0,
            "checks": {"valid_report": False},
            "error": str(exc),
        }

    expected_issues = set(oracle.get("required_issue_codes", []))
    expected_hard = set(oracle.get("hard_gate_issue_codes", []))
    reported_issues = set(report.blocking_issues)
    issue_recall = (
        len(expected_issues.intersection(reported_issues)) / len(expected_issues)
        if expected_issues
        else float(not reported_issues)
    )
    trace_tools = {str(item.get("tool_name")) for item in trace}
    required_tools = set(oracle.get("required_tools", []))
    diagnostic_tools = set(oracle.get("diagnostic_evidence_tools", []))
    required_documents = set(oracle.get("required_documents", []))
    read_documents = {
        str(item.get("arguments", {}).get("document_id"))
        for item in trace
        if item.get("tool_name") == "get_document"
    }
    evidence_to_tool = {str(item.get("evidence_id")): str(item.get("tool_name")) for item in trace}
    cited_evidence = {
        evidence_id for claim in report.claim_evidence for evidence_id in claim.evidence_ids
    }
    cited_tools = {evidence_to_tool[item] for item in cited_evidence if item in evidence_to_tool}
    state_steps = [
        int(item.get("step", 0))
        for item in trace
        if item.get("tool_name") == "update_investigation_state"
    ]
    submit_steps = [
        int(item.get("step", 0)) for item in trace if item.get("tool_name") == "submit_review"
    ]
    state_before_submit = bool(state_steps) and (
        not submit_steps or min(state_steps) < min(submit_steps)
    )
    checks = {
        "valid_report": True,
        "decision": report.decision.value == oracle.get("expected_decision"),
        "issue_coverage": issue_recall >= 0.8,
        "hard_gates": expected_hard.issubset(reported_issues),
        "tool_coverage": required_tools.issubset(trace_tools),
        "state_before_submit": state_before_submit,
        "evidence_grounded": bool(cited_evidence) and cited_evidence.issubset(evidence_to_tool),
        "diagnostic_evidence": bool(diagnostic_tools.intersection(cited_tools)),
        "document_coverage": required_documents.issubset(read_documents),
        "human_boundary": report.requires_human_approval,
    }
    weights = {
        "valid_report": 0.05,
        "decision": 0.15,
        "issue_coverage": 0.15,
        "hard_gates": 0.15,
        "tool_coverage": 0.10,
        "state_before_submit": 0.10,
        "evidence_grounded": 0.10,
        "diagnostic_evidence": 0.10,
        "document_coverage": 0.05,
        "human_boundary": 0.05,
    }
    score = sum(weights[name] for name, passed in checks.items() if passed)
    passed = score >= 0.85 and all(
        checks[name]
        for name in (
            "decision",
            "hard_gates",
            "diagnostic_evidence",
            "human_boundary",
        )
    )
    return {
        "passed": passed,
        "score": score,
        "checks": checks,
        "issue_recall": issue_recall,
        "missing_issues": sorted(expected_issues.difference(reported_issues)),
        "missing_tools": sorted(required_tools.difference(trace_tools)),
        "unknown_evidence_ids": sorted(cited_evidence.difference(evidence_to_tool)),
    }
