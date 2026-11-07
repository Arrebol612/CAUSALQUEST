"""Rubric and verifier for completed experiment decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from causalquest.schemas import ReviewReport, RubricNode, RubricTree, TaskType


def build_post_experiment_rubric() -> RubricTree:
    return RubricTree(
        version="post-experiment-rubric-v1",
        root=RubricNode(
            node_id="post_experiment_decision",
            weight=1.0,
            description="实验后证据与决策评审总分",
            children=(
                RubricNode("validity", 0.20, "遵守随机化风险与隔离硬门禁", hard_gate=True),
                RubricNode("estimand", 0.10, "只做同一 test 内声明对照的聚合点击率比较"),
                RubricNode("statistics", 0.20, "正确使用区间、检验和多重比较"),
                RubricNode("practicality", 0.15, "区分统计方向与登记 MDE"),
                RubricNode("trajectory", 0.10, "先调查并更新状态，再提交结论"),
                RubricNode("evidence", 0.15, "关键结论引用真实工具证据"),
                RubricNode("decision", 0.05, "建议与联合门禁一致"),
                RubricNode("human_boundary", 0.05, "保留人工上线审批", hard_gate=True),
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


def verify_post_experiment(
    response: Mapping[str, Any],
    oracle: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        report = ReviewReport.from_mapping(response)
    except (TypeError, ValueError) as exc:
        return {"passed": False, "score": 0.0, "checks": {"valid_report": False}, "error": str(exc)}
    if report.task_type is not TaskType.POST_EXPERIMENT_DECISION:
        return {
            "passed": False,
            "score": 0.0,
            "checks": {"valid_report": False},
            "error": "wrong task_type",
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
    evidence_to_tool = {str(item.get("evidence_id")): str(item.get("tool_name")) for item in trace}
    cited_evidence = {
        evidence_id for claim in report.claim_evidence for evidence_id in claim.evidence_ids
    }
    cited_tools = {evidence_to_tool[item] for item in cited_evidence if item in evidence_to_tool}
    diagnostic_tools = set(oracle.get("diagnostic_evidence_tools", []))
    required_documents = set(oracle.get("required_documents", []))
    read_documents = {
        str(item.get("arguments", {}).get("document_id"))
        for item in trace
        if item.get("tool_name") == "get_document"
    }
    state_steps = [
        int(item.get("step", 0))
        for item in trace
        if item.get("tool_name") == "update_investigation_state"
    ]
    submit_steps = [
        int(item.get("step", 0)) for item in trace if item.get("tool_name") == "submit_review"
    ]
    checks = {
        "valid_report": True,
        "decision": report.decision.value == oracle.get("expected_decision"),
        "issue_coverage": issue_recall >= 0.8,
        "hard_gates": expected_hard.issubset(reported_issues),
        "tool_coverage": required_tools.issubset(trace_tools),
        "state_before_submit": bool(state_steps)
        and (not submit_steps or min(state_steps) < min(submit_steps)),
        "evidence_grounded": bool(cited_evidence) and cited_evidence.issubset(evidence_to_tool),
        "diagnostic_evidence": bool(diagnostic_tools.intersection(cited_tools)),
        "document_coverage": required_documents.issubset(read_documents),
        "human_boundary": report.requires_human_approval,
    }
    weights = {
        "valid_report": 0.05,
        "decision": 0.10,
        "issue_coverage": 0.10,
        "hard_gates": 0.20,
        "tool_coverage": 0.15,
        "state_before_submit": 0.10,
        "evidence_grounded": 0.10,
        "diagnostic_evidence": 0.10,
        "document_coverage": 0.05,
        "human_boundary": 0.05,
    }
    score = sum(weights[name] for name, passed in checks.items() if passed)
    passed = score >= 0.85 and all(
        checks[name] for name in ("decision", "hard_gates", "diagnostic_evidence", "human_boundary")
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
