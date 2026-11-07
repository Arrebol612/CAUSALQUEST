"""Authored synthetic unit fixtures; no real archive records."""
import duckdb

def _source_database(path):
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE snapshot_metadata AS SELECT 'upworthy-canonical-v1' schema_version, "
            "'source-snapshot' snapshot_id"
        )
        connection.execute(
            """
            CREATE TABLE upworthy_tests AS SELECT
                'confirmatory'::VARCHAR source_partition,
                'test-1'::VARCHAR test_id,
                'component-1'::VARCHAR component_id,
                'candidate_train'::VARCHAR assigned_pool,
                false::BOOLEAN randomization_risk,
                NULL::VARCHAR quarantine_reasons,
                true::BOOLEAN schema_eligible,
                true::BOOLEAN eligible_for_positive_causal_case
            """
        )
        connection.execute(
            """
            CREATE TABLE upworthy_arms AS
            SELECT * FROM (VALUES
              ('confirmatory', 'test-1', 'ref', 1, 'reference headline', 'ref-slug',
               'image-a', 100000, 10000,
               0.10, '0.0', false, false),
              ('confirmatory', 'test-1', 'candidate', 2, 'candidate headline', 'candidate-slug',
               'image-b', 100000, 14000,
               0.14, '999.0', true, true)
            ) AS t(source_partition, test_id, arm_id, source_row_index, headline, slug,
                   eyecatcher_id, impressions, clicks, ctr, source_significance_undocumented,
                   editor_first_place, editor_winner)
            """
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

def _raw_case():
    return {
        "case_id": "post_test_001",
        "test_id": "test-1",
        "title": "标题方案实验后评审",
        "business_context": "编辑团队需要判断新标题是否达到登记收益。",
        "scenario_family": "content_package_selection",
        "template_family_id": "post_clear_practical_win",
        "decision_policy": {
            "reference_arm_id": "ref",
            "candidate_arm_id": "candidate",
            "mde_absolute": 0.01,
            "alpha": 0.05,
            "target_power": 0.80,
        },
    }
