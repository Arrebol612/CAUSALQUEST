"""Build a small authored CSV -> audited canonical DuckDB tutorial, offline."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from causalquest.data.upworthy import audit_upworthy_snapshot, build_upworthy_database


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('output already exists')
    raw = output / 'dataset/raw/tutorial'
    raw.mkdir(parents=True)
    columns = ['', 'created_at', 'updated_at', 'clickability_test_id', 'excerpt', 'headline',
               'lede', 'slug', 'eyecatcher_id', 'impressions', 'clicks', 'significance',
               'first_place', 'winner', 'share_text', 'square', 'test_week']
    files = []
    for role in ('confirmatory', 'holdout'):
        path = raw / (role + '.csv')
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for arm in range(2):
                row = dict.fromkeys(columns, '')
                row.update({'': str(arm), 'created_at': '2015-01-20 00:00:00',
                            'updated_at': '2015-01-21 00:00:00',
                            'clickability_test_id': role + '_authored_test',
                            'headline': role + '_authored_headline_' + str(arm),
                            'slug': role + '_authored_story', 'eyecatcher_id': role + '_image',
                            'impressions': '1000', 'clicks': str(100 + arm * 20),
                            'test_week': '201504', 'first_place': 'False', 'winner': 'False'})
                writer.writerow(row)
        files.append({'role': role, 'name': path.name, 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'expected_packages': 2, 'expected_tests': 1})
    config = {'dataset_id': 'authored_archive_tutorial', 'version': 'tutorial-v1',
              'acquisition': {'destination': 'dataset/raw/tutorial', 'files': files},
              'expected_schema': {'row_unit': 'package_or_experiment_arm',
                                  'experiment_key': 'clickability_test_id',
                                  'randomization_risk_aliases': ['problem']},
              'quality_policy': {'risky_period_start': '2013-06-25',
                                 'risky_period_end': '2014-01-10'}}
    audit = audit_upworthy_snapshot(config, output)
    summary = build_upworthy_database(config, output, 'dataset/processed/tutorial.duckdb')
    summary.update(authored_fixture_only=True, downloaded_source_data=False)
    for name, value in [('source_config.json', config), ('audit.json', audit), ('summary.json', summary)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False, indent=2))
