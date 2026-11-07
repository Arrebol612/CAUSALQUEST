from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from causalquest.framework.artifacts import loads,save,read


def test_duplicate_keys_nonfinite_and_concurrent_overwrite_are_rejected(tmp_path):
    for text in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}']:
        with pytest.raises(ValueError):loads(text)
    path=tmp_path/'record.json'
    def write(value):
        try:save(path,{'winner':value});return True
        except (FileExistsError,ValueError):return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes=list(pool.map(write,[1,2]))
    assert sum(outcomes)==1 and read(path)['winner'] in (1,2)
    with pytest.raises(ValueError):save(path,{'overwritten':True})

