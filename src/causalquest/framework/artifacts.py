"""Strict JSON and atomic, immutable experiment artifacts."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(value):
    def bad(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=bad)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'output already exists: ' + str(path))
    lock = path.with_name('.' + path.name + '.writer-lock')
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    fd, name = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), 'output appeared during write')
        # A per-artifact exclusive writer lock protects cooperating framework writers.
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
        os.close(lock_fd)
        os.unlink(lock)


def save(path, value):
    data = canonical(value) + b'\n'
    atomic(path, lambda stream: stream.write(data))


def jsonl(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_jsonl(path, rows):
    atomic(path, lambda stream: stream.write(b''.join(canonical(row) + b'\n' for row in rows)))


def reserve(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    save(path / 'started.json', {'schema': 'cq-workflow-v1', 'status': 'started'})
    return path


def implementation_digest():
    package = Path(__file__).resolve().parents[1]
    return digest({p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob('*.py'))})

