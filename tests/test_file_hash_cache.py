from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from plasmid_priority.utils.files import _file_sha256_cached, path_signature_with_hash


def test_path_signature_with_hash_reuses_sha256_cache_for_unchanged_file() -> None:
    _file_sha256_cached.cache_clear()
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "data.tsv"
        target.write_text("a\tb\n1\t2\n", encoding="utf-8")
        with mock.patch("builtins.open", wraps=open) as open_mock:
            sig1 = path_signature_with_hash(target, include_file_hash=True, max_file_size_mb=100.0)
            sig2 = path_signature_with_hash(target, include_file_hash=True, max_file_size_mb=100.0)
        assert sig1.get("sha256")
        assert sig1.get("sha256") == sig2.get("sha256")
        # Only one physical file read should occur due to cache reuse.
        assert open_mock.call_count == 1
