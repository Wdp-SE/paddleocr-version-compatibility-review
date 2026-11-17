"""Bounded process-local exact-result cache. No semantic matching or durable replay."""
from collections import OrderedDict
import copy
import hashlib
import json
from threading import Lock
import time


class ExactQueryCache:
    def __init__(self, *, max_entries=32, ttl=180, max_bytes=500_000, clock=time.monotonic):
        self.limit, self.ttl, self.max_bytes, self.clock = max_entries, ttl, max_bytes, clock
        self.rows, self.lock = OrderedDict(), Lock()

    def _key(self, identity):
        return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                        separators=(',', ':')).encode()).hexdigest()

    def get(self, identity):
        key = self._key(identity)
        with self.lock:
            row = self.rows.get(key)
            if row is None:
                return None
            if self.clock() - row[0] >= self.ttl:
                del self.rows[key]
                return None
            self.rows.move_to_end(key)
            return copy.deepcopy(row[1])

    def put(self, identity, value):
        if value.get('status') != 'OK':
            return False
        try:
            if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > self.max_bytes:
                return False
        except (ValueError, TypeError):
            return False
        with self.lock:
            self.rows[self._key(identity)] = self.clock(), copy.deepcopy(value)
            self.rows.move_to_end(self._key(identity))
            while len(self.rows) > self.limit:
                self.rows.popitem(last=False)
        return True
