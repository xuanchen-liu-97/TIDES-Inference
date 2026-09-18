"""Bounded exact-key reuse of structural columns and direct-solver results.

The cache is owned by one fixed inference problem. Blocks must not be mutated
during that problem's lifetime. No approximate parameter matching, rank update,
solver substitution, or cross-convention reuse is performed.
"""
from collections import OrderedDict
import numpy as np


class StructuralLinearCache:
    def __init__(self, enabled=True, max_bytes=32 * 1024**2):
        self.enabled = bool(enabled)
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative.")
        self._entries = OrderedDict()
        self._bytes = 0
        self._blocks = {}  # identity keys cannot be recycled while cached
        self.hits = {'column': 0, 'svd': 0, 'scaled_lstsq': 0}
        self.misses = dict(self.hits)

    @staticmethod
    def array_key(x):
        x = np.asarray(x, dtype=float)
        return (x.shape, x.tobytes())

    def key(self, kind, blocks, theta, y=None):
        for block in blocks:
            self._blocks[id(block)] = block
        return (kind, tuple(id(b) for b in blocks), self.array_key(theta),
                None if y is None else self.array_key(y))

    def get(self, key):
        if self.enabled and key in self._entries:
            self.hits[key[0]] += 1
            self._entries.move_to_end(key)
            return self._entries[key][0]
        self.misses[key[0]] += 1
        return None

    def put(self, key, value):
        if not self.enabled:
            return value
        arrays = [value] if isinstance(value, np.ndarray) else [v for v in value if isinstance(v, np.ndarray)]
        size = sum(a.nbytes for a in arrays)
        if size > self.max_bytes:
            return value
        if key in self._entries:
            self._bytes -= self._entries.pop(key)[1]
        while self._entries and self._bytes + size > self.max_bytes:
            _, (_, old_size) = self._entries.popitem(last=False)
            self._bytes -= old_size
        for a in arrays:
            a.flags.writeable = False
        self._entries[key] = (value, size)
        self._bytes += size
        return value

    def column(self, block, theta):
        key = self.key('column', (block,), theta)
        cached = self.get(key)
        return self.put(key, block @ theta) if cached is None else cached

    def matrix(self, blocks, theta, *, cache_columns=True):
        # Retain np.column_stack's original memory layout and arithmetic path.
        if not cache_columns:
            return np.column_stack([b @ theta for b in blocks])
        return np.column_stack([self.column(b, theta) for b in blocks])

    def stats(self):
        return {'enabled': self.enabled, 'max_bytes': self.max_bytes,
                'stored_bytes': self._bytes, 'hits': dict(self.hits),
                'misses': dict(self.misses)}
