import hashlib
import gmpy2
from gmpy2 import invert


class ArithmeticTensor(object):
    def __init__(self, block_table, n, shape=None):
        self._n = n
        self._tensor = block_table

    def hash(self, method=None):
        hash_func = get_hash_func(method)

        def _execute_hash(value, n):
            if isinstance(value, (int, float)):
                return int(hash_func(bytes(value), encoding="utf-8").hexdigest(), base=16) % n

        self._tensor.mapValues(lambda value: _execute_hash(value, self._n))

    def __truediv__(self, other):
        def _div(lhs, rhs):
            """
            rhs * x = lhs % self._n
            """
            return gmpy2.divm(rhs, lhs, self._n)

        self._tensor.join(lambda lhs, rhs: _div(lhs, rhs))

    def __mul__(self, other):
        if isinstance(other, other):
            pass


def get_hash_func(method="sha256"):
    if hasattr(hashlib, method):
        return getattr(hashlib, method)
