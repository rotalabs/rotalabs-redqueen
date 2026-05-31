"""Canonical seedable PRNG for reproducible, cross-language evolution.

Implements the redqueen-spec canonical generator:
  generator : xoshiro256++  (Blackman & Vigna, 2019; public domain)
  seeding   : SplitMix64 from a single u64 seed
  double    : (u64 >> 11) / 2**53  in [0, 1)
  below(n)  : Lemire nearly-divisionless (unbiased) in [0, n)
  shuffle   : Fisher-Yates, descending  (j = below(i+1) for i in len-1..1)

This replaces numpy's Generator so that a given seed produces the *same* stream
on every implementation (Python / TypeScript / Rust). The spec, reference
implementation, and cross-validated test vectors live in
`_dev/redqueen-spec/conformance/prng/`; `tests/test_prng.py` checks this module
reproduces those vectors exactly.

Derived helpers (`random`, `uniform`, `integers`, `choice`) are defined purely in
terms of the normative primitives, so they stay deterministic and portable. They
also mirror the subset of the numpy Generator API the engine relied on, which
keeps call sites unchanged. Gaussian/`normal` sampling is intentionally NOT
provided: the spec defers it until a genome needs it (so the draw order can be
pinned at that point).
"""

from __future__ import annotations

import os

_MASK = (1 << 64) - 1
_TWO53 = float(1 << 53)


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    z = z ^ (z >> 31)
    return state, z


def _seed_state(seed: int) -> list[int]:
    st = seed & _MASK
    s: list[int] = []
    for _ in range(4):
        st, z = _splitmix64(st)
        s.append(z)
    return s


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & _MASK


class Rng:
    """xoshiro256++ generator seeded via SplitMix64.

    Args:
        seed: A 64-bit integer seed. ``None`` draws an OS-entropy seed (for
            non-deterministic production runs); conformance runs MUST pass an
            explicit seed.
    """

    __slots__ = ("s",)

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little")
        self.s = _seed_state(seed)

    # ---- normative primitives -------------------------------------------------

    def next_u64(self) -> int:
        s = self.s
        result = (_rotl((s[0] + s[3]) & _MASK, 23) + s[0]) & _MASK
        t = (s[1] << 17) & _MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def next_double(self) -> float:
        """Uniform double in [0, 1)."""
        return (self.next_u64() >> 11) / _TWO53

    def below(self, n: int) -> int:
        """Uniform integer in [0, n), unbiased (Lemire nearly-divisionless)."""
        if n <= 0:
            return 0
        x = self.next_u64()
        m = x * n
        low = m & _MASK
        if low < n:
            t = (1 << 64) % n
            while low < t:
                x = self.next_u64()
                m = x * n
                low = m & _MASK
        return m >> 64

    def shuffle(self, a: list) -> list:
        """In-place Fisher-Yates shuffle (descending). Returns ``a``."""
        for i in range(len(a) - 1, 0, -1):
            j = self.below(i + 1)
            a[i], a[j] = a[j], a[i]
        return a

    # ---- derived helpers (deterministic; mirror the used numpy API) -----------

    def random(self) -> float:
        """Alias of :meth:`next_double` (matches ``numpy.Generator.random``)."""
        return self.next_double()

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Uniform float in [low, high)."""
        return low + (high - low) * self.next_double()

    def integers(self, low: int, high: int | None = None) -> int:
        """Uniform integer. ``integers(n)`` -> [0, n); ``integers(a, b)`` -> [a, b)."""
        if high is None:
            return self.below(low)
        return low + self.below(high - low)

    def choice(self, n: int, size: int | None = None, replace: bool = False) -> int | list[int]:
        """Sample index/indices from range(n).

        ``size=None`` returns a single index. With ``replace=False`` this is a
        partial Fisher-Yates draw (consumes exactly ``size`` ``below`` calls).
        """
        if size is None:
            return self.below(n)
        if replace:
            return [self.below(n) for _ in range(size)]
        pool = list(range(n))
        out: list[int] = []
        for i in range(size):
            j = i + self.below(n - i)
            pool[i], pool[j] = pool[j], pool[i]
            out.append(pool[i])
        return out
