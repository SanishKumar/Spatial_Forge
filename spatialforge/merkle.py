"""Merkle trees as RFC 6962 defines them.

One hash stands for a list, and any one item of the list can be shown to
belong to it with a handful of other hashes, its *path*, without the rest
of the list.

The tree is the one Certificate Transparency uses, chosen because it is
written down exactly and has published test vectors:

    MTH({})       = SHA-256()
    MTH({d0})     = SHA-256(0x00 || d0)
    MTH(D[0:n])   = SHA-256(0x01 || MTH(D[0:k]) || MTH(D[k:n]))

where ``k`` is the largest power of two smaller than ``n``. A leaf and an
inner node are hashed with different first bytes, so the hash of two
children can never be passed off as the hash of an item.

Everything here works on leaf *hashes*. Turning an item into bytes is the
caller's business; ``merkle_leaf_hash`` then gives the leaf.

The tree is built by pairing a level's nodes left to right and carrying an
unpaired last node up unchanged, which gives the same tree as the
definition above. That is tested against the definition itself, written
out recursively in the test, for every size up to 130.

Several leaves of one tree share most of their paths. Their *shared
proof* is the hashes of the largest subtrees of the definition's own
recursion that hold none of them, from left to right. With the leaves it
gives the root, and it is never longer than their paths together and
usually far shorter: every leaf of a tree needs none at all.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from typing import Sequence

MERKLE_HASH_BYTES = 32
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def merkle_leaf_hash(data: bytes | bytearray | memoryview) -> bytes:
    """The hash of one item of the list."""

    digest = hashlib.sha256(_LEAF_PREFIX)
    digest.update(data)
    return digest.digest()


def merkle_node_hash(left: bytes, right: bytes) -> bytes:
    """The hash of an inner node from its two children."""

    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def merkle_root(leaf_hashes: Sequence[bytes]) -> bytes:
    """The root of the tree over these leaves, in this order."""

    level = _validated(leaf_hashes)
    if not level:
        return hashlib.sha256().digest()
    while len(level) > 1:
        level = _level_above(level)
    return level[0]


def merkle_path_length(index: int, size: int) -> int:
    """How many hashes a leaf's path holds, in a tree of ``size`` leaves."""

    _validate_position(index, size)
    length = 0
    while size > 1:
        if not (index == size - 1 and size % 2):
            length += 1
        index //= 2
        size = (size + 1) // 2
    return length


def merkle_root_from_path(
    leaf_hash: bytes,
    index: int,
    size: int,
    path: Sequence[bytes],
) -> bytes:
    """The root a leaf and its path lead to.

    Whoever checks compares this with the root they hold. A path of the
    wrong length for that leaf in a tree of that size is refused, since
    no tree could have produced it.
    """

    _validate_position(index, size)
    node = _validate_hash(leaf_hash, "leaf hash")
    steps = [_validate_hash(step, "path entry") for step in path]
    if len(steps) != merkle_path_length(index, size):
        raise ValueError(
            f"a path for leaf {index} of {size} holds "
            f"{merkle_path_length(index, size)} hashes, not {len(steps)}"
        )
    taken = 0
    while size > 1:
        if not (index == size - 1 and size % 2):
            sibling = steps[taken]
            taken += 1
            node = (
                merkle_node_hash(sibling, node)
                if index % 2
                else merkle_node_hash(node, sibling)
            )
        index //= 2
        size = (size + 1) // 2
    return node


def merkle_shared_proof_length(indices: Sequence[int], size: int) -> int:
    """How many hashes the shared proof of these leaves holds.

    ``indices`` are the leaves' places, in increasing order.
    """

    chosen = _validated_indices(indices, size)

    def count(low: int, high: int, first: int, last: int) -> int:
        if first == last:
            return 1
        if high - low == 1:
            return 0
        middle_leaf = low + _split(high - low)
        middle = bisect_left(chosen, middle_leaf, first, last)
        return count(low, middle_leaf, first, middle) + count(
            middle_leaf, high, middle, last
        )

    return count(0, size, 0, len(chosen))


def merkle_root_from_shared_proof(
    leaf_hashes: Sequence[bytes],
    indices: Sequence[int],
    size: int,
    proof: Sequence[bytes],
) -> bytes:
    """The root several leaves and their shared proof lead to.

    ``leaf_hashes`` are the leaves at ``indices``, in the same increasing
    order. A proof of the wrong length for those places in a tree of
    that size is refused, since no tree could have produced it.
    """

    chosen = _validated_indices(indices, size)
    leaves = [_validate_hash(leaf, "leaf hash") for leaf in leaf_hashes]
    if len(leaves) != len(chosen):
        raise ValueError("a shared proof needs one leaf hash for each place")
    steps = [_validate_hash(step, "proof entry") for step in proof]
    needed = merkle_shared_proof_length(chosen, size)
    if len(steps) != needed:
        raise ValueError(
            f"a shared proof for {len(chosen)} leaves of {size} at "
            f"these places holds {needed} hashes, not {len(steps)}"
        )
    remaining = iter(steps)

    def root(low: int, high: int, first: int, last: int) -> bytes:
        if first == last:
            return next(remaining)
        if high - low == 1:
            return leaves[first]
        middle_leaf = low + _split(high - low)
        middle = bisect_left(chosen, middle_leaf, first, last)
        left = root(low, middle_leaf, first, middle)
        return merkle_node_hash(
            left, root(middle_leaf, high, middle, last)
        )

    return root(0, size, 0, len(chosen))


class MerkleTree:
    """A tree kept whole, so that any leaf's path can be read from it."""

    __slots__ = ("_levels",)

    def __init__(self, leaf_hashes: Sequence[bytes]) -> None:
        level = _validated(leaf_hashes)
        if not level:
            raise ValueError("a Merkle tree needs at least one leaf")
        levels = [level]
        while len(level) > 1:
            level = _level_above(level)
            levels.append(level)
        self._levels = tuple(levels)

    @property
    def size(self) -> int:
        return len(self._levels[0])

    @property
    def root(self) -> bytes:
        return self._levels[-1][0]

    def leaf(self, index: int) -> bytes:
        _validate_position(index, self.size)
        return self._levels[0][index]

    def path(self, index: int) -> tuple[bytes, ...]:
        """The siblings from leaf ``index`` up to the root, lowest first."""

        _validate_position(index, self.size)
        path = []
        for level in self._levels[:-1]:
            if not (index == len(level) - 1 and len(level) % 2):
                path.append(level[index ^ 1])
            index //= 2
        return tuple(path)

    def shared_proof(self, indices: Sequence[int]) -> tuple[bytes, ...]:
        """The shared proof of the leaves at ``indices``, left to right."""

        chosen = _validated_indices(indices, self.size)
        proof: list[bytes] = []

        def walk(low: int, high: int, first: int, last: int) -> None:
            if first == last:
                # A range the definition's recursion arrives at is one
                # node of the tree as it is kept: the one that many
                # levels up whose first leaf is the range's.
                level = (high - low - 1).bit_length()
                proof.append(self._levels[level][low >> level])
                return
            if high - low == 1:
                return
            middle_leaf = low + _split(high - low)
            middle = bisect_left(chosen, middle_leaf, first, last)
            walk(low, middle_leaf, first, middle)
            walk(middle_leaf, high, middle, last)

        walk(0, self.size, 0, len(chosen))
        return tuple(proof)


def _level_above(level: tuple[bytes, ...]) -> tuple[bytes, ...]:
    above = [
        merkle_node_hash(level[position], level[position + 1])
        for position in range(0, len(level) - 1, 2)
    ]
    if len(level) % 2:
        above.append(level[-1])
    return tuple(above)


def _validate_hash(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != MERKLE_HASH_BYTES:
        raise ValueError(f"a Merkle {label} is {MERKLE_HASH_BYTES} bytes")
    return value


def _validated(leaf_hashes: Sequence[bytes]) -> tuple[bytes, ...]:
    if isinstance(leaf_hashes, (bytes, bytearray, str)):
        raise ValueError("Merkle leaves are a sequence of hashes")
    return tuple(_validate_hash(leaf, "leaf hash") for leaf in leaf_hashes)


def _split(count: int) -> int:
    """The largest power of two smaller than ``count``, which is over one."""

    return 1 << ((count - 1).bit_length() - 1)


def _validated_indices(indices: Sequence[int], size: object) -> list[int]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("a Merkle tree size is a positive integer")
    if isinstance(indices, (bytes, bytearray, str)):
        raise ValueError("the places of leaves are a sequence of integers")
    chosen = list(indices)
    if not chosen:
        raise ValueError("a shared proof is of at least one leaf")
    previous = -1
    for index in chosen:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not previous < index < size
        ):
            raise ValueError(
                f"the places of leaves in a tree of {size} are distinct, "
                "in increasing order, and inside it"
            )
        previous = index
    return chosen


def _validate_position(index: object, size: object) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("a Merkle tree size is a positive integer")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < size
    ):
        raise ValueError(f"a tree of {size} leaves has no leaf {index!r}")
