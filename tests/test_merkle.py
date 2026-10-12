"""The Merkle tree, held to RFC 6962 as the RFC writes it.

The module builds its tree bottom up, pairing a level's nodes and carrying
an odd last one. The RFC defines the tree top down, by splitting a list at
the largest power of two smaller than its length. Those are two different
programs, so the definition is written out here a second time, recursively
and from the item bytes, and the module is held to it: every root for
every size up to 130, and every leaf's path for every size up to 66.

The roots are also pinned to the eight that Certificate Transparency
publishes as test vectors, which is what says the definition written here
is the RFC's and not a private one that merely agrees with itself.

A proof shared between several leaves is held to the same definition: the
hashes of the largest subtrees of its recursion that hold none of the
leaves, found here from the items, for every set of leaves of every tree
up to nine and for sets drawn at random up to seventy.
"""

from __future__ import annotations

import hashlib
import random
import unittest
from itertools import combinations

from spatialforge.merkle import (
    MERKLE_HASH_BYTES,
    MerkleTree,
    merkle_leaf_hash,
    merkle_node_hash,
    merkle_path_length,
    merkle_root,
    merkle_root_from_path,
    merkle_root_from_shared_proof,
    merkle_shared_proof_length,
)

# The leaves and roots of the Certificate Transparency reference test
# vectors: the root over the first one, the first two, and so on.
CT_LEAVES = tuple(
    bytes.fromhex(leaf)
    for leaf in (
        "",
        "00",
        "10",
        "2021",
        "3031",
        "40414243",
        "5051525354555657",
        "606162636465666768696a6b6c6d6e6f",
    )
)
CT_ROOTS = (
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
)
EMPTY_ROOT = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


def split_point(count: int) -> int:
    """The largest power of two smaller than ``count``."""

    power = 1
    while power * 2 < count:
        power *= 2
    return power


def rfc_root(items: list[bytes]) -> bytes:
    """``MTH`` of RFC 6962 section 2.1, from the items themselves."""

    if not items:
        return hashlib.sha256().digest()
    if len(items) == 1:
        return hashlib.sha256(b"\x00" + items[0]).digest()
    k = split_point(len(items))
    return hashlib.sha256(
        b"\x01" + rfc_root(items[:k]) + rfc_root(items[k:])
    ).digest()


def rfc_path(index: int, items: list[bytes]) -> list[bytes]:
    """``PATH`` of RFC 6962 section 2.1.1, lowest sibling first."""

    if len(items) == 1:
        return []
    k = split_point(len(items))
    if index < k:
        return rfc_path(index, items[:k]) + [rfc_root(items[k:])]
    return rfc_path(index - k, items[k:]) + [rfc_root(items[:k])]


def rfc_shared_proof(
    indices: list[int], items: list[bytes], first: int = 0
) -> list[bytes]:
    """What several leaves share, by the recursion of ``MTH`` itself.

    Where the recursion reaches a subtree holding none of the leaves it
    stops and takes that subtree's hash. Left to right.
    """

    if not any(first <= index < first + len(items) for index in indices):
        return [rfc_root(items)]
    if len(items) == 1:
        return []
    k = split_point(len(items))
    return rfc_shared_proof(indices, items[:k], first) + rfc_shared_proof(
        indices, items[k:], first + k
    )


def items_of(count: int) -> list[bytes]:
    """Distinct items of differing lengths, one of them empty."""

    return [b"item" * (index % 5) + bytes([index]) * index
            for index in range(count)]


def leaves_of(count: int) -> list[bytes]:
    return [merkle_leaf_hash(item) for item in items_of(count)]


class MerkleRootTests(unittest.TestCase):
    def test_the_certificate_transparency_vectors(self) -> None:
        for count, expected in enumerate(CT_ROOTS, start=1):
            with self.subTest(leaves=count):
                hashes = [merkle_leaf_hash(leaf) for leaf in CT_LEAVES[:count]]
                self.assertEqual(merkle_root(hashes).hex(), expected)
                self.assertEqual(MerkleTree(hashes).root.hex(), expected)
                self.assertEqual(rfc_root(list(CT_LEAVES[:count])).hex(),
                                 expected)

    def test_the_root_of_no_leaves_is_the_hash_of_nothing(self) -> None:
        self.assertEqual(merkle_root([]).hex(), EMPTY_ROOT)
        self.assertEqual(merkle_root(()).hex(), EMPTY_ROOT)

    def test_every_size_up_to_130_is_the_tree_the_rfc_defines(self) -> None:
        for count in range(131):
            with self.subTest(leaves=count):
                items = items_of(count)
                hashes = [merkle_leaf_hash(item) for item in items]
                expected = rfc_root(items)
                self.assertEqual(merkle_root(hashes), expected)
                self.assertEqual(merkle_root(tuple(hashes)), expected)
                if count:
                    tree = MerkleTree(hashes)
                    self.assertEqual(tree.root, expected)
                    self.assertEqual(tree.size, count)
                    self.assertEqual(
                        [tree.leaf(index) for index in range(count)], hashes
                    )

    def test_a_leaf_and_a_node_are_hashed_apart(self) -> None:
        left, right = leaves_of(3)[1:]
        self.assertEqual(
            merkle_leaf_hash(b"").hex(),
            hashlib.sha256(b"\x00").hexdigest(),
        )
        self.assertEqual(
            merkle_node_hash(left, right),
            hashlib.sha256(b"\x01" + left + right).digest(),
        )
        # Two children handed over as one item do not make their parent.
        self.assertNotEqual(
            merkle_leaf_hash(left + right), merkle_node_hash(left, right)
        )
        self.assertNotEqual(
            merkle_root([merkle_leaf_hash(left + right)]),
            merkle_root([left, right]),
        )
        self.assertNotEqual(
            merkle_node_hash(left, right), merkle_node_hash(right, left)
        )

    def test_the_root_depends_on_every_leaf_and_on_their_order(self) -> None:
        hashes = leaves_of(11)
        root = merkle_root(hashes)
        seen = {root}
        for index in range(11):
            changed = list(hashes)
            changed[index] = merkle_leaf_hash(b"another item")
            seen.add(merkle_root(changed))
            if index:
                exchanged = list(hashes)
                exchanged[index - 1], exchanged[index] = (
                    exchanged[index], exchanged[index - 1]
                )
                seen.add(merkle_root(exchanged))
            seen.add(merkle_root(hashes[:index] + hashes[index + 1:]))
        seen.add(merkle_root(hashes + hashes[-1:]))
        # One root, eleven changed, ten exchanged, eleven shortened and
        # one lengthened by repeating its last leaf: all different.
        self.assertEqual(len(seen), 1 + 11 + 10 + 11 + 1)

    def test_a_leaf_hash_reads_bytes_however_they_are_held(self) -> None:
        data = b"SFTILE01" + bytes(range(64))
        expected = merkle_leaf_hash(data)
        self.assertEqual(len(expected), MERKLE_HASH_BYTES)
        self.assertEqual(merkle_leaf_hash(bytearray(data)), expected)
        self.assertEqual(merkle_leaf_hash(memoryview(data)), expected)


class MerklePathTests(unittest.TestCase):
    def test_every_path_up_to_66_leaves_is_the_path_the_rfc_defines(
        self,
    ) -> None:
        for count in (*range(1, 67), 127, 128, 129, 130):
            items = items_of(count)
            hashes = [merkle_leaf_hash(item) for item in items]
            tree = MerkleTree(hashes)
            indices = (
                range(count)
                if count <= 66
                else (0, 1, 63, 64, count - 2, count - 1)
            )
            for index in indices:
                with self.subTest(leaves=count, leaf=index):
                    path = tree.path(index)
                    self.assertEqual(list(path), rfc_path(index, items))
                    self.assertEqual(
                        merkle_path_length(index, count), len(path)
                    )
                    self.assertEqual(
                        merkle_root_from_path(
                            hashes[index], index, count, path
                        ),
                        tree.root,
                    )
                    self.assertEqual(
                        merkle_root_from_path(
                            hashes[index], index, count, list(path)
                        ),
                        tree.root,
                    )

    def test_the_path_lengths_of_the_rfcs_seven_leaf_example(self) -> None:
        # RFC 6962 section 2.1.3: the paths for d0, d3, d4 and d6 are
        # [b, h, l], [c, g, l], [f, j, k] and [i, k].
        tree = MerkleTree(leaves_of(7))
        self.assertEqual(
            [len(tree.path(index)) for index in (0, 3, 4, 6)], [3, 3, 3, 2]
        )
        self.assertEqual(
            [merkle_path_length(index, 7) for index in range(7)],
            [3, 3, 3, 3, 3, 3, 2],
        )
        # The two paths that end on l share it, and the last leaf's
        # second step is the first four leaves' root.
        self.assertEqual(tree.path(0)[2], tree.path(3)[2])
        self.assertEqual(tree.path(4)[2], tree.path(6)[1])
        self.assertEqual(tree.path(6)[1], merkle_root(leaves_of(7)[:4]))

    def test_a_single_leaf_is_its_own_root_and_has_no_path(self) -> None:
        (leaf,) = leaves_of(1)
        tree = MerkleTree([leaf])
        self.assertEqual(tree.root, leaf)
        self.assertEqual(tree.path(0), ())
        self.assertEqual(merkle_path_length(0, 1), 0)
        self.assertEqual(merkle_root_from_path(leaf, 0, 1, ()), leaf)

    def test_a_path_leads_to_the_root_for_its_own_leaf_and_place_only(
        self,
    ) -> None:
        for count in range(2, 20):
            hashes = leaves_of(count)
            tree = MerkleTree(hashes)
            for index in range(count):
                path = tree.path(index)
                with self.subTest(leaves=count, leaf=index):
                    # Another leaf in this one's place.
                    self.assertNotEqual(
                        merkle_root_from_path(
                            merkle_leaf_hash(b"forged"), index, count, path
                        ),
                        tree.root,
                    )
                    # This leaf in every other place.
                    for other in range(count):
                        if other == index:
                            continue
                        if merkle_path_length(other, count) != len(path):
                            with self.assertRaisesRegex(
                                ValueError, "a path for leaf"
                            ):
                                merkle_root_from_path(
                                    hashes[index], other, count, path
                                )
                            continue
                        self.assertNotEqual(
                            merkle_root_from_path(
                                hashes[index], other, count, path
                            ),
                            tree.root,
                        )
                    # Every step of the path, changed in turn.
                    for step in range(len(path)):
                        changed = list(path)
                        changed[step] = merkle_leaf_hash(b"forged step")
                        self.assertNotEqual(
                            merkle_root_from_path(
                                hashes[index], index, count, changed
                            ),
                            tree.root,
                        )

    def test_a_root_does_not_say_how_many_leaves_it_is_over(self) -> None:
        # The first leaf of five and the first leaf of six climb the
        # same three steps on the same side, so one path serves either
        # count. Whoever checks must be told the count by something
        # other than the root, which is why a statement carries it.
        hashes = leaves_of(5)
        tree = MerkleTree(hashes)
        path = tree.path(0)
        self.assertEqual(len(path), 3)
        self.assertEqual(
            merkle_root_from_path(hashes[0], 0, 6, path), tree.root
        )
        # A count that would give the path another length is refused.
        for claimed in (1, 2, 3, 4, 9, 16, 17):
            with self.subTest(claimed=claimed):
                with self.assertRaisesRegex(ValueError, "a path for leaf"):
                    merkle_root_from_path(hashes[0], 0, claimed, path)
        # Under another count a leaf can even sit at another place: the
        # last leaf of five climbs one step, and so does the second
        # leaf of two.
        last = tree.path(4)
        self.assertEqual(last, (merkle_root(hashes[:4]),))
        self.assertEqual(
            merkle_root_from_path(hashes[4], 1, 2, last), tree.root
        )
        # At the true count no other place will do.
        for index in range(4):
            with self.subTest(place=index):
                with self.assertRaisesRegex(ValueError, "a path for leaf"):
                    merkle_root_from_path(hashes[4], index, 5, last)

    def test_a_path_of_the_wrong_length_is_refused(self) -> None:
        hashes = leaves_of(13)
        tree = MerkleTree(hashes)
        for index in range(13):
            path = tree.path(index)
            with self.subTest(leaf=index):
                with self.assertRaisesRegex(
                    ValueError,
                    f"a path for leaf {index} of 13 holds {len(path)} "
                    f"hashes, not {len(path) - 1}",
                ):
                    merkle_root_from_path(hashes[index], index, 13, path[:-1])
                with self.assertRaisesRegex(
                    ValueError, f"hashes, not {len(path) + 1}"
                ):
                    merkle_root_from_path(
                        hashes[index], index, 13, path + path[-1:]
                    )


class MerkleSharedProofTests(unittest.TestCase):
    def test_every_set_of_leaves_of_every_small_tree(self) -> None:
        for count in range(1, 10):
            items = items_of(count)
            hashes = [merkle_leaf_hash(item) for item in items]
            tree = MerkleTree(hashes)
            for held in range(1, count + 1):
                for chosen in combinations(range(count), held):
                    with self.subTest(leaves=count, chosen=chosen):
                        proof = tree.shared_proof(chosen)
                        self.assertEqual(
                            list(proof),
                            rfc_shared_proof(list(chosen), items),
                        )
                        self.assertEqual(
                            merkle_shared_proof_length(chosen, count),
                            len(proof),
                        )
                        self.assertEqual(
                            merkle_root_from_shared_proof(
                                [hashes[index] for index in chosen],
                                chosen,
                                count,
                                proof,
                            ),
                            tree.root,
                        )

    def test_sets_drawn_at_random_from_larger_trees(self) -> None:
        draw = random.Random(6962)
        for count in (*range(10, 71), 127, 128, 129, 1200):
            items = [
                place.to_bytes(2, "little") * (place % 7)
                for place in range(count)
            ]
            hashes = [merkle_leaf_hash(item) for item in items]
            tree = MerkleTree(hashes)
            for _ in range(6):
                chosen = sorted(
                    draw.sample(range(count), draw.randint(1, count))
                )
                with self.subTest(leaves=count, held=len(chosen)):
                    proof = tree.shared_proof(chosen)
                    self.assertEqual(
                        list(proof), rfc_shared_proof(chosen, items)
                    )
                    self.assertEqual(
                        merkle_shared_proof_length(chosen, count), len(proof)
                    )
                    self.assertEqual(
                        merkle_root_from_shared_proof(
                            [hashes[index] for index in chosen],
                            chosen,
                            count,
                            proof,
                        ),
                        tree.root,
                    )
                    # Never more than the leaves' paths one by one.
                    self.assertLessEqual(
                        len(proof),
                        sum(
                            merkle_path_length(index, count)
                            for index in chosen
                        ),
                    )

    def test_one_leaf_shares_with_nobody_and_all_of_them_need_nothing(
        self,
    ) -> None:
        for count in (1, 2, 5, 12, 33):
            hashes = leaves_of(count)
            tree = MerkleTree(hashes)
            for index in range(count):
                with self.subTest(leaves=count, leaf=index):
                    # The same hashes as its path, in another order: a
                    # path climbs, and this reads left to right.
                    self.assertEqual(
                        sorted(tree.shared_proof([index])),
                        sorted(tree.path(index)),
                    )
            self.assertEqual(tree.shared_proof(range(count)), ())
            self.assertEqual(
                merkle_root_from_shared_proof(
                    hashes, range(count), count, ()
                ),
                tree.root,
            )

    def test_tiles_that_lie_together_share_nearly_everything(self) -> None:
        # A frame of forty by thirty tiles, as a real one is at sixteen
        # pixels, and a block of twenty-six by twenty of them.
        hashes = [
            merkle_leaf_hash(place.to_bytes(2, "little"))
            for place in range(1200)
        ]
        tree = MerkleTree(hashes)
        region = sorted(
            row * 40 + column
            for row in range(5, 25)
            for column in range(7, 33)
        )
        self.assertEqual(len(region), 520)
        self.assertEqual(len(tree.shared_proof(region)), 126)
        self.assertEqual(
            sum(merkle_path_length(place, 1200) for place in region), 5720
        )
        corner = sorted(
            row * 40 + column
            for row in range(10, 13)
            for column in range(20, 23)
        )
        self.assertEqual(len(tree.shared_proof(corner)), 17)
        self.assertEqual(
            sum(merkle_path_length(place, 1200) for place in corner), 99
        )

    def test_a_shared_proof_leads_to_the_root_for_its_own_leaves_only(
        self,
    ) -> None:
        for count in range(2, 10):
            hashes = leaves_of(count)
            tree = MerkleTree(hashes)
            forged = merkle_leaf_hash(b"forged")
            for held in range(1, count):
                for chosen in combinations(range(count), held):
                    proof = tree.shared_proof(chosen)
                    leaves = [hashes[index] for index in chosen]
                    with self.subTest(leaves=count, chosen=chosen):
                        # Each leaf replaced in turn.
                        for place in range(held):
                            changed = list(leaves)
                            changed[place] = forged
                            self.assertNotEqual(
                                merkle_root_from_shared_proof(
                                    changed, chosen, count, proof
                                ),
                                tree.root,
                            )
                        # Each hash of the proof replaced in turn.
                        for step in range(len(proof)):
                            changed = list(proof)
                            changed[step] = forged
                            self.assertNotEqual(
                                merkle_root_from_shared_proof(
                                    leaves, chosen, count, changed
                                ),
                                tree.root,
                            )
                        # The same leaves and proof said to sit at any
                        # other places: refused for its length, or
                        # another root.
                        for other in combinations(range(count), held):
                            if other == chosen:
                                continue
                            if merkle_shared_proof_length(
                                other, count
                            ) != len(proof):
                                with self.assertRaisesRegex(
                                    ValueError, "a shared proof for"
                                ):
                                    merkle_root_from_shared_proof(
                                        leaves, other, count, proof
                                    )
                                continue
                            self.assertNotEqual(
                                merkle_root_from_shared_proof(
                                    leaves, other, count, proof
                                ),
                                tree.root,
                            )

    def test_what_is_not_a_shared_proof_is_refused(self) -> None:
        hashes = leaves_of(7)
        tree = MerkleTree(hashes)
        proof = tree.shared_proof([1, 4])
        leaves = [hashes[1], hashes[4]]
        self.assertEqual(
            merkle_root_from_shared_proof(leaves, [1, 4], 7, proof),
            tree.root,
        )
        places = {
            "none": [],
            "out of order": [4, 1],
            "one twice": [1, 1],
            "below the first": [-1, 4],
            "past the last": [1, 7],
            "a flag": [True, 4],
            "a fraction": [1.0, 4],
            "text": "14",
        }
        for name, indices in places.items():
            with self.subTest(places=name):
                with self.assertRaisesRegex(ValueError, "leaf|leaves"):
                    tree.shared_proof(indices)
                with self.assertRaisesRegex(ValueError, "leaf|leaves"):
                    merkle_shared_proof_length(indices, 7)
                with self.assertRaisesRegex(ValueError, "leaf|leaves"):
                    merkle_root_from_shared_proof(leaves, indices, 7, proof)
        for size in (0, -7, True, 7.0, "7"):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "tree size"):
                    merkle_shared_proof_length([1, 4], size)
        with self.assertRaisesRegex(
            ValueError,
            "a shared proof for 2 leaves of 7 at these places holds "
            f"{len(proof)} hashes, not {len(proof) - 1}",
        ):
            merkle_root_from_shared_proof(leaves, [1, 4], 7, proof[:-1])
        with self.assertRaisesRegex(ValueError, "hashes, not"):
            merkle_root_from_shared_proof(
                leaves, [1, 4], 7, proof + proof[:1]
            )
        with self.assertRaisesRegex(ValueError, "one leaf hash for each"):
            merkle_root_from_shared_proof(leaves[:1], [1, 4], 7, proof)
        with self.assertRaisesRegex(ValueError, "leaf hash"):
            merkle_root_from_shared_proof(
                [hashes[1], hashes[4][:31]], [1, 4], 7, proof
            )
        with self.assertRaisesRegex(ValueError, "proof entry"):
            merkle_root_from_shared_proof(
                leaves, [1, 4], 7, (proof[0].hex(),) + proof[1:]
            )


class MerkleInputTests(unittest.TestCase):
    def test_leaves_that_are_not_hashes_are_refused(self) -> None:
        good = leaves_of(3)
        cases = {
            "a short hash": good[:2] + [good[2][:31]],
            "a long hash": good[:2] + [good[2] + b"\x00"],
            "hex text": good[:2] + [good[2].hex()],
            "a bytearray": good[:2] + [bytearray(good[2])],
            "nothing": good[:2] + [None],
            "one hash, not a list of them": good[0],
            "text": "ab" * 32,
        }
        for name, leaves in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "Merkle"):
                    merkle_root(leaves)
                with self.assertRaisesRegex(ValueError, "Merkle"):
                    MerkleTree(leaves)

    def test_a_tree_kept_whole_needs_a_leaf(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one leaf"):
            MerkleTree([])

    def test_places_outside_the_tree_are_refused(self) -> None:
        hashes = leaves_of(5)
        tree = MerkleTree(hashes)
        for index in (-1, 5, 6, True, 1.0, "1", None):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "has no leaf"):
                    tree.path(index)
                with self.assertRaisesRegex(ValueError, "has no leaf"):
                    tree.leaf(index)
                with self.assertRaisesRegex(ValueError, "has no leaf"):
                    merkle_path_length(index, 5)
                with self.assertRaisesRegex(ValueError, "has no leaf"):
                    merkle_root_from_path(hashes[0], index, 5, tree.path(0))
        for size in (0, -3, True, 5.0, "5", None):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "tree size"):
                    merkle_path_length(0, size)
                with self.assertRaisesRegex(ValueError, "tree size"):
                    merkle_root_from_path(hashes[0], 0, size, tree.path(0))

    def test_a_path_holding_something_that_is_not_a_hash_is_refused(
        self,
    ) -> None:
        hashes = leaves_of(5)
        tree = MerkleTree(hashes)
        path = tree.path(2)
        for name, entry in (
            ("short", path[0][:31]),
            ("text", path[0].hex()),
            ("nothing", None),
        ):
            with self.subTest(entry=name):
                with self.assertRaisesRegex(ValueError, "path entry"):
                    merkle_root_from_path(
                        hashes[2], 2, 5, (entry,) + path[1:]
                    )
        with self.assertRaisesRegex(ValueError, "leaf hash"):
            merkle_root_from_path(hashes[2][:31], 2, 5, path)
        with self.assertRaisesRegex(ValueError, "leaf hash"):
            merkle_root_from_path(hashes[2].hex(), 2, 5, path)


if __name__ == "__main__":
    unittest.main()
