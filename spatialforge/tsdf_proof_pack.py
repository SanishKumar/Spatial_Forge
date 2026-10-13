"""A proof packed for carrying.

A proof's bytes are exact. The same proof is the same bytes on every
machine, and their SHA-256 names it. In a certificate or an audit most of
those bytes are raw depth samples, and raw depth compresses: a route's
certificate of 84 MB is 12 MB through zlib.

A packed proof is those same bytes inside a zlib stream, behind a
signature and their length. Packing is for carrying a proof and for
keeping it. It is no part of what a proof is:

- **Nothing here hashes packed bytes.** Two zlib libraries need not pack
  the same bytes the same way, and CPython is not built with the same one
  everywhere. A proof's name is the SHA-256 of what it unpacks to.
- **Any stream that unpacks to a proof's bytes carries that proof.** How
  hard it was squeezed, and by what, is not asked.
- **What is unpacked is read as strictly as if it had never been
  packed.** A packed file is opened, and then it is a proof's bytes or it
  is refused.

Unpacking is bounded before it starts. The length a packed proof declares
is held to the most its kind may be, no more than that many bytes are
inflated, and the stream has to end exactly there: no more to come, and
nothing after it.
"""

from __future__ import annotations

import struct
import zlib

from .errors import TsdfError

TSDF_PACKED_PROOF_MAGIC = b"SFTPACK1"
# zlib's own default. On real certificates a harder squeeze is no
# smaller and takes more than twice as long.
TSDF_PROOF_PACK_LEVEL = 6

_LENGTH = struct.Struct("<Q")
_PREAMBLE_BYTES = len(TSDF_PACKED_PROOF_MAGIC) + _LENGTH.size


def tsdf_proof_bytes_are_packed(data: bytes) -> bool:
    """Whether these bytes start as a packed proof does."""

    return type(data) is bytes and data.startswith(TSDF_PACKED_PROOF_MAGIC)


def pack_tsdf_proof_bytes(encoded: bytes) -> bytes:
    """A proof's bytes, packed. Not bytes to compare between machines."""

    if type(encoded) is not bytes or not encoded:
        raise TsdfError("a TSDF proof is packed from its bytes")
    if tsdf_proof_bytes_are_packed(encoded):
        raise TsdfError("a packed TSDF proof is not packed again")
    return b"".join(
        (
            TSDF_PACKED_PROOF_MAGIC,
            _LENGTH.pack(len(encoded)),
            zlib.compress(encoded, TSDF_PROOF_PACK_LEVEL),
        )
    )


def unpack_tsdf_proof_bytes(packed: bytes, *, most_bytes: int) -> bytes:
    """The bytes a packed proof carries, or a refusal.

    ``most_bytes`` is the most a proof of the kind expected may be. A
    packed proof that declares more is refused before anything is
    inflated, and one that inflates to more or less than it declares is
    refused after no more than that.
    """

    if (
        isinstance(most_bytes, bool)
        or not isinstance(most_bytes, int)
        or most_bytes < 1
    ):
        raise TsdfError("the most a TSDF proof may be is a positive count")
    if not tsdf_proof_bytes_are_packed(packed):
        raise TsdfError("packed TSDF proof has an unrecognized signature")
    if len(packed) < _PREAMBLE_BYTES:
        raise TsdfError("packed TSDF proof is cut short before its length")
    (declared,) = _LENGTH.unpack_from(packed, len(TSDF_PACKED_PROOF_MAGIC))
    if not 1 <= declared <= most_bytes:
        raise TsdfError(
            f"packed TSDF proof declares {declared} bytes; one of this "
            f"kind is from 1 to {most_bytes}"
        )
    inflater = zlib.decompressobj()
    try:
        unpacked = inflater.decompress(
            memoryview(packed)[_PREAMBLE_BYTES:], declared
        )
        if not inflater.eof and inflater.decompress(
            inflater.unconsumed_tail, 1
        ):
            raise TsdfError(
                "packed TSDF proof inflates to more than the "
                f"{declared} bytes it declares"
            )
    except zlib.error as error:
        raise TsdfError(
            f"packed TSDF proof cannot be inflated: {error}"
        ) from error
    if not inflater.eof:
        raise TsdfError("packed TSDF proof is cut short")
    if inflater.unused_data:
        raise TsdfError(
            f"packed TSDF proof holds {len(inflater.unused_data)} bytes "
            "after its end"
        )
    if len(unpacked) != declared:
        raise TsdfError(
            f"packed TSDF proof inflates to {len(unpacked)} bytes and "
            f"declares {declared}"
        )
    return unpacked


def _unpacked_if_packed(data: bytes, most_bytes: int) -> bytes:
    """A file's bytes as a proof's: unpacked if they were packed."""

    if tsdf_proof_bytes_are_packed(data):
        return unpack_tsdf_proof_bytes(data, most_bytes=most_bytes)
    return data
