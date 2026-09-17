"""
ldmb.py — memory-mappable binary storage for large lists of dicts whose
contents are arbitrary Python objects (nested dicts/lists/tuples, str, int,
float, bool, None).

File layout (little-endian):

    +--------+-------------------------+---------------------+---------+
    | header | record 0 ... record N-1 | index (16*N bytes)  | trailer |
    +--------+-------------------------+---------------------+---------+

    header  (8 B)  : magic b"LDMB" | u16 version | u8 flags | u8 zlib level
    records        : each dict pickled independently (optionally zlib-compressed)
    index          : N x (u64 absolute file offset, u64 byte length)
    trailer (20 B) : u64 index_offset | u64 count | magic b"LDMB"

Every record has its own offset/length in the footer index, so a file can be
memory-mapped and individual records read lazily, and files can be merged by
copying raw payload bytes and rewriting only the offsets.
"""

from __future__ import annotations

import mmap
import os
import pickle
import struct
import zlib
from array import array
from collections.abc import Sequence

try:
    import numpy as _np          # optional: vectorizes merge index math
except ImportError:
    _np = None

__all__ = ["save", "load", "open_ldd", "merge", "MemmapList"]

_MAGIC    = b"LDMB"
_VERSION  = 1
_ZLIB     = 0b1
_HEADER   = struct.Struct("<4sHBB")    # magic, version, flags, zlib level
_TRAILER  = struct.Struct("<QQ4s")     # index offset, count, magic
_HDR_SIZE = _HEADER.size               # 8
_MIN_SIZE = _HDR_SIZE + _TRAILER.size  # 28
_CHUNK    = 1 << 22                    # 4 MiB I/O granularity
_PROTOCOL = pickle.HIGHEST_PROTOCOL    # C pickler, fastest protocol


# --------------------------------------------------------------------------- #
#  Lazy mmap-backed random-access view
# --------------------------------------------------------------------------- #
class MemmapList(Sequence):
    """Random-access, lazy view of an LDMB file backed by an mmap.

    Reading records[i] touches only that record's 16-byte index entry and its
    own bytes — nothing else in the file is parsed. Safe for concurrent readers.
    """

    __slots__ = ("path", "_f", "_mm", "_n", "_idx", "_zlvl")

    def __init__(self, path):
        self.path = os.fspath(path)
        self._f = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
            size = self._mm.size()
            if size < _MIN_SIZE:
                raise ValueError(f"{self.path!r} is not a valid LDMB file")
            index_offset, count, magic = _TRAILER.unpack(
                self._mm[size - _TRAILER.size:])
            if magic != _MAGIC or index_offset + count * 16 + _TRAILER.size != size:
                raise ValueError(f"{self.path!r} is corrupt or not an LDMB file")
            magic, version, flags, zlvl = _HEADER.unpack(self._mm[:_HDR_SIZE])
            if magic != _MAGIC:
                raise ValueError(f"{self.path!r} is not an LDMB file")
            if version != _VERSION:
                raise ValueError(f"{self.path!r}: LDMB v{version} not supported")
            self._n = count
            self._zlvl = zlvl if flags & _ZLIB else 0
            self._idx = array("Q")                      # 16 bytes/record in RAM
            self._idx.frombytes(self._mm[index_offset:index_offset + count * 16])
        except Exception:
            self.close()
            raise

    def __len__(self):
        return self._n

    def _getone(self, i):
        n = self._n
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        j = i + i
        off = self._idx[j]
        b = self._mm[off:off + self._idx[j + 1]]
        if self._zlvl:
            b = zlib.decompress(b)
        return pickle.loads(b)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._getone(j) for j in range(*i.indices(self._n))]
        return self._getone(i)

    def __iter__(self):
        mm, idx, n = self._mm, self._idx, self._n
        loads = pickle.loads
        decompress = zlib.decompress if self._zlvl else None
        for j in range(0, n + n, 2):
            off = idx[j]
            b = mm[off:off + idx[j + 1]]
            if decompress is not None:
                b = decompress(b)
            yield loads(b)

    def to_list(self):
        return list(self)

    def close(self):
        mm = getattr(self, "_mm", None)
        if mm is not None:
            self._mm = None
            try:
                mm.close()
            except BufferError:
                pass
        f = getattr(self, "_f", None)
        if f is not None:
            self._f = None
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        n = getattr(self, "_n", "?")
        closed = getattr(self, "_mm", None) is None
        return f"<MemmapList {getattr(self, 'path', '?')!r} n={n}{' closed' if closed else ''}>"


# --------------------------------------------------------------------------- #
#  save / load / open
# --------------------------------------------------------------------------- #
def save_ldmb(records, path, *, compress=None) -> int:
    """Serialize an iterable of dicts to `path` as an LDMB file. Streams.

    compress=None/0 -> raw pickle: fastest, best for memmap-style access.
    compress=1..9   -> per-record zlib: smaller file, still random-access and
                       mergeable.
    Returns the number of records written.
    """
    if compress is True:
        compress = 6
    zlvl = int(compress) if compress else 0
    flags = _ZLIB if zlvl else 0
    n = 0
    with open(path, "wb", buffering=_CHUNK) as f:
        f.write(_HEADER.pack(_MAGIC, _VERSION, flags, zlvl))
        idx = array("Q")
        ia = idx.append
        dumps = pickle.dumps
        zcompress = zlib.compress
        pos = _HDR_SIZE
        for rec in records:
            b = dumps(rec, _PROTOCOL)
            if zlvl:
                b = zcompress(b, zlvl)
            ia(pos)
            ia(len(b))
            f.write(b)
            pos += len(b)
            n += 1
        f.write(idx.tobytes())
        f.write(_TRAILER.pack(pos, n, _MAGIC))
    return n


def load_ldmb(path) -> list:
    """Materialize a whole LDMB file into a plain list of dicts (fast path)."""
    ml = path if isinstance(path, MemmapList) else MemmapList(path)
    owned = ml is not path
    try:
        mm, idx, n = ml._mm, ml._idx, ml._n
        loads = pickle.loads
        decompress = zlib.decompress if ml._zlvl else None
        out = []
        append = out.append
        for j in range(0, n + n, 2):
            off = idx[j]
            b = mm[off:off + idx[j + 1]]
            if decompress is not None:
                b = decompress(b)
            append(loads(b))
        return out
    finally:
        if owned:
            ml.close()


def open_ldmb(path) -> MemmapList:
    """Open an LDMB file as a lazy, mmap-backed random-access sequence."""
    return MemmapList(path)


# --------------------------------------------------------------------------- #
#  merge
# --------------------------------------------------------------------------- #
class _FileSource:
    __slots__ = ("f", "n", "idx", "flags", "zlvl", "payload_size")

    def __init__(self, path):
        self.f = open(path, "rb")
        try:
            f = self.f
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size < _MIN_SIZE:
                raise ValueError(f"{path!r} is not a valid LDMB file")
            f.seek(size - _TRAILER.size)
            index_offset, count, magic = _TRAILER.unpack(f.read(_TRAILER.size))
            if magic != _MAGIC or index_offset + count * 16 + _TRAILER.size != size:
                raise ValueError(f"{path!r} is corrupt or not an LDMB file")
            f.seek(0)
            magic, version, flags, zlvl = _HEADER.unpack(f.read(_HDR_SIZE))
            if magic != _MAGIC:
                raise ValueError(f"{path!r} is not an LDMB file")
            if version != _VERSION:
                raise ValueError(f"{path!r}: LDMB v{version} not supported")
            f.seek(index_offset)
            self.idx = array("Q")
            self.idx.frombytes(f.read(count * 16))
            self.n, self.flags, self.zlvl = count, flags, zlvl
            self.payload_size = index_offset - _HDR_SIZE
        except Exception:
            self.f.close()
            raise

    def copy_payload(self, out):
        f = self.f
        f.seek(_HDR_SIZE)
        remaining = self.payload_size
        while remaining:
            b = f.read(min(_CHUNK, remaining))
            if not b:
                raise EOFError("unexpected EOF while copying payload")
            out.write(b)
            remaining -= len(b)

    def write_records(self, out, pos, out_zlvl):
        """Re-emit records, converting compression (used when formats differ)."""
        idx = array("Q")
        ia = idx.append
        f, src_idx = self.f, self.idx
        decompress = zlib.decompress if self.zlvl else None
        compress = zlib.compress if out_zlvl else None
        for j in range(0, self.n + self.n, 2):
            f.seek(src_idx[j])
            b = f.read(src_idx[j + 1])
            if decompress is not None:
                b = decompress(b)
            if compress is not None:
                b = compress(b, out_zlvl)
            ia(pos)
            ia(len(b))
            out.write(b)
            pos += len(b)
        return idx, pos

    def close(self):
        self.f.close()


class _ListSource:
    __slots__ = ("records", "n")

    def __init__(self, records):
        try:
            self.n = len(records)
        except TypeError:
            raise TypeError("merge() needs sized list sources; use save() "
                            "for generators") from None
        self.records = records

    def write_records(self, out, pos, out_zlvl):
        idx = array("Q")
        ia = idx.append
        dumps = pickle.dumps
        compress = zlib.compress if out_zlvl else None
        for rec in self.records:
            b = dumps(rec, _PROTOCOL)
            if compress is not None:
                b = compress(b, out_zlvl)
            ia(pos)
            ia(len(b))
            out.write(b)
            pos += len(b)
        return idx, pos


def _shift_index(idx, delta):
    if not len(idx) or not delta:
        return idx.tobytes()
    if _np is not None:
        a = _np.frombuffer(idx, dtype=_np.uint64).copy()
        a[0::2] += delta
        return a.tobytes()
    shifted = array("Q", idx)
    shifted[0::2] = array("Q", [o + delta for o in idx[0::2]])
    return shifted.tobytes()


def merge_ldmb(sources, out_path, *, compress=None) -> int:
    """Merge LDMB files, lists of dicts and/or MemmapList views into one file.

    Sources stored with the same compression settings as the output are merged
    at the byte level: payloads are block-copied in 4 MiB chunks and only the
    offsets in the index are rewritten (numpy-vectorized). Mixed sources are
    converted per record on the fly. Returns the record count written.

    Tip: use this to append, too:  merge(["old.ldmb", new_records], "new.ldmb")
    """
    out_path = os.fspath(out_path)
    out_abs = os.path.abspath(out_path)

    def _path_of(s):
        return s.path if isinstance(s, MemmapList) else s

    # Output compression: explicit arg, else inherited from the first file
    # source, so that merging same-format files is a pure byte copy.
    if compress is True:
        compress = 6
    if compress is None:
        out_zlvl = 0
        for s in sources:
            p = _path_of(s)
            if isinstance(p, (str, os.PathLike)):
                try:
                    with open(p, "rb") as f:
                        hdr = f.read(_HDR_SIZE)
                    if len(hdr) == _HDR_SIZE:
                        magic, _, flags, zlvl = _HEADER.unpack(hdr)
                        if magic == _MAGIC and flags & _ZLIB:
                            out_zlvl = zlvl
                except OSError:
                    pass
                break
    else:
        out_zlvl = int(compress) or 0
    out_flags = _ZLIB if out_zlvl else 0

    plan, opened = [], []
    total = 0
    try:
        for s in sources:
            p = _path_of(s)
            if isinstance(p, (str, os.PathLike)):
                if os.path.abspath(os.fspath(p)) == out_abs:
                    raise ValueError(f"refusing to merge {p!r} into itself")
                src = _FileSource(os.fspath(p))
                opened.append(src)
                plan.append(src)
            elif isinstance(s, (list, tuple)):
                plan.append(_ListSource(s))
            else:
                raise TypeError(f"unsupported source: {type(s).__name__}")
            total += plan[-1].n

        with open(out_path, "wb", buffering=_CHUNK) as out:
            out.write(_HEADER.pack(_MAGIC, _VERSION, out_flags, out_zlvl))
            pos = _HDR_SIZE
            index_chunks = []
            for src in plan:
                if (isinstance(src, _FileSource)
                        and src.flags == out_flags and src.zlvl == out_zlvl):
                    index_chunks.append(_shift_index(src.idx, pos - _HDR_SIZE))
                    src.copy_payload(out)
                    pos += src.payload_size
                else:
                    idx, pos = src.write_records(out, pos, out_zlvl)
                    index_chunks.append(idx.tobytes())
            for chunk in index_chunks:
                out.write(chunk)
            out.write(_TRAILER.pack(pos, total, _MAGIC))
        return total
    finally:
        for src in opened:
            src.close()


# --------------------------------------------------------------------------- #
#  Round-trip / merge self-test + micro-benchmark
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    def make_record(i):
        return {
            "aligned": {"dtime_ms": 10, "aligned": 58, "total": 271},
            "all_chords_good": bool(i % 2),
            "bad_durs": {"bad": 0, "zero": 0, "total": 324, "counts": {}},
            "features_counts": {k: (i + k) % 7 for k in range(200)},   # int keys
            "midi_md5": f"{i:032x}",
            "mono_mels": {},
            "pitches_patches_counts": {(36 + k, 0): k for k in range(60)},  # tuple keys
            "text_lyric_latin": None if i % 3 else "abc",
            "mixed_list": [i, (i, i + 1), {"k": i}],
            "score_notes": 324,
        }

    data = [make_record(i) for i in range(50_000)]

    t = time.perf_counter(); save(data, "bench.ldmb");            t_save = time.perf_counter() - t
    t = time.perf_counter(); out = load("bench.ldmb");            t_load = time.perf_counter() - t
    assert out == data

    with open_ldd("bench.ldmb") as ml:                            # lazy mmap access
        assert len(ml) == len(data)
        assert ml[123] == data[123] and ml[-1] == data[-1]
        assert ml[10:20] == data[10:20]

    save(data[:20_000], "a.ldmb")
    save(data[20_000:35_000], "b.ldmb", compress=6)               # mixed formats
    merge(["a.ldmb", "b.ldmb", data[35_000:]], "ab.ldmb")
    assert load("ab.ldmb") == data

    print(f"{len(data):,} records | save {t_save:.2f}s | load {t_load:.2f}s | "
          f"{os.path.getsize('bench.ldmb') / 1e6:.1f} MB")