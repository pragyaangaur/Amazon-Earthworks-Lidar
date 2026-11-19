"""Minimal, dependency-free reader for R ``.rds`` files.

Purpose: the record check must still run when ``pyreadr`` is absent and R is not
installed.  This reader covers exactly what the known-earthwork table needs --
a ``data.frame`` of atomic columns (doubles, integers, logicals, characters,
factors, complexes, raw) plus its attributes -- and refuses anything else with a
named error instead of guessing.

It reproduces the layout observed from files written by R 4.5.3 (see
``tests/fixtures/make_probe.R``): gzip/bzip2/xz compression, XDR (big-endian)
stream, serialization version 2 or 3, symbols interned in the stream's reference
table, attributes as tagged pairlists, ``row.names`` in R's compact form,
``NA_real_`` as the NaN bit pattern 0x7ff00000000007a2 and ``NA_integer_`` as
INT_MIN.

Deliberate limits, both reported as ``RDSNotSupported``:
  * ASCII and native-binary streams (write with the default XDR stream),
  * ALTREP compact vectors (serialized as SEXP type 238),
  * environments, closures, language objects, external pointers, S4 objects.
A caller that hits one should use the pyreadr or R backend.

Verified by parity against R's own ``write.csv`` output and against pyreadr on
the synthetic fixtures (tests/test_rds_parity.py).
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import math
import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = ["read_rds", "to_dataframe", "to_python", "RDSFormatError", "RDSNotSupported"]

# SEXP type codes as used in the serialized stream (Rinternals.h)
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP = 0, 1, 2, 3, 4
PROMSXP, LANGSXP, SPECIALSXP, BUILTINSXP, CHARSXP = 5, 6, 7, 8, 9
LGLSXP, INTSXP, REALSXP, CPLXSXP, STRSXP = 10, 13, 14, 15, 16
DOTSXP, ANYSXP, VECSXP, EXPRSXP, BCODESXP = 17, 18, 19, 20, 21
EXTPTRSXP, WEAKREFSXP, RAWSXP, S4SXP = 22, 23, 24, 25
ALTREP_SXP = 238
BASEENV_SXP, EMPTYENV_SXP = 241, 242
NAMESPACESXP, PACKAGESXP = 249, 250
MISSINGARG_SXP, UNBOUNDVALUE_SXP, GLOBALENV_SXP, NILVALUE_SXP, REFSXP = 251, 252, 253, 254, 255

TYPE_NAMES = {
    NILSXP: "NILSXP", SYMSXP: "SYMSXP", LISTSXP: "LISTSXP", CLOSXP: "CLOSXP", ENVSXP: "ENVSXP",
    PROMSXP: "PROMSXP", LANGSXP: "LANGSXP", SPECIALSXP: "SPECIALSXP", BUILTINSXP: "BUILTINSXP",
    CHARSXP: "CHARSXP", LGLSXP: "LGLSXP", INTSXP: "INTSXP", REALSXP: "REALSXP", CPLXSXP: "CPLXSXP",
    STRSXP: "STRSXP", DOTSXP: "DOTSXP", ANYSXP: "ANYSXP", VECSXP: "VECSXP", EXPRSXP: "EXPRSXP",
    BCODESXP: "BCODESXP", EXTPTRSXP: "EXTPTRSXP", WEAKREFSXP: "WEAKREFSXP", RAWSXP: "RAWSXP",
    S4SXP: "S4SXP", ALTREP_SXP: "ALTREP_SXP", NILVALUE_SXP: "NILVALUE_SXP", REFSXP: "REFSXP",
}

# CHARSXP "levels" field carries the encoding flags
BYTES_MASK, LATIN1_MASK, UTF8_MASK, ASCII_MASK = 2, 4, 8, 64

NA_REAL_BYTES = b"\x7f\xf0\x00\x00\x00\x00\x07\xa2"
NA_INT = -2147483648


class RDSFormatError(Exception):
    """The stream is malformed or not an .rds stream this reader understands."""


class RDSNotSupported(Exception):
    """The stream is valid R serialization but uses a feature outside this reader."""


class _Nil:
    """R's NULL, distinct from NA (which is None in the converted output)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NULL"

    def __bool__(self) -> bool:
        return False


NIL = _Nil()


@dataclass
class RSymbol:
    name: str

    def __repr__(self) -> str:
        return f"<sym {self.name}>"


@dataclass
class RVector:
    """Any atomic vector or generic vector, with its attributes."""

    rtype: int
    values: object
    attrs: dict = field(default_factory=dict)

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.rtype, f"SEXP({self.rtype})")


@dataclass
class RPairlist:
    """Tagged pairlist: ordered (tag, value) pairs plus attributes."""

    items: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


class _Reader:
    def __init__(self, buf: bytes, verbose: bool = False):
        self.buf = buf
        self.pos = 0
        self.refs: list[object] = []
        self.verbose = verbose

    # -- primitives --------------------------------------------------------
    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise RDSFormatError(f"unexpected end of stream at byte {self.pos} (wanted {n} bytes)")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def int32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def uint32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def doubles(self, n: int) -> np.ndarray:
        raw = self.take(8 * n)
        out = np.empty(n, dtype=float)
        for i in range(n):
            chunk = raw[8 * i:8 * i + 8]
            out[i] = math.nan if chunk == NA_REAL_BYTES else struct.unpack(">d", chunk)[0]
        return out

    # -- items -------------------------------------------------------------
    def read_item(self):
        raw_flags = self.uint32()
        # R packs a reference as  REFSXP | (index << 8); index 0 means the index
        # follows as a separate int (used for indices above INT_MAX >> 8).
        if (raw_flags & 0xFF) == REFSXP:
            idx = raw_flags >> 8
            if idx == 0:
                idx = self.int32()
            if not (1 <= idx <= len(self.refs)):
                raise RDSFormatError(f"reference index {idx} outside table of {len(self.refs)}")
            return self.refs[idx - 1]

        rtype = raw_flags & 0xFF
        levels = raw_flags >> 12
        has_attr = bool(raw_flags & (1 << 9))
        has_tag = bool(raw_flags & (1 << 10))
        object_bit = bool(raw_flags & (1 << 8))

        if rtype == NILVALUE_SXP:
            return NIL
        if rtype == SYMSXP:
            name = self.read_item()
            if not isinstance(name, str):
                raise RDSFormatError(f"symbol print name is {type(name).__name__}, not a string")
            sym = RSymbol(name)
            self.refs.append(sym)
            return sym
        if rtype == CHARSXP:
            return self._read_charsxp(levels)
        if rtype in (LISTSXP, LANGSXP, DOTSXP):
            return self._read_pairlist(raw_flags, rtype, has_attr, has_tag)
        if rtype in (LGLSXP, INTSXP, REALSXP, CPLXSXP, STRSXP, VECSXP, EXPRSXP):
            vec = self._read_vector(rtype, levels)
            if has_attr:
                vec.attrs = self._read_attrs()
            return vec
        if rtype == RAWSXP:
            n = self.int32()
            vec = RVector(RAWSXP, self.take(n))
            if has_attr:
                vec.attrs = self._read_attrs()
            return vec
        if rtype == ALTREP_SXP:
            raise RDSNotSupported(
                "ALTREP compact vector (SEXP type 238) at byte "
                f"{self.pos - 4}; use the pyreadr or R backend for this file")
        raise RDSNotSupported(
            f"SEXP type {rtype} ({TYPE_NAMES.get(rtype, '?')}) at byte {self.pos - 4}; "
            "use the pyreadr or R backend for this file")

    def _read_charsxp(self, levels: int):
        n = self.int32()
        if n == -1:
            return None  # NA_character_
        if n < 0:
            raise RDSFormatError(f"negative CHARSXP length {n}")
        raw = self.take(n)
        if levels & UTF8_MASK:
            enc = "utf-8"
        elif levels & LATIN1_MASK:
            enc = "latin-1"
        else:
            enc = "utf-8"
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    def _read_pairlist(self, raw_flags: int, rtype: int, has_attr: bool, has_tag: bool):
        node = RPairlist()
        if has_attr:
            node.attrs = self._read_attrs()
        while True:
            tag = self.read_item() if has_tag else NIL
            value = self.read_item()
            if rtype == LISTSXP:
                node.items.append((None if tag is NIL else tag, value))
            else:
                node.items.append((None if tag is NIL else tag, value))
            nxt_raw = self.uint32()
            if (nxt_raw & 0xFF) == REFSXP:
                idx = nxt_raw >> 8
                if idx == 0:
                    idx = self.int32()
                nxt = self.refs[idx - 1]
                if isinstance(nxt, RPairlist):
                    node.items.extend(nxt.items)
                break
            nxt_type = nxt_raw & 0xFF
            has_attr = bool(nxt_raw & (1 << 9))
            has_tag = bool(nxt_raw & (1 << 10))
            if nxt_type == NILVALUE_SXP:
                break
            if nxt_type != rtype:
                raise RDSNotSupported(
                    f"mixed pairlist tail at byte {self.pos - 4}: "
                    f"{TYPE_NAMES.get(nxt_type, nxt_type)}")
        return node

    def _read_vector(self, rtype: int, levels: int) -> RVector:
        n = self.int32()
        if n < 0:
            raise RDSFormatError(f"negative vector length {n} for {TYPE_NAMES.get(rtype)}")
        if rtype == REALSXP:
            return RVector(rtype, self.doubles(n))
        if rtype == INTSXP:
            vals = list(struct.unpack(f">{n}i", self.take(4 * n))) if n else []
            return RVector(rtype, [None if v == NA_INT else int(v) for v in vals])
        if rtype == LGLSXP:
            vals = list(struct.unpack(f">{n}i", self.take(4 * n))) if n else []
            return RVector(rtype, [None if v == NA_INT else bool(v) for v in vals])
        if rtype == CPLXSXP:
            re = self.doubles(n)
            im = self.doubles(n)
            return RVector(rtype, [None if (math.isnan(a) or math.isnan(b)) else complex(a, b)
                                   for a, b in zip(re, im)])
        if rtype == STRSXP:
            return RVector(rtype, [self.read_item() for _ in range(n)])
        if rtype in (VECSXP, EXPRSXP):
            return RVector(rtype, [self.read_item() for _ in range(n)])
        raise RDSNotSupported(f"vector type {TYPE_NAMES.get(rtype, rtype)}")

    def _read_attrs(self) -> dict:
        node = self.read_item()
        if not isinstance(node, RPairlist):
            raise RDSFormatError(f"attributes are {type(node).__name__}, not a pairlist")
        out = {}
        for tag, value in node.items:
            key = tag.name if isinstance(tag, RSymbol) else str(tag)
            out[key] = value
        return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def _decompress(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    if raw[:3] == b"BZh":
        return bz2.decompress(raw)
    if raw[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(raw)
    return raw


def read_rds(path: str, verbose: bool = False):
    """Read one ``.rds`` file and return its R object as Python/RSymbol/RVector."""
    with open(path, "rb") as fh:
        raw = fh.read()
    buf = _decompress(raw)
    marker = buf[:2]
    if marker != b"X\n":
        if marker in (b"A\n", b"B\n"):
            raise RDSNotSupported(
                f"{path}: stream format {marker!r} (ascii/native-binary) is not supported; "
                "write the file with the default XDR stream or use the pyreadr/R backend")
        raise RDSFormatError(f"{path}: not an RDS stream (starts with {buf[:8]!r})")
    r = _Reader(buf, verbose=verbose)
    r.pos = 2
    version = r.int32()
    if version not in (2, 3):
        raise RDSNotSupported(f"{path}: serialization version {version} not supported (2 or 3)")
    r.int32()  # writer R version
    r.int32()  # minimum reader R version
    if version >= 3:
        n = r.int32()
        r.take(n)  # native encoding of the writing session
    obj = r.read_item()
    if r.pos != len(buf):
        raise RDSFormatError(
            f"{path}: {len(buf) - r.pos} trailing byte(s) after the object at byte {r.pos} "
            f"(next bytes {buf[r.pos:r.pos + 8]!r}); the stream was not fully understood")
    return obj


def _vector_to_python(vec: RVector):
    """Convert an atomic vector, honouring factor/integer64/Date attributes where cheap."""
    if vec.attrs.get("class") and "factor" in [str(c) for c in _as_class_list(vec.attrs["class"])]:
        codes = vec.values
        levs = vec.attrs.get("levels")
        labels = levs.values if isinstance(levs, RVector) else []
        out = []
        for c in codes:
            if c is None or c < 1 or c > len(labels):
                out.append(None)
            else:
                out.append(labels[c - 1])
        return out
    if vec.rtype == REALSXP:
        vals = vec.values
        return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                for v in np.asarray(vals, dtype=float)]
    if vec.rtype in (INTSXP, LGLSXP):
        return None if vec.values is None else list(vec.values)
    if vec.rtype == STRSXP:
        return list(vec.values)
    if vec.rtype == VECSXP:
        return [to_python(v) for v in vec.values]
    if vec.rtype == RAWSXP:
        return vec.values
    return vec.values


def _as_class_list(node):
    if isinstance(node, RVector) and node.rtype in (STRSXP, VECSXP):
        return list(node.values)
    return []


def to_python(obj):
    """Recursively convert the parsed R object to plain Python values."""
    if isinstance(obj, RVector):
        return _vector_to_python(obj)
    if isinstance(obj, RPairlist):
        return {k: to_python(v) for k, v in obj.items}
    if isinstance(obj, RSymbol):
        return obj.name
    if obj is NIL:
        return None
    return obj


def _row_names(attrs: dict, n: int) -> list[str] | None:
    rn = attrs.get("row.names")
    if not isinstance(rn, RVector):
        return None
    vals = rn.values
    # R's compact row.names form is c(NA, n) -- observed with both signs: a
    # data.frame() call writes -n, a subsetted frame writes +n.  Either way the
    # labels are the default sequence, so return None.
    if rn.rtype == INTSXP and len(vals) == 2 and vals[0] is None:
        return None
    if rn.rtype in (STRSXP, INTSXP):
        return [str(v) if v is not None else "" for v in vals]
    return None


def to_dataframe(obj):
    """Convert a parsed R ``data.frame`` to a pandas DataFrame."""
    import pandas as pd

    if not isinstance(obj, RVector) or obj.rtype != VECSXP:
        raise RDSFormatError(f"top-level object is {type(obj).__name__}, not a list/data.frame")
    classes = [str(c) for c in _as_class_list(obj.attrs.get("class", None))]
    if "data.frame" not in classes:
        raise RDSFormatError(f"top-level object has class {classes or 'none'}, not 'data.frame'")
    names = _as_class_list(obj.attrs.get("names"))
    if len(names) != len(obj.values):
        raise RDSFormatError(f"data.frame has {len(obj.values)} columns but {len(names)} names")
    data = {}
    n = None
    for name, column in zip(names, obj.values):
        vals = to_python(column)
        if not isinstance(vals, list):
            raise RDSFormatError(f"column {name!r} is not a vector")
        if n is None:
            n = len(vals)
        elif len(vals) != n:
            raise RDSFormatError(f"column {name!r} has length {len(vals)}, expected {n}")
        if all(v is None or isinstance(v, (int, float, bool, str)) for v in vals):
            data[name] = pd.Series(vals, dtype=object if any(isinstance(v, str) for v in vals) else None)
        else:
            data[name] = pd.Series(vals, dtype=object)
    df = pd.DataFrame(data)
    rn = _row_names(obj.attrs, n or 0)
    if rn is not None:
        df.index = rn
    return df
