#!/usr/bin/env python3
"""Add the current P3 DNS criterion to the methodology PDF without replacing its pages."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "HELIX_spall_detection_methodology.pdf"
MARKER = b"P3 positivity DNS update"


def _object_body(pdf: bytes, object_number: int) -> bytes:
    pattern = rb"(?m)^" + str(object_number).encode() + rb" 0 obj\n(.*?)\nendobj"
    match = re.search(pattern, pdf, re.DOTALL)
    if match is None:
        raise ValueError(f"PDF object {object_number} was not found")
    return match.group(1)


def _write_object(buffer: bytearray, offsets: dict[int, int], object_number: int, body: bytes) -> None:
    offsets[object_number] = len(buffer)
    buffer.extend(f"{object_number} 0 obj\n".encode())
    buffer.extend(body)
    buffer.extend(b"\nendobj\n")


def update_pdf(input_path: Path, output_path: Path) -> None:
    pdf = input_path.read_bytes()
    if MARKER in pdf:
        raise ValueError(f"{input_path.name} already contains the P3 DNS update")

    startxref_match = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", pdf)
    if startxref_match is None:
        raise ValueError("PDF has no readable xref table")
    previous_xref = int(startxref_match.group(1))

    trailer = pdf[previous_xref:]
    size_match = re.search(rb"/Size\s+(\d+)", trailer)
    root_match = re.search(rb"/Root\s+(\d+\s+\d+\s+R)", trailer)
    info_match = re.search(rb"/Info\s+(\d+\s+\d+\s+R)", trailer)
    if size_match is None or root_match is None:
        raise ValueError("PDF trailer is missing required Size or Root entries")

    next_object = int(size_match.group(1))
    overlay_object = next_object
    font_object = next_object + 1

    # Revise the shared resource dictionary with a built-in Type 1 font.  The
    # existing Matplotlib fonts and all page resources remain unchanged.
    resources = _object_body(pdf, 8)
    original_font_ref = b"/Font 3 0 R"
    if original_font_ref not in resources:
        raise ValueError("Unexpected PDF resource dictionary; cannot add overlay font")
    revised_resources = resources.replace(
        original_font_ref,
        b"/Font << /F1 39 0 R /F4 99 0 R /F2 142 0 R /F3 237 0 R "
        + f"/F5 {font_object} 0 R >>".encode(),
        1,
    )

    # The first page has open space above its footer.  This overlay is a
    # permanent PDF content stream, not a viewer-only note.
    overlay_stream = b"""% P3 positivity DNS update
q
0.98 0.94 0.78 rg
35 130 542 91 re f
0.76 0.52 0.08 RG
1.0 w
35 130 542 91 re S
BT
/F5 12 Tf
0.10 0.20 0.35 rg
48 199 Td
(DNS update: P3 velocity must be > 0 m/s) Tj
0 -18 Td
/F5 9.5 Tf
(Any trace with a pullback minimum P3 at zero or below zero velocity is DNS.) Tj
0 -14 Td
(This physical criterion is checked before the P4 recompression gates.) Tj
0 -14 Td
(No spall strength is reported for these traces; the diagnostic fit is retained.) Tj
ET
Q
"""
    overlay = b"<< /Length " + str(len(overlay_stream)).encode() + b" >>\nstream\n" + overlay_stream + b"endstream"

    first_page = _object_body(pdf, 11)
    contents_ref = b"/Contents 9 0 R"
    if contents_ref not in first_page:
        raise ValueError("Unexpected first-page contents; cannot add overlay")
    revised_first_page = first_page.replace(
        contents_ref, f"/Contents [ 9 0 R {overlay_object} 0 R]".encode(), 1
    )

    update = bytearray(pdf)
    if not update.endswith(b"\n"):
        update.extend(b"\n")
    offsets: dict[int, int] = {}
    _write_object(update, offsets, overlay_object, overlay)
    _write_object(update, offsets, font_object, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    _write_object(update, offsets, 8, revised_resources)
    _write_object(update, offsets, 11, revised_first_page)

    xref_position = len(update)
    update.extend(b"xref\n")
    for object_number in (8, 11):
        update.extend(f"{object_number} 1\n{offsets[object_number]:010d} 00000 n \n".encode())
    update.extend(f"{overlay_object} 2\n".encode())
    for object_number in (overlay_object, font_object):
        update.extend(f"{offsets[object_number]:010d} 00000 n \n".encode())

    info_entry = b"" if info_match is None else b" /Info " + info_match.group(1)
    update.extend(
        b"trailer\n<< /Size "
        + str(font_object + 1).encode()
        + b" /Root "
        + root_match.group(1)
        + info_entry
        + b" /Prev "
        + str(previous_xref).encode()
        + b" >>\nstartxref\n"
        + str(xref_position).encode()
        + b"\n%%EOF\n"
    )

    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:
        temporary.write(update)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    update_pdf(args.input.resolve(), args.output.resolve())
    print(f"Updated {args.output}")


if __name__ == "__main__":
    main()
