import io

import numpy as np
import pytest
from PIL import Image

from server.paths import SAMPLES
from server.preprocess import (ImageError, Square, crop, default_square, open_image, parse_square, to_batch,
                               view_thumbs)


def test_default_crop_matches_predict_py():
    from predict import to_crop
    for name in ["crack_01.jpg", "dent_02.jpg", "missing-head_03.jpg"]:
        p = SAMPLES / name
        img = open_image(p.read_bytes())
        ours = np.asarray(crop(img, default_square(*img.size)))
        ref = np.asarray(to_crop(p))
        assert ours.shape == (224, 224, 3)
        assert np.array_equal(ours, ref)


def test_exif_orientation_applied():
    base = Image.new("RGB", (300, 200), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    base.save(buf, "JPEG", exif=exif.tobytes())
    img = open_image(buf.getvalue())
    assert img.size == (200, 300)


def test_tiny_image_still_crops():
    buf = io.BytesIO()
    Image.new("RGB", (50, 80), "grey").save(buf, "PNG")
    img = open_image(buf.getvalue())
    sq = default_square(*img.size)
    assert sq == Square(0, 15, 50)
    assert crop(img, sq).size == (224, 224)


def test_non_image_raises():
    with pytest.raises(ImageError):
        open_image(b"definitely not an image")


def test_parse_square_clamps():
    assert parse_square("-40,-10,100", 300, 200) == Square(0, 0, 100)
    assert parse_square("250,150,100", 300, 200) == Square(200, 100, 100)
    assert parse_square("10,10,5000", 300, 200) == Square(10, 0, 200)     # side capped at min(w, h); left still fits
    assert parse_square("10,10,2", 300, 200).side == 16                    # minimum side


@pytest.mark.parametrize("bad", ["", "1,2", "a,b,c", "1,2,3,4", "nan,1,1"])
def test_parse_square_malformed(bad):
    with pytest.raises(ValueError):
        parse_square(bad, 300, 200)


def test_views_and_batch_shapes():
    img = open_image((SAMPLES / "crack_01.jpg").read_bytes())
    c = crop(img, default_square(*img.size))
    views = view_thumbs(c)
    assert len(views) == 8 and all(v.startswith("data:image/jpeg;base64,") for v in views)
    assert tuple(to_batch([c, c]).shape) == (2, 3, 224, 224)


def _png_header_only(w, h):
    import struct
    import zlib
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def test_decompression_bomb_is_an_image_error():
    with pytest.raises(ImageError):
        open_image(_png_header_only(40000, 40000))


def test_huge_photo_is_downscaled_before_cropping():
    from server.preprocess import MAX_SIDE
    buf = io.BytesIO()
    Image.new("RGB", (MAX_SIDE * 2, 300), "grey").save(buf, "JPEG")
    img = open_image(buf.getvalue())
    assert max(img.size) == MAX_SIDE
