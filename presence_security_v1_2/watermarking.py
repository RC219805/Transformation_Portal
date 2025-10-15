# watermarking.py — LSB and DCT-based watermarking (educational reference, v1.2)
# Notes: This is a simple demonstrator for concept and testing; tune thresholds for production.

import hashlib

import numpy as np
from PIL import Image


def _bytes_from_ids(manifest_hash_hex: str, session_id: str):
    mh = bytes.fromhex(manifest_hash_hex)[:16]  # 16 bytes
    sid = hashlib.sha256(session_id.encode("utf-8")).digest()[:16]
    return mh + sid  # 32 bytes payload


def embed_lsb_rgb(img: Image.Image, manifest_hash_hex: str, session_id: str) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    payload = _bytes_from_ids(manifest_hash_hex, session_id)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    h, w, _ = arr.shape
    total = h * w * 3
    if bits.size > total:
        raise ValueError("Image too small for LSB payload")
    flat = arr.reshape(-1)
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    out = flat.reshape(arr.shape)
    return Image.fromarray(out)


# 8x8 DCT helpers (JPEG-like) — naive implementation.
# WARNING: These DCT/IDCT functions use nested loops and have O(N²) time complexity.
# They are suitable for educational/reference use only and are inefficient for large data or production.
# For production or performance-critical code, use scipy.fft.dct and scipy.fft.idct instead.
def _dct_1d(x):
    N = x.shape[0]
    X = np.zeros_like(x, dtype=np.float64)
    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
        c = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        X[k] = c * s
    return X


def _idct_1d(X):
    N = X.shape[0]
    x = np.zeros_like(X, dtype=np.float64)
    for n in range(N):
        s = 0.0
        for k in range(N):
            c = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            s += c * X[k] * np.cos(np.pi * (n + 0.5) * k / N)
        x[n] = s
    return x


def _dct2(block):
    return np.apply_along_axis(_dct_1d, 0, np.apply_along_axis(_dct_1d, 1, block))


def _idct2(block):
    return np.apply_along_axis(_idct_1d, 0, np.apply_along_axis(_idct_1d, 1, block))


def embed_dct_luma(img: Image.Image, manifest_hash_hex: str, session_id: str, strength=2.0) -> Image.Image:
    X = np.array(img.convert("YCbCr"), dtype=np.float32)
    Y = X[:, :, 0]
    payload = _bytes_from_ids(manifest_hash_hex, session_id)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    h, w = Y.shape
    # Process 8x8 blocks; use DCT coefficient (3,2) for embedding.
    # (3,2) is a mid-frequency coefficient (in zig-zag order) chosen as a compromise:
    # - Low-frequency coefficients (top-left) are more robust to compression but changes are more visible.
    # - High-frequency coefficients (bottom-right) are less robust (often quantized away in JPEG).
    # - Mid-frequency coefficients like (3,2) balance robustness (survive compression) and imperceptibility (less visible).
    # This choice is common in watermarking literature for educational and practical reasons.
    bi = 0
    outY = Y.copy().astype(np.float64)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            if y + 8 > h or x + 8 > w or bi >= bits.size:
                continue
            block = outY[y : y + 8, x : x + 8]
            D = _dct2(block)
            bit = 1 if bits[bi] else -1
            D[3, 2] += strength * bit
            block2 = _idct2(D)
            outY[y : y + 8, x : x + 8] = block2
            bi += 1
    X[:, :, 0] = np.clip(outY, 16, 235).astype(np.uint8)
    return Image.fromarray(X.astype(np.uint8), mode="YCbCr").convert("RGB")


def extract_lsb_rgb(img: Image.Image, bitlen=256) -> bytes:
    arr = np.array(img.convert("RGB"))
    flat = arr.reshape(-1)
    bits = flat[:bitlen] & 1
    b = np.packbits(bits).tobytes()
    return b[: bitlen // 8]


def manifest_session_from_lsb(img: Image.Image):
    b = extract_lsb_rgb(img, bitlen=256)  # 32 bytes
    return b[:16], b[16:32]  # (manifest_hash16, session_id16)


def sha3_manifest_hex(manifest_bytes: bytes) -> str:
    return hashlib.sha3_256(manifest_bytes).hexdigest()
