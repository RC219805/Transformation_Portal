# watermarking.py — LSB and DCT-based watermarking (production-ready, v1.3)
# Notes: Uses scipy for efficient DCT/IDCT operations. Tune thresholds for production.

import hashlib

import numpy as np
from PIL import Image
from scipy.fft import dct, idct


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


# 8x8 DCT helpers (JPEG-like) — production implementation using scipy.
# Uses scipy.fft.dct/idct for O(N log N) performance instead of O(N²) naive loops.
# Type-2 DCT with orthonormal normalization matches JPEG standard.
def _dct2(block: np.ndarray) -> np.ndarray:
    """Compute 2D DCT-II of an 8x8 block using scipy.

    Args:
        block: 8x8 numpy array

    Returns:
        8x8 DCT coefficients array
    """
    return dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')


def _idct2(block: np.ndarray) -> np.ndarray:
    """Compute 2D inverse DCT-II of an 8x8 block using scipy.

    Args:
        block: 8x8 DCT coefficients array

    Returns:
        8x8 reconstructed block
    """
    return idct(idct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')


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
