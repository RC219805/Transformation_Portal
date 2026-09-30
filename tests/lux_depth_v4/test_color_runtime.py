"""CMS identity cannot borrow authority from another wheel or unchanged version."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from transformation_portal.lux_depth_v4 import color_runtime

pytestmark = pytest.mark.unit


@pytest.fixture
def cms_wheel(tmp_path, monkeypatch):
    package = tmp_path / "imagecodecs"
    package.mkdir()
    facade = package / "__init__.py"
    extension = package / "_cms.abi3.so"
    library = package / ".dylibs" / "liblcms2.2.dylib"
    library.parent.mkdir()
    for path in (facade, extension, library):
        path.write_bytes(path.name.encode())
    modules = {
        "imagecodecs": SimpleNamespace(__file__=str(facade), cms_version=lambda: "lcms2 2.18.0"),
        "imagecodecs._cms": SimpleNamespace(__file__=str(extension)),
    }
    files = [facade, extension, library]

    def distribution(*args, **kwargs):
        digests = [color_runtime.identity._hash_regular_file(path)[0] for path in files]
        return {
            "version": "2026.3.6",
            "record_sha256": "a" * 64,
            "installed_files_sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
        }

    monkeypatch.setattr(color_runtime.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(color_runtime.identity, "_distribution_record", distribution)
    return SimpleNamespace(modules=modules, extension=extension, library=library, files=files)


@pytest.mark.parametrize("artifact", ["extension", "library"])
def test_same_version_native_byte_change_changes_cms_identity(cms_wheel, artifact):
    before = color_runtime.color_runtime_identity()
    path = getattr(cms_wheel, artifact)
    path.write_bytes(path.read_bytes() + b"changed implementation")
    after = color_runtime.color_runtime_identity()
    assert after["imagecodecs_version"] == before["imagecodecs_version"]
    assert after["lcms_version"] == before["lcms_version"]
    assert after["wheel_files_sha256"] != before["wheel_files_sha256"]
    field = "cms_extension_sha256" if artifact == "extension" else "bundled_lcms"
    assert after[field] != before[field]


@pytest.mark.parametrize("module", ["imagecodecs", "imagecodecs._cms"])
def test_shadow_cms_cannot_borrow_installed_wheel_identity(cms_wheel, module, tmp_path):
    shadow = tmp_path / "shadow.so"
    shadow.write_bytes(b"unrelated module")
    cms_wheel.modules[module].__file__ = str(shadow)
    with pytest.raises(ValueError, match="outside its materialized wheel"):
        color_runtime.color_runtime_identity()


@pytest.mark.parametrize("engine", ["pillow 2.18.0", "", "x" * 129])
def test_unrecognized_cms_engine_cannot_authorize_replay(cms_wheel, engine):
    cms_wheel.modules["imagecodecs"].cms_version = lambda: engine
    with pytest.raises(ValueError, match="CMS"):
        color_runtime.color_runtime_identity()


def test_cms_requires_a_wheel_owned_lcms_library(cms_wheel):
    # A wheel without a retained native LCMS binding cannot substitute a host
    # library discovered elsewhere on the filesystem.
    cms_wheel.files.remove(cms_wheel.library)
    with pytest.raises(ValueError, match="wheel-bundled LCMS"):
        color_runtime.color_runtime_identity()
