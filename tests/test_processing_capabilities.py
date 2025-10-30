from luxury_tiff_batch_processor import LuxuryGradeException, ProcessingCapabilities
import pytest

pytest.importorskip("numpy")


class _StubTiffFile:
    def __init__(self, *, supports_hdr=True, provide_writer=True):
        self.supports_hdr = supports_hdr
        if provide_writer:
            self.imwrite = object()


def test_capabilities_without_tifffile_dependency():
    capabilities = ProcessingCapabilities(tifffile_module=None)

    assert capabilities.bit_depth == 8
    assert capabilities.hdr_capable is False

    with pytest.raises(LuxuryGradeException):
        capabilities.assert_luxury_grade()


def test_capabilities_with_hdr_supporting_dependency():
    capabilities = ProcessingCapabilities(tifffile_module=_StubTiffFile())

    assert capabilities.bit_depth == 16
    assert capabilities.hdr_capable is True

    # No exception should be raised when the environment meets the requirements.
    capabilities.assert_luxury_grade()


def test_capabilities_detect_hdr_limitations():
    capabilities = ProcessingCapabilities(
        tifffile_module=_StubTiffFile(supports_hdr=False)
    )

    assert capabilities.bit_depth == 16
    assert capabilities.hdr_capable is False

    with pytest.raises(LuxuryGradeException):
        capabilities.assert_luxury_grade()


def test_capabilities_detect_writer_absence():
    capabilities = ProcessingCapabilities(
        tifffile_module=_StubTiffFile(provide_writer=False)
    )

    assert capabilities.bit_depth == 8
    assert capabilities.hdr_capable is False

    with pytest.raises(LuxuryGradeException):
        capabilities.assert_luxury_grade()
