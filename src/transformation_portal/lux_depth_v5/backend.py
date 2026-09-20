"""V5 depth/sky transport on the governed cancellable photographic session."""

from __future__ import annotations

from typing import ClassVar, Mapping

import numpy as np

from transformation_portal.lux_depth_v4.backend import DA3Session as GovernedDA3Session
from transformation_portal.lux_depth_v4.backend import probe_device, require_process_supervisor

INFERENCE_RECIPE = "tp.da3.explicit_precision_sky.v1"
SKY_MASK_POLICY = "da3_mono_sky_ge_0_3"


class DA3Session(GovernedDA3Session):
    """Carry positive-domain evidence without fabricating an unavailable sky mask."""

    _WORKER_MODULE = "transformation_portal.lux_depth_v5.worker"
    _ARRAY_DTYPES: ClassVar[Mapping[str, np.dtype]] = {
        "native_depth": np.dtype("float32"),
        "sky_mask": np.dtype("bool"),
    }

    def compute(self, proxy: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        arrays, response = super().compute(proxy)
        precision = self.plan.to_payload()["configuration"]["depth"]["precision"]
        if (
            response.get("precision") != precision
            or response.get("inference_recipe") != INFERENCE_RECIPE
            or response.get("native_semantics") != "da3_metric_uncalibrated"
            or response.get("sky_mask_policy") != SKY_MASK_POLICY
            or type(response.get("sky_available")) is not bool
            or response["sky_available"] != ("sky_mask" in arrays)
            or response.get("confidence_available") is not False
        ):
            raise RuntimeError("V5 worker inference receipt differs from the prepared depth contract")
        return arrays, response


__all__ = ["DA3Session", "INFERENCE_RECIPE", "SKY_MASK_POLICY", "probe_device", "require_process_supervisor"]
