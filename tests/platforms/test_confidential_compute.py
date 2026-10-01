# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NVIDIA Confidential Computing detection and stock defaults."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

# platforms.cuda imports compiled extensions at module import time. Stub them
# so these mocked unit tests run without a full CUDA build.
for _mod in (
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm._qutlass_C",
):
    sys.modules.setdefault(_mod, ModuleType(_mod))

from vllm.platforms.interface import ConfidentialComputeProfile  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cc_caches():
    """Detection is cached; clear between tests."""
    from vllm.platforms import cuda as cuda_plat

    cuda_plat._detect_confidential_compute_profile.cache_clear()
    yield
    cuda_plat._detect_confidential_compute_profile.cache_clear()
    try:
        from vllm.v1 import conf_compute_utils

        conf_compute_utils.confidential_compute_enabled.cache_clear()
    except Exception:
        pass


def test_profile_ppcie_without_cc_feature_is_bounce_buffer():
    """Hopper PPCIe: ccFeature OFF but multiGpuMode PROTECTED_PCIE."""
    from vllm.platforms import cuda as cuda_plat
    from vllm.third_party import pynvml

    profile = cuda_plat._profile_from_nvml_fields(
        cc_feature=pynvml.NVML_CC_SYSTEM_FEATURE_DISABLED,
        multi_gpu_mode=pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE,
        devtools=pynvml.NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF,
        source="nvml_settings",
    )
    assert profile.host_device_copies_synchronous is True
    assert profile.cc_feature_enabled is False
    assert profile.multi_gpu_mode == pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE
    assert profile.nvlink_encrypted is False
    assert profile.enabled is True


def test_profile_classic_cc_and_devtools():
    from vllm.platforms import cuda as cuda_plat
    from vllm.third_party import pynvml

    classic = cuda_plat._profile_from_nvml_fields(
        cc_feature=pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED,
        multi_gpu_mode=pynvml.NVML_CC_SYSTEM_MULTIGPU_NONE,
        devtools=pynvml.NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF,
        source="nvml_settings",
    )
    assert classic.host_device_copies_synchronous is True
    assert classic.nvlink_encrypted is True

    # DevTools alone still counts as bounce-buffer CC.
    devtools = cuda_plat._profile_from_nvml_fields(
        cc_feature=pynvml.NVML_CC_SYSTEM_FEATURE_DISABLED,
        multi_gpu_mode=pynvml.NVML_CC_SYSTEM_MULTIGPU_NONE,
        devtools=pynvml.NVML_CC_SYSTEM_DEVTOOLS_MODE_ON,
        source="nvml_settings",
    )
    assert devtools.host_device_copies_synchronous is True
    assert devtools.devtools is True


def test_profile_disabled_when_no_cc_signals():
    from vllm.platforms import cuda as cuda_plat
    from vllm.third_party import pynvml

    profile = cuda_plat._profile_from_nvml_fields(
        cc_feature=pynvml.NVML_CC_SYSTEM_FEATURE_DISABLED,
        multi_gpu_mode=pynvml.NVML_CC_SYSTEM_MULTIGPU_NONE,
        devtools=pynvml.NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF,
        source="nvml_settings",
    )
    assert profile.host_device_copies_synchronous is False
    assert profile.enabled is False


def test_profile_nvle_constant():
    from vllm.platforms import cuda as cuda_plat
    from vllm.third_party import pynvml

    assert pynvml.NVML_CC_SYSTEM_MULTIGPU_NVLE == 2

    profile = cuda_plat._profile_from_nvml_fields(
        cc_feature=pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED,
        multi_gpu_mode=pynvml.NVML_CC_SYSTEM_MULTIGPU_NVLE,
        devtools=pynvml.NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF,
        source="nvml_settings",
    )
    assert profile.host_device_copies_synchronous is True
    assert profile.nvlink_encrypted is True


def test_env_override_forces_profile(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms import cuda as cuda_plat

    monkeypatch.setenv("VLLM_CONFIDENTIAL_COMPUTE", "1")
    cuda_plat._detect_confidential_compute_profile.cache_clear()
    on = cuda_plat._detect_confidential_compute_profile()
    assert on.host_device_copies_synchronous is True
    assert on.source == "env"

    monkeypatch.setenv("VLLM_CONFIDENTIAL_COMPUTE", "0")
    cuda_plat._detect_confidential_compute_profile.cache_clear()
    off = cuda_plat._detect_confidential_compute_profile()
    assert off.host_device_copies_synchronous is False
    assert off.source == "env"


def test_detection_uses_settings_profile(monkeypatch: pytest.MonkeyPatch):
    """Cached detection should consume the Settings/PPCIe-aware profile."""
    from vllm.platforms import cuda as cuda_plat
    from vllm.third_party import pynvml

    monkeypatch.delenv("VLLM_CONFIDENTIAL_COMPUTE", raising=False)
    monkeypatch.setattr(cuda_plat.torch.cuda, "is_available", lambda: True)

    expected = ConfidentialComputeProfile(
        host_device_copies_synchronous=True,
        cc_feature_enabled=False,
        multi_gpu_mode=pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE,
        source="nvml_settings",
    )
    monkeypatch.setattr(cuda_plat, "_nvml_read_conf_compute_profile", lambda: expected)

    cuda_plat._detect_confidential_compute_profile.cache_clear()
    profile = cuda_plat._detect_confidential_compute_profile()
    assert profile is expected
    assert cuda_plat.CudaPlatformBase.is_confidential_compute() is True
    assert (
        cuda_plat.CudaPlatformBase.get_confidential_compute_profile().source
        == "nvml_settings"
    )


def test_platform_default_profile_disabled():
    from vllm.platforms.interface import Platform

    profile = Platform.get_confidential_compute_profile()
    assert isinstance(profile, ConfidentialComputeProfile)
    assert profile.host_device_copies_synchronous is False
    assert Platform.is_confidential_compute() is False


@pytest.fixture
def resolve_stock_cc_defaults(monkeypatch: pytest.MonkeyPatch):
    """Run the real ``VllmConfig`` default-resolution code on a stand-in config.

    ``use_v2_model_runner`` and ``_async_scheduling_off_by_cc_default`` are the
    real implementations, bound to a ``SimpleNamespace`` so no model, GPU, or
    full ``VllmConfig`` is needed. Only their inputs are faked: the platform,
    ``VLLM_USE_V2_MODEL_RUNNER``, Triton availability, and which features each
    runner supports.
    """
    from vllm.config import VllmConfig
    from vllm.config import vllm as vllm_config_mod

    def _resolve(
        *,
        cc: bool,
        is_cuda: bool = True,
        user_v2: bool | None = None,
        has_triton: bool = True,
        v1_unsupported: tuple[str, ...] = (),
        v2_unsupported: tuple[str, ...] = (),
    ) -> tuple[bool, bool]:
        """Return ``(use_v2_model_runner, async_off_by_cc_default)``."""
        if user_v2 is None:
            monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
        else:
            monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if user_v2 else "0")
        monkeypatch.setattr(vllm_config_mod, "HAS_TRITON", has_triton)

        platform = MagicMock()
        platform.is_rocm.return_value = False
        platform.is_cuda.return_value = is_cuda
        platform.is_confidential_compute.return_value = cc
        monkeypatch.setattr("vllm.platforms.current_platform", platform)

        cfg = SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None),
            model_config=None,
            _get_v1_model_runner_unsupported_features=lambda: list(v1_unsupported),
            _get_v2_model_runner_unsupported_features=lambda: list(v2_unsupported),
        )
        use_v2 = VllmConfig.use_v2_model_runner.fget(cfg)
        cfg.use_v2_model_runner = use_v2
        async_off = VllmConfig._async_scheduling_off_by_cc_default(cfg)
        return use_v2, async_off

    return _resolve


def test_cc_stock_defaults_prefer_v1_and_keep_async_on(resolve_stock_cc_defaults):
    """Stock CC serve: V1 has the #52226 copy opts, so keep async scheduling."""
    use_v2, async_off = resolve_stock_cc_defaults(cc=True)
    assert use_v2 is False
    assert async_off is False


def test_cc_forced_v2_defaults_async_off(resolve_stock_cc_defaults):
    """Explicit V2 under CC has no copy opts: async defaults off."""
    use_v2, async_off = resolve_stock_cc_defaults(cc=True, user_v2=True)
    assert use_v2 is True
    assert async_off is True


def test_cc_v1_unsupported_falls_back_to_v2_and_async_off(resolve_stock_cc_defaults):
    """If V1 cannot serve the config, do not force it: fall back to V2."""
    use_v2, async_off = resolve_stock_cc_defaults(
        cc=True, v1_unsupported=("some_feature",)
    )
    assert use_v2 is True
    assert async_off is True


def test_cc_explicit_v1_env_is_respected(resolve_stock_cc_defaults):
    use_v2, async_off = resolve_stock_cc_defaults(cc=True, user_v2=False)
    assert use_v2 is False
    assert async_off is False


def test_non_cc_defaults_unchanged(resolve_stock_cc_defaults):
    """Without CC the stock runner stays V2 and async is not turned off."""
    use_v2, async_off = resolve_stock_cc_defaults(cc=False)
    assert use_v2 is True
    assert async_off is False


def test_cc_default_ignored_off_cuda(resolve_stock_cc_defaults):
    """The V1-under-CC runner default is CUDA-only."""
    use_v2, _ = resolve_stock_cc_defaults(cc=True, is_cuda=False)
    assert use_v2 is True


def test_cuda_check_and_update_config_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    from vllm.platforms import cuda as cuda_plat

    monkeypatch.setattr(
        cuda_plat.CudaPlatformBase,
        "is_confidential_compute",
        classmethod(lambda cls: True),
    )

    model_config = MagicMock()
    model_config.enforce_eager = True
    model_config.is_mm_prefix_lm = False

    offload = MagicMock()
    offload.uva.cpu_offload_gb = 4.0
    offload.uva.cpu_offload_params = set()
    offload.prefetch.offload_group_size = 0

    vllm_config = MagicMock()
    vllm_config.parallel_config.worker_cls = "already_set"
    vllm_config.model_config = model_config
    vllm_config.scheduler_config.is_multimodal_model = False
    vllm_config.scheduler_config.disable_chunked_mm_input = True
    vllm_config.offload_config = offload
    vllm_config.compilation_config.cudagraph_mode = False

    with caplog.at_level("WARNING"):
        cuda_plat.CudaPlatformBase.check_and_update_config(vllm_config)

    text = "\n".join(r.message for r in caplog.records)
    assert "enforce-eager" in text
    assert "CPU offload" in text
