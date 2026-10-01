# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NVIDIA Confidential Computing detection and stock defaults."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.platforms.interface import ConfidentialComputeProfile


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
    monkeypatch.setattr(
        cuda_plat, "_nvml_read_conf_compute_profile", lambda: expected
    )

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


def test_cc_stock_policy_table():
    """Documented stock policy: async on iff bounce-buffer copy opts active.

    Copy opts ship on Model Runner V1 only today (#52226). Prefer V1 under CC
    when unset; if V2 is forced, default async off.
    """

    def resolve_async(*, cc: bool, use_v2: bool, user_async: bool | None) -> bool:
        if user_async is not None:
            return user_async
        if cc and use_v2:
            return False
        return True

    def resolve_v2(*, cc: bool, user_v2: bool | None, v1_ok: bool) -> bool:
        if user_v2 is not None:
            return user_v2
        if cc and v1_ok:
            return False
        return True

    # Stock CC serve: prefer V1, keep async on.
    use_v2 = resolve_v2(cc=True, user_v2=None, v1_ok=True)
    assert use_v2 is False
    assert resolve_async(cc=True, use_v2=use_v2, user_async=None) is True

    # Forced V2 under CC: async defaults off.
    use_v2 = resolve_v2(cc=True, user_v2=True, v1_ok=True)
    assert use_v2 is True
    assert resolve_async(cc=True, use_v2=use_v2, user_async=None) is False

    # Explicit user async wins either way.
    assert resolve_async(cc=True, use_v2=True, user_async=True) is True
    assert resolve_async(cc=True, use_v2=False, user_async=False) is False

    # Non-CC unchanged preference toward async on.
    assert resolve_async(cc=False, use_v2=True, user_async=None) is True


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
