# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from tests.quantization import utils


class _DummyCapability:
    def __init__(self, value: int):
        self._value = value

    def to_int(self) -> int:
        return self._value


def _dummy_quant_config(min_capability: int):
    class _Config:
        @classmethod
        def get_min_capability(cls) -> int:
            return min_capability

    return _Config


def _dummy_platform(
    *,
    is_cuda: bool = False,
    is_rocm: bool = False,
    is_xpu: bool = False,
    capability: _DummyCapability | None = None,
):
    return SimpleNamespace(
        is_cuda=lambda: is_cuda,
        is_rocm=lambda: is_rocm,
        is_xpu=lambda: is_xpu,
        verify_quantization=lambda quant_method: None,
        get_device_capability=lambda: capability,
    )


def test_is_quant_method_supported_for_xpu_gguf(monkeypatch):
    monkeypatch.setattr(
        utils,
        "current_platform",
        _dummy_platform(is_xpu=True, capability=None),
    )
    monkeypatch.setattr(
        utils,
        "get_quantization_config",
        lambda _: _dummy_quant_config(60),
    )

    assert utils.is_quant_method_supported("gguf")


def test_is_quant_method_supported_for_xpu_non_gguf(monkeypatch):
    monkeypatch.setattr(
        utils,
        "current_platform",
        _dummy_platform(is_xpu=True, capability=None),
    )
    monkeypatch.setattr(
        utils,
        "get_quantization_config",
        lambda _: _dummy_quant_config(60),
    )

    assert not utils.is_quant_method_supported("fp8")
