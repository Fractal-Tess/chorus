from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chorus.devices import DevicePolicy
    from chorus.models import ModelCatalog, ModelSpec

CHANNELS = ("cpu", "gpu")


class ChannelUnavailableError(RuntimeError):
    """A supported and enabled logical channel has no usable hardware."""


class GpuQueueFullError(RuntimeError):
    """The bounded wait queue for GPU work is full."""


@dataclass(frozen=True)
class ChannelRule:
    channels: tuple[str, ...]
    default_channel: str


class ChannelPolicy:
    """Immutable per-model logical-channel policy loaded at startup."""

    def __init__(
        self,
        catalog: ModelCatalog,
        devices: DevicePolicy,
        config_path: Path | None = None,
    ):
        self._devices = devices
        environment_path = os.environ.get("TTS_CHANNEL_CONFIG")
        self.path = (
            config_path
            or (Path(environment_path) if environment_path else _default_path())
        ).resolve()
        self._rules = load_channel_config(
            self.path,
            catalog,
            explicit=config_path is not None or bool(environment_path),
        )

    def rule(self, spec: ModelSpec) -> ChannelRule:
        configured = self._rules.get(f"{spec.engine}/{spec.name}")
        if configured is not None:
            return configured
        supported = tuple(channel for channel in CHANNELS if _supports(spec, channel))
        if not supported:
            raise ValueError(f"Model {spec.engine}/{spec.name} supports no channels")
        return ChannelRule(supported, self._derived_default(supported))

    def status(self, spec: ModelSpec) -> dict[str, object]:
        rule = self.rule(spec)
        return {
            "default_channel": rule.default_channel,
            "channels": [
                {
                    "id": channel,
                    "supported": _supports(spec, channel),
                    "enabled": channel in rule.channels,
                    "available": (
                        _supports(spec, channel)
                        and channel in rule.channels
                        and self._available(channel)
                    ),
                }
                for channel in CHANNELS
            ],
        }

    def resolve(self, spec: ModelSpec, requested: str | None = None) -> str:
        rule = self.rule(spec)
        channel = rule.default_channel if requested is None else requested
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel {channel}; expected cpu or gpu")
        if not _supports(spec, channel):
            raise ValueError(
                f"Model {spec.engine}/{spec.name} does not support {channel}"
            )
        if channel not in rule.channels:
            raise ValueError(
                f"Channel {channel} is not enabled for {spec.engine}/{spec.name}"
            )
        if not self._available(channel):
            raise ChannelUnavailableError(
                f"Channel {channel} is unavailable: no enabled compatible hardware"
            )
        return channel

    def physical_devices(self, channel: str) -> list[str]:
        if channel == "cpu":
            return ["cpu"] if "cpu" in self._devices.allowed else []
        if channel == "gpu":
            return [d for d in self._devices.allowed if d.startswith("cuda:")]
        raise ValueError(f"Unknown channel {channel}; expected cpu or gpu")

    def _available(self, channel: str) -> bool:
        return bool(self.physical_devices(channel))

    def _derived_default(self, supported: tuple[str, ...]) -> str:
        if "gpu" in supported and self._available("gpu"):
            return "gpu"
        if "cpu" in supported and self._available("cpu"):
            return "cpu"
        # Preserve the preferred GPU choice when hardware is unavailable so the
        # request reports 503 rather than silently crossing channels.
        if "gpu" in supported:
            return "gpu"
        return "cpu"


def _default_path() -> Path:
    from chorus.models import ROOT

    return ROOT / "channels.toml"


def _supports(spec: ModelSpec, channel: str) -> bool:
    return "cpu" in spec.devices if channel == "cpu" else "cuda" in spec.devices


def load_channel_config(
    path: Path,
    catalog: ModelCatalog,
    *,
    explicit: bool = False,
) -> dict[str, ChannelRule]:
    """Load and strictly validate a TOML model-channel allowlist."""
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError:
        if explicit:
            raise
        return {}
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid channel configuration {path}: {error}") from error
    if not isinstance(data, dict) or set(data) != {"models"}:
        raise ValueError("Channel configuration must contain only a [models] table")
    models = data["models"]
    if not isinstance(models, dict):
        raise ValueError("Channel configuration [models] must be a table")

    rules: dict[str, ChannelRule] = {}
    for selector, raw in models.items():
        if not isinstance(selector, str):
            raise ValueError("Channel configuration model names must be strings")
        if not isinstance(raw, dict):
            raise ValueError(f"Channel configuration for {selector} must be a table")
        if set(raw) != {"channels", "default_channel"}:
            unknown = sorted(set(raw) - {"channels", "default_channel"})
            missing = sorted({"channels", "default_channel"} - set(raw))
            detail = (
                f"unknown keys: {', '.join(unknown)}"
                if unknown
                else f"missing keys: {', '.join(missing)}"
            )
            raise ValueError(f"Invalid channel configuration for {selector} ({detail})")
        engine, separator, model = selector.partition("/")
        if not separator or not engine or not model:
            raise ValueError(f"Unknown model in channel configuration: {selector}")
        try:
            spec = catalog.resolve(engine, model)
        except ValueError as error:
            raise ValueError(
                f"Unknown model in channel configuration: {selector}"
            ) from error
        channels = raw["channels"]
        if (
            not isinstance(channels, list)
            or not channels
            or any(not isinstance(channel, str) for channel in channels)
        ):
            raise ValueError(
                f"channels for {selector} must be a non-empty list of strings"
            )
        if len(channels) != len(set(channels)):
            raise ValueError(f"Duplicate channels for {selector}")
        if any(channel not in CHANNELS for channel in channels):
            raise ValueError(f"Unsupported channel for {selector}; expected cpu or gpu")
        for channel in channels:
            if not _supports(spec, channel):
                raise ValueError(f"Model {selector} does not support channel {channel}")
        default_channel = raw["default_channel"]
        if not isinstance(default_channel, str) or default_channel not in channels:
            raise ValueError(f"Invalid default channel for {selector}")
        rules[selector] = ChannelRule(tuple(channels), default_channel)
    return rules
