"""Standard-library-only named-time authority for SkyGAN runtime adapters."""

from collections.abc import Mapping
from types import MappingProxyType

SKYGAN_TIME_OF_DAY_HOURS: Mapping[str, float] = MappingProxyType(
    {
        "sunrise": 6.5,
        "morning": 9.0,
        "midday": 12.0,
        "golden_hour": 17.0,
        "sunset": 18.5,
        "twilight": 19.5,
    }
)


def resolve_skygan_time_of_day(time_of_day: str, *, hours: Mapping[str, float] = SKYGAN_TIME_OF_DAY_HOURS) -> float:
    """Resolve a named slot, preserving adapter/subclass mapping order in errors."""
    try:
        return hours[time_of_day]
    except KeyError:
        raise ValueError(f"Unknown time_of_day {time_of_day!r}; expected one of {list(hours)}") from None
