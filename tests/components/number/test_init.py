"""The tests for the Analog Switch component."""
from unittest.mock import MagicMock

from homeassistant.components.number import NumberEntity


class MockNumberEntity(NumberEntity):
    """Mock NumberEntity device to use in tests."""

    @property
    def max_value(self) -> float:
        """Return the max value."""
        return 1.0

    @property
    def state(self):
        """Return the current value."""
        return "0.5"


async def test_step(hass):
    """Test if calculation of step."""
    number = NumberEntity()
    assert number.step == 1.0

    number_2 = MockNumberEntity()
    assert number_2.step == 0.1


async def test_sync_set_value(hass):
    """Test if async set_value calls sync set_value."""
    number = NumberEntity()
    number.hass = hass

    number.set_value = MagicMock()
    await number.async_set_value(42)

    assert number.set_value.called
    assert number.set_value.call_args[0][0] == 42
