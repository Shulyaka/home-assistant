"""Demo platform that offers a fake humidity device."""
from homeassistant.components.humidity import HumidityDevice
from homeassistant.components.humidity.const import (
    CURRENT_HUMIDIFIER_DRY,
    CURRENT_HUMIDIFIER_HUMIDIFY,
    HUMIDIFIER_MODE_DRY,
    HUMIDIFIER_MODE_HUMIDIFY,
    HUMIDIFIER_MODE_HUMIDIFY_DRY,
    HUMIDIFIER_MODE_OFF,
    SUPPORT_PRESET_MODE,
    SUPPORT_TARGET_HUMIDITY,
)

SUPPORT_FLAGS = 0


def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the Demo humidity devices."""
    add_entities(
        [
            DemoHumidity(
                name="Humidifier",
                preset=None,
                target_humidity=68,
                current_humidity=77,
                humidifier_mode=HUMIDIFIER_MODE_HUMIDIFY,
                humidifier_action=CURRENT_HUMIDIFIER_HUMIDIFY,
                humidifier_modes=[HUMIDIFIER_MODE_HUMIDIFY, HUMIDIFIER_MODE_OFF],
            ),
            DemoHumidity(
                name="Dehumidifier",
                preset=None,
                target_humidity=54,
                current_humidity=67,
                humidifier_mode=HUMIDIFIER_MODE_DRY,
                humidifier_action=CURRENT_HUMIDIFIER_DRY,
                humidifier_modes=[HUMIDIFIER_MODE_DRY, HUMIDIFIER_MODE_OFF],
            ),
            DemoHumidity(
                name="Hygrostat",
                preset="home",
                preset_modes=["home", "eco"],
                target_humidity=50,
                current_humidity=49,
                humidifier_mode=HUMIDIFIER_MODE_HUMIDIFY_DRY,
                humidifier_action=None,
                humidifier_modes=[HUMIDIFIER_MODE_HUMIDIFY_DRY, HUMIDIFIER_MODE_DRY, HUMIDIFIER_MODE_HUMIDIFY],
            ),
        ]
    )


class DemoHumidity(HumidityDevice):
    """Representation of a demo humidity device."""

    def __init__(
        self,
        name,
        preset,
        target_humidity,
        current_humidity,
        humidifier_mode,
        humidifier_action,
        humidifier_modes,
        preset_modes=None,
    ):
        """Initialize the humidity device."""
        self._name = name
        self._support_flags = SUPPORT_FLAGS
        if preset is not None:
            self._support_flags = self._support_flags | SUPPORT_PRESET_MODE
        if target_humidity is not None:
            self._support_flags = self._support_flags | SUPPORT_TARGET_HUMIDITY
        self._target_humidity = target_humidity
        self._preset = preset
        self._preset_modes = preset_modes
        self._current_humidity = current_humidity
        self._humidifier_action = humidifier_action
        self._humidifier_mode = humidifier_mode
        self._humidifier_modes = humidifier_modes

    @property
    def supported_features(self):
        """Return the list of supported features."""
        return self._support_flags

    @property
    def should_poll(self):
        """Return the polling state."""
        return False

    @property
    def name(self):
        """Return the name of the humidity device."""
        return self._name

    @property
    def current_humidity(self):
        """Return the current humidity."""
        return self._current_humidity

    @property
    def target_humidity(self):
        """Return the humidity we try to reach."""
        return self._target_humidity

    @property
    def humidifier_action(self):
        """Return current operation ie. heat, cool, idle."""
        return self._humidifier_action

    @property
    def humidifier_mode(self):
        """Return humidifier target humidifier state."""
        return self._humidifier_mode

    @property
    def humidifier_modes(self):
        """Return the list of available operation modes."""
        return self._humidifier_modes

    @property
    def preset_mode(self):
        """Return preset mode."""
        return self._preset

    @property
    def preset_modes(self):
        """Return preset modes."""
        return self._preset_modes

    async def async_set_humidity(self, humidity):
        """Set new humidity level."""
        self._target_humidity = humidity
        self.async_write_ha_state()

    async def async_set_humidifier_mode(self, humidifier_mode):
        """Set new operation mode."""
        self._humidifier_mode = humidifier_mode
        self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode):
        """Update preset_mode on."""
        self._preset = preset_mode
        self.async_write_ha_state()
