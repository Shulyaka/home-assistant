"""The tests for the generic_hygrostat."""
import datetime

from asynctest import mock
import pytest
import pytz
import voluptuous as vol

from homeassistant.components import input_boolean, switch
from homeassistant.components.climate.const import (
    ATTR_PRESET_MODE,
    ATTR_HUMIDITY,
    DOMAIN,
    HVAC_MODE_DRY,
#    HVAC_MODE_HUMIDIFY,
    HVAC_MODE_OFF,
    PRESET_AWAY,
    PRESET_NONE,
)
from homeassistant.const import (
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
import homeassistant.core as ha
from homeassistant.core import DOMAIN as HASS_DOMAIN, CoreState, State, callback
from homeassistant.setup import async_setup_component
from homeassistant.util.unit_system import METRIC_SYSTEM

from tests.common import assert_setup_component, mock_restore_cache
from tests.components.climate import common

#Currently the climate platform does not support humidifiers, pretend them to be dehumidifiers for now
HVAC_MODE_HUMIDIFY = HVAC_MODE_DRY

ENTITY = "climate.test"
ENT_SENSOR = "sensor.test"
ENT_SWITCH = "switch.test"
HUMIDIFY_ENTITY = "climate.test_heat"
DRY_ENTITY = "climate.test_cool"
ATTR_AWAY_MODE = "away_mode"
MIN_HUMIDITY = 20.0
MAX_HUMIDITY = 65.0
TARGET_HUMIDITY = 42.0
DRY_TOLERANCE = 0.5
WET_TOLERANCE = 0.5


async def test_setup_missing_conf(hass):
    """Test set up heat_control with missing config values."""
    config = {
        "platform": "generic_hygrostat",
        "name": "test",
        "target_sensor": ENT_SENSOR,
    }
    with assert_setup_component(0):
        await async_setup_component(hass, "climate", {"climate": config})


async def test_valid_conf(hass):
    """Test set up generic_hygrostat with valid config values."""
    assert await async_setup_component(
        hass,
        "climate",
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test",
                "humidifier": ENT_SWITCH,
                "target_sensor": ENT_SENSOR,
            }
        },
    )


@pytest.fixture
def setup_comp_1(hass):
    """Initialize components."""
    hass.config.units = METRIC_SYSTEM
    assert hass.loop.run_until_complete(
        async_setup_component(hass, "homeassistant", {})
    )


async def test_humidifier_input_boolean(hass, setup_comp_1):
    """Test humidifier switching input_boolean."""
    humidifier_switch = "input_boolean.test"
    assert await async_setup_component(
        hass, input_boolean.DOMAIN, {"input_boolean": {"test": None}}
    )

    assert await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test",
                "humidifier": humidifier_switch,
                "target_sensor": ENT_SENSOR,
                "initial_hvac_mode": HVAC_MODE_HUMIDIFY,
            }
        },
    )

    assert STATE_OFF == hass.states.get(humidifier_switch).state

    _setup_sensor(hass, 23)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 32)

    assert STATE_ON == hass.states.get(humidifier_switch).state


async def test_humidifier_switch(hass, setup_comp_1):
    """Test humidifier switching test switch."""
    platform = getattr(hass.components, "test.switch")
    platform.init()
    switch_1 = platform.DEVICES[1]
    assert await async_setup_component(
        hass, switch.DOMAIN, {"switch": {"platform": "test"}}
    )
    humidifier_switch = switch_1.entity_id

    assert await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test",
                "humidifier": humidifier_switch,
                "target_sensor": ENT_SENSOR,
                "initial_hvac_mode": HVAC_MODE_HUMIDIFY,
            }
        },
    )

    await hass.async_block_till_done()
    assert STATE_OFF == hass.states.get(humidifier_switch).state

    _setup_sensor(hass, 23)
    await common.async_set_humidity(hass, 32)
    await hass.async_block_till_done()

    assert STATE_ON == hass.states.get(humidifier_switch).state


def _setup_sensor(hass, humidity):
    """Set up the test sensor."""
    hass.states.async_set(ENT_SENSOR, humidity)


@pytest.fixture
def setup_comp_2(hass):
    """Initialize components."""
    hass.config.units = METRIC_SYSTEM
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 2,
                    "wet_tolerance": 4,
                    "humidifier": ENT_SWITCH,
                    "target_sensor": ENT_SENSOR,
                    "away_humidity": 35,
                    "initial_hvac_mode": HVAC_MODE_HUMIDIFY,
                }
            },
        )
    )


async def test_setup_defaults_to_unknown(hass):
    """Test the setting of defaults to unknown."""
    hass.config.units = METRIC_SYSTEM
    await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test",
                "dry_tolerance": 2,
                "wet_tolerance": 4,
                "humidifier": ENT_SWITCH,
                "target_sensor": ENT_SENSOR,
                "away_humidity": 35,
            }
        },
    )
    assert HVAC_MODE_OFF == hass.states.get(ENTITY).state


async def test_default_setup_params(hass, setup_comp_2):
    """Test the setup with default parameters."""
    state = hass.states.get(ENTITY)
    assert 30 == state.attributes.get("min_humidity")
    assert 99 == state.attributes.get("max_humidity")
    assert 30 == state.attributes.get("humidity")


async def test_get_hvac_modes(hass, setup_comp_2):
    """Test that the operation list returns the correct modes."""
    state = hass.states.get(ENTITY)
    modes = state.attributes.get("hvac_modes")
    assert [HVAC_MODE_HUMIDIFY, HVAC_MODE_OFF] == modes


async def test_set_target_humidity(hass, setup_comp_2):
    """Test the setting of the target humidity."""
    await common.async_set_humidity(hass, 40)
    state = hass.states.get(ENTITY)
    assert 40.0 == state.attributes.get("humidity")
    with pytest.raises(vol.Invalid):
        await common.async_set_humidity(hass, None)
    state = hass.states.get(ENTITY)
    assert 40.0 == state.attributes.get("humidity")


async def test_set_away_mode(hass, setup_comp_2):
    """Test the setting away mode."""
    await common.async_set_humidity(hass, 45)
    await common.async_set_preset_mode(hass, PRESET_AWAY)
    state = hass.states.get(ENTITY)
    assert 35 == state.attributes.get("humidity")


async def test_set_away_mode_and_restore_prev_humidity(hass, setup_comp_2):
    """Test the setting and removing away mode.

    Verify original humidity is restored.
    """
    await common.async_set_humidity(hass, 45)
    await common.async_set_preset_mode(hass, PRESET_AWAY)
    state = hass.states.get(ENTITY)
    assert 35 == state.attributes.get("humidity")
    await common.async_set_preset_mode(hass, PRESET_NONE)
    state = hass.states.get(ENTITY)
    assert 45 == state.attributes.get("humidity")


async def test_set_away_mode_twice_and_restore_prev_humidity(hass, setup_comp_2):
    """Test the setting away mode twice in a row.

    Verify original humidity is restored.
    """
    await common.async_set_humidity(hass, 45)
    await common.async_set_preset_mode(hass, PRESET_AWAY)
    await common.async_set_preset_mode(hass, PRESET_AWAY)
    state = hass.states.get(ENTITY)
    assert 35 == state.attributes.get("humidity")
    await common.async_set_preset_mode(hass, PRESET_NONE)
    state = hass.states.get(ENTITY)
    assert 45 == state.attributes.get("humidity")


async def test_sensor_bad_value(hass, setup_comp_2):
    """Test sensor that have None as state."""
    state = hass.states.get(ENTITY)
    humidity = state.attributes.get("current_humidity")

    _setup_sensor(hass, None)
    await hass.async_block_till_done()

    state = hass.states.get(ENTITY)
    assert humidity == state.attributes.get("current_humidity")


async def test_set_target_humidity_humidifier_on(hass, setup_comp_2):
    """Test if target humidity turn humidifier on."""
    calls = _setup_switch(hass, False)
    _setup_sensor(hass, 36)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 45)
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_set_target_humidity_humidifier_off(hass, setup_comp_2):
    """Test if target humidity turn humidifier off."""
    calls = _setup_switch(hass, True)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 36)
    assert 2 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_humidifier_on_within_tolerance(hass, setup_comp_2):
    """Test if humidity change doesn't turn on within tolerance."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 44)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_humidifier_on_outside_tolerance(hass, setup_comp_2):
    """Test if humidity change turn humidifier on outside cold tolerance."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 42)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_humidifier_off_within_tolerance(hass, setup_comp_2):
    """Test if humidity change doesn't turn off within tolerance."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 48)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_humidifier_off_outside_tolerance(hass, setup_comp_2):
    """Test if humidity change turn humidifier off outside hot tolerance."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 50)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_running_when_hvac_mode_is_off(hass, setup_comp_2):
    """Test that the switch turns off when enabled is set False."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_no_state_change_when_hvac_mode_off(hass, setup_comp_2):
    """Test that the switch doesn't turn on when enabled is False."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_hvac_mode_heat(hass, setup_comp_2):
    """Test change mode from OFF to HUMIDIFY.

    Switch turns on when humidity below setpoint and mode changes.
    """
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    calls = _setup_switch(hass, False)
    await common.async_set_hvac_mode(hass, HVAC_MODE_HUMIDIFY)
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


def _setup_switch(hass, is_on):
    """Set up the test switch."""
    hass.states.async_set(ENT_SWITCH, STATE_ON if is_on else STATE_OFF)
    calls = []

    @callback
    def log_call(call):
        """Log service calls."""
        calls.append(call)

    hass.services.async_register(ha.DOMAIN, SERVICE_TURN_ON, log_call)
    hass.services.async_register(ha.DOMAIN, SERVICE_TURN_OFF, log_call)

    return calls


@pytest.fixture
def setup_comp_3(hass):
    """Initialize components."""
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 2,
                    "wet_tolerance": 4,
                    "away_humidity": 30,
                    "humidifier": ENT_SWITCH,
                    "target_sensor": ENT_SENSOR,
                    "dry_mode": True,
                    "initial_hvac_mode": HVAC_MODE_DRY,
                }
            },
        )
    )


async def test_set_target_humidity_dry_off(hass, setup_comp_3):
    """Test if target humidity turn dry off."""
    calls = _setup_switch(hass, True)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 45)
    assert 2 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_turn_away_mode_on_cooling(hass, setup_comp_3):
    """Test the setting away mode when cooling."""
    _setup_switch(hass, True)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 34)
    await common.async_set_preset_mode(hass, PRESET_AWAY)
    state = hass.states.get(ENTITY)
    assert 30 == state.attributes.get("humidity")


async def test_hvac_mode_cool(hass, setup_comp_3):
    """Test change mode from OFF to DRY.

    Switch turns on when humidity below setpoint and mode changes.
    """
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    calls = _setup_switch(hass, False)
    await common.async_set_hvac_mode(hass, HVAC_MODE_DRY)
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_set_target_humidity_dry_on(hass, setup_comp_3):
    """Test if target humidity turn dry on."""
    calls = _setup_switch(hass, False)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 40)
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_dry_off_within_tolerance(hass, setup_comp_3):
    """Test if humidity change doesn't turn dry off within tolerance."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 44.8)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_set_humidity_change_dry_off_outside_tolerance(hass, setup_comp_3):
    """Test if humidity change turn dry off."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 42)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_dry_on_within_tolerance(hass, setup_comp_3):
    """Test if humidity change doesn't turn dry on within tolerance."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 40.2)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_dry_on_outside_tolerance(hass, setup_comp_3):
    """Test if humidity change turn dry on."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_running_when_operating_mode_is_off_2(hass, setup_comp_3):
    """Test that the switch turns off when enabled is set False."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_no_state_change_when_operation_mode_off_2(hass, setup_comp_3):
    """Test that the switch doesn't turn on when enabled is False."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    _setup_sensor(hass, 50)
    await hass.async_block_till_done()
    assert 0 == len(calls)


@pytest.fixture
def setup_comp_4(hass):
    """Initialize components."""
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 0.3,
                    "wet_tolerance": 0.3,
                    "humidifier": ENT_SWITCH,
                    "target_sensor": ENT_SENSOR,
                    "dry_mode": True,
                    "min_cycle_duration": datetime.timedelta(minutes=10),
                    "initial_hvac_mode": HVAC_MODE_DRY,
                }
            },
        )
    )


async def test_humidity_change_dry_trigger_on_not_long_enough(hass, setup_comp_4):
    """Test if humidity change turn dry on."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_dry_trigger_on_long_enough(hass, setup_comp_4):
    """Test if humidity change turn dry on."""
    fake_changed = datetime.datetime(
        1918, 11, 11, 11, 11, 11, tzinfo=datetime.timezone.utc
    )
    with mock.patch(
        "homeassistant.helpers.condition.dt_util.utcnow", return_value=fake_changed
    ):
        calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_dry_trigger_off_not_long_enough(hass, setup_comp_4):
    """Test if humidity change turn dry on."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_dry_trigger_off_long_enough(hass, setup_comp_4):
    """Test if humidity change turn dry on."""
    fake_changed = datetime.datetime(
        1918, 11, 11, 11, 11, 11, tzinfo=datetime.timezone.utc
    )
    with mock.patch(
        "homeassistant.helpers.condition.dt_util.utcnow", return_value=fake_changed
    ):
        calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_mode_change_dry_trigger_off_not_long_enough(hass, setup_comp_4):
    """Test if mode change turns dry off despite minimum cycle."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    assert 1 == len(calls)
    call = calls[0]
    assert "homeassistant" == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_mode_change_dry_trigger_on_not_long_enough(hass, setup_comp_4):
    """Test if mode change turns dry on despite minimum cycle."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    await common.async_set_hvac_mode(hass, HVAC_MODE_HUMIDIFY)
    assert 1 == len(calls)
    call = calls[0]
    assert "homeassistant" == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


@pytest.fixture
def setup_comp_5(hass):
    """Initialize components."""
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 0.3,
                    "wet_tolerance": 0.3,
                    "humidifier": ENT_SWITCH,
                    "target_sensor": ENT_SENSOR,
                    "dry_mode": True,
                    "min_cycle_duration": datetime.timedelta(minutes=10),
                    "initial_hvac_mode": HVAC_MODE_DRY,
                }
            },
        )
    )


async def test_humidity_change_dry_trigger_on_not_long_enough_2(hass, setup_comp_5):
    """Test if humidity change turn dry on."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_dry_trigger_on_long_enough_2(hass, setup_comp_5):
    """Test if humidity change turn dry on."""
    fake_changed = datetime.datetime(
        1918, 11, 11, 11, 11, 11, tzinfo=datetime.timezone.utc
    )
    with mock.patch(
        "homeassistant.helpers.condition.dt_util.utcnow", return_value=fake_changed
    ):
        calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_dry_trigger_off_not_long_enough_2(hass, setup_comp_5):
    """Test if humidity change turn dry on."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_dry_trigger_off_long_enough_2(hass, setup_comp_5):
    """Test if humidity change turn dry on."""
    fake_changed = datetime.datetime(
        1918, 11, 11, 11, 11, 11, tzinfo=datetime.timezone.utc
    )
    with mock.patch(
        "homeassistant.helpers.condition.dt_util.utcnow", return_value=fake_changed
    ):
        calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_mode_change_dry_trigger_off_not_long_enough_2(hass, setup_comp_5):
    """Test if mode change turns dry off despite minimum cycle."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    assert 1 == len(calls)
    call = calls[0]
    assert "homeassistant" == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_mode_change_dry_trigger_on_not_long_enough_2(hass, setup_comp_5):
    """Test if mode change turns dry on despite minimum cycle."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    await common.async_set_hvac_mode(hass, HVAC_MODE_HUMIDIFY)
    assert 1 == len(calls)
    call = calls[0]
    assert "homeassistant" == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


@pytest.fixture
def setup_comp_6(hass):
    """Initialize components."""
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 0.3,
                    "wet_tolerance": 0.3,
                    "humidifier": ENT_SWITCH,
                    "target_sensor": ENT_SENSOR,
                    "min_cycle_duration": datetime.timedelta(minutes=10),
                    "initial_hvac_mode": HVAC_MODE_HUMIDIFY,
                }
            },
        )
    )


async def test_humidity_change_humidifier_trigger_off_not_long_enough(hass, setup_comp_6):
    """Test if humidity change doesn't turn humidifier off because of time."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_humidifier_trigger_on_not_long_enough(hass, setup_comp_6):
    """Test if humidity change doesn't turn humidifier on because of time."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)


async def test_humidity_change_humidifier_trigger_on_long_enough(hass, setup_comp_6):
    """Test if humidity change turn humidifier on after min cycle."""
    fake_changed = datetime.datetime(
        1918, 11, 11, 11, 11, 11, tzinfo=datetime.timezone.utc
    )
    with mock.patch(
        "homeassistant.helpers.condition.dt_util.utcnow", return_value=fake_changed
    ):
        calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_humidifier_trigger_off_long_enough(hass, setup_comp_6):
    """Test if humidity change turn humidifier off after min cycle."""
    fake_changed = datetime.datetime(
        1918, 11, 11, 11, 11, 11, tzinfo=datetime.timezone.utc
    )
    with mock.patch(
        "homeassistant.helpers.condition.dt_util.utcnow", return_value=fake_changed
    ):
        calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_mode_change_humidifier_trigger_off_not_long_enough(hass, setup_comp_6):
    """Test if mode change turns humidifier off despite minimum cycle."""
    calls = _setup_switch(hass, True)
    await common.async_set_humidity(hass, 40)
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    await common.async_set_hvac_mode(hass, HVAC_MODE_OFF)
    assert 1 == len(calls)
    call = calls[0]
    assert "homeassistant" == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_mode_change_humidifier_trigger_on_not_long_enough(hass, setup_comp_6):
    """Test if mode change turns humidifier on despite minimum cycle."""
    calls = _setup_switch(hass, False)
    await common.async_set_humidity(hass, 45)
    _setup_sensor(hass, 40)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    await common.async_set_hvac_mode(hass, HVAC_MODE_HUMIDIFY)
    assert 1 == len(calls)
    call = calls[0]
    assert "homeassistant" == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


@pytest.fixture
def setup_comp_7(hass):
    """Initialize components."""
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 0.3,
                    "wet_tolerance": 0.3,
                    "humidifier": ENT_SWITCH,
                    "target_humidity": 25,
                    "target_sensor": ENT_SENSOR,
                    "dry_mode": True,
                    "min_cycle_duration": datetime.timedelta(minutes=15),
                    "keep_alive": datetime.timedelta(minutes=10),
                    "initial_hvac_mode": HVAC_MODE_DRY,
                }
            },
        )
    )


async def test_humidity_change_dry_trigger_on_long_enough_3(hass, setup_comp_7):
    """Test if turn on signal is sent at keep-alive intervals."""
    calls = _setup_switch(hass, True)
    await hass.async_block_till_done()
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 40)
    test_time = datetime.datetime.now(pytz.UTC)
    _send_time_changed(hass, test_time)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=5))
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=10))
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_dry_trigger_off_long_enough_3(hass, setup_comp_7):
    """Test if turn on signal is sent at keep-alive intervals."""
    calls = _setup_switch(hass, False)
    await hass.async_block_till_done()
    _setup_sensor(hass, 35)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 40)
    test_time = datetime.datetime.now(pytz.UTC)
    _send_time_changed(hass, test_time)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=5))
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=10))
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


def _send_time_changed(hass, now):
    """Send a time changed event."""
    hass.bus.async_fire(ha.EVENT_TIME_CHANGED, {ha.ATTR_NOW: now})


@pytest.fixture
def setup_comp_8(hass):
    """Initialize components."""
    assert hass.loop.run_until_complete(
        async_setup_component(
            hass,
            DOMAIN,
            {
                "climate": {
                    "platform": "generic_hygrostat",
                    "name": "test",
                    "dry_tolerance": 0.3,
                    "wet_tolerance": 0.3,
                    "target_humidity": 25,
                    "humidifier": ENT_SWITCH,
                    "target_sensor": ENT_SENSOR,
                    "min_cycle_duration": datetime.timedelta(minutes=15),
                    "keep_alive": datetime.timedelta(minutes=10),
                    "initial_hvac_mode": HVAC_MODE_HUMIDIFY,
                }
            },
        )
    )


async def test_humidity_change_humidifier_trigger_on_long_enough_2(hass, setup_comp_8):
    """Test if turn on signal is sent at keep-alive intervals."""
    calls = _setup_switch(hass, True)
    await hass.async_block_till_done()
    _setup_sensor(hass, 35)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 40)
    test_time = datetime.datetime.now(pytz.UTC)
    _send_time_changed(hass, test_time)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=5))
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=10))
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_ON == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_humidity_change_humidifier_trigger_off_long_enough_2(hass, setup_comp_8):
    """Test if turn on signal is sent at keep-alive intervals."""
    calls = _setup_switch(hass, False)
    await hass.async_block_till_done()
    _setup_sensor(hass, 45)
    await hass.async_block_till_done()
    await common.async_set_humidity(hass, 40)
    test_time = datetime.datetime.now(pytz.UTC)
    _send_time_changed(hass, test_time)
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=5))
    await hass.async_block_till_done()
    assert 0 == len(calls)
    _send_time_changed(hass, test_time + datetime.timedelta(minutes=10))
    await hass.async_block_till_done()
    assert 1 == len(calls)
    call = calls[0]
    assert HASS_DOMAIN == call.domain
    assert SERVICE_TURN_OFF == call.service
    assert ENT_SWITCH == call.data["entity_id"]


async def test_custom_setup_params(hass):
    """Test the setup with custom parameters."""
    result = await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test",
                "humidifier": ENT_SWITCH,
                "target_sensor": ENT_SENSOR,
                "min_humidity": MIN_HUMIDITY,
                "max_humidity": MAX_HUMIDITY,
                "target_humidity": TARGET_HUMIDITY,
            }
        },
    )
    assert result
    state = hass.states.get(ENTITY)
    assert state.attributes.get("min_humidity") == MIN_HUMIDITY
    assert state.attributes.get("max_humidity") == MAX_HUMIDITY
    assert state.attributes.get("humidity") == TARGET_HUMIDITY


async def test_restore_state(hass):
    """Ensure states are restored on startup."""
    mock_restore_cache(
        hass,
        (
            State(
                "climate.test_hygrostat",
                HVAC_MODE_OFF,
                {ATTR_HUMIDITY: "40", ATTR_PRESET_MODE: PRESET_AWAY},
            ),
        ),
    )

    hass.state = CoreState.starting

    await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test_hygrostat",
                "humidifier": ENT_SWITCH,
                "target_sensor": ENT_SENSOR,
                "away_humidity": 32,
            }
        },
    )

    state = hass.states.get("climate.test_hygrostat")
    assert state.attributes[ATTR_HUMIDITY] == 40
    assert state.attributes[ATTR_PRESET_MODE] == PRESET_AWAY
    assert state.state == HVAC_MODE_OFF


async def test_no_restore_state(hass):
    """Ensure states are restored on startup if they exist.

    Allows for graceful reboot.
    """
    mock_restore_cache(
        hass,
        (
            State(
                "climate.test_hygrostat",
                HVAC_MODE_OFF,
                {ATTR_HUMIDITY: "40", ATTR_PRESET_MODE: PRESET_AWAY},
            ),
        ),
    )

    hass.state = CoreState.starting

    await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test_hygrostat",
                "humidifier": ENT_SWITCH,
                "target_sensor": ENT_SENSOR,
                "target_humidity": 42,
            }
        },
    )

    state = hass.states.get("climate.test_hygrostat")
    assert state.attributes[ATTR_HUMIDITY] == 42
    assert state.state == HVAC_MODE_OFF


async def test_restore_state_uncoherence_case(hass):
    """
    Test restore from a strange state.

    - Turn the generic hygrostat off
    - Restart HA and restore state from DB
    """
    _mock_restore_cache(hass, humidity=40)

    calls = _setup_switch(hass, False)
    _setup_sensor(hass, 35)
    await _setup_climate(hass)
    await hass.async_block_till_done()

    state = hass.states.get(ENTITY)
    assert 40 == state.attributes[ATTR_HUMIDITY]
    assert HVAC_MODE_OFF == state.state
    assert 0 == len(calls)

    calls = _setup_switch(hass, False)
    await hass.async_block_till_done()
    state = hass.states.get(ENTITY)
    assert HVAC_MODE_OFF == state.state


async def _setup_climate(hass):
    assert await async_setup_component(
        hass,
        DOMAIN,
        {
            "climate": {
                "platform": "generic_hygrostat",
                "name": "test",
                "dry_tolerance": 2,
                "wet_tolerance": 4,
                "away_humidity": 32,
                "humidifier": ENT_SWITCH,
                "target_sensor": ENT_SENSOR,
                "dry_mode": True,
            }
        },
    )


def _mock_restore_cache(hass, humidity=40, hvac_mode=HVAC_MODE_OFF):
    mock_restore_cache(
        hass,
        (
            State(
                ENTITY,
                hvac_mode,
                {ATTR_HUMIDITY: str(humidity), ATTR_PRESET_MODE: PRESET_AWAY},
            ),
        ),
    )
