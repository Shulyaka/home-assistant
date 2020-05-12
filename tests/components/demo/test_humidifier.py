"""The tests for the demo humidifier component."""

import pytest
import voluptuous as vol

from homeassistant.components.humidifier.const import (
    ATTR_CURRENT_HUMIDITY,
    ATTR_CURRENT_TEMPERATURE,
    ATTR_HUMIDITY,
    ATTR_MAX_HUMIDITY,
    ATTR_MIN_HUMIDITY,
    ATTR_PRESET_MODE,
    DOMAIN,
    PRESET_AWAY,
    PRESET_ECO,
    SERVICE_SET_HUMIDITY,
    SERVICE_SET_PRESET_MODE,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
)
from homeassistant.setup import async_setup_component

ENTITY_DEHUMIDIFIER = "humidifier.dehumidifier"
ENTITY_HYGROSTAT = "humidifier.hygrostat"
ENTITY_HUMIDIFIER = "humidifier.humidifier"


@pytest.fixture(autouse=True)
async def setup_demo_humidifier(hass):
    """Initialize setup demo humidifier."""
    assert await async_setup_component(
        hass, DOMAIN, {"humidifier": {"platform": "demo"}}
    )


def test_setup_params(hass):
    """Test the initial parameters."""
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.state == STATE_ON
    assert state.attributes.get(ATTR_HUMIDITY) == 54
    assert state.attributes.get(ATTR_CURRENT_HUMIDITY) == 67
    assert state.attributes.get(ATTR_CURRENT_TEMPERATURE) == 25


def test_default_setup_params(hass):
    """Test the setup with default parameters."""
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.attributes.get(ATTR_MIN_HUMIDITY) == 0
    assert state.attributes.get(ATTR_MAX_HUMIDITY) == 100


async def test_set_target_humidity_bad_attr(hass):
    """Test setting the target humidity without required attribute."""
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.attributes.get(ATTR_HUMIDITY) == 54

    with pytest.raises(vol.Invalid):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_SET_HUMIDITY,
            {ATTR_HUMIDITY: None, ATTR_ENTITY_ID: ENTITY_DEHUMIDIFIER},
            blocking=True,
        )
    await hass.async_block_till_done()

    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.attributes.get(ATTR_HUMIDITY) == 54


async def test_set_target_humidity(hass):
    """Test the setting of the target humidity."""
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.attributes.get(ATTR_HUMIDITY) == 54

    await hass.services.async_call(
        DOMAIN,
        SERVICE_SET_HUMIDITY,
        {ATTR_HUMIDITY: 64, ATTR_ENTITY_ID: ENTITY_DEHUMIDIFIER},
        blocking=True,
    )
    await hass.async_block_till_done()

    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.attributes.get(ATTR_HUMIDITY) == 64


async def test_set_hold_mode_away(hass):
    """Test setting the hold mode away."""
    await hass.services.async_call(
        DOMAIN,
        SERVICE_SET_PRESET_MODE,
        {ATTR_PRESET_MODE: PRESET_AWAY, ATTR_ENTITY_ID: ENTITY_HYGROSTAT},
        blocking=True,
    )
    await hass.async_block_till_done()

    state = hass.states.get(ENTITY_HYGROSTAT)
    assert state.attributes.get(ATTR_PRESET_MODE) == PRESET_AWAY


async def test_set_hold_mode_eco(hass):
    """Test setting the hold mode eco."""
    await hass.services.async_call(
        DOMAIN,
        SERVICE_SET_PRESET_MODE,
        {ATTR_PRESET_MODE: PRESET_ECO, ATTR_ENTITY_ID: ENTITY_HYGROSTAT},
        blocking=True,
    )
    await hass.async_block_till_done()

    state = hass.states.get(ENTITY_HYGROSTAT)
    assert state.attributes.get(ATTR_PRESET_MODE) == PRESET_ECO


async def test_turn_on(hass):
    """Test turn on device."""
    await hass.services.async_call(
        DOMAIN, SERVICE_TURN_OFF, {ATTR_ENTITY_ID: ENTITY_DEHUMIDIFIER}, blocking=True
    )
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.state == STATE_OFF

    await hass.services.async_call(
        DOMAIN, SERVICE_TURN_ON, {ATTR_ENTITY_ID: ENTITY_DEHUMIDIFIER}, blocking=True
    )
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.state == STATE_ON


async def test_turn_off(hass):
    """Test turn off device."""
    await hass.services.async_call(
        DOMAIN, SERVICE_TURN_ON, {ATTR_ENTITY_ID: ENTITY_DEHUMIDIFIER}, blocking=True
    )
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.state == STATE_ON

    await hass.services.async_call(
        DOMAIN, SERVICE_TURN_OFF, {ATTR_ENTITY_ID: ENTITY_DEHUMIDIFIER}, blocking=True
    )
    state = hass.states.get(ENTITY_DEHUMIDIFIER)
    assert state.state == STATE_OFF
