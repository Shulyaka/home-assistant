"""The tests for reproduction of state."""

import pytest

from homeassistant.components.humidifier.reproduce_state import async_reproduce_states
from homeassistant.components.humidifier.const import (
    ATTR_HUMIDITY,
    ATTR_PRESET_MODE,
    DOMAIN,
    OPERATION_MODE_AUTO,
    OPERATION_MODE_HUMIDIFY,
    OPERATION_MODE_OFF,
    SERVICE_SET_HUMIDITY,
    SERVICE_SET_OPERATION_MODE,
    SERVICE_SET_PRESET_MODE,
)
from homeassistant.core import Context, State

from tests.common import async_mock_service

ENTITY_1 = "humidifier.test1"
ENTITY_2 = "humidifier.test2"


@pytest.mark.parametrize(
    "state", [OPERATION_MODE_AUTO, OPERATION_MODE_HUMIDIFY, OPERATION_MODE_OFF]
)
async def test_with_operation_mode(hass, state):
    """Test that state different humidifier states."""
    calls = async_mock_service(hass, DOMAIN, SERVICE_SET_OPERATION_MODE)

    await async_reproduce_states(hass, [State(ENTITY_1, state)])

    await hass.async_block_till_done()

    assert len(calls) == 1
    assert calls[0].data == {"entity_id": ENTITY_1, "operation_mode": state}


async def test_multiple_state(hass):
    """Test that multiple states gets calls."""
    calls_1 = async_mock_service(hass, DOMAIN, SERVICE_SET_OPERATION_MODE)

    await async_reproduce_states(
        hass,
        [
            State(ENTITY_1, OPERATION_MODE_HUMIDIFY),
            State(ENTITY_2, OPERATION_MODE_AUTO),
        ],
    )

    await hass.async_block_till_done()

    assert len(calls_1) == 2
    # order is not guaranteed
    assert any(
        call.data == {"entity_id": ENTITY_1, "operation_mode": OPERATION_MODE_HUMIDIFY}
        for call in calls_1
    )
    assert any(
        call.data == {"entity_id": ENTITY_2, "operation_mode": OPERATION_MODE_AUTO}
        for call in calls_1
    )


async def test_state_with_none(hass):
    """Test that none is not a humidifier state."""
    calls = async_mock_service(hass, DOMAIN, SERVICE_SET_OPERATION_MODE)

    await async_reproduce_states(hass, [State(ENTITY_1, None)])

    await hass.async_block_till_done()

    assert len(calls) == 0


async def test_state_with_context(hass):
    """Test that context is forwarded."""
    calls = async_mock_service(hass, DOMAIN, SERVICE_SET_OPERATION_MODE)

    context = Context()

    await async_reproduce_states(
        hass, [State(ENTITY_1, OPERATION_MODE_HUMIDIFY)], context
    )

    await hass.async_block_till_done()

    assert len(calls) == 1
    assert calls[0].data == {
        "entity_id": ENTITY_1,
        "operation_mode": OPERATION_MODE_HUMIDIFY,
    }
    assert calls[0].context == context


@pytest.mark.parametrize(
    "service,attribute",
    [
        (SERVICE_SET_PRESET_MODE, ATTR_PRESET_MODE),
        (SERVICE_SET_HUMIDITY, ATTR_HUMIDITY),
    ],
)
async def test_attribute(hass, service, attribute):
    """Test that service call is made for each attribute."""
    calls_1 = async_mock_service(hass, DOMAIN, service)

    value = "dummy"

    await async_reproduce_states(hass, [State(ENTITY_1, None, {attribute: value})])

    await hass.async_block_till_done()

    assert len(calls_1) == 1
    assert calls_1[0].data == {"entity_id": ENTITY_1, attribute: value}
