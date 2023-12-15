"""OpenAI functions to be called from GPT."""

import json
import logging
from asyncio import create_task, shield, timeout
from typing import Any

from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.script import Script
from homeassistant.util import dt as dt_util
from homeassistant.util.yaml.loader import parse_yaml

from .const import DOMAIN, EXPORTED_ATTRIBUTES

_LOGGER = logging.getLogger(__name__)

SCRIPT_TIMEOUT = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "entity_registry_inquiry",
            "description": "Get entities defined in Home Assistant",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Return the entity with this name only.",
                    },
                    "area": {
                        "type": "string",
                        "description": "Return entities in this area only.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Return entities with this domain only. Only valid Home Assistant domains are accepted.",
                    },
                    "device_class": {
                        "type": "string",
                        "description": "Return entities with this device class. Only valid Home Assistant device classes are accepted.",
                    },
                    "state": {
                        "type": "string",
                        "description": "Return entities with this state only.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "homeassistant_script",
            "description": "Execute any script in this Home Assistant instance",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The script in Home Assistant yaml script format. This is the same format as the action part of a HA automation. Don't use milliseconds for delay time.",
                    }
                },
                "required": ["script"],
            },
        },
    },
]


async def async_call_function(
    hass: HomeAssistant, function_name: str, function_args: str
) -> str:
    """Wrap the function call to parse the arguments and handle exceptions."""

    available_functions = {
        "entity_registry_inquiry": entity_registry_inquiry,
        "homeassistant_script": homeassistant_script,
    }

    _LOGGER.debug("Function call: %s(%s)", function_name, function_args)

    try:
        function_to_call = available_functions[function_name]
        parsed_args = json.loads(function_args)
        response = await function_to_call(hass, **parsed_args)
        response_str = json.dumps(response)

    except Exception as e:  # pylint: disable=broad-exception-caught
        response = {"error": type(e).__name__}
        if str(e):
            response["error_text"] = str(e)
        response_str = json.dumps(response)

    _LOGGER.debug("Function response: %s", response_str)

    return response_str


async def entity_registry_inquiry(
    hass: HomeAssistant,
    name: str | None = None,
    area: str | None = None,
    domain: str | None = None,
    device_class: str | None = None,
    state: str | None = None,
) -> dict:
    """Get entities defined in Home Assistant."""

    states = intent.async_match_states(
        hass,
        name=name,
        area_name=area,
        domains=[domain] if domain is not None else None,
        device_classes=[device_class] if device_class is not None else None,
        assistant=CONVERSATION_DOMAIN,
    )

    entity_registry = er.async_get(hass)
    result = []

    for entity_state in states:
        entity_state = template.TemplateState(hass, entity_state, collect=False)

        if (
            state is not None
            and entity_state.state != state
            and entity_state.state_with_unit != state
        ):
            continue

        entry: dict["str", Any] = {
            "name": entity_state.name,
            "entity_id": entity_state.entity_id,
            "state": entity_state.state_with_unit,
            "last_changed": dt_util.get_age(entity_state.last_changed) + " ago",
        }

        if registry_entry := entity_registry.async_get(entity_state.entity_id):
            if area_name := template.area_name(hass, entity_state.entity_id):
                entry["area"] = area_name
            if len(registry_entry.aliases):
                entry["aliases"] = list(registry_entry.aliases)

        attributes = {}
        for attribute, value in entity_state.attributes.items():
            if attribute in EXPORTED_ATTRIBUTES:
                attributes[attribute] = value
        if attributes:
            entry["attributes"] = attributes

        result.append(entry)

    if result:
        return {"entities": result}

    error_text = "Entities matching the criteria are not found or not exposed"
    if device_class:
        error_text += (
            ". Please note that not all entities have device_class set up,"
            " so you may want to repeat the function call without device_class parameter"
            " if the expected entities were not found."
        )
    return {"error": error_text}


async def homeassistant_script(
    hass: HomeAssistant,
    script: Any,
) -> dict:
    """Execute a script in Home Assistant."""

    if isinstance(script, str):
        script = parse_yaml(script)

    if isinstance(script, list) and len(script) == 1:
        script = script[0]

    try:
        # check if AI decided to list actions at top level or inside a sequence key
        action = cv.determine_script_action(script)
        if "sequence" in script:
            raise RuntimeError(
                f'The "{action}" action should be inside the "sequence" list, please rewrite the script'
            )
        sequence = [script]
    except ValueError:
        if "trigger" in script:
            raise RuntimeError("This is a script, not an automation. Please rewrite without triggers.")
        sequence = script["sequence"]

    _LOGGER.debug("Parsed sequence: %s", sequence)

    script = Script(hass, sequence=sequence, name="convesation_scipt", domain=DOMAIN)

    for entity_id in script.referenced_entities:
        if not async_should_expose(hass, CONVERSATION_DOMAIN, entity_id):
            raise RuntimeError(
                f"Referencing unknown or not exposed entity {entity_id}, please rewrite the script"
            )

    try:
        async with timeout(SCRIPT_TIMEOUT):
            result = await shield(create_task(script.async_run()))
    except TimeoutError:
        return {
            "success": True,
            "message": "The script is scheduled to execute in background",
        }

    if result.service_response:
        return {"service_response": result.service_response}

    return {"success": True}
