"""Module to coordinate llm tools."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
import importlib
import logging
from typing import Any

import voluptuous as vol

from homeassistant.const import ATTR_DEVICE_CLASS
from homeassistant.core import Context, HomeAssistant, State, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from . import entity_registry, intent, template

_LOGGER = logging.getLogger(__name__)

IGNORE_INTENTS = [intent.INTENT_NEVERMIND]


@dataclass(slots=True)
class ToolInput:
    """Tool input to be processed."""

    tool_name: str
    tool_args: dict[str, Any]
    platform: str
    context: Context | None
    user_prompt: str | None
    language: str | None
    assistant: str | None


class Tool:
    """LLM Tool base class."""

    name: str
    description: str | None = None
    parameters: vol.Schema = vol.Schema({})

    @abstractmethod
    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput
    ) -> JsonObjectType:
        """Call the tool."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Represent a string of a Tool."""
        return f"<{self.__class__.__name__} - {self.name}>"


@callback
def async_get_tools(hass: HomeAssistant) -> Iterable[Tool]:
    """Return a list of LLM tools."""
    for intent_handler in intent.async_get(hass):
        if intent_handler.intent_type not in IGNORE_INTENTS:
            yield IntentTool(intent_handler)


@callback
async def async_call_tool(hass: HomeAssistant, tool_input: ToolInput) -> JsonObjectType:
    """Call a LLM tool, validate args and return the response."""
    for tool in async_get_tools(hass):
        if tool.name == tool_input.tool_name:
            break
    else:
        raise HomeAssistantError(f'Tool "{tool_input.tool_name}" not found')

    _tool_input = ToolInput(
        tool_name=tool.name,
        tool_args=tool.parameters(tool_input.tool_args),
        platform=tool_input.platform,
        context=tool_input.context or Context(),
        user_prompt=tool_input.user_prompt,
        language=tool_input.language,
        assistant=tool_input.assistant,
    )

    return await tool.async_call(hass, _tool_input)


COMMON_LLM_ATTRIBUTES = [ATTR_DEVICE_CLASS]


@cache
def _domain_llm_attributes(domain: str) -> list[str]:
    """Return a cached list of attributes to be included."""
    module = importlib.import_module(f"homeassistant.components.{domain}")
    return getattr(module, "LLM_ATTRIBUTES", [])


def _format_state(hass: HomeAssistant, entity_state: State) -> dict[str, Any]:
    """Format state for better understanding by a LLM."""
    er = entity_registry.async_get(hass)
    entity_state = template.TemplateState(hass, entity_state, collect=False)

    result: dict[str, Any] = {
        "name": entity_state.name,
        "entity_id": entity_state.entity_id,
        "state": entity_state.state_with_unit,
        "last_changed": dt_util.get_age(entity_state.last_changed) + " ago",
    }

    if registry_entry := er.async_get(entity_state.entity_id):
        if area_name := template.area_name(hass, entity_state.entity_id):
            result["area"] = area_name
        if floor_name := template.floor_name(hass, entity_state.entity_id):
            result["floor"] = floor_name
        if len(registry_entry.aliases):
            result["aliases"] = list(registry_entry.aliases)

    domain = split_entity_id(entity_state.entity_id)[0]
    attributes: dict[str, Any] = {}
    for attribute, value in entity_state.attributes.items():
        if (
            attribute in _domain_llm_attributes(domain)
            or attribute in COMMON_LLM_ATTRIBUTES
        ):
            attributes[attribute] = value
    if attributes:
        result["attributes"] = attributes

    return result


class IntentTool(Tool):
    """LLM Tool representing an Intent."""

    def __init__(
        self,
        intent_handler: intent.IntentHandler,
    ) -> None:
        """Init the class."""
        self.name = intent_handler.intent_type
        self.description = f"Execute Home Assistant {self.name} intent"
        if slot_schema := intent_handler.slot_schema:
            self.parameters = vol.Schema(slot_schema)

    async def async_call(
        self, hass: HomeAssistant, tool_input: ToolInput
    ) -> JsonObjectType:
        """Handle the intent."""
        slots = {key: {"value": val} for key, val in tool_input.tool_args.items()}

        intent_response = await intent.async_handle(
            hass,
            tool_input.platform,
            self.name,
            slots,
            tool_input.user_prompt,
            tool_input.context,
            tool_input.language,
            tool_input.assistant,
        )
        response = intent_response.as_dict()
        if intent_response.matched_states:
            response["data"]["matched_states"] = [
                _format_state(hass, state) for state in intent_response.matched_states
            ]
        if intent_response.unmatched_states:
            response["data"]["unmatched_states"] = [
                _format_state(hass, state) for state in intent_response.unmatched_states
            ]
        return response
