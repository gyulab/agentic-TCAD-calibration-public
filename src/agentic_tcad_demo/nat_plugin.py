"""NVIDIA NeMo Agent Toolkit (NAT) plugin: the calibration tools as NAT functions.

Install with `pip install -e ".[nat]"` (Python 3.11-3.13). NAT discovers this module through the
`nat.plugins` entry point in pyproject.toml. A NAT agent such as `react_agent` or `tool_calling_agent`
then drives the calibration by calling four tools:

    tcad_observe                                  current J, residual terms, OFAT gains, eligible components
    tcad_update_group(group, rationale)           OFAT + bounded Gauss-Newton trial on one parameter group
    tcad_activate_component(component, rationale) request a model component (approval + J-gated trial)
    tcad_stop(rationale)                          end calibration, run held-out evaluation, write outputs

All four tools of one workflow share a CalibrationSession keyed by (config_file, out_dir). The NAT agent
only chooses actions; bounds, the accept rule, activation eligibility and approval stay in the controller.
Every decision is recorded in <out_dir>/action_log.jsonl and can be replayed without the LLM:

    python -m agentic_tcad_demo run --planner replay --replay-log <out_dir>/action_log.jsonl
"""

from __future__ import annotations

import asyncio
import json

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.llm import LLMBaseConfig

from .session import CalibrationSession

_SESSIONS: dict[tuple[str, str], CalibrationSession] = {}


class _SessionConfig(FunctionBaseConfig):
    config_file: str = "configs/synthetic_demo.yaml"
    out_dir: str = "outputs/nat_run"
    planner_label: str = "nat"


def _session(config: _SessionConfig) -> CalibrationSession:
    key = (config.config_file, config.out_dir)
    if key not in _SESSIONS or _SESSIONS[key].finished and _SESSIONS[key].result is None:
        _SESSIONS[key] = CalibrationSession(config.config_file, config.out_dir, f"nat({config.planner_label})")
    return _SESSIONS[key]


def _dump(obj: dict) -> str:
    return json.dumps(obj, default=float)


async def _close(config: _SessionConfig) -> None:
    session = _SESSIONS.pop((config.config_file, config.out_dir), None)
    if session is not None and not session.finished:
        await asyncio.to_thread(session.close)


class ObserveConfig(_SessionConfig, name="tcad_observe"):
    pass


class UpdateGroupConfig(_SessionConfig, name="tcad_update_group"):
    pass


class ActivateComponentConfig(_SessionConfig, name="tcad_activate_component"):
    pass


class StopConfig(_SessionConfig, name="tcad_stop"):
    pass


@register_function(config_type=ObserveConfig)
async def tcad_observe(config: ObserveConfig, builder: Builder):
    async def observe(request: str = "") -> str:
        """Return the current calibration state: objective J, its weighted residual terms, the dominant
        term, current parameters, each parameter group's OFAT-predicted gain, and the model components
        that are currently eligible for activation. Call this first."""
        session = await asyncio.to_thread(_session, config)
        return _dump(session.observe())

    yield FunctionInfo.from_fn(observe, description=observe.__doc__)


@register_function(config_type=UpdateGroupConfig)
async def tcad_update_group(config: UpdateGroupConfig, builder: Builder):
    async def update_group(group: str, rationale: str) -> str:
        """Run OFAT sensitivity and one bounded Gauss-Newton trial on an untried parameter group
        (for example contact, electrostatics, transport). The trial is kept only if the simulation
        converges and J decreases. Returns the outcome and the next observation."""
        session = await asyncio.to_thread(_session, config)
        return _dump(await asyncio.to_thread(session.act, "update_group", group, rationale))

    try:
        yield FunctionInfo.from_fn(update_group, description=update_group.__doc__)
    finally:
        await _close(config)  # if the agent ended without tcad_stop, stop and write outputs anyway


@register_function(config_type=ActivateComponentConfig)
async def tcad_activate_component(config: ActivateComponentConfig, builder: Builder):
    async def activate_component(component: str, rationale: str) -> str:
        """Request activation of a model component listed in eligible_components (for example
        fixed_charge, tail_states). It needs an approval record and is kept only if J decreases."""
        session = await asyncio.to_thread(_session, config)
        return _dump(await asyncio.to_thread(session.act, "activate_component", component, rationale))

    yield FunctionInfo.from_fn(activate_component, description=activate_component.__doc__)


@register_function(config_type=StopConfig)
async def tcad_stop(config: StopConfig, builder: Builder):
    async def stop(rationale: str) -> str:
        """End calibration. The final model is held fixed, evaluated on held-out drain-bias and
        channel-length conditions, and the log, summary and parameters are written."""
        session = await asyncio.to_thread(_session, config)
        return _dump(await asyncio.to_thread(session.act, "stop", None, rationale))

    yield FunctionInfo.from_fn(stop, description=stop.__doc__)


# -- Offline scripted LLM ----------------------------------------------------------------------------
# For tests and CI only: replies with a fixed list of messages, so the complete NAT path
# (agent -> tools -> controller) runs without a network or API key. It does not reason.
# (NAT's own nat_test_llm is not a LangChain Runnable in NAT 1.9, so react_agent cannot use it.)


class ScriptedLLMConfig(LLMBaseConfig, name="tcad_scripted_llm"):
    responses: list[str] = []


@register_llm_provider(config_type=ScriptedLLMConfig)
async def tcad_scripted_llm(config: ScriptedLLMConfig, builder: Builder):
    yield LLMProviderInfo(config=config, description="Scripted offline LLM for integration tests")


@register_llm_client(config_type=ScriptedLLMConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def tcad_scripted_llm_langchain(config: ScriptedLLMConfig, builder: Builder):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    yield FakeListChatModel(responses=list(config.responses))
