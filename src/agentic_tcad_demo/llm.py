"""Optional LLM planner (off by default).

The LLM sees the same observation as the rule planner and may only pick one of the currently legal
actions; it never proposes numbers. Steps still come from OFAT + Eq. (3), and acceptance is still
decided by the controller on J. The backend is any OpenAI-compatible chat-completions endpoint
(hosted APIs, NVIDIA NIM, vLLM, Ollama, ...). Configure it only through environment variables:

    AGENTIC_TCAD_LLM_BASE_URL   e.g. https://api.openai.com/v1  or  http://localhost:11434/v1
    AGENTIC_TCAD_LLM_MODEL      model name understood by that endpoint
    AGENTIC_TCAD_LLM_API_KEY    API key (optional for local servers)

If the LLM output is invalid, the controller logs a planner error; it does NOT silently fall back to
the rule planner, so a run labelled "llm" is never secretly a rule-based run.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

from .controller import Action
from .tools import ACTION_SCHEMA

SYSTEM_PROMPT = """You are the planning agent of a TCAD calibration workflow running on SYNTHETIC data.
You do not compute parameter values. Each turn you receive a JSON observation with the objective J,
its weighted residual terms, the OFAT-predicted gain of each available parameter group, and the model
components that are currently eligible for activation. Choose exactly one legal action:
""" + "\n".join(f"- {k}: {v}" for k, v in ACTION_SCHEMA.items()) + """
Rules: "update_group" targets must be untried groups from the observation; "activate_component" targets
must be listed in eligible_components. Prefer the action that addresses the dominant residual.
Answer with one JSON object only: {"action": "...", "target": "... or null", "rationale": "one sentence"}"""

Transport = Callable[[str, dict, dict, float], dict]


class LLMConfigError(RuntimeError):
    pass


def parse_action(text: str) -> Action:
    """Parse one JSON action. Raises ValueError for anything outside the action schema."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(text)
    action, target = data["action"], data.get("target")
    if action not in ACTION_SCHEMA or not (target is None or isinstance(target, str)):
        raise ValueError(f"unexpected action/target: {action!r} / {target!r}")
    return Action(action, target, str(data.get("rationale", "")))


def http_post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (user-configured URL)
        return json.loads(resp.read().decode())


class LLMPlanner:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 transport: Transport = http_post_json, timeout: float = 60.0):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.transport, self.timeout = transport, timeout
        self.name = f"llm(model={model})"

    @classmethod
    def from_env(cls, transport: Transport = http_post_json) -> "LLMPlanner":
        base = os.environ.get("AGENTIC_TCAD_LLM_BASE_URL")
        model = os.environ.get("AGENTIC_TCAD_LLM_MODEL")
        if not base or not model:
            raise LLMConfigError("the LLM planner needs AGENTIC_TCAD_LLM_BASE_URL and AGENTIC_TCAD_LLM_MODEL "
                                 "(and usually AGENTIC_TCAD_LLM_API_KEY); see .env.example. "
                                 "The default deterministic planner needs none: use --planner rule.")
        return cls(base, model, os.environ.get("AGENTIC_TCAD_LLM_API_KEY"), transport)

    def choose(self, observation: dict) -> Action:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": json.dumps(observation, default=float)}]}
        try:
            reply = self.transport(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
            return parse_action(reply["choices"][0]["message"]["content"])
        except Exception as exc:  # malformed output or transport error: surfaced as an illegal action
            return Action("invalid", None, f"LLM output could not be used: {type(exc).__name__}: {exc}")
