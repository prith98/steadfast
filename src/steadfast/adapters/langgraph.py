"""Adapter for LangGraph compiled state-graph agents.

Wraps a compiled :class:`langgraph.graph.state.CompiledStateGraph` so it
conforms to the Steadfast :class:`steadfast.agent.Agent` protocol. The
adapter invokes the graph via ``ainvoke`` with a single human message
built from the task input, extracts the final answer from the last AI
message in ``state.messages``, and collects trajectory entries from the
``tool_calls`` field of any intermediate AI messages.

The trajectory contract is fixed by ADR-0006 §A:

* Populate :attr:`AgentResponse.trajectory` from tool calls when the
  graph exposes them.
* When the graph never invokes a tool, ``trajectory`` is the empty list
  ``[]`` — *not* ``None``. Trajectory consistency falls through to the
  same N/A path toolless agents already use per ADR-0004 §G.
* Confidence comes from the standard
  :func:`steadfast.perturbations.confidence.parse_verbalized_confidence`
  parser applied to the final AI message text per ADR-0005 §B-C. Unlike
  :class:`steadfast.agent.SimplePromptingAgent`, the adapter does not
  re-invoke the graph on parse failure — the parse-fail soft-fail
  surface (confidence ``None``, ``parse_ok=False`` in metadata) is
  consistent with ADR-0005 §C, but the retry path would require
  re-running the full graph (potentially many tool calls) and is
  deferred to v0.2 if metric distributions on the LangGraph adapter
  surface a systematic parse-failure bias.

The adapter assumes the ``state.messages`` shape that LangGraph's
prebuilt agents and most idiomatic custom graphs use: a flat
``list[BaseMessage]`` keyed under ``"messages"`` (overridable via
``messages_key``). Exotic graphs (subgraphs, custom reducers, non-list
message containers) may require subclassing the adapter to override
:meth:`_extract_messages` / :meth:`_extract_trajectory`.

LangGraph is an **optional dependency** — install via
``pip install steadfast[langgraph]``. The adapter raises a clear
:class:`ImportError` at construction time when the extra is not
installed.

References:

* ADR-0006 §A — LangGraph trajectory contract; closes auto-memory Q2.
* ADR-0002 §A.2 — trajectory optionality (empty list, not None).
* ADR-0005 §B-C — confidence parser contract.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from steadfast.agent import Agent, AgentResponse, MetadataValue, Task, ToolCall
from steadfast.perturbations.confidence import parse_verbalized_confidence

if TYPE_CHECKING:
    # These types are only used in stringified annotations; runtime use
    # of the LangGraph API goes through the duck-typed ``Any`` paths in
    # the adapter so the import is optional. The TYPE_CHECKING-only
    # import keeps mypy happy without requiring langgraph at runtime.
    from langgraph.graph.state import CompiledStateGraph


def _scalarize(value: Any) -> MetadataValue:
    """Project an arbitrary value into the :data:`MetadataValue` scalar union.

    ``ToolCall.args`` is typed as ``dict[str, MetadataValue]`` per the
    Q5 metadata typing policy (ADR-0002 §A.3 / project kickoff); rich
    nested structures from LangChain tool-call args (lists, dicts) are
    stringified so they round-trip cleanly into the public type
    contract. Stringification is the simplest faithful interpretation:
    diagnostic readers see the original shape via ``repr``-style output;
    structural retry matching in
    :func:`steadfast.metrics.robustness._has_post_corruption_retry`
    works on whatever scalar form the agent emits consistently.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _extract_text(message: Any) -> str:
    """Best-effort extraction of the text content from a LangChain message.

    LangChain's ``BaseMessage.content`` can be either a ``str`` or a
    list of content blocks (the multimodal shape). When the content is a
    list, we concatenate any block whose ``type == "text"``. Non-text
    blocks (images, etc.) are dropped — the Steadfast surface is
    text-only for v0.1.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # Multimodal block shape: {"type": "text", "text": "..."}.
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _is_ai_message(message: Any) -> bool:
    """Return True iff ``message`` is a LangChain ``AIMessage``.

    Duck-typed: checks the class name rather than importing
    ``langchain_core.messages.AIMessage`` so the adapter file remains
    importable when ``langchain-core`` is not installed (the
    construction-time gate raises a clear ImportError; module import is
    deliberately lazy).
    """
    return type(message).__name__ == "AIMessage"


def _is_tool_message(message: Any) -> bool:
    """Return True iff ``message`` is a LangChain ``ToolMessage``."""
    return type(message).__name__ == "ToolMessage"


class LangGraphAdapter(Agent):
    """Wrap a compiled LangGraph state-graph agent as a Steadfast :class:`Agent`.

    Parameters
    ----------
    graph:
        A compiled LangGraph state graph, typically the result of
        ``StateGraph(...).compile()`` or
        ``langgraph.prebuilt.create_react_agent(model, tools)``. The
        graph must expose an ``ainvoke`` coroutine and accept a state
        dict keyed by ``messages_key``.
    messages_key:
        State key under which the graph stores its message list.
        Defaults to ``"messages"`` (the LangGraph prebuilt-agent
        convention).

    Notes
    -----
    The adapter does **not** mutate the supplied graph. It is safe to
    reuse the same adapter instance across many concurrent ``arun``
    calls; LangGraph's ``ainvoke`` is task-isolated.

    Confidence parsing follows the same surface as
    :class:`steadfast.agent.SimplePromptingAgent`: when
    :attr:`Task.confidence_suffix` is set, the adapter appends the
    suffix to the user message and parses the trailing
    ``ANSWER:``/``CONFIDENCE:`` block from the final AI message text.
    On parse failure the rep stays ``COMPLETED`` with
    ``confidence=None`` and ``metadata["elicitation_parse_ok"]=False``
    — the metric layer skips None-confidence reps for calibration but
    keeps them in consistency / robustness / format pools per
    ADR-0005 §C.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[Any],
        *,
        messages_key: str = "messages",
    ) -> None:
        try:
            import langchain_core  # noqa: F401
            import langgraph  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LangGraphAdapter requires the 'langgraph' optional extra. "
                "Install with: pip install 'steadfast[langgraph]'"
            ) from exc
        # Stored as ``Any`` (not ``CompiledStateGraph[Any]``) because
        # mypy's strict mode requires the type parameter to be concrete;
        # the stringified hint above documents the expected type for
        # readers while keeping mypy strict-compatible at runtime.
        self._graph: Any = graph
        self._messages_key = messages_key

    async def arun(self, task: Task) -> AgentResponse:
        # Build the human message text. When the task carries a
        # confidence suffix (calibration co-requested), append it so the
        # graph's LLM sees the standard ANSWER/CONFIDENCE tail
        # instructions. The graph's prompt template (if any) is the
        # graph's concern; we feed it the prompt the way the rest of
        # Steadfast does.
        if task.confidence_suffix:
            prompt = f"{task.input}\n\n{task.confidence_suffix}"
        else:
            prompt = task.input

        # The "messages" input convention: a list of tuples
        # ``[(role, content), ...]`` is the most portable shape across
        # LangGraph's prebuilt and custom graphs. LangChain-Core
        # auto-converts tuples to ``HumanMessage`` / ``SystemMessage`` /
        # etc. at the graph entry point.
        state_in: dict[str, Any] = {self._messages_key: [("human", prompt)]}

        # ``ainvoke`` returns the final state dict; we read the message
        # log from it. Any exception raised by the graph propagates —
        # the metric layer catches it via ``return_exceptions=True`` on
        # asyncio.gather and excludes the failing rep, matching the
        # contradiction/typo/distractor patterns.
        final_state: dict[str, Any] = await self._graph.ainvoke(state_in)
        messages: list[Any] = self._extract_messages(final_state)

        final_text = self._extract_final_text(messages)
        trajectory = self._extract_trajectory(messages)

        # Confidence parsing only when the elicitation suffix is set.
        # Soft-fail on parse failure per ADR-0005 §C: rep still
        # COMPLETED, ``confidence=None``, ``parse_ok=False`` in
        # metadata. Unlike SimplePromptingAgent we don't retry — see
        # the module docstring rationale.
        confidence: float | None
        refused: bool
        answer: str
        parse_ok: bool
        if task.confidence_suffix:
            parsed = parse_verbalized_confidence(final_text)
            parse_ok = parsed.parse_ok
            if parse_ok:
                answer = parsed.answer
                confidence = parsed.confidence
                refused = parsed.refused
            else:
                answer = final_text.strip()
                confidence = None
                refused = False
        else:
            answer = final_text.strip()
            confidence = None
            refused = False
            parse_ok = True

        metadata: dict[str, MetadataValue] = {
            "adapter": "langgraph",
            "elicitation_parse_ok": parse_ok,
            "n_messages": len(messages),
            "n_tool_calls": len(trajectory),
        }

        return AgentResponse(
            answer=answer,
            confidence=confidence,
            refused=refused,
            trajectory=trajectory,
            raw_output=final_text,
            cost_usd=Decimal("0"),
            metadata=metadata,
        )

    def _extract_messages(self, state: dict[str, Any]) -> list[Any]:
        """Pull the message list out of the final state dict.

        Defaults to indexing by ``messages_key`` and treating the value
        as a list. Subclasses can override for graphs that store
        messages under nested keys or use custom message containers.
        """
        raw = state.get(self._messages_key, [])
        if isinstance(raw, list):
            return raw
        # A single-message graph might return a bare message instead of
        # a list; wrap defensively so downstream extraction works.
        return [raw]

    def _extract_final_text(self, messages: list[Any]) -> str:
        """Return the text of the last AIMessage, or the empty string."""
        for message in reversed(messages):
            if _is_ai_message(message):
                return _extract_text(message)
        return ""

    def _extract_trajectory(self, messages: list[Any]) -> list[ToolCall]:
        """Build a flat trajectory from AIMessage.tool_calls and ToolMessage results.

        Iterates ``messages`` in invocation order. For each AIMessage
        with a populated ``tool_calls`` attribute, emits one
        :class:`ToolCall` per call. Where a subsequent ToolMessage
        carries a matching ``tool_call_id``, its content populates
        ``ToolCall.result``; otherwise ``result`` is ``None``.

        LangChain tool-call entries can be either dicts (the standard
        shape: ``{"name": str, "args": dict, "id": str}``) or pydantic
        objects with attribute access — we accept both.
        """
        # Index tool results by their ``tool_call_id`` for fast lookup.
        # The ToolMessage shape (langchain-core 0.3+) puts the id on
        # ``.tool_call_id``.
        results_by_id: dict[str, str] = {}
        for message in messages:
            if not _is_tool_message(message):
                continue
            tool_call_id = getattr(message, "tool_call_id", None)
            if not isinstance(tool_call_id, str):
                continue
            results_by_id[tool_call_id] = _extract_text(message)

        trajectory: list[ToolCall] = []
        for message in messages:
            if not _is_ai_message(message):
                continue
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                continue
            for raw_call in tool_calls:
                if isinstance(raw_call, dict):
                    name = raw_call.get("name", "")
                    args = raw_call.get("args", {}) or {}
                    call_id = raw_call.get("id", "")
                else:
                    name = getattr(raw_call, "name", "")
                    args = getattr(raw_call, "args", {}) or {}
                    call_id = getattr(raw_call, "id", "")
                scalar_args: dict[str, MetadataValue] = {
                    str(k): _scalarize(v) for k, v in dict(args).items()
                }
                result = results_by_id.get(call_id) if isinstance(call_id, str) else None
                trajectory.append(
                    ToolCall(
                        name=str(name),
                        args=scalar_args,
                        result=result,
                    )
                )
        return trajectory


__all__ = ["LangGraphAdapter"]
