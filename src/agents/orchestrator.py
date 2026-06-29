from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    set_default_openai_key,
)
from src.agents.agents import OrchestratorAgent
from src.agents.context import RAGRunContext
from src.agents.schemas import DirectResponseOutput, GenerationOutput
from src.config.settings import get_config, get_secrets
from src.evaluation.cost_latency_eval import CostLatencyEvaluator, LatencyTimer
from src.generation.generator import GenerationResult
from src.utils.logger import get_logger, set_correlation_id

if TYPE_CHECKING:
    from src.core.container import AppContainer

logger = get_logger(__name__)
_cfg = get_config()
_sec = get_secrets()

_PREF_DIR = Path(_cfg.log.get("preferences_dir", "logs/user_preferences"))

_DETECT_EXTRACT_PROMPT = """\
You are a behavioral instruction detector for an AI research assistant.

Classify the user message and return EXACTLY ONE of these formats:

1. Pure behavioral instruction — applies going forward only, no immediate action:
   PREF: <concise imperative sentence>
   Use when the user sets a general preference for future responses.
   Examples: "be concise" → PREF: Be concise.

2. Behavioral instruction + re-apply to the PREVIOUS answer immediately:
   PREF: <instruction> | REDO
   Use when the instruction is about HOW the last answer was presented and the user
   clearly wants it redone (e.g. they say "now", "for the answer", "show them",
   "return them", "make it shorter", "translate it").
   Examples: "show citations now" → PREF: Show citations. | REDO
             "return citations for the answer" → PREF: Show citations. | REDO
             "make it shorter" → PREF: Be concise. | REDO
             
3. Behavioral instruction COMBINED with a new question to answer:
   PREF: <instruction> | QUERY: <standalone question>
   Use when the message contains BOTH a preference AND a distinct new question.
   Examples: "answer in bullet points. what is the methodology" →
             PREF: Answer in bullet points. | QUERY: What is the methodology?
             "cite the context. what are the findings" →
             PREF: Include citations in answers. | QUERY: What are the findings?

4. Reset / clear all preferences:
   CLEAR
   Use for: "forget everything I said", "reset your instructions", "start fresh", \
"clear your memory", "ignore all previous preferences".

5. Regular question, greeting, or follow-up — NO behavioral instruction:
   NULL
   Examples: "ok", "thanks", "got it", "what is MCI?" → NULL

Rules:
- A behavioral instruction tells the assistant HOW to behave.
- A regular question asks for information.
- "ok", "thanks", "got it", "great" alone → NULL
- Return ONLY the classified output — no explanation, no prose."""


def _pref_path(namespace: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", namespace)
    return _PREF_DIR / f"{safe}.json"


def _load_preferences(namespace: str) -> list[str]:
    path = _pref_path(namespace)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not load preferences for namespace '%s': %s", namespace, exc)
        return []


def _save_preferences(namespace: str, preferences: list[str]) -> None:
    path = _pref_path(namespace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preferences, indent=2, ensure_ascii=False))
        logger.info("Preferences saved → %s  contents=%s", path, preferences)
    except Exception as exc:
        logger.warning("Could not save preferences for namespace '%s': %s", namespace, exc)


class _PrefDetection:
    __slots__ = ("preference", "follow_on_query", "redo", "clear_all")

    def __init__(
        self,
        preference: str | None = None,
        follow_on_query: str | None = None,
        redo: bool = False,
        clear_all: bool = False,
    ) -> None:
        self.preference = preference
        self.follow_on_query = follow_on_query
        self.redo = redo
        self.clear_all = clear_all

    @property
    def has_action(self) -> bool:
        return bool(self.preference or self.clear_all)


def _detect_preference(query: str, openai_client) -> _PrefDetection:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _DETECT_EXTRACT_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Preference detection failed (%s); skipping.", exc)
        return _PrefDetection()

    if not raw or raw.upper() == "NULL":
        return _PrefDetection()

    if raw.upper() == "CLEAR":
        return _PrefDetection(clear_all=True)

    if raw.upper().startswith("PREF:"):
        body = raw[5:].strip()
        if "|" in body:
            parts = body.split("|", 1)
            pref = parts[0].strip()
            suffix = parts[1].strip()
            if suffix.upper() == "REDO":
                return _PrefDetection(preference=pref, redo=True)
            if suffix.upper().startswith("QUERY:"):
                suffix = suffix[6:].strip()
            return _PrefDetection(preference=pref, follow_on_query=suffix or None)
        return _PrefDetection(preference=body)

    return _PrefDetection(preference=raw)


_RECONCILE_PROMPT = """\
You manage a persistent list of behavioral instructions for an AI assistant.

Existing instructions (0-indexed):
{existing}

New instruction: "{new}"

Choose ONE action:
- REPLACE N  — the new instruction directly contradicts or supersedes instruction N \
(e.g. "Show citations" supersedes "Do not show citations"). N is the 0-based index.
- ADD         — the new instruction is genuinely new and does not conflict with any existing one.
- SKIP        — the new instruction is already covered by an existing one (duplicate or subset).

Return ONLY one of these exact strings: REPLACE <N>, ADD, or SKIP. Nothing else."""


def _reconcile_preferences(new_pref: str, existing: list[str], openai_client) -> list[str]:
    if not existing:
        return [new_pref]

    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(existing))
    prompt = _RECONCILE_PROMPT.format(existing=numbered, new=new_pref)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        decision = (response.choices[0].message.content or "").strip().upper()

        if decision == "SKIP":
            return existing
        if decision.startswith("REPLACE"):
            parts = decision.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 0 <= idx < len(existing):
                    updated = list(existing)
                    updated[idx] = new_pref
                    logger.info(
                        "Preference reconciliation: replaced index %d ('%s') with '%s'",
                        idx,
                        existing[idx],
                        new_pref,
                    )
                    return updated
    except Exception as exc:
        logger.warning("Preference reconciliation failed (%s); appending.", exc)

    return existing + [new_pref]


class RAGOrchestrator:

    def __init__(self, container: AppContainer) -> None:
        self._container = container
        self._pii_guard = container.pii_guard
        self._acl = container.acl
        self._qu = container.query_understanding
        self._max_turns = _cfg.query.get("agents", {}).get("max_turns", 15)

        set_default_openai_key(_sec.openai_api_key)

        self._agent = OrchestratorAgent
        self._cl_evaluator = CostLatencyEvaluator(
            score_ledger_path=_cfg.log.get("cost_latency_ledger", "logs/cost_latency.jsonl")
        )

    async def run(
        self,
        raw_query: str,
        auth_token: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        namespace: str | None = None,
    ) -> GenerationResult:

        correlation_id = set_correlation_id()
        timer = LatencyTimer()
        logger.info(
            "RAGOrchestrator.run started",
            extra={"query": raw_query[:120], "correlation_id": correlation_id},
        )

        if namespace is None:
            namespace = "default"
            try:
                claims = self._acl.authenticate(auth_token or "")
                namespace = self._acl.get_namespace(claims)
            except Exception as exc:
                logger.warning("ACL auth failed (%s); using default namespace.", exc)

        user_preferences = _load_preferences(namespace)
        openai_client = self._container.generator._openai

        routing_intent = self._qu.get_routing_intent(raw_query, conversation_history or [])

        extracted_pref: str | None = None
        effective_query = raw_query

        if routing_intent in ("retrieval", "conversational"):
            detection = _detect_preference(raw_query, openai_client)

            if detection.clear_all:
                if user_preferences:
                    user_preferences = []
                    _save_preferences(namespace, user_preferences)
                    logger.info("User preferences cleared for namespace '%s'.", namespace)
                extracted_pref = "__CLEARED__"
                routing_intent = "conversational"

            elif detection.preference:
                extracted_pref = detection.preference
                updated = _reconcile_preferences(extracted_pref, user_preferences, openai_client)
                if updated != user_preferences:
                    user_preferences = updated
                    _save_preferences(namespace, user_preferences)

                if detection.follow_on_query:
                    effective_query = detection.follow_on_query
                    routing_intent = "retrieval"
                    logger.info(
                        "Mixed message — routing question '%s' to retrieval.", effective_query[:80]
                    )
                elif detection.redo:
                    last_question = next(
                        (
                            m["content"]
                            for m in reversed(conversation_history or [])
                            if m.get("role") == "user"
                        ),
                        None,
                    )
                    if last_question:
                        effective_query = last_question
                        routing_intent = "retrieval"
                        logger.info("REDO — re-running previous query '%s'.", effective_query[:80])
                    else:
                        routing_intent = "conversational"
                else:
                    routing_intent = "conversational"

        base_history = self._qu.compress_history(list(conversation_history or []))
        history_with_pref = list(base_history)
        if extracted_pref == "__CLEARED__":
            history_with_pref = history_with_pref + [
                {
                    "role": "system",
                    "content": "[USER PREFERENCE RESET] All previous preferences have been cleared.",
                }
            ]
        elif extracted_pref:
            history_with_pref = history_with_pref + [
                {"role": "system", "content": f"[USER PREFERENCE STORED] {extracted_pref}"}
            ]

        with timer.stage("condense"):
            condensed_query = self._qu.condense_with_history(effective_query, history_with_pref)

        if condensed_query != raw_query and history_with_pref and routing_intent == "retrieval":
            routing_intent = "followup"

        ctx = RAGRunContext(
            raw_query=raw_query,
            conversation_history=history_with_pref,
            processed_query=condensed_query,  # type: ignore[arg-type]
            query_routing_intent=routing_intent,
            auth_token=auth_token,
            correlation_id=correlation_id,
            namespace=namespace,
            container=self._container,
            user_preferences=user_preferences,
        )

        try:
            with timer.stage("agent"):
                run_result = await Runner.run(
                    self._agent,
                    input=raw_query,
                    context=ctx,
                    max_turns=self._max_turns,
                )
        except InputGuardrailTripwireTriggered as exc:
            logger.warning(
                "Input guardrail tripped: %s", exc, extra={"correlation_id": correlation_id}
            )
            return GenerationResult(
                answer=(
                    "Your query could not be processed. " "Please check the input and try again."
                ),
                model_used="guardrail",
            )
        except OutputGuardrailTripwireTriggered as exc:
            logger.warning(
                "Output guardrail tripped: %s", exc, extra={"correlation_id": correlation_id}
            )
            return GenerationResult(
                answer=(
                    "I was unable to generate a sufficiently reliable answer "
                    "from the available sources. Please try rephrasing your question."
                ),
                model_used="guardrail",
            )
        except Exception as exc:
            logger.error(
                "Agent pipeline error: %s",
                exc,
                extra={"correlation_id": correlation_id},
                exc_info=True,
            )
            return GenerationResult(
                answer="An internal error occurred. Please try again.",
                model_used="error",
            )

        result = self._extract_result(run_result, ctx)

        try:
            result.answer = self._pii_guard.redact(result.answer, context="output")
        except Exception as exc:
            logger.warning("Output PII scan failed (non-fatal): %s", exc)

        try:
            self._cl_evaluator.record_request(
                query=raw_query,
                namespace=namespace,
                timer=timer,
                generation_result=result,
            )
        except Exception as exc:
            logger.warning("Cost/latency recording failed (non-fatal): %s", exc)

        logger.info(
            "RAGOrchestrator.run complete",
            extra={
                "correlation_id": correlation_id,
                "last_agent": run_result.last_agent.name if run_result.last_agent else "unknown",
                "agent_trace": ctx.agent_trace,
                "answer_length": len(result.answer),
                "latency_ms": round(timer.total_ms, 1),
            },
        )
        return result

    def _extract_result(self, run_result, ctx: RAGRunContext) -> GenerationResult:
        last_agent_name = run_result.last_agent.name if run_result.last_agent else "unknown"

        if ctx.generation_result is not None:
            ctx.generation_result.model_used = last_agent_name
            return ctx.generation_result

        raw_output = getattr(run_result, "final_output", None)

        if isinstance(raw_output, GenerationOutput):
            return GenerationResult(
                answer=raw_output.answer,
                citations=[c.model_dump() for c in (raw_output.citations or [])],
                faithfulness_score=raw_output.faithfulness_score,
                has_conflict=raw_output.has_conflict,
                model_used=last_agent_name,
            )

        if isinstance(raw_output, DirectResponseOutput):
            return GenerationResult(answer=raw_output.answer, model_used=last_agent_name)

        if isinstance(raw_output, str) and raw_output.strip():
            return GenerationResult(answer=raw_output.strip(), model_used=last_agent_name)

        for cls in (GenerationOutput, DirectResponseOutput):
            try:
                output = run_result.final_output_as(cls, raise_if_incorrect_type=False)
                if output:
                    if isinstance(output, GenerationOutput):
                        return GenerationResult(
                            answer=output.answer,
                            citations=[c.model_dump() for c in (output.citations or [])],
                            faithfulness_score=output.faithfulness_score,
                            has_conflict=output.has_conflict,
                            model_used=last_agent_name,
                        )
                    return GenerationResult(answer=output.answer, model_used=last_agent_name)
            except Exception:
                pass

        logger.warning(
            "Unrecognised final_output type from agent '%s': %s — value: %s",
            last_agent_name,
            type(raw_output),
            repr(raw_output)[:200],
        )
        return GenerationResult(
            answer="I'm sorry, I wasn't able to process that. Please try again.",
            model_used=last_agent_name,
        )
