from __future__ import annotations

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
)
from src.agents.context import RAGRunContext
from src.agents.schemas import GenerationOutput
from src.config.settings import get_config
from src.operations.ops_middleware import PIIGuard
from src.utils.logger import get_logger

logger = get_logger(__name__)
_cfg = get_config()
_pii_guard = PIIGuard()


@input_guardrail
async def pii_input_guardrail(
    ctx: RunContextWrapper[RAGRunContext],
    agent: Agent,
    input: str,
) -> GuardrailFunctionOutput:

    guardrail_cfg = _cfg.operations.get("pii", {})
    block_on_pii = guardrail_cfg.get("pii_block_on_input", False)

    try:
        result = _pii_guard.redact(
            text=input,
            context="query",
        )

    except Exception as exc:
        logger.error(
            "PII guard failed — blocking query as precaution",
            extra={
                "error": str(exc),
                "correlation_id": ctx.context.correlation_id,
            },
        )

        return GuardrailFunctionOutput(
            output_info="PII scan failure — query blocked.",
            tripwire_triggered=True,
        )

    if result == input:
        logger.info(
            "PII scan clean",
            extra={"correlation_id": ctx.context.correlation_id},
        )

        return GuardrailFunctionOutput(
            output_info="pii_scan_ok",
            tripwire_triggered=False,
        )

    if block_on_pii:
        logger.warning(
            "PHI detected — query blocked",
            extra={
                "correlation_id": ctx.context.correlation_id,
            },
        )

        ctx.context.record(
            "pii_input_guardrail",
            "BLOCKED — PII detected",
        )

        return GuardrailFunctionOutput(
            output_info="pii_blocked",
            tripwire_triggered=True,
        )


@output_guardrail
async def output_faithfulness_guardrail(
    ctx: RunContextWrapper[RAGRunContext],
    agent: Agent,
    output: GenerationOutput,
) -> GuardrailFunctionOutput:

    min_score: float = _cfg.query.get("guardrails", {}).get("min_faithfulness_score", 0.40)

    score = output.faithfulness_score
    if score is not None and score < min_score:
        logger.warning(
            "Output faithfulness guardrail tripped",
            extra={
                "faithfulness_score": score,
                "min_required": min_score,
                "correlation_id": ctx.context.correlation_id,
            },
        )
        return GuardrailFunctionOutput(
            output_info=(f"Faithfulness score {score:.2f} below minimum {min_score:.2f}."),
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info=f"faithfulness_ok: {score}", tripwire_triggered=False
    )


@output_guardrail
async def output_length_guardrail(
    ctx: RunContextWrapper[RAGRunContext],
    agent: Agent,
    output: GenerationOutput,
) -> GuardrailFunctionOutput:

    min_tokens: int = _cfg.query.get("guardrails", {}).get("min_answer_tokens", 20)
    max_tokens: int = _cfg.query.get("guardrails", {}).get("max_answer_tokens", 8000)

    length = len(output.answer)

    if length < min_tokens:
        logger.warning(
            "Output too short — possible refusal or empty answer",
            extra={"length": length, "min": min_tokens},
        )
        return GuardrailFunctionOutput(
            output_info=f"Answer too short ({length} tokens).",
            tripwire_triggered=True,
        )

    if length > max_tokens:
        logger.warning(
            "Output too long — possible runaway generation",
            extra={"length": length, "max": max_tokens},
        )
        return GuardrailFunctionOutput(
            output_info=f"Answer too long ({length} tokens).",
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(output_info=f"length_ok: {length}", tripwire_triggered=False)
