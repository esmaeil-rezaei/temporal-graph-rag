from __future__ import annotations

from agents import Agent, ModelSettings, RunContextWrapper, handoff
from src.agents.context import RAGRunContext
from src.agents.graph_tools import graph_query
from src.agents.guardrails import (
    output_faithfulness_guardrail,
    output_length_guardrail,
    pii_input_guardrail,
)
from src.agents.schemas import DirectResponseOutput, GenerationOutput
from src.agents.tools import (
    generate_answer,
    generate_from_history,
    get_routing_intent,
    prepare_query,
    retrieve_context,
)
from src.config.settings import get_config

_cfg = get_config()
_orchestrator_model = _cfg.query.get("agents", {}).get("orchestrator_model", "gpt-4o")
_worker_model = _cfg.query.get("agents", {}).get("worker_model", "gpt-4o")


def _conversational_instructions(
    ctx: RunContextWrapper[RAGRunContext],
    agent: Agent[RAGRunContext],
) -> str:
    """
    Dynamic instructions for ConversationalAgent.
    Injects the last few conversation turns so the agent can answer
    follow-up requests like 'summarize it' or 'the last answer'.
    """
    base = (
        "You are a friendly and concise conversational assistant.\n\n"
        "Respond naturally, conversationally, and briefly.\n\n"
        "Rules:\n"
        "- Keep responses short and warm.\n"
        "- Do not fabricate factual information.\n"
        "- If the user is asking you to summarize or reference a PREVIOUS answer, "
        "use the conversation history provided below.\n"
        "- If the user appears to ask a knowledge-intensive question not covered by "
        "the history, encourage them to provide more details.\n"
        "- Never claim to have retrieved information.\n"
        "- If the conversation history contains a [USER PREFERENCE STORED] system "
        "message, you were routed here because the user gave a behavioral instruction. "
        "Acknowledge it explicitly — e.g. 'Got it, I\\'ll answer in bullet points from "
        "now on.' or 'Noted — I\\'ll include citations in all answers going forward.' "
        "Be specific about WHAT was stored. Do NOT give a generic response.\n"
        "- If the conversation history contains a [USER PREFERENCE RESET] system "
        "message, confirm the reset — e.g. 'Done — all preferences cleared. "
        "Starting fresh.'"
    )
    preferences = getattr(ctx.context, "user_preferences", None) or []
    if preferences:
        pref_lines = "\n".join(f"- {p}" for p in preferences)
        base += (
            "\n\n---\nUSER PREFERENCES (always follow these across the entire session):\n"
            + pref_lines
            + "\n---"
        )

    history = getattr(ctx.context, "conversation_history", None) or []
    if not history:
        return base
    recent = history[-6:]
    history_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in recent)
    return base + "\n\n---\nCONVERSATION HISTORY (most recent turns):\n" + history_text + "\n---"


ConversationalAgent: Agent[RAGRunContext] = Agent(
    name="ConversationalAgent",
    handoff_description=(
        "Handles greetings, casual conversation, and general non-retrieval interactions. "
        "Also handles requests to summarize or reference the previous answer."
    ),
    instructions=_conversational_instructions,
    model=_orchestrator_model,
    model_settings=ModelSettings(temperature=0.7),
    output_type=DirectResponseOutput,
    input_guardrails=[pii_input_guardrail],
)


def _retrieval_instructions(
    ctx: RunContextWrapper[RAGRunContext],
    agent: Agent[RAGRunContext],
) -> str:
    base = (
        "You are a retrieval-augmented generation (RAG) assistant.\n\n"
        "Your task is to answer the user's question using ONLY retrieved context.\n\n"
        "Execution flow (mandatory):\n\n"
        "STEP 1 — Call prepare_query.\n"
        "STEP 2 — Call retrieve_context.\n"
        "STEP 3 — Optionally call graph_query.\n"
        "STEP 4 — Call generate_answer.\n\n"
        "Rules:\n"
        "- NEVER use prior knowledge.\n"
        "- NEVER fabricate information.\n"
        "- NEVER say 'I am retrieving', 'let me search', or narrate your steps.\n"
        "- NEVER produce any output before generate_answer completes.\n"
        "- Preserve all [CITE:...] markers exactly.\n"
        "- If retrieval is insufficient, return the generated fallback response exactly.\n"
        "- If conflict information exists, include it verbatim.\n"
        "- Do not add extra commentary, disclaimers, or preambles."
    )
    preferences = getattr(ctx.context, "user_preferences", None) or []
    if preferences:
        pref_lines = "\n".join(f"- {p}" for p in preferences)
        base += "\n\n---\nUSER PREFERENCES (always follow these):\n" + pref_lines + "\n---"
    return base


RetrievalAgent: Agent[RAGRunContext] = Agent(
    name="RetrievalAgent",
    handoff_description=("Handles queries requiring retrieval from external knowledge sources."),
    instructions=_retrieval_instructions,
    tools=[prepare_query, retrieve_context, graph_query, generate_answer],
    model=_worker_model,
    model_settings=ModelSettings(temperature=0.0),
    output_type=GenerationOutput,
    input_guardrails=[pii_input_guardrail],
    output_guardrails=[
        output_faithfulness_guardrail,
        output_length_guardrail,
    ],
)

FollowUpAgent: Agent[RAGRunContext] = Agent(
    name="FollowUpAgent",
    instructions=(
        "You are a follow-up resolution agent. Execute these steps in order:\n\n"
        "STEP 1: Call generate_from_history.\n\n"
        "STEP 2: Read the tool result:\n"
        "  - If needs_retrieval is false → you are done. Do NOT output any text.\n"
        "  - If needs_retrieval is true → hand off to RetrievalAgent.\n\n"
        "ABSOLUTE RULES:\n"
        "  - Never produce any text yourself.\n"
        "  - Never skip steps.\n"
        "  - Never ask the user for clarification.\n"
        "  - Hand off silently."
    ),
    tools=[generate_from_history],
    handoffs=[handoff(RetrievalAgent)],
    model=_worker_model,
    model_settings=ModelSettings(temperature=0.0),
    # NOTE: output_type intentionally omitted — output_type forces structured JSON
    # output mode on the model, which conflicts with handoff tool calls. An agent
    # that might hand off must produce plain tool calls, not a typed final output.
    # generate_from_history stores its result on ctx.generation_result directly.
)

OrchestratorAgent: Agent[RAGRunContext] = Agent(
    name="OrchestratorAgent",
    instructions=(
        "You are the routing controller for a multi-agent RAG system.\n\n"
        "Your ONLY job is to hand off to the correct agent. Never answer directly.\n\n"
        "STEP 1 — Call get_routing_intent. It returns a pre-computed intent value.\n\n"
        "STEP 2 — Hand off immediately based on the returned intent:\n\n"
        "  'followup'       → hand off to FollowUpAgent\n"
        "  'conversational' → hand off to ConversationalAgent\n"
        "  'retrieval'      → hand off to RetrievalAgent\n\n"
        "Never answer directly. Never skip the tool call. Never reason about the query yourself."
    ),
    tools=[get_routing_intent],
    handoffs=[
        handoff(FollowUpAgent),
        handoff(ConversationalAgent),
        handoff(RetrievalAgent),
    ],
    model=_orchestrator_model,
    model_settings=ModelSettings(temperature=0.0),
    input_guardrails=[pii_input_guardrail],
)
