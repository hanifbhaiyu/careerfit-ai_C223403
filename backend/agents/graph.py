"""The multi-agent workflow, assembled as a LangGraph state machine.

Shape of the graph:

                        ┌──────────────┐
                        │    router    │   supervisor: one cheap LLM call
                        └──────┬───────┘
             ┌────────────┬────┴─────┬─────────────┬──────────────┐
             ▼            ▼          ▼             ▼              ▼
        cv_analyst   job_scout   skill_gap   market_intel   general_advice
             │            │          │             │              │
             └────────────┴────┬─────┴─────────────┴──────────────┘
                               ▼
                        ┌──────────────┐
                        │  synthesiser │   merges findings into one answer
                        └──────┬───────┘
                               ▼
                              END

Why a graph rather than a chain of if-statements: the router's branch is data,
not control flow, so LangSmith records it as a traceable decision; state merging
is handled by the reducer on `findings`; and adding a specialist later means
adding one node and one edge rather than editing a dispatch function.

The `cv_analyst` path also fans out to `skill_gap`, because a CV review is much
more useful when it says what to learn next. That is the one place where two
specialists run for a single request, and it is why the synthesiser exists.
"""

from __future__ import annotations

import logging
import uuid

from langgraph.graph import END, StateGraph

from backend.agents.advisor_agent import give_general_advice, synthesise_answer
from backend.agents.cv_analyst_agent import analyse_cv
from backend.agents.job_scout_agent import find_matching_jobs
from backend.agents.market_intel_agent import gather_market_intel
from backend.agents.router_agent import route_request, select_branch
from backend.agents.skill_gap_agent import analyse_skill_gap
from backend.agents.state import CareerState
from backend.core.tracing import run_config

logger = logging.getLogger(__name__)

_compiled_graph = None


def build_graph():
    """Construct and compile the workflow."""
    workflow = StateGraph(CareerState)

    workflow.add_node("router", route_request)
    workflow.add_node("cv_analyst", analyse_cv)
    workflow.add_node("job_scout", find_matching_jobs)
    workflow.add_node("skill_gap", analyse_skill_gap)
    workflow.add_node("market_intel", gather_market_intel)
    workflow.add_node("general_advice", give_general_advice)
    workflow.add_node("synthesiser", synthesise_answer)

    workflow.set_entry_point("router")

    # The router's returned route string selects the next node.
    workflow.add_conditional_edges(
        "router",
        select_branch,
        {
            "cv_review": "cv_analyst",
            "job_match": "job_scout",
            "skill_gap": "skill_gap",
            "market_intel": "market_intel",
            "general_advice": "general_advice",
        },
    )

    # A CV review continues into skill gap analysis: knowing the CV is weak is
    # only half an answer, the candidate also needs to know what to fix.
    workflow.add_edge("cv_analyst", "skill_gap")

    workflow.add_edge("skill_gap", "synthesiser")
    workflow.add_edge("job_scout", "synthesiser")
    workflow.add_edge("market_intel", "synthesiser")
    workflow.add_edge("general_advice", "synthesiser")
    workflow.add_edge("synthesiser", END)

    return workflow.compile()


def get_graph():
    """Return the compiled graph, building it once per process."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("Compiled the CareerFit agent graph.")
    return _compiled_graph


def run_workflow(
    query: str,
    session_id: str | None = None,
    cv_text: str = "",
    cv_profile: dict | None = None,
    target_role: str = "",
    location: str = "",
    ocr_engine: str = "",
) -> dict:
    """Execute the full workflow for one request.

    Returns the final state, which carries the answer plus all the evidence
    (retrieved chunks, search results, grounding citations) so the UI can show
    the user what the answer was built from.
    """
    session = session_id or str(uuid.uuid4())

    initial_state: CareerState = {
        "query": query,
        "session_id": session,
        "cv_text": cv_text,
        "cv_profile": cv_profile or {},
        "target_role": target_role,
        "location": location,
        "ocr_engine": ocr_engine,
        "findings": {},
        "retrieved_chunks": [],
        "search_results": [],
        "grounding_sources": [],
        "grounding_queries": [],
        "errors": [],
    }

    entrypoint = "cv_upload" if cv_text else "chat"
    config = run_config(session_id=session, entrypoint=entrypoint)

    logger.info("Running workflow for session %s", session)
    final_state = get_graph().invoke(initial_state, config=config)

    return final_state
