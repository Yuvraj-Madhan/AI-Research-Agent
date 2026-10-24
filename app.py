
from typing import Literal, TypedDict

from ollama import chat
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

from src.logger import get_logger
from src.tools import (
    research_query,
    coding_query,
    calcuator_query,
    writing_query,
    response_formatter,
    general_query,

)
from src.rag_node import rag_node

logger = get_logger("ai_research_agent")


# -------------------- STATE --------------------

class State(TypedDict, total=False):
    user_query: str
    query_category: str
    llm_result: str
    needs_calculation: bool
    is_calculator_query: bool


class RouteResponse(BaseModel):
    category: Literal[
        "research_query",
        "coding_query",
        "calculator_query",
        "writer_query",
        "rag",
        "general_qa",
    ]
    needs_calculation: bool


class ExpressionResponse(BaseModel):
    expression: str


# -------------------- ROUTER --------------------

def route_query(state: State) -> dict:
    user_query = state["user_query"]

    
    system_prompt = """
    You are a query classification system for an AI agent.

    Classify the user's query into exactly ONE category:
    research_query, coding_query, calculator_query, writer_query, rag, or general_qa.

    IMPORTANT CLASSIFICATION PRIORITY:
    1. If the user asks about information in their PDF, uploaded file,
    private document, indexed knowledge base, or vector database,
    choose rag.
    2. Otherwise, if the query requires current information or external
    web research, choose research_query.
    3. Otherwise, if the user requests code or programming help,
    choose coding_query.
    4. Otherwise, if the user requests a writing artifact, choose writer_query.
    5. Otherwise, if the user asks for standalone arithmetic or mathematics,
    choose calculator_query.
    6. Otherwise, choose general_qa.

    CATEGORY DEFINITIONS:

    rag:
    Questions that require searching the user's private documents,
    uploaded PDFs, indexed files, or vector database.

    This category takes priority over general_qa, even when the question
    asks for a definition, explanation, summary, or educational concept.

    Examples:
    - "Explain attention mechanism from the PDF"
    -> category="rag", needs_calculation=false
    - "What does my PDF say about transformers?"
    -> category="rag", needs_calculation=false
    - "Summarize the uploaded document"
    -> category="rag", needs_calculation=false
    - "According to my document, what is RAG?"
    -> category="rag", needs_calculation=false

    general_qa:
    General knowledge questions, definitions, educational explanations,
    and conceptual questions that can be answered without searching
    the user's private documents.

    Examples:
    - "What is attention mechanism?"
    -> category="general_qa", needs_calculation=false
    - "Explain transformers in simple words"
    -> category="general_qa", needs_calculation=false
    - "What is machine learning?"
    -> category="general_qa", needs_calculation=false

    research_query:
    Questions requiring current information, latest news, current prices,
    stock prices, or external web research.

    Examples:
    - "What is the latest iPhone price?"
    -> category="research_query", needs_calculation=false
    - "Find the current iPhone price and calculate its price before 18% GST"
    -> category="research_query", needs_calculation=true

    coding_query:
    Writing, debugging, fixing, or explaining code.

    Example:
    - "Fix this Python ModuleNotFoundError"
    -> category="coding_query", needs_calculation=false

    calculator_query:
    Standalone arithmetic, percentages, GST, and mathematical calculations
    that do not require web research.

    Example:
    - "Calculate 1000 / 1.18"
    -> category="calculator_query", needs_calculation=false

    writer_query:
    Requests to create or transform a writing artifact, such as an email,
    letter, article, report, or rewriting existing text.

    Example:
    - "Write an email requesting leave"
    -> category="writer_query", needs_calculation=false

    CALCULATION RULE:
    Set needs_calculation=true only when external web research must happen
    before a calculation can be performed.
    For all other queries, set needs_calculation=false.

    Do not answer the user's question.
    Return only JSON matching the provided schema.
    """

    
    response = chat(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        format=RouteResponse.model_json_schema(),
    )

    result = RouteResponse.model_validate_json(
        response.message.content.strip()
    )

    logger.info(
        "Selected category: %s; needs_calculation: %s",
        result.category,
        result.needs_calculation,
    )

    return {
        "query_category": result.category,
        "needs_calculation": result.needs_calculation,
        "is_calculator_query": result.needs_calculation,
    }


# -------------------- ROUTING --------------------
def choose_route(state: State) -> str:
    routes = {
        "research_query": "researcher",
        "coding_query": "coder",
        "calculator_query": "calculator",
        "writer_query": "writer",
        "rag": "rag",
        "general_qa": "general_qa",
    }

    return routes.get(
        state.get("query_category", ""),
        "general_qa",
    )


def route_after_research(state: State) -> str:
    if state.get("needs_calculation", False):
        return "calculator"

    return "end"


# -------------------- CALCULATOR --------------------

def calculator_handler(state: State) -> dict:
    print("⚠️ Calculator: Calculating ")
    user_query = state["user_query"]
    research_result = state.get("llm_result", "")

    response = chat(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": """
                Extract a valid mathematical expression for the user's
                requested calculation.

                You will receive the user's question and, when available,
                the research results containing the values needed.

                Use the research results to identify the relevant numbers.
                Use the user's requested calculation to construct the
                expression.

                For example:
                If the price including 18% GST is 164900 and the user
                asks for the price before GST, return:
                {"expression": "164900 / 1.18"}

                For removing a percentage-based GST rate, divide the
                GST-inclusive price by (1 + rate / 100).

                Return only JSON with the field "expression".
                Use only numbers, parentheses, and +, -, *, /, **.
                Do not calculate the answer yourself.

                If a required value cannot be established, return an
                empty expression. Never invent missing numbers.
                """,
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{user_query}\n\n"
                    f"Research results, if available:\n{research_result}"
                ),
            },
        ],
        format=ExpressionResponse.model_json_schema(),
    )

    parsed = ExpressionResponse.model_validate_json(
        response.message.content.strip()
    )

    expression = parsed.expression.strip()

    if not expression:
        return {
            "llm_result": (
                f"{research_result}\n\n"
                "I couldn't identify enough information to perform "
                "the requested calculation."
                if research_result
                else "Please provide enough information to calculate."
            )
        }

    calculation_result = calcuator_query.invoke(
        {"experssion": expression}
    )

    if str(calculation_result).startswith("Calculation error:"):
        return {
            "llm_result": (
                f"{research_result}\n\n"
                f"Calculation failed: {calculation_result}"
                if research_result
                else str(calculation_result)
            )
        }

    if research_result:
        final_result = (
            f"{research_result}\n\n"
            f"Calculation:\n"
            f"Expression: {expression}\n"
            f"Result: {calculation_result}"
        )
    else:
        final_result = (
            f"Expression: {expression}\n"
            f"Result: {calculation_result}"
        )

    return {"llm_result": final_result}


# -------------------- RAG --------------------

def rag_handler(state: State) -> dict:
    logger.warning("RAG tool has not been connected yet.")

    return {
        "llm_result": (
            "RAG is not connected yet. Add your RAG tool "
            "to src/tools.py before using this route."
        )
    }


# -------------------- GRAPH --------------------

graph = StateGraph(State)

graph.add_node("general_qa", general_query)
graph.add_node("router", route_query)
graph.add_node("researcher", research_query)
graph.add_node("coder", coding_query)
graph.add_node("calculator", calculator_handler)
graph.add_node("writer", writing_query)
graph.add_node("rag", rag_node)
graph.add_node("response_formatter",response_formatter)

graph.add_edge(START, "router")

# Initial routing
graph.add_conditional_edges(
    "router",
    choose_route,
    {
        "researcher": "researcher",
        "coder": "coder",
        "calculator": "calculator",
        "writer": "writer",
        "rag": "rag",
        "general_qa": "general_qa",
    },
)

# After research, decide whether calculation is needed
graph.add_conditional_edges(
    "researcher",
    route_after_research,
    {
        "calculator": "calculator",
        "end": "response_formatter",
    },
)

# Format the final result
graph.add_edge("calculator", "response_formatter")
graph.add_edge("coder", "response_formatter")
graph.add_edge("writer", "response_formatter")
graph.add_edge("rag", "response_formatter")
graph.add_edge("general_qa", "response_formatter")
graph.add_edge("response_formatter", END)

app = graph.compile()


# -------------------- MAIN --------------------

def main():
    while True:
        user_query = input("\nEnter your query: ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        try:
            initial_state: State = {
                "user_query": user_query,
                "llm_result": "",
                "needs_calculation": False,
                "is_calculator_query": False,
            }

            result = app.invoke(initial_state)

            print(
                "\nSelected category:",
                result.get("query_category", "unknown"),
            )
            print(
                "\nAnswer:\n",
                result.get("llm_result", "No result returned."),
            )

        except Exception as e:
            logger.exception("Agent execution failed")
            print("Error:", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
