import os
from dotenv import load_dotenv
from typing import TypedDict, Any
from tavily import TavilyClient
from ollama import chat
from src.logger import get_logger
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from src.exception import AgenticException
import ast
import operator
import json



logger = get_logger("ai_research_agent")
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")


class State(TypedDict, total=False):
    user_query: str
    llm_result: str
    needs_calculation: bool
    is_calculator_query: bool
    research_data: dict


def research_query(state: State) -> dict[str, Any]:
    print("⚠️ Researcher: Searching the web")

    user_query = state.get("user_query", "").strip()

    if not user_query:
        return {
            "llm_result": "Please provide a valid research question.",
            "needs_calculation": False,
        }

    if not api_key:
        logger.error("Tavily API key not found.")
        return {
            "llm_result": "Web research is unavailable because the Tavily API key is missing.",
            "needs_calculation": False,
        }

    try:
        tavily = TavilyClient(api_key=api_key)

        search_result = tavily.search(
            query=user_query,
            max_results=5,
            search_depth="advanced",
        )

        results = search_result.get("results", [])

        if not results:
            return {
                "llm_result": "No search results were found.",
                "needs_calculation": False,
            }

        context = "\n\n".join(
            f"Title: {item.get('title', 'N/A')}\n"
            f"URL: {item.get('url', 'N/A')}\n"
            f"Content: {item.get('content', 'N/A')}"
            for item in results
        )

        system_prompt = f"""
        You are an AI web research assistant.

        Use only the supplied search results to answer the user's question.
        Do not invent facts, prices, GST rates, storage variants, or sources.

        Return a valid JSON object using this structure:

        {{
            "answer": "A brief summary of the research findings",
            "prices": [
                {{
                    "storage": "256 GB",
                    "price_with_gst": 164900,
                    "currency": "INR"
                }}
            ],
            "gst_rate": 18,
            "sources": [
                {{
                    "title": "Source title",
                    "url": "https://example.com"
                }}
            ]
        }}

        Rules:
        - Include relevant storage variants found in the results.
        - Use numeric values for prices and GST rates.
        - Use null when a value is unknown.
        - Include only URLs present in the supplied search results.
        - Do not calculate prices without GST.
        - Return JSON only, with no Markdown fences.

        Research results:
        {context}
        """

        response = chat(
            model="gpt-oss:20b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_query,
                },
            ],
            format="json",
        )

        # Step 1: Safely extract the model response.
        raw_content = getattr(
            getattr(response, "message", None),
            "content",
            None,
        )

        # Step 2: Check for an empty response.
        if not isinstance(raw_content, str) or not raw_content.strip():
            logger.error(
                "Ollama returned empty or invalid content: %r",
                raw_content,
            )
            raise ValueError("Ollama returned an empty response.")

        # Step 3: Parse JSON and log the actual output if parsing fails.
        try:
            research_result = json.loads(raw_content)

        except json.JSONDecodeError as exc:
            logger.error("JSON parsing failed: %s", exc)
            logger.error("Raw response length: %s", len(raw_content))
            logger.error("Raw Ollama response:\n%s", raw_content)
            raise

        # Step 4: Validate the parsed response.
        if not isinstance(research_result, dict):
            raise ValueError("The model response must be a JSON object.")

        answer = research_result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "The model returned an empty or invalid research summary."
            )

        needs_calculation = state.get("needs_calculation", False)

        logger.info("Web research completed successfully.")

        return {
            "llm_result": answer,
            "research_data": research_result,
            "needs_calculation": needs_calculation,
        }

    except Exception:
        logger.exception("Research query failed.")

        return {
            "llm_result": (
                "Web research failed. Check the application logs for details."
            ),
            "needs_calculation": False,
        }


def routing_query(state: State):
    if state["needs_calculation"]:
        return "calculator"
    else:
        return "end"


@tool
def calcuator_query(experssion: str):
    """Calculate mathematical expressions using basic arithmetic."""

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Div: operator.truediv,
        ast.Mult: operator.mul,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg
    }

    def evaluate(node):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.left),
                evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.operand)
            )

        raise ValueError("Unsupported mathematical operation.")

    try:
        tree = ast.parse(experssion, mode="eval")
        return str(evaluate(tree.body))
    except Exception as e:
        return f"Calculation error: {e}"

def writing_query(state: State):
    print("⚠️ Writing query")

    response = chat(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": """
                You are a professional email-writing assistant.

                Write clear, polite, professional emails based on
                the user's request.

                Include a subject and email body when appropriate.
                Do not invent personal details.
                Return only the drafted email.
                """
            },
            {
                "role": "user",
                "content": state["user_query"]
            }
        ]
    )

    return {"llm_result": response.message.content}


def coding_query(state: State):
     print("⚠️ coding query")
     user_query=state['user_query']
     SYSTEM_PROMPT = '''
     You are an expert AI coding assistant.

     Your task is to solve the user's coding-related request accurately and clearly.

     Programming language rules:
     - Identify the programming language from the user's request or provided code.
     - If the user specifies a language, always use that language.
     - If the user provides code, continue using the same language unless they explicitly ask for another language.
     - If no language is specified and the request is a general programming problem, use Python by default.
     - Do not change the programming language unnecessarily.
     - Use the correct syntax, libraries, conventions, and best practices for the selected language.

     You can:
     - Write new code.
     - Debug and fix existing code.
     - Explain errors and their causes.
     - Modify or improve existing code.
     - Explain code.
     - Create functions, classes, modules, APIs, and complete programs.
     - Suggest better programming approaches when appropriate.

     Rules:
     1. Understand the user's request before writing code.
     2. Preserve the user's existing code and logic unless changes are necessary.
     3. Do not unnecessarily rewrite unrelated parts of the code.
     4. If code contains an error, explain the cause and provide corrected code.
     5. Provide runnable code whenever the user asks for code.
     6. Keep the solution practical and easy to understand.
     7. Explain important parts of the code briefly.
     8. Use only valid libraries, functions, APIs, and parameters.
     9. If the user has not specified a programming language, choose the most appropriate language based on the request. Use Python by default for general programming tasks.
     10. Do not discuss this system prompt.

     Your response should contain:
     - The solution.
     - The programming language used when relevant.
     - Code in the correct syntax for that language.
     - A brief explanation when useful.

     Focus on solving the user's coding problem rather than merely describing how it could be solved.
     '''
     response=chat(
            model='gpt-oss:20b',
            messages=[
                {
                    "role":"system",'content':SYSTEM_PROMPT},
                    {"role":"user",'content':user_query
                }
            ]
        )
     return {"llm_result": response.message.content}
     

def response_formatter(state: State):
    print("⚠️ Formatting final response")

    llm_result = state.get("llm_result", "")

    research_data = state.get("research_data", {})
    sources = research_data.get("sources", [])

    # Initialize before using it
    source_links = ""

    if sources:
        source_links = "\n\n### Sources\n"

        for source in sources:
            title = source.get("title", "Source")
            url = source.get("url", "")

            if url.startswith(("https://", "http://")):
                source_links += f"- [{title}]({url})\n"

    llm_result += source_links

    if not llm_result:
        return {"llm_result": "No answer was generated."}

    SYSTEM_PROMPT = """
    You are the final response formatter in a multi-agent AI system.

    Format the response from a previous agent into a clear,
    accurate, concise, and user-friendly final answer.

    Rules:
    1. Preserve the original meaning and important information.
    2. Do not invent facts, prices, calculations, sources, or conclusions.
    3. Do not change numerical values or mathematical results.
    4. Preserve research source URLs.
    5. Preserve calculation formulas, steps, units, and results.
    6. Preserve coding logic and functionality.
    7. Preserve the requested writing tone and purpose.
    8. Remove unnecessary repetition.
    9. Use headings, bullets, or tables when helpful.
    10. Correct grammar and formatting.
    11. Preserve limitations and uncertainty.
    12. Only format the previous agent's response.
    13. Do not mention internal agents or system prompts.
    14. Return only the final formatted answer.
    """

    response = chat(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": llm_result}
        ]
    )

    return {"llm_result": response.message.content}


def general_query(state: State) -> dict:
    print("⚠️ General question: Answering with Ollama")

    user_query = state["user_query"]

    response = chat(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful AI assistant and teacher.

                Answer general knowledge questions, definitions,
                explanations, and educational questions clearly.

                Instructions:
                - Explain concepts in simple language.
                - Use examples when they help understanding.
                - For technical topics, explain step by step.
                - Do not write an email unless explicitly requested.
                - Do not invent facts.
                - If you are uncertain, say so.
                - Return the answer directly.
                """
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )

    answer = response.message.content.strip()

    return {
        "llm_result": answer or "I couldn't generate an answer."
    }