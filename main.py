
from src.exception import AgenticException
from app import app
from src.logger import get_logger

logger = get_logger("ai_research_agent")


def main():
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()

            if user_query.lower() in {"exit", "quit"}:
                print("Exiting...")
                break

            if not user_query:
                print("Please enter a valid question.")
                continue

            state = {
                "user_query": user_query,
                "llm_result": "",
                "needs_calculation": False,
                "is_calculator_query": False,
            }

            result = app.invoke(state)

            print("\nSelected category:", result.get("query_category", "unknown"))
            print("\nAnswer:", result.get("llm_result", "No result returned."))

        except AgenticException as e:
            logger.error("Agent error: %s", str(e))
            print("The agent encountered an error. Check the log file.")

        except Exception:
            logger.exception("Unexpected error in the application")
            print("An unexpected error occurred. Check the log file.")


if __name__ == "__main__":
    main()
