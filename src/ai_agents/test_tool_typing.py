from typing import assert_type

from ai_agents.tool import tool


@tool(description="A typed greeting tool")
def typed_greeting(name: str, excited: bool = False) -> str:
    suffix = "!" if excited else "."
    return f"Hello, {name}{suffix}"


assert_type(typed_greeting("Alice"), str)
