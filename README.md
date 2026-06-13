# AI Agents

`ai_agents` is a small Python library for exposing typed Python callables as LLM tools across OpenAI, Anthropic, and Google Gemini.

It helps you define a tool once, generate provider-ready schemas from type hints, validate model-supplied arguments, and execute returned tool calls through a consistent interface.

## Install

```bash
pip install ai-agents
```

## Quick Example

```python
from typing import Annotated

from pydantic import Field

from ai_agents.tool import tool
from ai_agents.tool_collection_openai import ToolCollectionOpenAI


@tool()
def say_hello(name: Annotated[str, Field(description="Name of the person to greet")]) -> str:
    """
    Say hello to someone.
    """
    return f"Hello, {name}!"


collection = ToolCollectionOpenAI(say_hello)
tools = collection.tools()
```

Pass `tools` to your provider request. When the provider returns a tool call, execute it with:

```python
result = collection.invoke_fn(tool_call)
print(result["say_hello"].result)
```

## Documentation

The full documentation is maintained in the project wiki:

https://github.com/jasonrig/ai_agents/wiki

Use the wiki for current guidance on:

- creating and decorating tools
- OpenAI, Anthropic, and Gemini provider collections
- tool invocation
- sync and async tools
- hooks for arguments, results, and tool disclosure

## Project Status

This project is public and usable, but it is primarily maintained to support my own projects. Contributions and issue reports are welcome and reviewed on a best-effort basis.
