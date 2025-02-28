# AI Agents

A simple Python decorator-based framework for creating function-calling agents compatible with major LLM providers (OpenAI, Anthropic, and Google Gemini).

## Project Status

This project is made public to benefit the community, but please note that it's primarily maintained to support my own projects. While contributions and issue reports are welcome, they will be reviewed on a best-effort basis.

## Documentation Guide

Agents can be documented in several ways:

### Using Docstrings

The docstring of the function becomes the agent's description:

```python
@agent()
def calculate_price(quantity: int, price: float):
    """
    Calculate the total price including tax for a given quantity of items.
    This description will be provided to the LLM to help it understand when to use this function.
    """
    return quantity * price * 1.1  # 10% tax
```

### Using Field Annotations

Parameters can be documented using Pydantic's Field annotations:

```python
@agent()
def format_currency(
    amount: Annotated[float, Field(
        description="The amount to format"
    )],
    currency: Annotated[str, Field(
        description="The three-letter currency code"
    )]
):
    """
    Format a monetary amount with its currency symbol
    """
    return f"${amount:.2f}" if currency == "USD" else f"€{amount:.2f}"
```

### Using the Decorator

You can also provide a name and description in the decorator itself:

```python
@agent(
    name="convert_temperature",  # Override the function name
    description="Convert temperatures between Celsius and Fahrenheit"
)
def temp_convert(
    temp: Annotated[float, Field(description="Temperature value to convert")],
    from_unit: Annotated[str, Field(description="Current temperature unit (C or F)")],
    to_unit: Annotated[str, Field(description="Target temperature unit (C or F)")]
):
    if from_unit == to_unit:
        return temp
    if from_unit == "C" and to_unit == "F":
        return (temp * 9/5) + 32
    if from_unit == "F" and to_unit == "C":
        return (temp - 32) * 5/9
```

## Basic Examples

- Single decorator interface for creating function-calling agents
- Support for synchronous and asynchronous functions
- Automatic JSON schema generation from Python type hints
- Compatible with OpenAI, Anthropic, and Google Gemini APIs
- Support for both function and class-based agents
- Pydantic integration for input validation

## Usage

### Basic Examples

#### Single Agent
```python
from typing import Annotated
from pydantic import Field
from ai_agents.agent import agent

@agent()
def say_hello(name: Annotated[str, Field(description="Name of the person to greet")]):
    """
    Use this function to say hello to someone
    """
    return f"Hello, {name}!"
```

#### Multiple Agents
```python
from typing import Annotated
from pydantic import Field
from ai_agents.agent import agent

@agent()
def calculate_area(length: Annotated[float, Field(description="Length of the rectangle")],
                  width: Annotated[float, Field(description="Width of the rectangle")]):
    """
    Calculate the area of a rectangle
    """
    return length * width

@agent()
def format_result(number: Annotated[float, Field(description="Number to format")],
                 unit: Annotated[str, Field(description="Unit of measurement")]):
    """
    Format a number with its unit of measurement
    """
    return f"{number:.2f} {unit}"

# Create collection with both agents
collection = AgentCollectionOpenAI(calculate_area, format_result)
```

### Function Return Values

When invoking functions through the agent collection, the results are wrapped in a `FunctionOutputPayload` class:

```python
@dataclasses.dataclass
class FunctionOutputPayload:
    result: FunctionOutput  # The actual return value from the function
    extras: Optional[Any] = None  # Additional metadata like tool_call_id
```

This structure serves two important purposes:
1. It provides access to the actual return value of your function via the `result` field
2. It exposes LLM-specific metadata through the `extras` field

The `extras` field is particularly useful as it contains provider-specific information such as:
- For OpenAI: The `tool_call_id` which is needed when responding to function calls
- For Anthropic: The `tool_call_id` from Claude's tool use blocks
- For other providers: Any additional metadata that might be useful for tracking or debugging

This design allows you to maintain provider-specific context while working with a consistent interface across different LLM platforms.

### Using with OpenAI

```python
from openai import OpenAI
from ai_agents.agent_collection_openai import AgentCollectionOpenAI

# Create an agent collection
collection = AgentCollectionOpenAI(say_hello)

# Get tools for OpenAI
tools = collection.tools()

# Use with OpenAI client
client = OpenAI()
message = client.chat.completions.create(
    model="gpt-4",
    tools=tools,
    messages=[{"role": "user", "content": "Say hello to Alice"}]
)

# Execute the function call
result = collection.invoke_fn(message.choices[0].message.tool_calls[0])
# Returns: {"say_hello": FunctionOutputPayload(result="Hello, Alice!", extras={"tool_call_id": "..."})}

# Access the actual function output
greeting = result["say_hello"].result  # "Hello, Alice!"

# Access the tool_call_id for use in assistant messages
tool_call_id = result["say_hello"].extras["tool_call_id"]
# Use in response: {"tool_call_id": tool_call_id, "role": "tool", "content": greeting}
```

### Using with Anthropic

```python
import anthropic
from ai_agents.agent_collection_anthropic import AgentCollectionAnthropic

# Create an agent collection
collection = AgentCollectionAnthropic(say_hello)

# Get tools for Anthropic
tools = collection.tools()

# Use with Anthropic client
client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-haiku-20240307",
    tools=tools,
    messages=[{"role": "user", "content": "Say hello to Alice"}]
)

# Get the tool use block from the response
tool_use_block = next(filter(lambda x: isinstance(x, ToolUseBlock), message.content))

# Execute the function call
result = collection.invoke_fn(tool_use_block)
# Returns: {"say_hello": FunctionOutputPayload(result="Hello, Alice!", extras={"tool_call_id": "..."})}

# Access the actual function output
greeting = result["say_hello"].result  # "Hello, Alice!"
```

### Using with Google Gemini

```python
from google import genai
from google.genai import types
from ai_agents.agent_collection_gemini import AgentCollectionGemini

# Create an agent collection
collection = AgentCollectionGemini(say_hello)

# Get tools and config for Gemini
tools = collection.tools()
model_config = types.GenerateContentConfig(tools=tools)

# Use with Gemini client
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello to Alice",
    config=model_config,
)

# Execute the function call
result = collection.invoke_fn(response.function_calls[0])
# Returns: {"say_hello": FunctionOutputPayload(result="Hello, Alice!", extras=None)}

# Access the actual function output
greeting = result["say_hello"].result  # "Hello, Alice!"
```

### Multiple Function Calls

You can invoke multiple functions in a single call:

```python
results = collection.invoke_fn(
    function_call_1,
    function_call_2
)

# Access individual results
result1 = results["function_name_1"].result
result2 = results["function_name_2"].result
```

### Advanced Features

#### Async Functions

```python
@agent()
async def async_hello(name: str):
    """
    Asynchronous greeting function
    """
    await asyncio.sleep(1)
    return f"Hello, {name}!"
```

#### Class-based Agents

```python
@agent()
class Greeter:
    def __call__(self, name: str):
        return f"Hello, {name}!"
```

#### Complex Input Types

```python
from pydantic import BaseModel

class Name(BaseModel):
    first_name: str
    last_name: str

@agent()
def greet_full_name(name: Annotated[Name, Field(description="Full name of the person")]):
    """
    Greet someone using their full name
    """
    return f"Hello, {name.first_name} {name.last_name}!"
```

## License

MIT License
## Contributing

While this project is primarily maintained for personal use, contributions are welcome via pull requests. Please note that review times may vary depending on availability.