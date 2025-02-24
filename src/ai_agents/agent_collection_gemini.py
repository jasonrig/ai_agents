from typing import List

from google.genai.types import Tool, FunctionDeclaration, FunctionCall

from ai_agents.agent import AgentCollection, input_schema, FunctionPayload


class AgentCollectionGemini(AgentCollection[Tool, FunctionCall]):
    def tools(self) -> List[Tool]:
        tools = list()
        for metadata, agent in self._agents.values():
            schema = input_schema(agent)
            del schema["description"]
            tools.append(Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=metadata.name,
                        description=metadata.description,
                        parameters=schema
                    )
                ]
            ))
        return tools

    def extract_parameters(self, inp: FunctionCall) -> FunctionPayload:
        return FunctionPayload(inp.name, inp.args)
