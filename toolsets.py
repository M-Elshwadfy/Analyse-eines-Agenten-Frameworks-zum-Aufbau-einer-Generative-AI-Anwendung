from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

def agent_tool():
    return "I'm registered directly on the agent"
def extra_tool():
    return "I'm passed as an extra tool for a specific run"
def override_tool():
    return "I override all other tools"

agent_toolset = FunctionToolset(tools=[agent_tool]) 
extra_toolset = FunctionToolset(tools=[extra_tool])
override_toolset = FunctionToolset(tools=[override_tool])

test_model = TestModel() 
agent = Agent(test_model, toolsets=[agent_toolset])

result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool']

result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool', 'extra_tool']

with agent.override(toolsets=[override_toolset]):
    result = agent.run_sync('What tools are available?', toolsets=[extra_toolset]) 
    print([t.name for t in test_model.last_model_request_parameters.function_tools])
    #> ['override_tool']