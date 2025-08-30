from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.llms.base import LLM
from euri_llm import generate_completion


# Tool 1 it's going to generate some sort of poem 
@tool
def expert_writer(input):
    """Writes a meaningful poem on the given topic."""
    message=[{"role":"user",
              "content": f"write a meaningful poem on {input}"}]
    return generate_completion(message)

#tool 2 its going to be an expert mathematician
@tool
def expert_mathematician(input):
    
    """Solves mathematical expressions."""
    result=eval(input,{"__builtins__":{}}, {})
    return result 

tools=[
    expert_writer,expert_mathematician
]

class EuriaiLLM(LLM):
    def _call(self, prompt, stop=None):
        result = generate_completion(prompt)
        if result is None:
            raise ValueError("euri_completion returned None. Check API status or input.")
        return result
    
    @property
    def _llm_type(self):
        return "euri-llm"
    
prompt_template = PromptTemplate.from_template(
    """You are a helpful assistant with access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
)


agent = create_react_agent(
    llm=EuriaiLLM(),
    tools=tools,
    prompt=prompt_template,
    output_parser=ReActSingleInputOutputParser()
)

# agent created its time to execute the agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

response=agent_executor.invoke({"input":"write a meaningful poem on the beauty of Space Black hole and solve 50*866"})  # Example input
print(response['output'])  # Print the output of the agent's response