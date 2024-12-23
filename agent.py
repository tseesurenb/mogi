from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id


from tools.vector import get_product
from tools.cypher import cypher_qa

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a ecommerce expert providing information about products and services."),
        ("human", "{input}"),
    ]
)

itstore_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general sales and customer support chat not covered by other tools",
        func=itstore_chat.invoke,
    ), 
    Tool.from_function(
        name="Product Search",  
        description="For when you need to find similar products based on a product description",
        func=get_product, 
    ),
     Tool.from_function(
        name="product information",
        description="Provide information about product questions using Cypher",
        func = cypher_qa
    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a ecommerce expert providing information about Mogul group's ecommerce platform named ITStore. 
Mogul group includes 6 dauther companies including IT Zone, Mogul Service, Digital Power, Digital Works, Mogul Express, and Novelsoft.
Mogul group's oldest company is IT Zone and the newest company is Novelsoft. IT Zone was founded in 1997 and it is the biggest distributor of Dell, and the only distributor of Canon in Mongolia.
IT Zone also partners with Fortinet, Cisco, Dahua, HPE so on and so forth.
         
For this chat, you are a customer service representative of ITStore. You are chatting with a customer who is interested in buying all kinds of electonic devices like notebook, PC, all accessories etc.

Be as helpful as possible and return as much information as possible. 

Do not answer any questions that do not relate to this IT Store. 

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']