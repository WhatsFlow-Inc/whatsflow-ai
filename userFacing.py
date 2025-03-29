from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
from wapFlowComposer import FlowComposerAgent
from reactFlowComposer import ReactFlowComposerAgent
from editWapFlow import FlowTransformerAgent
import json
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# System prompts
USER_FACING_SYSTEM_PROMPT = """
You are a senior WhatsApp Flow Architect and a conversational assistant.
Your role is to interactively help users design robust WhatsApp Flow JSONs.
For every user input, you should:
- Ask clarifying questions if necessary.
- Explain your thought process and design decisions.
- Propose a draft of the plan of the WhatsApp Flow
- Ask the user if they would like any changes or additional details.
Maintain a friendly and conversational tone, and remember all previous exchanges. This is an agent facing a SMB owner, I don't want anything included that might scare them or be very new for them, like JSONs, or anything out of context or very technical.
"""

PLANNER_SYSTEM_PROMPT = """
You are an expert conversational flow planner for WhatsApp automation.

Your job is to read the conversation between a user and an assistant and from that conversation:
1. Identify the user's intent.
2. Extract all necessary information (slots) required to design a WhatsApp form (Flow).
3. Structure the form into a clear sequence of steps.
4. Include conditional logic where applicable (e.g., "If the user selects X, then show Y").
5. Present the flow as a well-formatted block of text, step-by-step, suitable to pass to a developer for implementation.

Always assume missing details where appropriate, but mention these assumptions clearly.
"""

wap_flow_agent = FlowComposerAgent()
react_flow_agent = ReactFlowComposerAgent()

# Session memory per thread_id
session_store = {}

class UserSession:
    def __init__(self):
        self.system_prompt = USER_FACING_SYSTEM_PROMPT
        self.messages = []  # Remove system message from messages array
        self.plan = None  # Initialize plan attribute
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self):
        return self.messages

def agent_respond(user_input, session: UserSession):
    session.add_message("user", user_input)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": USER_FACING_SYSTEM_PROMPT}
        ] + session.get_messages(),  # Ensure the system prompt is included
        max_tokens=1024,
        temperature=0.7,
    )
    
    agent_reply = response.choices[0].message.content
    session.add_message("assistant", agent_reply)
    
    return agent_reply

def planner_generate_flow(session: UserSession):
    conversation_history = format_conversation(session.get_messages())
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the conversation history:\n{conversation_history}\nPlease generate the flow as per the instructions."}
        ],
        max_tokens=1024,
        temperature=0.5,
    )
    plan = response.choices[0].message.content
    session.plan = plan
    return plan

def format_conversation(messages):
    formatted = ""
    for msg in messages:
        if msg["role"] != "system":  # skip system prompts in conversation history
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return formatted

# Define request/response models
class ChatRequest(BaseModel):
    thread_id: str
    user_input: str

class PlanRequest(BaseModel):
    thread_id: str

class ChatResponse(BaseModel):
    reply: str

class PlanResponse(BaseModel):
    flow_plan: str

class FlowsResponse(BaseModel):
    wap_json: Dict
    react_json: Dict

class FlowRequest(BaseModel):
    thread_id: str

class EditFlowRequest(BaseModel):
    react_flow_json: Dict
    example_react_flow: Dict
    example_whatsapp_flow: Dict

class EditFlowResponse(BaseModel):
    wap_json: Dict

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.thread_id or not request.user_input:
        raise HTTPException(status_code=400, detail="Missing 'thread_id' or 'user_input'")

    # Get or create a session for this thread_id
    session = session_store.get(request.thread_id)
    if not session:
        session = UserSession()
        session_store[request.thread_id] = session
    
    # Get agent's response
    reply = agent_respond(request.user_input, session)
    
    return ChatResponse(reply=reply)

@app.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    if not request.thread_id:
        raise HTTPException(status_code=400, detail="Missing 'thread_id'")

    session = session_store.get(request.thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="No conversation found for this thread_id")

    # Generate planning output and store it in session
    flow_plan = planner_generate_flow(session)
    
    return PlanResponse(flow_plan=flow_plan)

@app.get("/plan", response_model=PlanResponse)
async def get_plan(thread_id: str):
    if not thread_id:
        raise HTTPException(status_code=400, detail="Missing 'thread_id'")

    session = session_store.get(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="No conversation found for this thread_id")
        
    if not session.plan:
        raise HTTPException(status_code=404, detail="No plan generated yet for this thread_id")

    return PlanResponse(flow_plan=session.plan)

@app.post("/get_flows", response_model=FlowsResponse)
async def get_flows(request: FlowRequest):
    session = session_store.get(request.thread_id)
    if not session:
        raise HTTPException(status_code=404, detail="No conversation found for this thread_id")

    wap_json_output = wap_flow_agent.compose_flow(session.plan)
    react_flow_output = react_flow_agent.compose_flow_json(session.plan)
    
    # Parse the JSON string from wap_json_output (removing markdown formatting if present)
    if isinstance(wap_json_output, str):
        # Remove markdown code block formatting if present
        wap_json_output = wap_json_output.replace('```json\n', '').replace('\n```', '')
        # Parse the JSON string into a dictionary
        wap_json_output = json.loads(wap_json_output)

    return FlowsResponse(wap_json=wap_json_output, react_json=react_flow_output)

@app.post("/edit-wap-flow", response_model=EditFlowResponse)
async def edit_wap_flow(request: EditFlowRequest):
    # Initialize the FlowTransformerAgent
    transformer = FlowTransformerAgent()
    
    # Transform React Flow JSON to WhatsApp Flow JSON using examples
    wap_json_output = transformer.transform_flow(
        request.example_react_flow,
        request.example_whatsapp_flow,
        request.react_flow_json
    )
    
    # Parse the JSON string (removing markdown formatting if present)
    if isinstance(wap_json_output, str):
        # Remove markdown code block formatting if present
        wap_json_output = wap_json_output.replace('```json\n', '').replace('\n```', '')
        wap_json_output = json.loads(wap_json_output)
    
    # If wap_json_output is a list, wrap it in a dictionary
    if isinstance(wap_json_output, list):
        wap_json_output = {"flow": wap_json_output}
    
    return EditFlowResponse(wap_json=wap_json_output)

if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok

    # Open a ngrok tunnel to the HTTP server
    public_url = ngrok.connect(5000).public_url
    print(f"Public URL: {public_url}")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=5000)
