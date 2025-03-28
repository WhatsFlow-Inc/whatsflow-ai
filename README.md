# WhatsApp Flow Designer API

A FastAPI-based service that helps businesses design WhatsApp conversation flows through an interactive AI assistant. The service converts natural language conversations into structured WhatsApp flows and React Flow diagrams.

## Features

- Interactive AI assistant to help design WhatsApp flows
- Conversation memory management with thread-based sessions
- Automated flow planning and generation
- Dual output support: WhatsApp JSON format and React Flow JSON format
- RESTful API endpoints for chat, planning, and flow generation

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install fastapi uvicorn pydantic python-dotenv openai
```
3. Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

## API Endpoints

### POST /chat
Start or continue a conversation with the AI assistant.
```json
{
    "thread_id": "unique_session_id",
    "user_input": "I want to create a flow for pizza ordering"
}
```

### POST /plan
Generate a structured flow plan based on the conversation.
```json
{
    "thread_id": "unique_session_id"
}
```

### GET /plan
Retrieve the previously generated plan for a specific thread.
- Query parameter: `thread_id`

### POST /get_flows
Generate both WhatsApp and React Flow JSON representations.
- Query parameter: `thread_id`

## Usage Example

1. Start a conversation:
```bash
curl -X POST "http://localhost:5000/chat" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "123", "user_input": "I need a WhatsApp flow for taking pizza orders"}'
```

2. Generate a plan:
```bash
curl -X POST "http://localhost:5000/plan" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "123"}'
```

3. Get the flow JSONs:
```bash
curl -X POST "http://localhost:5000/get_flows?thread_id=123"
```

## Running the Server

Start the server using:
```bash
python userFacing.py
```
The API will be available at `http://localhost:5000`

## Architecture

- `UserSession`: Manages conversation history and plan storage per thread
- `FlowComposerAgent`: Generates WhatsApp-compatible JSON flows
- `ReactFlowComposerAgent`: Generates React Flow diagram JSON
- FastAPI endpoints handle request routing and response formatting

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Dependencies

- FastAPI
- Pydantic
- OpenAI
- python-dotenv
- uvicorn

