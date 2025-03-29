import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class ReactFlowComposerAgent:
    def __init__(self, model="claude-3-sonnet-20240229", temperature=0.5):
        """
        Initialize the agent with the specified model and temperature.
        """
        self.model = model
        self.temperature = temperature
        self.client = anthropic

    def compose_flow_json(self, plan_text: str) -> dict:
        """
        Generate a React Flow JSON from the provided planning text.
        
        Args:
            plan_text (str): The planning output as plain text.
        
        Returns:
            dict: A dictionary representing the React Flow JSON.
        """
        prompt = f"""
Below is a list of custom React Flow components available for building the flow:

-------------------------------
Components Reference:

1. Greeting Node:
   - Type: "greeting"
   - Example:
   {{
       "id": "<unique_id>",
       "type": "greeting",
       "position": {{"x": 100, "y": 50}},
       "data": {{
           "label": "Welcome to our WhatsApp Flow!",
           "style": {{"background": "#e6f7ff", "border": "1px solid #91d5ff"}}
       }}
   }}

2. Choice Node:
   - Type: "choice"
   - Example:
   {{
       "id": "<unique_id>",
       "type": "choice",
       "position": {{"x": 100, "y": 150}},
       "data": {{
           "label": "Please choose an option:",
           "options": ["Track an order", "Browse products", "Make an inquiry"],
           "style": {{"background": "#fffbe6", "border": "1px solid #ffe58f"}}
       }}
   }}

3. Input Node:
   - Type: "input"
   - Example:
   {{
       "id": "<unique_id>",
       "type": "input",
       "position": {{"x": 100, "y": 250}},
       "data": {{
           "label": "Please enter your order number:",
           "placeholder": "Order Number",
           "style": {{"background": "#f6ffed", "border": "1px solid #b7eb8f"}}
       }}
   }}

4. Link Node:
   - Type: "link"
   - Example:
   {{
       "id": "<unique_id>",
       "type": "link",
       "position": {{"x": 100, "y": 350}},
       "data": {{
           "label": "Visit our website for more products",
           "url": "https://example.com",
           "style": {{"background": "#fff0f6", "border": "1px solid #ffadd2"}}
       }}
   }}

5. Inquiry Node:
   - Type: "inquiry"
   - Example:
   {{
       "id": "<unique_id>",
       "type": "inquiry",
       "position": {{"x": 100, "y": 450}},
       "data": {{
           "label": "Please describe your inquiry:",
           "placeholder": "Type your inquiry here",
           "style": {{"background": "#e6fffb", "border": "1px solid #87e8de"}}
       }}
   }}
-------------------------------

Using the components above, generate a React Flow JSON that includes two arrays: "nodes" and "edges". 
The JSON should represent the following planning details:

{plan_text}

Requirements:
- Each node must have a unique "id", a "type" chosen from the available types (greeting, choice, input, link, inquiry), a "position" with x and y coordinates (spread evenly for clarity), and "data" including at least a "label".
- Each edge must have a unique "id", a "source" (the id of the starting node), a "target" (the id of the ending node), and a "label" describing the transition (e.g., conditions).
- The flow should be well-structured, visually appealing, and informative.
- Output with just the valid JSON.

Please just output valid JSON and no supporting text. ONLY JSON.
"""
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            system="You are a React Flow JSON generator.",
            max_tokens=4096,
            temperature=self.temperature,
        )

        flow_json = response.content[0].text.strip()
        flow_json = json.loads(flow_json)

        return flow_json

# Example usage:
if __name__ == "__main__":   
    agent = ReactFlowComposerAgent()
