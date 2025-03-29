import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class FlowTransformerAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"

    def _get_conversion_prompt(self, example_react_flow, example_whatsapp_flow, new_react_flow):
        return f"""You are a Flow Conversion Specialist that transforms React Flow JSON into WhatsApp Flow JSON.

EXAMPLE REACT FLOW:
{example_react_flow}

CORRESPONDING WHATSAPP FLOW:
{example_whatsapp_flow}

NEW REACT FLOW TO CONVERT:
{new_react_flow}

Guidelines:
- Use the example conversion as a reference for mapping patterns
- Preserve the flow logic and connections
- Follow the same structure as shown in the example
- Ensure all required WhatsApp Flow properties are included
- Map React Flow node types to appropriate WhatsApp components
- Only return the WhatsApp Flow JSON without any additional text or explanations

WHATSAPP FLOW JSON:
"""

    def transform_flow(self, example_react_flow, example_whatsapp_flow, new_react_flow):
        formatted_prompt = self._get_conversion_prompt(
            example_react_flow,
            example_whatsapp_flow,
            new_react_flow
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Flow Conversion Specialist that transforms React Flow JSON into WhatsApp Flow JSON."},
                {"role": "user", "content": formatted_prompt}
            ],
            max_tokens=4096,
            temperature=0
        )
        return response.choices[0].message.content.strip()

# Main execution
if __name__ == "__main__":
    agent = FlowTransformerAgent()
    # Example usage:
    # example_react = {...}  # Your example React Flow JSON
    # example_whatsapp = {...}  # Corresponding WhatsApp Flow JSON
    # new_react_flow = {...}  # New React Flow JSON to convert
    # whatsapp_flow = agent.transform_flow(example_react, example_whatsapp, new_react_flow)