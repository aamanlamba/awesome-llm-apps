from anthropic import Anthropic
import os
from dotenv import load_dotenv
import logging
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize the client with your API key
anthropic = Anthropic(
    api_key= os.environ.get("ANTHROPIC_API_KEY")
)
logging.info(f"{anthropic.api_key}")
# Create a message
message = anthropic.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

# Get the response
print(message.content)
logging.info(message.content)