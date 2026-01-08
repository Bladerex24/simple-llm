from llm import LLM
import json

# Initialize the model with specified parameters
model = LLM("./gpt-oss-120b", max_num_seqs=10, max_seq_len=1024)

# List of prompts to be sent to the model for batch generation
prompts = [
    "Calculate 5 plus 3",
    "What color is the sky?",
    "Who painted the Sistine Chapel ceiling?",
    "What is the capital of France?",
    "How many days are in a year?",
    "Which country is known as the Land of the Rising Sun?",
    "What is the meaning of life?",
    "Who built the pyramids?",
    "Give me a joke",
    "Write a haiku about the ocean",
]

# Generate responses for the prompts; returns a Future object
# This is a non-blocking operation; the model will start generating responses in the background
request = model.generate(prompts, max_tokens=1000, ignore_eos=False)

# Wait for the results
results = request.result()
output_list = []
for result in results:
    output_list.append({
        "reasoning": result.reasoning,
        "answer": result.text,
        "raw_text": result.raw_text
    })

print(json.dumps(output_list, indent=2, ensure_ascii=False))

# Stop/cleanup the model to free resources
model.stop()
