import os
os.environ["HUGGINGFACE_TOKEN"] = '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'

# #################################################################
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_auth_token=True,  # Use the Hugging Face token for gated models
    device=0  # Ensure you have a GPU available
)


prompts = [
    {"role": "user", "content": "Tell me a story about a brave knight."},
    {"role": "user", "content": "Explain quantum physics in simple terms."}
]

for prompt in prompts:
    result = text_generation_pipeline(prompt["content"])
    print(f"Prompt: {prompt['content']}")
    print(f"Response: {result[0]['generated_text']}")
