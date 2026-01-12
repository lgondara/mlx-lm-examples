from mlx_lm import load, generate

# Load base model + adapters
model, tokenizer = load(
    "mlx-community/Qwen3-4B-4bit",
    adapter_path="adapters/qwen3-4b-finance-20260111_164747"
)

# Generate
prompt = "What are the tax implications of a self-employed business owner hiring an employee?"
conversation = [{"role": "user", "content": prompt}]

# Transform the prompt into the chat template
prompt = tokenizer.apply_chat_template(
    conversation=conversation,
    add_generation_prompt=True,
)

# Specify the maximum number of tokens
max_tokens = 1000

# Specify if tokens and timing information will be printed
verbose = True

# Generate a response with the specified settings
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=max_tokens,
    verbose=verbose,
)