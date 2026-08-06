"""Adaptadores de LLM para simulate.py. El backend se elige con ECON_BACKEND
(openai | ollama | bedrock) y cada uno expone la misma firma de get_completion.
"""
import multiprocessing
import os
from functools import partial
from time import sleep

from dotenv import load_dotenv

load_dotenv()

ECON_BACKEND = os.getenv("ECON_BACKEND", "openai").lower()
MAX_RETRIES = 20
RETRY_DELAY_SECONDS = 6

# Precio por 1k tokens (prompt, completion). Modelos ausentes -> costo 0
# (backends locales como Ollama, o modelos de Bedrock sin tarifa cargada).
PRICING = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-3.5-turbo-0613": (0.001, 0.002),
    "gpt-4o": (0.0025, 0.01),
}


def _cost(model, prompt_tokens, completion_tokens):
    prompt_cost_1k, completion_cost_1k = PRICING.get(model, (0.0, 0.0))
    return prompt_tokens / 1000 * prompt_cost_1k + completion_tokens / 1000 * completion_cost_1k


def _with_retries(call):
    for i in range(MAX_RETRIES):
        try:
            return call()
        except Exception as e:
            if i < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error", 0.0


def _complete_openai(dialogs, temperature, max_tokens):
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def call():
        response = client.chat.completions.create(
            model=model,
            messages=dialogs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cost = _cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content, cost

    return _with_retries(call)


def _complete_ollama(dialogs, temperature, max_tokens):
    import ollama

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    def call():
        response = client.chat(
            model=model,
            messages=dialogs,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"], 0.0  # backend local, sin costo

    return _with_retries(call)


def _to_bedrock_messages(dialogs):
    system = [{"text": m["content"]} for m in dialogs if m["role"] == "system"]
    messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in dialogs
        if m["role"] != "system"
    ]
    return system, messages


def _complete_bedrock(dialogs, temperature, max_tokens):
    import boto3

    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    system, messages = _to_bedrock_messages(dialogs)

    def call():
        response = client.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        usage = response["usage"]
        cost = _cost(model_id, usage["inputTokens"], usage["outputTokens"])
        text = response["output"]["message"]["content"][0]["text"]
        return text, cost

    return _with_retries(call)


_BACKENDS = {
    "openai": _complete_openai,
    "ollama": _complete_ollama,
    "bedrock": _complete_bedrock,
}


def get_completion(dialogs, temperature=0, max_tokens=100):
    try:
        backend = _BACKENDS[ECON_BACKEND]
    except KeyError:
        raise ValueError(
            f"ECON_BACKEND='{ECON_BACKEND}' desconocido. Opciones: {list(_BACKENDS)}"
        )
    return backend(dialogs, temperature, max_tokens)


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100):
    get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    total_cost = sum(cost for _, cost in results)
    return [response for response, _ in results], total_cost
