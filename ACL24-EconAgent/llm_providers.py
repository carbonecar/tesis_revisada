"""Adaptadores de LLM para simulate.py. El backend se elige con ECON_BACKEND
(openai | ollama | bedrock) y cada uno expone la misma firma de get_completion.
"""
import logging
import multiprocessing
import os
from functools import partial
from time import sleep

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ECON_BACKEND = os.getenv("ECON_BACKEND", "openai").lower()
MAX_RETRIES = 20
RETRY_DELAY_SECONDS = 6
MAX_RETRY_DELAY_SECONDS = 60

# Precio por 1k tokens (prompt, completion). Modelos ausentes -> costo 0
# a menos que PRICE_INPUT_PER_1M / PRICE_OUTPUT_PER_1M estén seteadas (ver _cost).
PRICING = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-3.5-turbo-0613": (0.001, 0.002),
    "gpt-4o": (0.0025, 0.01),
    "us.anthropic.claude-sonnet-4-6": (0.003, 0.015),
}

# Override manual de precio (USD por 1M tokens), útil para Bedrock u otros
# modelos ausentes de PRICING. Si están seteadas, tienen prioridad sobre PRICING.
_PRICE_INPUT_PER_1M = os.getenv("PRICE_INPUT_PER_1M")
_PRICE_OUTPUT_PER_1M = os.getenv("PRICE_OUTPUT_PER_1M")

_warned_models: set[str] = set()

# Acumuladores globales de uso real (tokens), visibles vía get_usage_summary().
_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def _cost(model, prompt_tokens, completion_tokens):
    _usage_totals["prompt_tokens"] += prompt_tokens
    _usage_totals["completion_tokens"] += completion_tokens
    _usage_totals["calls"] += 1

    if _PRICE_INPUT_PER_1M is not None and _PRICE_OUTPUT_PER_1M is not None:
        price_in_1m = float(_PRICE_INPUT_PER_1M)
        price_out_1m = float(_PRICE_OUTPUT_PER_1M)
        return prompt_tokens / 1_000_000 * price_in_1m + completion_tokens / 1_000_000 * price_out_1m

    if model not in PRICING and model not in _warned_models:
        _warned_models.add(model)
        logger.warning(
            "Sin precio cargado para modelo=%r — costo se reporta como $0. "
            "Seteá PRICE_INPUT_PER_1M / PRICE_OUTPUT_PER_1M para estimar costo real.",
            model,
        )
    prompt_cost_1k, completion_cost_1k = PRICING.get(model, (0.0, 0.0))
    return prompt_tokens / 1000 * prompt_cost_1k + completion_tokens / 1000 * completion_cost_1k


def get_usage_summary():
    """Totales acumulados de tokens/llamadas desde que se importó el módulo."""
    return dict(_usage_totals)


def print_usage_summary():
    u = _usage_totals
    print(
        f"Uso total: {u['calls']} llamadas, "
        f"{u['prompt_tokens']} tokens input, {u['completion_tokens']} tokens output"
    )


class DailyQuotaExhausted(RuntimeError):
    """El backend reportó un tope duro por día (no un throttle transitorio de
    minuto). Reintentar con el backoff normal no sirve — hay que parar."""


# Frases que distinguen un throttle "por día" (no se resuelve reintentando en
# minutos) de un throttle transitorio de minuto/segundo (sí se resuelve).
_DAILY_QUOTA_MARKERS = ("tokens per day", "requests per day", "per-day")


def _is_daily_quota_error(exception) -> bool:
    return any(marker in str(exception).lower() for marker in _DAILY_QUOTA_MARKERS)


def _retry_delay(attempt, exception):
    # Los SDK de OpenAI/boto3 exponen la respuesta HTTP en .response; si el
    # servidor nos dice cuánto esperar (429 con Retry-After), lo respetamos.
    response = getattr(exception, "response", None)
    if response is not None:
        retry_after = getattr(response, "headers", {}).get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        if getattr(response, "status_code", None) == 429:
            return min(RETRY_DELAY_SECONDS * (2 ** attempt), MAX_RETRY_DELAY_SECONDS)
    return RETRY_DELAY_SECONDS


def _with_retries(call):
    for i in range(MAX_RETRIES):
        try:
            return call()
        except Exception as e:
            if _is_daily_quota_error(e):
                # No tiene sentido reintentar un tope de 24hs con backoff de
                # segundos, y seguir haciéndolo termina rellenando el resto
                # de la corrida con la acción de fallback en silencio.
                raise DailyQuotaExhausted(str(e)) from e
            if i < MAX_RETRIES - 1:
                delay = _retry_delay(i, e)
                logger.debug(
                    "Retry %d/%d after %s: %s (esperando %.1fs)",
                    i + 1, MAX_RETRIES, type(e).__name__, e, delay,
                )
                sleep(delay)
            else:
                logger.warning("Gave up after %d retries: %s: %s", MAX_RETRIES, type(e).__name__, e)
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
        _usage_totals["prompt_tokens"] += response.get("prompt_eval_count", 0)
        _usage_totals["completion_tokens"] += response.get("eval_count", 0)
        _usage_totals["calls"] += 1
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
    logger.debug("-> [%s] dialogs=%r", ECON_BACKEND, dialogs)
    response, cost = backend(dialogs, temperature, max_tokens)
    logger.debug("<- [%s] cost=%.5f response=%r", ECON_BACKEND, cost, response)
    return response, cost


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100):
    get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    total_cost = sum(cost for _, cost in results)
    return [response for response, _ in results], total_cost
