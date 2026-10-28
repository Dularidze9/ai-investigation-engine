"""USD per 1M tokens (input, output). Source: OpenAI pricing page, checked during planning (Oct 2026)."""

PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
}
DEFAULT_MODEL = "gpt-6-luna"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
