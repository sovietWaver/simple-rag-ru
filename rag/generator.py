import ollama

PROMPT_TEMPLATE = """Ты — ассистент, отвечающий строго на основе контекста ниже.
Если в контексте нет ответа — честно скажи об этом.
Не выдумывай факты. Отвечай кратко и по делу на русском языке.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

# gemma4 is loaded with num_ctx=4096; requesting more triggers a reload that
# fails with 503 when RAM is tight on CPU-only inference.
_NUM_CTX = 4096
_CHAR_WARN_THRESHOLD = 7000  # ~3500 tokens for Russian ≈ borderline for 4096


def _build_prompt(context: str, question: str) -> str:
    # .replace() instead of .format() — chunks may contain { } (code, JSON)
    return (
        PROMPT_TEMPLATE
        .replace("{context}", context)
        .replace("{question}", question)
    )


def ask_llm(
    context: str,
    question: str,
    model: str = "gemma4:latest",
) -> str:
    prompt = _build_prompt(context, question)
    char_len = len(prompt)
    print(f"Prompt length: {char_len} chars", end="")
    if char_len > _CHAR_WARN_THRESHOLD:
        print(f"  ⚠  близко к лимиту контекста ({_NUM_CTX} токенов)!", end="")
    print()

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": _NUM_CTX,
            "temperature": 0.2,
            "num_predict": 1024,
        },
    )
    return response["message"]["content"]


def ask_llm_stream(
    context: str,
    question: str,
    model: str = "gemma4:latest",
) -> str:
    prompt = _build_prompt(context, question)
    char_len = len(prompt)
    print(f"Prompt length: {char_len} chars", end="")
    if char_len > _CHAR_WARN_THRESHOLD:
        print(f"  ⚠  близко к лимиту контекста ({_NUM_CTX} токенов)!", end="")
    print()

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": _NUM_CTX,
            "temperature": 0.2,
            "num_predict": 1024,
        },
        stream=True,
    )

    collected = []
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        collected.append(token)
    print()
    return "".join(collected)
