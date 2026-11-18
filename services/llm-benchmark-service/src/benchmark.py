import json
import os

from groq import Groq


def benchmark_schedule(targets, available_time):
    """
    Ask Groq to select observation targets within the available
    telescope time.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    client = Groq(api_key=api_key)

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    prompt = f"""
You are an observation scheduling assistant.

Select the best combination of observation targets.

Rules:
1. Total observation duration must not exceed available telescope time.
2. Maximize the sum of target priorities.
3. Select only targets from the supplied list.
4. Do not invent targets.
5. Return valid JSON only.

Available telescope time: {available_time} hours.

Targets:
{json.dumps(targets)}

Required JSON format:
{{
  "selected_targets": ["Target name"],
  "reasoning": "Brief explanation"
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise scientific observation scheduling assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    return {
        "model": model,
        "selected_targets": result.get("selected_targets", []),
        "reasoning": result.get("reasoning", "")
    }
