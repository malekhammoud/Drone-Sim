import json
import urllib.request
from typing import Optional, Dict, Any

def query_tether(prompt: str, api_url: str = "http://localhost:11434/v1/chat/completions", model: str = "llama3.2:1b") -> Optional[Dict[str, Any]]:
    """Query local Ollama model for tactical edge-case decision making."""
    system_prompt = (
        "You are an autonomous UAV tactical supervisor in the Arctic. "
        "Analyze operator console inputs or tactical events (e.g., active attacks, closed zones). "
        "Respond ONLY with a JSON object describing the action. "
        "Schema:\n"
        "{\n"
        '  "action": "CLOSED_ZONE" | "INTERCEPT" | "ABORT",\n'
        '  "lat": float,\n'
        '  "lon": float,\n'
        '  "radius_m": float,\n'
        '  "reason": "string"\n'
        "}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        print(f"[Tether Error] Failed to process event: {e}")
        return None