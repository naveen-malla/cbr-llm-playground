import os, requests

def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "phi3:mini")

def generate_with_ollama(prompt: str, max_tokens: int = 256) -> str:
    # Requires: `ollama serve` running (Ollama app) and model pulled
    url = "http://localhost:11434/api/generate"
    payload = {"model": ollama_model(), "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def propose_adaptation(query_text: str, retrieved_case_text: str, retrieved_solution_text: str) -> dict:
    sys = (
      "You are a careful, concise assistant. "
      "Given a new problem, and a similar past case with its solution, adapt the solution to fit the new problem. "
      "Explain the key differences and why the adapted steps make sense. Return plain text."
    )
    prompt = (
      f"{sys}\n\n"
      f"NEW PROBLEM:\n{query_text}\n\n"
      f"SIMILAR PAST CASE:\n{retrieved_case_text}\n\n"
      f"PAST SOLUTION:\n{retrieved_solution_text}\n\n"
      f"ADAPTED SOLUTION:"
    )
    out = generate_with_ollama(prompt)
    return {"adapted_solution": out}
