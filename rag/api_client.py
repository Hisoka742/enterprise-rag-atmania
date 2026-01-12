from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests


def try_fetch_questions(api_base: str, timeout: int = 20) -> Optional[List[Dict[str, Any]]]:
 
    candidates = [
        "/questions",
        "/api/questions",
        "/v1/questions",
        "/challenge/questions",
    ]
    for path in candidates:
        url = api_base.rstrip("/") + path
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
               
                if isinstance(data, list):
                    return data
    
                if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                    return data["questions"]
        except Exception:
            pass
    return None
