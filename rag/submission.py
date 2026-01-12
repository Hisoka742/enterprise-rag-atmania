from __future__ import annotations

from typing import Any, Dict, List
import json
import os


def make_reference(pdf_sha1: str, page_index: int) -> Dict[str, Any]:
    return {"pdf_sha1": pdf_sha1, "page_index": int(page_index)}


def write_submission(
    out_path: str,
    team_email: str,
    submission_name: str,
    answers: List[Dict[str, Any]],
) -> None:
    normalized: List[Dict[str, Any]] = []
    for a in answers:
        item: Dict[str, Any] = {
            "value": a.get("value"),
            "references": a.get("references", []),
        }
        if a.get("question_text"):
            item["question_text"] = a["question_text"]
        if a.get("kind"):
            item["kind"] = a["kind"]
        normalized.append(item)

    payload = {
        "team_email": team_email,
        "submission_name": submission_name,
        "answers": normalized,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)




