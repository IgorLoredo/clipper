"""File exchange for LLM calls answered by a Codex session."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


class CodexResponseRequired(RuntimeError):
    def __init__(self, requests: list[Path], stage: str = "selection"):
        self.requests = requests
        self.stage = stage
        super().__init__("Codex responses required for:\n" + "\n".join(map(str, requests)))


def _stage(schema: dict) -> str:
    properties = schema.get("properties", {})
    if "clips" in properties:
        return "visual_review"
    if "reviews" in properties:
        item_props = properties["reviews"].get("items", {}).get("properties", {})
        if "accepted" in item_props and "payoff" in item_props:
            return "selection"
        return "campaign_review"
    return "selection"


def write_status(requests: list[Path], command: list[str], stage: str) -> Path:
    directory = requests[0].parent
    payload = {
        "format": "clipper-codex-status-v1",
        "state": "awaiting_response",
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "pending_requests": [str(path.resolve()) for path in requests],
        "next_action": "Write each matching .response.json and repeat the same command.",
    }
    path = directory / "codex-status.json"
    _write_request(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def complete_status(directory: Path, command: list[str]) -> None:
    path = directory / "codex-status.json"
    if not directory.is_dir() and not path.is_file():
        return
    payload = {"format": "clipper-codex-status-v1", "state": "complete",
               "updated_at": datetime.now(timezone.utc).isoformat(),
               "command": command, "pending_requests": [], "next_action": "Review clips.json and the MP4 files."}
    _write_request(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _write_request(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(contents + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CodexExchange:
    def __init__(self, directory: Path, model: str):
        self.directory = directory
        self.model = model

    def __call__(self, system: str, user: str, schema: dict, *, images: list[dict] | None = None) -> dict:
        request = {"format": "clipper-codex-v1", "model": self.model,
                   "system": system, "user": user, "schema": schema}
        if images:
            request["images"] = images
        contents = json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False)
        request_id = hashlib.sha256(contents.encode("utf-8")).hexdigest()
        request_path = (self.directory / f"{request_id}.request.json").resolve()
        response_path = request_path.with_name(f"{request_id}.response.json")
        if not request_path.is_file():
            _write_request(request_path, contents)
        if not response_path.is_file():
            raise CodexResponseRequired([request_path], _stage(schema))

        from jsonschema import Draft202012Validator

        try:
            payload = json.loads(response_path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError) as exc:
            print(f"[codex] invalid JSON in {response_path}: {exc}", flush=True)
        else:
            error = next(Draft202012Validator(schema).iter_errors(payload), None)
            if error is None:
                # Reject NaN and Infinity, which Python's JSON parser accepts by default.
                try:
                    json.dumps(payload, allow_nan=False)
                except ValueError as exc:
                    print(f"[codex] invalid value in {response_path}: {exc}", flush=True)
                else:
                    return payload
            else:
                print(f"[codex] invalid response in {response_path} at {error.json_path}: {error.message}",
                      flush=True)
        raise CodexResponseRequired([request_path], _stage(schema))
