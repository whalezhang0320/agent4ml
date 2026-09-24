from __future__ import annotations

from fastapi.testclient import TestClient

from agent4ml.backend.agents.ml_research.task_store import InMemoryTaskStore
from agent4ml.backend.app.ml_service.api import create_app, encode_sse
from agent4ml.backend.app.ml_service.settings import MLServiceSettings


def _settings(tmp_path):
    return MLServiceSettings(
        redis_url="redis://unused",
        redis_namespace="test",
        work_root=tmp_path / "runs",
        allowed_root=tmp_path,
        host="127.0.0.1",
        port=8000,
        sse_heartbeat_seconds=0.01,
    )


def test_submit_get_and_cancel_task_over_http(tmp_path):
    app = create_app(store=InMemoryTaskStore(), settings=_settings(tmp_path))

    with TestClient(app) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        created = client.post(
            "/v1/tasks",
            json={"name": "demo", "command": ["python", "train.py"], "cwd": str(tmp_path)},
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        fetched = client.get(f"/v1/tasks/{task_id}")
        assert fetched.json()["status"] == "queued"

        cancelled = client.post(f"/v1/tasks/{task_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"


def test_sse_encoding_contains_replay_id_and_structured_payload(tmp_path):
    store = InMemoryTaskStore()
    app = create_app(store=store, settings=_settings(tmp_path))
    service = app.state.task_service
    task = service.submit(name="demo", command=["python"], cwd=tmp_path)
    event = store.read_events(task.task_id)[0]

    encoded = encode_sse(event)

    assert f"id: {event.event_id}\n" in encoded
    assert "event: task.created\n" in encoded
    assert f'"task_id":"{task.task_id}"' in encoded


def test_terminal_sse_stream_replays_only_events_after_last_event_id(tmp_path):
    store = InMemoryTaskStore()
    app = create_app(store=store, settings=_settings(tmp_path))

    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            json={"name": "demo", "command": ["python"], "cwd": str(tmp_path)},
        ).json()
        task_id = created["task_id"]
        first_event_id = store.read_events(task_id)[0].event_id
        client.post(f"/v1/tasks/{task_id}/cancel")

        replay = client.get(
            f"/v1/tasks/{task_id}/events",
            headers={"Last-Event-ID": first_event_id},
        )

    assert replay.status_code == 200
    assert "event: task.created" not in replay.text
    assert "event: task.queued" in replay.text
    assert "event: task.cancelled" in replay.text


def test_sse_rejects_invalid_replay_cursor(tmp_path):
    app = create_app(store=InMemoryTaskStore(), settings=_settings(tmp_path))

    with TestClient(app) as client:
        task_id = client.post(
            "/v1/tasks",
            json={"name": "demo", "command": ["python"], "cwd": str(tmp_path)},
        ).json()["task_id"]
        response = client.get(f"/v1/tasks/{task_id}/events?after=not-a-stream-id")

    assert response.status_code == 400
