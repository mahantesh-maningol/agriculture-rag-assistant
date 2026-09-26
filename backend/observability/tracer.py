import time
import uuid
from contextlib import contextmanager


class Tracer:

    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.spans = []

    @contextmanager
    def span(self, name: str, metadata: dict | None = None):

        start_time = time.perf_counter()

        span = {
            "name": name,
            "metadata": metadata or {},
        }

        try:
            yield span

        except Exception as exc:
            span["status"] = "error"
            span["error"] = str(exc)
            raise

        finally:
            duration = time.perf_counter() - start_time

            span["status"] = span.get("status", "success")
            span["latency_ms"] = round(duration * 1000, 2)

            self.spans.append(span)

    def get_trace(self):
        return {
            "trace_id": self.trace_id,
            "spans": self.spans,
        }
