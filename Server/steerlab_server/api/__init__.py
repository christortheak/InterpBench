"""HTTP service (parallel to Swift ``WebServer``), thin over the engine.

The existing Swift ``steerlab-cli serve`` + ``web/index.html`` are the
cluster-ready blueprint; this FastAPI app captures the same route shapes plus an
**async job model** for long GPU tasks (extract/sweep/run/validate take
minutes–hours on a shared box) and real token streaming over SSE.
"""
