"""Entry point: `uvicorn server:app` (the app lives in the asr_server package)."""

import uvicorn

from asr_server.app import create_app

app = create_app()

if __name__ == "__main__":
    # NOTE: for GPU models, keep workers=1 unless you deliberately replicate the model per worker.
    uvicorn.run(app, host="0.0.0.0", port=8000)
