"""`python -m app` starts the API on 0.0.0.0:$PORT (default 8000)."""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=get_settings().port, proxy_headers=True)
