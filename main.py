from __future__ import annotations

import uvicorn
from dotenv import load_dotenv

from modules.multimodal.app import create_app
from modules.multimodal.config import Settings


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bridge_host,
        port=settings.bridge_port,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
