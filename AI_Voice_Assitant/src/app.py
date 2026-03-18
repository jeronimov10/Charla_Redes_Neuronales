from pathlib import Path

import modal

from backend import Moshi, app  # noqa: F401 - Moshi must be imported to register its Modal endpoints

frontend = Path(__file__).with_name('frontend')
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('fastapi')
    .add_local_dir(frontend, '/assets')
    .add_local_python_source('backend')
)

@app.function(image=image, timeout=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    webapp = FastAPI()
    webapp.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
        allow_credentials=True,
    )

    webapp.mount('/', StaticFiles(directory='/assets', html=True))
    return webapp
