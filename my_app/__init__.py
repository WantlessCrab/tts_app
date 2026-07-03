"""tts_app my_app package.
Service entrypoints are imported in package mode, for example:
    uvicorn my_app.audio_server:app
    uvicorn my_app.pdf_processor.process:app
    uvicorn my_app.acquire.api:app
    uvicorn my_app.convert.api:app
    uvicorn my_app.doctr_service:app
"""