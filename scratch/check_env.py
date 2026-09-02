import sys

for pkg in ["fastapi", "uvicorn", "flask", "bottle", "gradio"]:
    try:
        __import__(pkg)
        print(f"[FOUND] {pkg}")
    except ImportError:
        print(f"[NOT FOUND] {pkg}")
