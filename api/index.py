"""Vercel Python entrypoint. Vercel's @vercel/python runtime detects the ASGI
`app` and serves it; all routes are rewritten here by vercel.json."""

import os
import sys

# Vercel does not always run the function from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402,F401
