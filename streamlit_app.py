"""Compatibility entrypoint for Streamlit Community Cloud.

Some deployments default to `streamlit_app.py`.
This file simply loads the main application from `app.py`.
"""

from app import *  # noqa: F401,F403
