#!/usr/bin/env python
#
# WHISPER FRONTEND APP
# Streamlit version
#

import os

try:
    import streamlit as st
    from dotenv import dotenv_values
except Exception as e:
    print(f"Caught fatal exception: {e}")

# local imports
from scriptdump.libs.settings import Properties

# resolve paths relative to this file's directory
_pkg_dir = os.path.dirname(os.path.abspath(__file__))

# load environment
config_env: dict = dotenv_values(os.path.join(_pkg_dir, ".env"))

# load app settings
config_filename: str = config_env.get(
    "CONFIG_FILE", os.path.join(_pkg_dir, "parameters.yaml")
)
print(config_filename)
appSettings = Properties(config_file=config_filename)

# define app pages
audio_page = st.Page(
    "pages/whisper_audio.py", title="Audio to Text", icon=":material/speaker:"
)
enabled_sections = [audio_page]

# setup application main page
st.logo(os.path.join(_pkg_dir, "assets", "redhat.png"))
pg = st.navigation(enabled_sections)
st.set_page_config(
    page_title="Whisper AI Audio Transcription", page_icon=":material/edit:"
)

# run app
pg.run()
