import streamlit as st

from onto_viewer.apps.viewer import OntoViewerApp

st.set_page_config(page_title="Ontology Viewer", page_icon="📊", layout="wide")

app = OntoViewerApp()
app.run()