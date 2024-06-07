import streamlit as st
from menu import menu
from utils.constants import DATASETS


st.set_page_config(layout="wide")

# Initialize st.session_state.role to None
if "dataset" not in st.session_state:
	st.session_state.dataset = None

if "root_dir" not in st.session_state:
	st.session_state.root_dir = None
	# st.session_state.root_dir = r"C:\Users\mariu\Downloads\NAKO-536\NAKO_tabular"

# Retrieve the dataset and rootdir from session state
st.session_state._dataset = st.session_state.dataset
st.session_state._root_dir = st.session_state.root_dir


def set_dataset():
	# Callback function to save the role selection to Session State
	st.session_state.dataset = st.session_state._dataset


def set_root_dir():
	st.session_state.root_dir = st.session_state._root_dir


st.header("Data analyzing...", divider=True)

st.text_input(
	"Provide the root path of your data:",
	key="_root_dir",
	placeholder="/home/data/NAKO_tabular...",
	on_change=set_root_dir,
)


# Selectbox to choose role
st.selectbox(
	"Select dataset:",
	[None] + DATASETS,
	key="_dataset",
	on_change=set_dataset,
)

menu()  # Render the dynamic menu!
