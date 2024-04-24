import streamlit as st

st.header("Define settings for your tooling")

# st.session_state.update(st.session_state)

st.header(":red[Does not work as wished yet]")


def keep(key):
	# Copy from temporary widget key to permanent key
	st.session_state[key] = st.session_state["_" + key]


def unkeep(key):
	# Copy from permanent key to temporary widget key
	st.session_state["_" + key] = st.session_state[key]


if "data_path" not in st.session_state:
	st.session_state["data_path"] = ""

data_path = st.text_input("Data path:", key="_data_path", on_change=keep, args=["data_path"])


# Init all the settings inside of session state
metadata_path = st.text_input("Metadata path:", key="metadata_path")
description_file_path = st.text_input("HTML description path:", key="description_file")
print(st.session_state)
