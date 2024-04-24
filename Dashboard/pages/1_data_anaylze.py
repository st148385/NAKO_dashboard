import csv
from pathlib import Path

import pandas as pd
import streamlit as st

# Hardcoding the data paths for now, might change in future
# data_path = st.file_uploader("Select folder containing data information")

st.header("30 K metadata dataset", divider=True)

# root = Path("/home/artur/Desktop/FA/data/NAKO_tabular/30k_metadata")
root = st.text_input(
	"Provide the root dir of 30 k metadata",
	"/home/artur/Desktop/FA/data/NAKO_tabular/30k_metadata",
	key="root_path",
	placeholder="/home/data/NAKO_tabular...",
)
if root:
	root = Path(root)

	# ------- Read Data --------- #
	metadata_path = root / "NAKO-536_2023-07-13_Erstuntersuchung_MRT_PSN_wie_194_Metadaten.csv"
	data_path = root / "NAKO-536_2023-07-13_Erstuntersuchung_MRT_PSN_wie_194.csv"
	data_description_path = root / "NAKO-536 - 2023-07-13 - Beschreibung des Übergabedatensatzes.html"
	data = pd.read_csv(data_path, sep=";", encoding="latin1", quoting=csv.QUOTE_NONE)
	st.dataframe(data)
