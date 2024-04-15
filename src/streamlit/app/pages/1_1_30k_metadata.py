import csv
from pathlib import Path

import pandas as pd
import streamlit as st
from utils.utils import extract_infos_given_datapath

# Hardcoding the data paths for now, might change in future
# data_path = st.file_uploader("Select folder containing data information")

root = Path("/home/artur/Desktop/FA/data/NAKO_tabular/30k_metadata")
metadata_path = root / "NAKO-536_2023-07-13_Erstuntersuchung_MRT_PSN_wie_194_Metadaten.csv"
data_path = root / "NAKO-536_2023-07-13_Erstuntersuchung_MRT_PSN_wie_194.csv"
data_description_path = root / "NAKO-536 - 2023-07-13 - Beschreibung des Übergabedatensatzes.html"

st.header("30 K metadata dataset", divider=True)


if data_path:
	st.markdown("Overview of the data")
	data = pd.read_csv(data_path, sep=";", encoding="latin1", quoting=csv.QUOTE_NONE)
	st.write(data.head())
	# with open(data_description_path, "r") as data_description_json:
	# 	data_description = json.load(data_description_json)

	# Start from 11 to discard "ID" as attribute
	features = data.columns[1:]
	col1, col2 = st.columns(2)
	with col1:
		option = st.selectbox("Choose the attribute you wish to get more info about.", features)
	with col2:
		attr_info = extract_infos_given_datapath(data_path, description_file=data_description_path)
		st.markdown("""
			**Data Description:**
					""")
		st.markdown(f"{attr_info[option].get('feature_information_text')}")
		st.write(attr_info[option].get("feature_stats"))
