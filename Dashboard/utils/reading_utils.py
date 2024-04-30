from pathlib import Path
from typing import Union

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


def read_html_from_path(path: Union[str, Path]) -> BeautifulSoup:
	"""Create soup object directly from path.

	:param path: path to html file
	:type path: Union[str, Path]
	:return: soup object which can be scraped
	:rtype: BeautifulSoup
	"""
	with open(str(path)) as file:
		html_content = file.read()
	return BeautifulSoup(html_content, "html.parser")


@st.cache_data
def read_csv_file_cached(path: Union[str, Path], sep: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
	"""Read CSV file using pandas.
	The backend remains exatcly the same, the only difference is
	that the function shall be cached due to usage in streamlit.

	:param path: path to the file to be read.
	:type path: Union[str, Path]
	:param sep: seperator used in the CSV file.
	:type sep: str
	:param encoding: encoding used.
	:type encoding: str
	:return: Dataframe similar to read_csv
	:rtype: pd.DataFrame
	"""
	return pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines="warn")
