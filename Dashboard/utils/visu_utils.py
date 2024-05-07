import copy
from typing import Any, Dict, List, Union

import pandas as pd
import plotly.express as px

from utils.constants import TICK_FREQUENCY


def create_plotly_histogram(
	data: pd.DataFrame,
	x_axis: str,
	feature_dict: Dict[str, Any],
	groupby: str = None,
	mapping_dict: Dict[int, str] = False,
):
	"""Create histogram figure

	:param data: dataframe
	:type data: pd.DataFrame
	:param x_axis: over which feature we want to compute the histogram
	:type x_axis: str
	:param groupby: Over which feature we want to group (e.g. colorize)
	:type groupby: str
	"""

	if not mapping_dict:
		mapping_dict = {}

	# Create plotly figure
	fig = px.histogram(
		data,
		x=x_axis,
		color=groupby,
		hover_data=data.columns,
		color_discrete_sequence=px.colors.qualitative.Safe,
		opacity=0.8,
		barmode="overlay",
	)

	fig.update_layout(
		title={
			"text": f"{x_axis}-distribution after filtering",
			"y": 0.9,
			"x": 0.5,
			"xanchor": "center",
			"yanchor": "top",
		}
	)
	if feature_dict.get(x_axis)["nominal/ordinal"]:
		tick_frequency = TICK_FREQUENCY if len(mapping_dict[x_axis].keys()) > TICK_FREQUENCY else 1
		fig.update_xaxes(
			tickvals=[key for i, (key, _) in enumerate(mapping_dict[x_axis].items()) if i % tick_frequency == 0],
			ticktext=[val for i, (_, val) in enumerate(mapping_dict[x_axis].items()) if i % tick_frequency == 0],
		)

	# Rename legend
	if mapping_dict.get(groupby):
		for label, label_name in mapping_dict.get(groupby).items():
			fig.update_traces(
				{"name": label_name.replace("'", "")},
				selector={"name": str(copy.copy(label))},
			)

	return fig


def create_plotly_scatterplot(
	data: pd.DataFrame,
	feature1: str,
	feature2: str,
	feature_dict: Dict[str, Any],
	groupby: Union[List[str], str],
	mapping_dict: Dict[int, str] = False,
):
	"""_summary_

	:param data: _description_
	:type data: pd.DataFrame
	:param feature1: _description_
	:type feature1: str
	:param feature2: _description_
	:type feature2: str
	:param groupby: _description_
	:type groupby: Union[List[str], str]
	:param mapping_dict: _description_, defaults to {}
	:type mapping_dict: Dict[int, str], optional
	"""

	if not mapping_dict:
		mapping_dict = {}

	# There's no beautiful solution to group the plot by multiple stuff.
	# Maybe only for binary features.
	copied_data = data.copy()
	if isinstance(groupby, str):
		groupby = [groupby]

	# To ensure discrete color in legend
	for group in groupby:
		copied_data[group] = copied_data[group].astype(str)

	fig = px.scatter(
		copied_data,
		x=feature1,
		y=feature2,
		color=groupby[0],
		color_discrete_sequence=px.colors.qualitative.Safe,
		opacity=0.1,
		marginal_x="box",
		marginal_y="box",
		trendline="ols",
		width=1000,
		height=800,
	)

	if feature_dict.get(feature1)["nominal/ordinal"]:
		tick_frequency = TICK_FREQUENCY if len(mapping_dict[feature1].keys()) > TICK_FREQUENCY else 1
		fig.update_xaxes(
			tickvals=[key for i, (key, _) in enumerate(mapping_dict[feature1].items()) if i % tick_frequency == 0],
			ticktext=[val for i, (_, val) in enumerate(mapping_dict[feature1].items()) if i % tick_frequency == 0],
		)

	if feature_dict.get(feature2)["nominal/ordinal"]:
		tick_frequency = TICK_FREQUENCY if len(mapping_dict[feature2].keys()) > TICK_FREQUENCY else 1
		fig.update_yaxes(
			tickvals=[key for i, (key, _) in enumerate(mapping_dict[feature2].items()) if i % tick_frequency == 0],
			ticktext=[val for i, (_, val) in enumerate(mapping_dict[feature2].items()) if i % tick_frequency == 0],
		)

	if mapping_dict.get(groupby[0]):
		# Rename legend
		for label, label_name in mapping_dict.get(groupby[0]).items():
			fig.update_traces(
				{"name": label_name.replace("'", "")},
				selector={"name": str(label)},
			)
	return fig
