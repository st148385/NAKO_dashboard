import dash
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

# ---- Load and preprocess data ------

data_path = "/home/artur/Desktop/FA/data/NAKO_tabular/30k_metadata"
######################################

app = dash.Dash(__name__)


app.layout = html.Div(
	children=[
		html.H1("Reasearch Thesis", style={"text-align": "center"}),
		html.P("Te"),
		dcc.Input(
			id="dataset-id",
			type="text",
			value="/home/artur/Desktop/FA/data/NAKO_tabular/30k_metadata",
		),
		dcc.Graph(id="Some"),
		html.Div(id="output_div"),
	]
)


@app.callback(Output("some", "children"), [], [State("dataset-id", "value")])
def get_text(text):
	return text


if __name__ == "__main__":
	app.run_server(debug=True)
