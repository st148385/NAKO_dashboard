import dash
from dash import dcc, html

dash.register_page(__name__)

layout = html.Div(
	[
		dcc.Input(id="input1", type="text", placeholder="", style={"marginRight": "10px"}),
	]
)
