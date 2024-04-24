import dash
from dash import dcc, html

dash.register_page(__name__)

layout = html.Div(
	[
		html.H1(children="Research Thesis - Artur", style={"textAlign": "center"}),
	]
)
