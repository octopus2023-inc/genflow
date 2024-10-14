# app.py

import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import os
import graph_builder
import inspect
from examples import functions, structured_output_schema
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load the stylesheet for styling the graph
default_stylesheet = [
    {'selector': 'node',
     'style': {
         'label': 'data(label)',
         'width': '60px',
         'height': '60px',
         'font-size': '12px',
         'background-color': '#888',
         'color': '#fff',
         'text-valign': 'center',
         'text-halign': 'center',
     }},
    {'selector': '.function_call',
     'style': {'background-color': '#0074D9'}},
    {'selector': '.llm_service',
     'style': {'background-color': '#2ECC40'}},
    {'selector': '.yml_flow',
     'style': {'background-color': '#FF851B'}},
    {'selector': '.entrypoint-node',
     'style': {
         'shape': 'rectangle',
         'border-width': '2px',
         'border-color': '#fff'
     }},
    {'selector': '.output-node',
     'style': {'shape': 'triangle', 'border-width': '2px', 'border-color': '#fff'}},
    {'selector': 'edge',
     'style': {
         'curve-style': 'bezier',
         'target-arrow-shape': 'vee',
         'target-arrow-color': '#fff',
         'arrow-scale': 2,
         'width': 2,
         'line-color': '#ccc',
     }},
    {'selector': '.edge-to-yml-flow',
     'style': {
         'line-style': 'dashed',
     }},
]

# Build the initial graph elements
yaml_file = 'examples/Market_report_generation/sample.yaml'  # Ensure this is the correct YAML file
try:
    elements = graph_builder.build_graph_data(yaml_file)
    logger.debug(f"Elements being passed to Cytoscape:\n{elements}")
except Exception as e:
    logger.error(f"Error building graph data: {e}")
    elements = []

# Log the elements
logger.info(f"Initial elements: {len(elements)}")
logger.debug(f"Elements: {elements}")

# Define the legend for the node types and shapes
legend = html.Div([
    html.H3('Legend', style={'color': '#fff', 'textAlign': 'center'}),
    html.Div([
        # Node type legends
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '20px',
                'backgroundColor': '#0074D9',
                'marginRight': '10px'
            }),
            html.Span('Function Call Node', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '20px',
                'backgroundColor': '#2ECC40',
                'marginRight': '10px'
            }),
            html.Span('LLM Service Node', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '20px',
                'backgroundColor': '#FF851B',
                'marginRight': '10px'
            }),
            html.Span('YAML Flow Node', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        # Node shape legends
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '20px',
                'backgroundColor': '#888',
                'marginRight': '10px',
                'border': '2px solid #fff',
                'borderRadius': '50%'
            }),
            html.Span('Regular Node', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '20px',
                'backgroundColor': '#0074D9',
                'marginRight': '10px',
                'border': '2px solid #fff',
                'borderRadius': '0%',
            }),
            html.Span('Entrypoint Node', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '0',
                'height': '0',
                'borderLeft': '10px solid transparent',
                'borderRight': '10px solid transparent',
                'borderBottom': '20px solid #0074D9',
                'marginRight': '10px',
            }),
            html.Span('Output Node', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        # Edge legends
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '2px',
                'backgroundColor': '#ccc',
                'marginRight': '10px'
            }),
            html.Span('Dependency Edge', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(style={
                'display': 'inline-block',
                'width': '20px',
                'height': '2px',
                'backgroundColor': '#ccc',
                'borderTop': '2px dashed #ccc',
                'marginRight': '10px'
            }),
            html.Span('Edge to yml_flow Node (Dashed)', style={'color': '#fff'})
        ], style={'marginBottom': '5px'}),
    ], style={'padding': '10px'})
], style={
    'position': 'absolute',
    'bottom': '10px',
    'right': '10px',
    'backgroundColor': '#1e1e1e',
    'padding': '10px',
    'border': '1px solid #fff',
    'borderRadius': '5px',
    'width': '300px',
    'zIndex': '1000',
    'color': '#fff'
})

# Define the layout
app.layout = html.Div([
    html.H1('GenFlow Workflow Visualizer', style={'color': '#fff'}),
    html.Div([
        html.Label('YAML File Path:', style={'color': '#fff'}),
        dcc.Input(id='yaml-file-input', value=yaml_file, type='text', style={'width': '300px'}),
        html.Button('Load', id='load-button', n_clicks=0, style={'marginLeft': '10px'}),
    ], style={'marginBottom': '20px'}),
    html.Div([
        cyto.Cytoscape(
            id='cytoscape',
            elements=elements if elements else [],
            layout={'name': 'breadthfirst'},
            style={'width': '100%', 'height': '600px', 'border': '1px solid black', 'background-color': '#1e1e1e'},
            stylesheet=default_stylesheet,
        ),
        legend,  # Add the legend here
    ], style={'position': 'relative'}),
    html.Div(id='node-data', style={'marginTop': '20px', 'whiteSpace': 'pre-wrap', 'color': '#fff'})
], style={'backgroundColor': '#2e2e2e', 'padding': '20px'})

# Callback to update the graph when a new YAML file is loaded
@app.callback(
    Output('cytoscape', 'elements'),
    [Input('load-button', 'n_clicks')],
    [State('yaml-file-input', 'value')]
)
def update_graph(n_clicks, yaml_file):
    if n_clicks:
        logger.debug(f"Load button clicked. Loading YAML file: {yaml_file}")
        if os.path.exists(yaml_file):
            try:
                elements = graph_builder.build_graph_data(yaml_file)
                logger.debug(f"Loaded elements: {elements}")
                return elements
            except Exception as e:
                logger.error(f"Error loading YAML file '{yaml_file}': {e}")
                return dash.no_update
        else:
            logger.error(f"YAML file '{yaml_file}' does not exist.")
            return dash.no_update
    else:
        return dash.no_update

# Callback to display node information
@app.callback(
    Output('node-data', 'children'),
    Input('cytoscape', 'tapNodeData')
)
def display_node_data(data):
    if data:
        logger.debug(f"Displaying data for node: {data}")
        node_id = data.get('id', 'N/A')
        node_label = data.get('label', node_id)
        node_type = data.get('type', 'N/A')
        params = data.get('params', {})
        outputs = data.get('outputs', [])

        content = [
            html.H4(f"Node: {node_label}", style={'color': '#fff'}),
            html.P(f"Type: {node_type}", style={'color': '#fff'})
        ]

        # Display input parameters
        if params:
            content.append(html.H5("Input Parameters:", style={'color': '#fff'}))
            for key, value in params.items():
                content.append(html.P(f"{key}: {value}", style={'color': '#fff'}))
        else:
            content.append(html.P("No input parameters.", style={'color': '#fff'}))

        # Display outputs
        if outputs:
            content.append(html.H5("Outputs:", style={'color': '#fff'}))
            for output in outputs:
                content.append(html.P(f"- {output}", style={'color': '#fff'}))
        else:
            content.append(html.P("No outputs.", style={'color': '#fff'}))

        # Existing logic for functions and tools
        if node_type == 'function_call':
            func_name = data.get('function', '')
            if func_name:
                if func_name.startswith('COMPOSIO.'):
                    # It's a COMPOSIO function
                    content.append(html.H5(f"Function: {func_name}", style={'color': '#fff'}))
                    content.append(html.P("This is a COMPOSIO function.", style={'color': '#fff'}))
                else:
                    func = getattr(functions, func_name, None)
                    if func:
                        try:
                            source = inspect.getsource(func)
                            content.append(html.H5(f"Function: {func_name}", style={'color': '#fff'}))
                            content.append(html.Pre(source, style={'color': '#fff', 'backgroundColor': '#333', 'padding': '10px'}))
                        except OSError:
                            content.append(html.P(f"Source code for function '{func_name}' not available.", style={'color': '#fff'}))
                    else:
                        content.append(html.P(f"Function '{func_name}' not found in 'functions.py'.", style={'color': '#fff'}))
            else:
                content.append(html.P("No function associated with this node.", style={'color': '#fff'}))

        elif node_type == 'llm_service':
            tools = data.get('tools', [])
            structured_output_schema_name = data.get('structured_output_schema', '')

            if tools:
                content.append(html.H5("Tools:", style={'color': '#fff'}))
                for tool_name in tools:
                    if not tool_name:
                        continue  # Skip empty tool names
                    if tool_name.startswith('COMPOSIO.'):
                        # It's a COMPOSIO function
                        content.append(html.P(f"COMPOSIO Function: {tool_name}", style={'color': '#fff'}))
                    else:
                        func = getattr(functions, tool_name, None)
                        if func:
                            try:
                                source = inspect.getsource(func)
                                content.append(html.P(f"Function: {tool_name}", style={'color': '#fff'}))
                                content.append(html.Pre(source, style={'color': '#fff', 'backgroundColor': '#333', 'padding': '10px'}))
                            except OSError:
                                content.append(html.P(f"Source code for function '{tool_name}' not available.", style={'color': '#fff'}))
                        else:
                            content.append(html.P(f"Function '{tool_name}' not found in 'functions.py'.", style={'color': '#fff'}))

            if structured_output_schema_name:
                schema = getattr(structured_output_schema, structured_output_schema_name, None)
                if schema:
                    try:
                        source = inspect.getsource(schema)
                        content.append(html.H5(f"Structured Output Schema: {structured_output_schema_name}", style={'color': '#fff'}))
                        content.append(html.Pre(source, style={'color': '#fff', 'backgroundColor': '#333', 'padding': '10px'}))
                    except OSError:
                        content.append(html.P(f"Source code for schema '{structured_output_schema_name}' not available.", style={'color': '#fff'}))
                else:
                    content.append(html.P(f"Schema '{structured_output_schema_name}' not found in 'structured_output_schema.py'.", style={'color': '#fff'}))

        elif node_type == 'yml_flow':
            content.append(html.P("This node represents a sub-flow.", style={'color': '#fff'}))

        # Additional node types can be handled here

        return content
    else:
        logger.debug("No node data to display.")
    return "Click on a node to see details."

if __name__ == '__main__':
    app.run_server(debug=False)
