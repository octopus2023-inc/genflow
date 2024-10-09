# GenFlow.py

import yaml
import jinja2
import networkx as nx
import importlib
import traceback
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import openai
from openai import OpenAI
import inspect
from typing import Union
from composio_openai import ComposioToolSet, App, Action
import textwrap

# Load environment variables
load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)

class GenFlow:
    """
    Class to parse YAML data, construct an execution graph, and execute nodes.
    """

    def __init__(self, yaml_data):
        self.yaml_data = yaml_data
        self.nodes = {}        # Maps node names to Node instances
        self.outputs = {}      # Stores outputs from nodes
        self.graph = nx.DiGraph()
        self.env = jinja2.Environment()
        self.variables = {}    # For storing variables across nodes
        self.variable_producers = {}  # Maps variable names to node names
        self.client = OpenAI()
        logger.debug("GenFlow initialized.")

    def parse_yaml(self):
        """
        Parses the YAML data and constructs nodes.
        """
        data = self.yaml_data
        logger.debug("Parsing YAML data")
        for node_data in data.get('nodes', []):
            # Check if node is an llm_service node with tools
            if node_data.get('type') == 'llm_service' and 'tools' in node_data:
                tools = node_data['tools']
                for i, tool in enumerate(tools):
                    if tool.startswith('COMPOSIO.'):
                        # Ensure 'composio_handle_tool_call' exists in functions.py
                        self.ensure_composio_handle_tool_call_exists()
            node = Node(node_data)
            logger.debug(f"Adding node '{node.name}' of type '{node.type}'")
            if node.name in self.nodes:
                raise ValueError(f"Duplicate node name '{node.name}' found.")
            self.nodes[node.name] = node

            # Record variables produced by the node
            produced_variables = node.get_produced_variables()
            for var_name in produced_variables:
                if var_name in self.variable_producers:
                    raise ValueError(f"Variable '{var_name}' is set by multiple nodes.")
                self.variable_producers[var_name] = node.name

        # Build the execution graph
        self.build_graph()

    def ensure_composio_handle_tool_call_exists(self):
        """
        Ensures that 'composio_handle_tool_call' function exists in functions.py.
        If not, it appends the function definition to functions.py.
        """

        functions_py_path = 'functions.py'
        function_name = 'composio_handle_tool_call'

        # Check if functions.py exists
        if not os.path.exists(functions_py_path):
            # Create functions.py
            with open(functions_py_path, 'w') as f:
                f.write('')
            logger.info(f"Created '{functions_py_path}'")

        # Read functions.py content
        with open(functions_py_path, 'r') as f:
            functions_py_content = f.read()

        # Check if function is already defined
        if f'def {function_name}(' in functions_py_content:
            logger.info(f"Function '{function_name}' already exists in '{functions_py_path}'")
        else:
            # Append the function definition
            function_code = '''
    from composio_openai import ComposioToolSet, App, Action

    def composio_handle_tool_call(response):
        composio_toolset = ComposioToolSet()
        return composio_toolset.handle_tool_calls(response)
    '''
            # Dedent the function code
            function_code = textwrap.dedent(function_code)

            with open(functions_py_path, 'a') as f:
                f.write('\n' + function_code)  # Ensure there's a newline before appending
            logger.info(f"Appended function '{function_name}' to '{functions_py_path}'")

    def build_graph(self):
        """
        Builds the execution graph based on node dependencies.
        """
        logger.debug("Building execution graph")
        for node in self.nodes.values():
            self.graph.add_node(node.name)
            logger.debug(f"Added node '{node.name}' to graph")

        for node in self.nodes.values():
            dependencies = node.get_dependencies()
            logger.debug(f"Node '{node.name}' dependencies: {dependencies}")
            for dep in dependencies:
                if dep in self.nodes:
                    self.graph.add_edge(dep, node.name)
                    logger.debug(f"Added edge from node '{dep}' to '{node.name}'")
                elif dep in self.variable_producers:
                    producer_node = self.variable_producers[dep]
                    self.graph.add_edge(producer_node, node.name)
                    logger.debug(f"Added edge from variable producer '{producer_node}' to '{node.name}'")
                else:
                    logger.error(f"Node '{node.name}' depends on undefined node or variable '{dep}'")
                    raise ValueError(f"Node '{node.name}' depends on undefined node or variable '{dep}'.")
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            logger.error("The execution graph has cycles. Cannot proceed.")
            raise ValueError("The execution graph has cycles. Cannot proceed.")

    def run(self):
        """
        Executes the nodes in topological order.
        """
        try:
            execution_order = list(nx.topological_sort(self.graph))
            logger.info(f"Execution order: {execution_order}")
        except nx.NetworkXUnfeasible:
            logger.error("Graph has cycles, cannot proceed.")
            raise Exception("Graph has cycles, cannot proceed.")

        for node_name in execution_order:
            node = self.nodes[node_name]
            node.set_flow(self)  # Set the reference to the GenFlow instance

            # Render parameters
            try:
                logger.debug(f"Rendering parameters for node '{node_name}'")
                params = node.render_params(self.outputs, self.env)
                logger.debug(f"Parameters for node '{node_name}': {params}")
            except Exception as e:
                logger.error(f"Error rendering params for node '{node_name}': {e}")
                traceback.print_exc()
                continue  # Skip execution of this node

            # Execute node
            try:
                logger.info(f"Executing node '{node_name}'")
                outputs = node.execute(params)
                # Validate outputs
                if not isinstance(outputs, dict):
                    raise ValueError(f"Node '{node_name}' did not return a dictionary of outputs.")
                expected_outputs = set(node.outputs)
                actual_outputs = set(outputs.keys())
                if expected_outputs != actual_outputs:
                    raise ValueError(
                        f"Node '{node_name}' outputs {actual_outputs} do not match expected outputs {expected_outputs}."
                    )
                # Save outputs
                self.outputs[node_name] = outputs
                logger.info(f"Outputs of node '{node_name}': {outputs}")
            except Exception as e:
                logger.error(f"Error executing node '{node_name}': {e}")
                traceback.print_exc()
                continue  # Skip saving outputs for this node

class Node:
    """
    Represents an operation node in the execution graph.
    """

    def __init__(self, node_data):
        self.name = node_data['name']
        self.type = node_data['type']
        self.node_data = node_data
        self.outputs = node_data.get('outputs', [])
        self.flow = None  # Reference to the GenFlow instance
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    def set_flow(self, flow):
        """
        Sets the reference to the GenFlow instance.

        Args:
            flow (GenFlow): The GenFlow instance.
        """
        self.flow = flow

    def get_dependencies(self):
        """
        Extracts dependencies from the parameters and variables.

        Returns:
            set: A set of node or variable names that this node depends on.
        """
        dependencies = set()
        params = self.node_data.get('params', {})
        for value in params.values():
            dependencies.update(self.extract_dependencies_from_value(value))

        return dependencies

    @staticmethod
    def extract_dependencies_from_value(value):
        """
        Extracts node and variable names from templated values.

        Args:
            value: The parameter value to inspect.

        Returns:
            set: A set of node or variable names referenced in the value.
        """
        import re
        pattern = r"\{\{\s*([\w_\.]+)\s*\}\}"
        matches = re.findall(pattern, str(value))
        dependencies = set()
        for match in matches:
            var_parts = match.split('.')
            dependencies.add(var_parts[0])
        return dependencies

    def get_produced_variables(self):
        """
        Returns a list of variables produced by this node.
        """
        if self.type == 'set_variable':
            return [self.node_data['variable_name']]
        else:
            return []

    def render_params(self, outputs, env):
        """
        Renders the parameters with values from previous outputs.

        Args:
            outputs (dict): Outputs from previously executed nodes.
            env (jinja2.Environment): Jinja2 environment for templating.

        Returns:
            dict: Rendered parameters ready for execution.
        """
        params = self.node_data.get('params', {})
        # Build the context with outputs accessible by node name
        context = {}
        for node_name, node_outputs in outputs.items():
            context[node_name] = node_outputs
        # Add variables to context
        context.update(self.flow.variables)
        # Render each parameter individually
        rendered_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Check if the value is a simple variable reference
                import re
                simple_var_pattern = r"^\{\{\s*([\w_\.]+)\s*\}\}$"
                match = re.match(simple_var_pattern, value)
                if match:
                    var_name = match.group(1)
                    # Split var_name to access nested variables
                    var_parts = var_name.split('.')
                    obj = context
                    try:
                        for part in var_parts:
                            if isinstance(obj, dict):
                                obj = obj[part]
                            else:
                                obj = getattr(obj, part)
                        rendered_params[key] = obj
                    except (KeyError, AttributeError, TypeError) as e:
                        # Variable not found in context
                        raise ValueError(f"Variable '{var_name}' not found in context.") from e
                else:
                    # Use Jinja2 rendering for other cases
                    template = env.from_string(value)
                    rendered_value = template.render(**context)
                    rendered_params[key] = rendered_value
            else:
                # Non-string params are used as is
                rendered_params[key] = value
        return rendered_params

    def execute(self, params):
        """
        Executes the node based on its type and parameters.

        Args:
            params (dict): Parameters for the node execution.

        Returns:
            dict: Outputs from the node execution.
        """
        self.logger.info(f"Executing node of type '{self.type}' with params: {params}")
        if self.type == 'function_call':
            return self.execute_function_call(params)
        elif self.type == 'llm_service':
            return self.execute_llm_service(params)
        else:
            raise NotImplementedError(f"Type '{self.type}' not implemented for node '{self.name}'.")

    def execute_function_call(self, params):
        """
        Executes a function call node.

        Args:
            params (dict): Parameters for the function.

        Returns:
            dict: Dictionary of outputs from the function.
        """
        try:
            module = importlib.import_module('functions')
            func = getattr(module, self.node_data['function'])
        except ImportError as e:
            raise ImportError(f"Error importing module 'functions': {e}")
        except AttributeError as e:
            raise AttributeError(f"Function '{self.node_data['function']}' not found in 'functions' module.")

        # Check if function has proper docstrings and type annotations
        if not func.__doc__:
            raise ValueError(f"Function '{self.node_data['function']}' must have a docstring.")
        signature = inspect.signature(func)
        for param in signature.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError(f"Parameter '{param.name}' in function '{func.__name__}' lacks type annotation.")

        # Execute function with parameters
        result = func(**params)
        if not isinstance(result, dict):
            raise ValueError(f"Function '{self.node_data['function']}' should return a dictionary of outputs.")
        # Ensure result keys match outputs
        expected_outputs = set(self.outputs)
        actual_outputs = set(result.keys())
        if expected_outputs != actual_outputs:
            raise ValueError(
                f"Function outputs {actual_outputs} do not match expected outputs {expected_outputs}."
            )
        return result

    def execute_llm_service(self, params):
        """
        Executes an LLM service node.

        Args:
            params (dict): Parameters for the LLM service.

        Returns:
            dict: Dictionary of outputs from the LLM service.
        """
        service = self.node_data['service']
        if service == 'openai':
            return self.execute_openai_service(params)
        else:
            raise NotImplementedError(f"LLM service '{service}' not implemented for node '{self.name}'.")

    def execute_openai_service(self, params):
        """
        Executes an OpenAI LLM service call using the chat completions API.

        Args:
            params (dict): Parameters for the LLM service.

        Returns:
            dict: Dictionary of outputs from the LLM service.
        """
        import inspect
        model = self.node_data.get('model')
        tools = self.node_data.get('tools')
        structured_output_schema_name = self.node_data.get('structured_output_schema')

        # Check if both 'tools' and 'structured_output_schema' are defined
        if tools and structured_output_schema_name:
            raise ValueError(f"Node '{self.name}' cannot have both 'tools' and 'structured_output_schema' defined.")

        # Prepare messages
        messages = [
            {"role": "user", "content": params['prompt']}
        ]

        if tools:
            # Function calling
            # Prepare the function definitions
            function_definitions = []
            available_functions = {}
            composio_tools=[]
            for tool_name in tools:
                # Check if function exists in functions.py
                if tool_name.split('.')[0]=='COMPOSIO':
                    tool_name=tool_name.split('COMPOSIO.')[1]
                    composio_toolset = ComposioToolSet()
                    composio_tools.append(tool_name)
                    self.logger.debug(f"Registering COMPOSIO tool for openai call: {tool_name}")
                    try:
                        function_schema = composio_toolset.get_tools(actions=[tool_name])[0]
                    except Exception as e:
                        print(e)
                    try:
                        module = importlib.import_module('functions')
                        func = getattr(module, 'composio_handle_tool_call')
                    except ImportError as e:
                        raise ImportError(f"Error importing module 'functions': {e}")
                    except AttributeError as e:
                        raise AttributeError(f"Function 'composio_handle_tool_call' not found in 'functions' module.")

                else:
                    try:
                        module = importlib.import_module('functions')
                        func = getattr(module, tool_name)
                    except ImportError as e:
                        raise ImportError(f"Error importing module 'functions': {e}")
                    except AttributeError as e:
                        raise AttributeError(f"Function '{tool_name}' not found in 'functions' module.")

                    # Check if function has proper docstrings and type annotations
                    if not func.__doc__:
                        raise ValueError(f"Function '{tool_name}' must have a docstring.")
                    signature = inspect.signature(func)
                    for param in signature.parameters.values():
                        if param.annotation == inspect.Parameter.empty:
                            raise ValueError(f"Parameter '{param.name}' in function '{func.__name__}' lacks type annotation.")

                    # Get the function schema from the function object
                    function_schema = get_function_schema(func)

                # Build the function definition
                function_definitions.append(function_schema)

                # Add to available functions
                available_functions[tool_name] = func

            # Call OpenAI API
            response = self.flow.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=function_definitions,
                tool_choice="auto"  # or "auto"
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if the assistant wants to call a function
            if assistant_message.tool_calls:
                tool_calls = assistant_message.tool_calls
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call=available_functions[function_name]
                    if not function_to_call:
                        raise ValueError(f"Function '{function_name}' is not available.")
                    if function_name in composio_tools:
                        edited_response=response
                        edited_response.choices[0].message.tool_calls=[tool_call]
                        function_response = composio_toolset.handle_tool_calls(edited_response)
                        edited_response.choices[0].message.tool_calls = tool_calls
                        self.logger.debug(f"Executing COMPOSIO tool: {function_name}")
                    else:
                        function_args = tool_call.function.arguments

                        # Parse the arguments
                        try:
                            arguments = json.loads(function_args)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Failed to parse function arguments: {e}")

                        # Get the function signature and call the function with given arguments
                        try:
                            sig = inspect.signature(function_to_call)
                            call_args = {}
                            for k, v in sig.parameters.items():
                                if k in arguments:
                                    call_args[k] = arguments[k]
                                elif v.default != inspect.Parameter.empty:
                                    call_args[k] = v.default
                                else:
                                    raise ValueError(f"Missing required argument '{k}' for function '{function_name}'.")

                            self.logger.debug(f"Calling function '{function_name}' with arguments {call_args}")
                            function_response = function_to_call(**call_args)
                            self.logger.debug(f"Function '{function_name}' returned: {function_response}")
                        except Exception as e:
                            raise Exception(f"Error executing function '{function_name}': {e}")

                    # Append the function's response to messages
                    tool_message = {
                        "tool_call_id":tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response)
                    }
                    print(f'tool message{tool_message}')
                    messages.append(tool_message)

                # Call the model again to get the final response
                second_response = self.flow.client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                assistant_final_message = second_response.choices[0].message
                result = assistant_final_message.content

                if len(self.outputs) != 1:
                    raise ValueError(
                        f"Node '{self.name}' expects {len(self.outputs)} outputs, but OpenAI service returned 1."
                    )
                return {self.outputs[0]: result}

            else:
                # The assistant did not call any function
                result = assistant_message.content
                if len(self.outputs) != 1:
                    raise ValueError(
                        f"Node '{self.name}' expects {len(self.outputs)} outputs, but OpenAI service returned 1."
                    )
                return {self.outputs[0]: result}

        elif structured_output_schema_name:
            # Structured outputs
            # Get the schema from structured_output_schema.py
            try:
                module = importlib.import_module('structured_output_schema')
                schema_class = getattr(module, structured_output_schema_name)
            except ImportError as e:
                raise ImportError(f"Error importing module 'structured_output_schema': {e}")
            except AttributeError as e:
                raise AttributeError(f"Schema '{structured_output_schema_name}' not found in 'structured_output_schema' module.")

            # Call OpenAI API with response_format
            try:
                response = self.flow.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema_class,
                )

                # Get the parsed result
                assistant_message = response.choices[0].message
                if assistant_message.refusal:
                    raise Exception(f"OpenAI refusal for structured outputs on node '{self.name}'. Refusal:{assistant_message.refusal}")
                else:
                    result = assistant_message.parsed
            except Exception as e:
                if type(e) == openai.LengthFinishReasonError:
                    raise Exception(f"Too many tokens were passed to openAi during structured output generation on node {self.name}")
                else:
                    raise Exception(f"Failed to parse structured output for node '{self.name}'.")
            if result is None:
                raise ValueError(f"Failed to parse structured output for node '{self.name}'. result coming as None")

            if len(self.outputs) != 1:
                raise ValueError(
                    f"Node '{self.name}' expects {len(self.outputs)} outputs, but OpenAI service returned 1."
                )
            return {self.outputs[0]: result}

        else:
            # Simple prompt completion
            response = self.flow.client.chat.completions.create(
                model=model,
                messages=messages
            )
            assistant_message = response.choices[0].message
            result = assistant_message.content
            if len(self.outputs) != 1:
                raise ValueError(
                    f"Node '{self.name}' expects {len(self.outputs)} outputs, but OpenAI service returned 1."
                )
            return {self.outputs[0]: result}

def get_function_schema(func):
    """
    Retrieves the function schema for a function object by inspecting its signature and docstring.

    Args:
        func (function): The function object.

    Returns:
        dict: The function definition, including name, description, parameters.
    """
    import inspect

    function_name = func.__name__
    docstring = inspect.getdoc(func) or ""
    signature = inspect.signature(func)

    # Build parameters schema
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    for param_name, param in signature.parameters.items():
        param_type = param.annotation
        if param_type == inspect.Parameter.empty:
            raise ValueError(f"Parameter '{param_name}' in function '{function_name}' is missing type annotation.")

        # Handle typing.Optional and typing.Union
        if getattr(param_type, '__origin__', None) is Union:
            # Get the non-None type
            param_type = [t for t in param_type.__args__ if t is not type(None)][0]

        if hasattr(param_type, '__origin__') and param_type.__origin__ == list:
            item_type = param_type.__args__[0]
            item_type_name = type_mapping.get(item_type, "string")
            param_schema = {
                "type": "array",
                "items": {"type": item_type_name}
            }
        else:
            param_type_name = type_mapping.get(param_type, "string")
            param_schema = {
                "type": param_type_name
            }

        # Optionally, extract parameter description from docstring (not implemented here)

        parameters["properties"][param_name] = param_schema

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    function_def = {"type": "function",
                    "function": {
                        "name": function_name,
                        "description": docstring,
                        "parameters": parameters
                                }
                    }

    return function_def
