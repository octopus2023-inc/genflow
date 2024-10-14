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
from typing import Union, Set
from composio_openai import ComposioToolSet, App, Action
from langchain_core.utils.function_calling import convert_to_openai_function
import textwrap
# Import YmlCompose
from yml_compose import YmlCompose

# Load environment variables
load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)

class GenFlow:
    """
    Class to parse YAML data, construct an execution graph, and execute nodes.
    """

    def __init__(self, yaml_file):
        self.yaml_file = yaml_file  # Path to the main YAML file
        self.yaml_data = self.load_yaml_file(yaml_file)
        self.nodes = {}        # Maps node names to Node instances
        self.outputs = {}      # Stores outputs from nodes
        self.graph = nx.DiGraph()
        self.env = jinja2.Environment()
        self.client = OpenAI()
        logger.debug("GenFlow initialized.")

    def load_yaml_file(self, yaml_file):
        """
        Loads the YAML data from a file.

        Args:
            yaml_file (str): Path to the YAML file.

        Returns:
            dict: The YAML data.
        """
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"YAML file '{yaml_file}' does not exist.")

        with open(yaml_file, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file '{yaml_file}': {e}")
        return data

    def has_yml_flow_nodes(self, yaml_data):
        """
        Recursively checks if the YAML data contains any 'yml_flow' nodes.

        Args:
            yaml_data (dict): The YAML data to check.

        Returns:
            bool: True if 'yml_flow' nodes are present, False otherwise.
        """
        nodes = yaml_data.get('nodes', [])
        for node_data in nodes:
            node_type = node_data.get('type')
            if node_type == 'yml_flow':
                return True
            elif node_type in ['function_call', 'llm_service']:
                continue
            else:
                # If node_type is not recognized, raise an error
                raise ValueError(f"Unknown node type '{node_type}' in node '{node_data.get('name')}'.")
        return False

    def parse_yaml(self):
        """
        Parses the YAML data and constructs nodes.
        """
        # Validate the YAML data before parsing

        self.validate_yaml(self.yaml_data, self.yaml_file)

        # Check if there are any 'yml_flow' nodes
        if self.has_yml_flow_nodes(self.yaml_data):
            logger.info("Detected 'yml_flow' nodes. Composing YAML files using YmlCompose.")
            # Compose the YAML data into a single flow
            yml_composer = YmlCompose(self.yaml_file)
            self.composed_yaml_data = yml_composer.compose()
            data = self.composed_yaml_data
        else:
            data = self.yaml_data
            logger.debug("No 'yml_flow' nodes detected. Proceeding with original YAML data.")

        logger.debug("Parsing YAML data")
        for node_data in data.get('nodes', []):
            node = Node(node_data)
            logger.debug(f"Adding node '{node.name}' of type '{node.type}'")
            if node.name in self.nodes:
                raise ValueError(f"Duplicate node name '{node.name}' found.")
            self.nodes[node.name] = node

        # Build the execution graph
        self.build_graph()

    def validate_yaml(self, yaml_data, yaml_file, parent_node_names=None, visited_files=None, parent_params=None):
        """
        Validates the YAML data and any associated sub-flows for consistency and errors.

        Args:
            yaml_data (dict): The YAML data to validate.
            yaml_file (str): Path to the YAML file being validated.
            parent_node_names (set): Set of node names from the parent flow to detect duplicates.
            visited_files (set): Set of visited YAML file paths to prevent circular references.
            parent_params (set): Set of parameter names passed from the parent flow.
        """
        if parent_node_names is None:
            parent_node_names = set()
        if visited_files is None:
            visited_files = set()
        if parent_params is None:
            parent_params = set()

        logger.debug(f"Validating YAML file '{yaml_file}'")

        # Get absolute path of the yaml_file
        yaml_file_abs = os.path.abspath(yaml_file)

        # Prevent circular references
        if yaml_file_abs in visited_files:
            raise ValueError(f"Circular reference detected in YAML file '{yaml_file}'.")
        visited_files.add(yaml_file_abs)

        # Check if yaml_data is a dictionary
        if not isinstance(yaml_data, dict):
            raise ValueError(f"YAML file '{yaml_file}' must contain a dictionary at the top level.")

        # Check if 'nodes' key exists
        if 'nodes' not in yaml_data:
            raise ValueError(f"YAML file '{yaml_file}' must contain a 'nodes' key.")

        nodes = yaml_data['nodes']

        # Check if 'nodes' is a list
        if not isinstance(nodes, list):
            raise ValueError(f"The 'nodes' key in YAML file '{yaml_file}' must be a list.")

        node_names = set()

        for node_data in nodes:
            # Check if node_data is a dictionary
            if not isinstance(node_data, dict):
                raise ValueError(f"A node in YAML file '{yaml_file}' is not a dictionary.")

            node_name = node_data.get('name')
            node_type = node_data.get('type')
            params = node_data.get('params', {})
            outputs = node_data.get('outputs', [])

            # Check for 'name' and 'type'
            if not node_name:
                raise ValueError(f"A node in YAML file '{yaml_file}' is missing the 'name' field.")
            if not node_type:
                raise ValueError(f"Node '{node_name}' in YAML file '{yaml_file}' is missing the 'type' field.")

            # Check for duplicate node names
            if node_name in node_names or node_name in parent_node_names:
                raise ValueError(f"Duplicate node name '{node_name}' found in YAML file '{yaml_file}'.")
            node_names.add(node_name)

            # Check for valid node types
            valid_node_types = {'function_call', 'llm_service', 'yml_flow'}
            if node_type not in valid_node_types:
                raise ValueError(f"Invalid node type '{node_type}' in node '{node_name}' in YAML file '{yaml_file}'.")

            # Validate 'function_call' nodes
            if node_type == 'function_call':
                function_name = node_data.get('function')
                if not function_name:
                    raise ValueError(f"Node '{node_name}' of type 'function_call' must have a 'function' field in YAML file '{yaml_file}'.")
                try:
                    module = importlib.import_module('functions')
                    func = getattr(module, function_name)
                except (ImportError, AttributeError):
                    raise ValueError(f"Function '{function_name}' not found in 'functions.py' for node '{node_name}' in YAML file '{yaml_file}'.")

            # Validate 'llm_service' nodes
            if node_type == 'llm_service':
                service = node_data.get('service')
                if not service:
                    raise ValueError(f"Node '{node_name}' of type 'llm_service' must have a 'service' field in YAML file '{yaml_file}'.")
                if 'tools' in node_data:
                    tools = node_data['tools']
                    for i, tool in enumerate(tools):
                        # Validate if COMPOSIO tools are valid
                        if tool.startswith('COMPOSIO.'):
                            tool_name = tool.split('COMPOSIO.')[1]
                            composio_toolset = ComposioToolSet()
                            try:
                                _ = composio_toolset.get_tools(actions=[tool_name])[0]
                            except Exception as e:
                                print(f"COMPOSIO tool {tool_name} not valid. Please check COMPOSIO documentation for valid tools. \n COMPOSIO error message: {e}")
                        # Validate if LANGCHAIN tools are valid
                        elif tool.startswith('LANGCHAIN.'):
                            tool_name = tool_name.split('LANGCHAIN.')[1]
                            module = importlib.import_module('langchain_community.tools')
                            try:
                                _ = getattr(module, tool_name)
                            except:
                                print(f"Unable to import {tool_name} from langchain_community.tools. Please check Langchain's documentation.\n Langchain error message: {e}")
            # Validate 'yml_flow' nodes
            if node_type == 'yml_flow':
                yml_file_sub = node_data.get('yml_file')
                if not yml_file_sub:
                    raise ValueError(f"Node '{node_name}' of type 'yml_flow' must have a 'yml_file' field in YAML file '{yaml_file}'.")
                # Resolve the subflow yaml file path relative to the current yaml file
                yml_file_sub_path = os.path.join(os.path.dirname(yaml_file), yml_file_sub)
                # Check if the YAML file exists
                if not os.path.exists(yml_file_sub_path):
                    raise FileNotFoundError(f"YAML file '{yml_file_sub_path}' specified in node '{node_name}' does not exist.")
                # Load the sub-flow YAML file
                sub_yaml_data = self.load_yaml_file(yml_file_sub_path)
                # Get the 'params' passed to the sub-flow
                sub_flow_params = node_data.get('params', {})
                sub_flow_param_names = set(sub_flow_params.keys())

                # Collect all parameter names used in the sub-flow
                used_params_in_sub_flow = self.collect_used_params(sub_yaml_data)

                # Check that each parameter passed in is used in the sub-flow
                unused_params = sub_flow_param_names - used_params_in_sub_flow
                if unused_params:
                    raise ValueError(f"Parameters {unused_params} passed to sub-flow '{yml_file_sub_path}' are not used in the sub-flow.")

                # Collect all parameters used in the sub-flow but not defined
                sub_flow_parent_params = parent_params.union(sub_flow_param_names)
                undefined_params = used_params_in_sub_flow - sub_flow_param_names - sub_flow_parent_params
                if undefined_params:
                    raise ValueError(f"Sub-flow '{yml_file_sub_path}' uses undefined parameters: {undefined_params}")

                # Pass down parent parameters to the sub-flow
                new_parent_params = sub_flow_parent_params
                # Recursively validate the sub-flow
                self.validate_yaml(sub_yaml_data, yml_file_sub_path, parent_node_names.union(node_names), visited_files, new_parent_params)

                # Check that outputs specified are produced by the sub-flow
                sub_flow_node_outputs = self.collect_sub_flow_outputs(sub_yaml_data)
                specified_outputs = node_data.get('outputs', [])
                missing_outputs = set(specified_outputs) - sub_flow_node_outputs
                if missing_outputs:
                    raise ValueError(f"Outputs {missing_outputs} specified in 'outputs' of 'yml_flow' node '{node_name}' are not produced by the sub-flow '{yml_file_sub_path}'.")

            # Validate parameters and outputs
            if not isinstance(params, dict):
                raise ValueError(f"Parameters in node '{node_name}' must be a dictionary in YAML file '{yaml_file}'.")
            if not isinstance(outputs, list):
                raise ValueError(f"Outputs in node '{node_name}' must be a list in YAML file '{yaml_file}'.")
            # Check for duplicate parameter names
            param_names = set(params.keys())
            if len(param_names) != len(params):
                raise ValueError(f"Duplicate parameter names found in node '{node_name}' in YAML file '{yaml_file}'.")
            # Check for duplicate output names
            if len(set(outputs)) != len(outputs):
                raise ValueError(f"Duplicate output names found in node '{node_name}' in YAML file '{yaml_file}'.")

        # After collecting all node names, check for references to undefined nodes
        for node_data in nodes:
            node_name = node_data['name']
            params = node_data.get('params', {})
            # Collect all referenced nodes in parameters
            referenced_nodes = self.collect_referenced_nodes(params)
            undefined_nodes = referenced_nodes - node_names - parent_node_names
            if undefined_nodes:
                raise ValueError(f"Node '{node_name}' in YAML file '{yaml_file}' references undefined nodes: {undefined_nodes}")

        # Build a temporary graph to check for cycles
        temp_graph = nx.DiGraph()
        for node_data in nodes:
            node_name = node_data['name']
            temp_graph.add_node(node_name)
        for node_data in nodes:
            node_name = node_data['name']
            params = node_data.get('params', {})
            referenced_nodes = self.collect_referenced_nodes(params)
            for ref_node in referenced_nodes:
                if ref_node in node_names:
                    temp_graph.add_edge(ref_node, node_name)
        if not nx.is_directed_acyclic_graph(temp_graph):
            raise ValueError(f"The workflow graph in YAML file '{yaml_file}' has cycles.")

        logger.info(f"YAML file {yaml_file} passed all validation checks.")

    def collect_used_params(self, yaml_data) -> Set[str]:
        """
        Collects all parameter names used in the YAML data.

        Args:
            yaml_data (dict): The YAML data.

        Returns:
            Set[str]: A set of parameter names used in the YAML data.
        """
        used_params = set()

        nodes = yaml_data.get('nodes', [])
        for node_data in nodes:
            params = node_data.get('params', {})
            used_params.update(self.collect_referenced_params(params))

        return used_params

    def collect_referenced_params(self, params) -> Set[str]:
        """
        Collects all parameter names used in the parameters.

        Args:
            params (dict): Parameters dictionary.

        Returns:
            Set[str]: A set of parameter names used.
        """
        referenced_params = set()

        def traverse(value):
            if isinstance(value, str):
                # Extract parameter references from templated strings
                import re
                pattern = r"\{\{\s*([\w_]+)\s*\}\}"
                matches = re.findall(pattern, value)
                referenced_params.update(matches)
            elif isinstance(value, dict):
                for v in value.values():
                    traverse(v)
            elif isinstance(value, list):
                for item in value:
                    traverse(item)

        traverse(params)
        return referenced_params

    def collect_sub_flow_outputs(self, yaml_data) -> Set[str]:
        """
        Collects all outputs produced by nodes in the sub-flow.

        Args:
            yaml_data (dict): The YAML data.

        Returns:
            Set[str]: A set of output names produced in the sub-flow.
        """
        outputs = set()

        nodes = yaml_data.get('nodes', [])
        for node_data in nodes:
            node_outputs = node_data.get('outputs', [])
            outputs.update(node_outputs)

        return outputs

    def collect_referenced_nodes(self, params) -> Set[str]:
        """
        Collects all node names referenced in the parameters.

        Args:
            params (dict): Parameters dictionary.

        Returns:
            Set[str]: A set of referenced node names.
        """
        referenced_nodes = set()

        def traverse(value):
            if isinstance(value, str):
                # Extract node references from templated strings
                import re
                pattern = r"\{\{\s*([\w_]+)\."
                matches = re.findall(pattern, value)
                referenced_nodes.update(matches)
            elif isinstance(value, dict):
                for v in value.values():
                    traverse(v)
            elif isinstance(value, list):
                for item in value:
                    traverse(item)

        traverse(params)
        return referenced_nodes

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
            langchain_tools=[]
            for tool_name in tools:
                # Register COMPOSIO tools
                if tool_name.split('.')[0]=='COMPOSIO':
                    tool_name=tool_name.split('COMPOSIO.')[1]
                    composio_toolset = ComposioToolSet()
                    composio_tools.append(tool_name)
                    self.logger.info(f"Registering COMPOSIO tool for openai call: {tool_name}")
                    try:
                        function_schema = composio_toolset.get_tools(actions=[tool_name])[0]
                    except Exception as e:
                        print(e)
                    try:
                        func = lambda response: ComposioToolSet().handle_tool_calls(response)
                    except Exception as e:
                            print(e)
                #Register LANGCHAIN tools
                elif tool_name.split('.')[0]=='LANGCHAIN':
                    tool_name=tool_name.split('LANGCHAIN.')[1]
                    module = importlib.import_module('langchain_community.tools')
                    try:
                        langchain_tool = getattr(module, tool_name)
                    except Exception as e:
                            print(f"Unable to import {tool_name} from langchain_community.tools. Please check Langchain's documentation.\n Langchain error message: {e}")
                    self.logger.info(f"Registering LANGCHAIN tool for openai call: {tool_name}")
                    langchain_tool_instance=langchain_tool()
                    langchain_openai_function_format=convert_to_openai_function(langchain_tool_instance)
                    function_schema={'type':'function','function':langchain_openai_function_format}
                    tool_name=function_schema['function']['name']
                    langchain_tools.append(tool_name)
                    self.logger.info(f"Langchain function schema: {function_schema}")
                    run_langchain_tool=lambda x: langchain_tool_instance.invoke(x)
                    func=run_langchain_tool
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
                    #Handle COMPOSIO function execution
                    if function_name in composio_tools:
                        function_response=self.handle_composio_tool_execution(response,tool_call,tool_calls,function_name,function_to_call)
                    #Handle LANGCHAIN function execution
                    elif function_name in langchain_tools:
                        function_response=self.handle_langchain_tool_execution(tool_call,function_name,function_to_call)
                    #Handle custom function execution
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
                            self.logger.info(f"Calling function '{function_name}' with arguments {call_args}")
                            function_response = function_to_call(**call_args)
                            self.logger.info(f"Function '{function_name}' returned: {function_response}")
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

    def handle_composio_tool_execution(self,tool_call_response,tool_call,tool_calls,function_name,function_to_call):
        """
        Executes COMPOSIO tools by parsing LLM response.

        Args:
            tool_call_response: LLM response object containing tool call arguments
            tool_call: COMPOSIO tool call object from tool_call_response
            tool_calls: All tool calls from LLM response
            function_name: registered name COMPOSIO function to be executed
            function_to_call: COMPOSIO function to be executed

        Returns:
            function_response: Output of COMPOSIO function
        """

        edited_response = tool_call_response
        edited_response.choices[0].message.tool_calls = [tool_call]
        self.logger.info(f"Executing COMPOSIO tool: {function_name}")
        function_response = function_to_call(edited_response)
        edited_response.choices[0].message.tool_calls = tool_calls

        return function_response

    def handle_langchain_tool_execution(self,tool_call,function_name,function_to_call):
        """
        Executes LANGCHAIN tools by parsing LLM response.

            Args:
                tool_call: LANGCHAIN tool call object from tool_call_response
                function_name: registered name of LANGCHAIN function to be executed
                function_to_call: LANGCHAIN function to be executed

            Returns:
                function_response: Output of LANGCHAIN function
        """
        try:
            function_args = tool_call.function.arguments
            arguments = json.loads(function_args)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse function arguments: {e}")
        self.logger.info(f"Executing langchain tool '{function_name}'.")
        try:
            self.logger.info(f"Calling function '{function_name}' with arguments {arguments}")
            function_response = function_to_call(arguments)
            self.logger.info(f"Langchain function '{function_name}' returned: {function_response}")
            return function_response
        except Exception as e:
            raise Exception(f"Error executing langchain function '{function_name}': {e}")

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
