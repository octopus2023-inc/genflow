# yml_compose.py

import yaml
import os
import logging
import re

# Module-level logger
print('name is')
print(__name__)
logger = logging.getLogger(__name__)
print(logger)

class YmlCompose:
    """
    Class to compose YAML files by resolving references to other YAML files.
    """

    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.combined_data = {'nodes': []}
        self.node_name_set = set()  # To keep track of all node names to avoid duplicates.

    def compose(self, save_combined_yaml=False, output_file='combined.yaml'):
        """
        Starts the composition process and returns the combined YAML data.

        Args:
            save_combined_yaml (bool): If True, saves the combined YAML data to a file.
            output_file (str): The filename to save the combined YAML data.
        """
        logger.info(f"Starting composition with root YAML file '{self.yaml_file}'")
        self._process_yaml_file(self.yaml_file)
        logger.info("Composition completed")

        if save_combined_yaml:
            with open(output_file, 'w') as f:
                yaml.dump(self.combined_data, f)
            logger.info(f"Combined YAML saved to '{output_file}'")

        return self.combined_data

    def _process_yaml_file(self, yaml_file, parent_prefix=''):
        """
        Recursively processes a YAML file, resolving any sub-flow references.

        Args:
            yaml_file (str): Path to the YAML file.
            parent_prefix (str): Prefix to be added to node names for uniqueness.
        """
        logger.info(f"Processing YAML file '{yaml_file}' with prefix '{parent_prefix}'")
        if not os.path.exists(yaml_file):
            logger.error(f"YAML file '{yaml_file}' does not exist.")
            raise FileNotFoundError(f"YAML file '{yaml_file}' does not exist.")

        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if 'nodes' not in data:
            logger.error(f"YAML file '{yaml_file}' must contain 'nodes'.")
            raise ValueError(f"YAML file '{yaml_file}' must contain 'nodes'.")

        nodes = data['nodes']
        i = 0
        while i < len(nodes):
            node_data = nodes[i]
            node_type = node_data.get('type')
            node_name = node_data.get('name')
            if not node_name:
                logger.error("A node without a 'name' was found.")
                raise ValueError("A node without a 'name' was found.")

            # Create a unique node name by prefixing
            unique_node_name = parent_prefix + node_name
            logger.info(f"Processing node '{node_name}' (unique name: '{unique_node_name}') of type '{node_type}'")
            if unique_node_name in self.node_name_set:
                logger.error(f"Duplicate node name '{unique_node_name}' detected.")
                raise ValueError(f"Duplicate node name '{unique_node_name}' detected.")

            self.node_name_set.add(unique_node_name)

            if node_type == 'yml_flow':
                # Handle sub-flow
                sub_flow_file = node_data.get('yml_file')
                if not sub_flow_file:
                    logger.error(f"Node '{unique_node_name}' of type 'yml_flow' must have a 'yml_file' field.")
                    raise ValueError(f"Node '{unique_node_name}' of type 'yml_flow' must have a 'yml_file' field.")

                # Compute the absolute path of the sub-flow file
                sub_flow_file_path = os.path.join(os.path.dirname(yaml_file), sub_flow_file)
                logger.info(f"Sub-flow file for node '{unique_node_name}' is '{sub_flow_file_path}'")

                # Get parameters and outputs
                sub_flow_params = node_data.get('params', {})
                sub_flow_outputs = node_data.get('outputs', [])

                # Remove the yml_flow node from the list
                nodes.pop(i)

                # Process sub-flow
                sub_flow_nodes = self._load_yaml_file(sub_flow_file_path)
                # Adjust sub-flow nodes
                adjusted_sub_flow_nodes = self._adjust_sub_flow_nodes(
                    sub_flow_nodes, unique_node_name + '__', sub_flow_params)

                # Add adjusted sub-flow nodes to combined data
                self.combined_data['nodes'].extend(adjusted_sub_flow_nodes)

                # Build output mapping for outputs specified in yml_flow node
                output_mapping = {}
                for output in sub_flow_outputs:
                    # Find the node in sub-flow that produces this output
                    found = False
                    for node in adjusted_sub_flow_nodes:
                        if output in node.get('outputs', []):
                            output_mapping[unique_node_name + '.' + output] = node['name'] + '.' + output
                            found = True
                            break
                    if not found:
                        logger.error(f"Output '{output}' specified in 'yml_flow' node '{unique_node_name}' not produced in sub-flow.")
                        raise ValueError(f"Output '{output}' specified in 'yml_flow' node '{unique_node_name}' not produced in sub-flow.")

                # Adjust references in remaining nodes
                self._adjust_references_in_nodes(nodes[i:], node_name, output_mapping)

                # Do not increment i, as we removed the current node
                continue
            else:
                # Copy node data and adjust the name
                node_data = node_data.copy()
                node_data['name'] = unique_node_name

                # Adjust parameter references to account for prefixed node names
                node_data['params'] = self._adjust_params(node_data.get('params', {}), parent_prefix, set())
                logger.debug(f"Adjusted parameters for node '{unique_node_name}': {node_data['params']}")

                # Add the node to the combined data
                self.combined_data['nodes'].append(node_data)
                logger.info(f"Added node '{unique_node_name}' to combined data")

                i +=1  # Move to the next node

    def _load_yaml_file(self, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        if 'nodes' not in data:
            logger.error(f"YAML file '{yaml_file}' must contain 'nodes'.")
            raise ValueError(f"YAML file '{yaml_file}' must contain 'nodes'.")
        return data['nodes']

    def _adjust_sub_flow_nodes(self, nodes, sub_flow_prefix, sub_flow_params):
        adjusted_nodes = []
        sub_flow_node_names = set()
        for node_data in nodes:
            node_type = node_data.get('type')
            node_name = node_data.get('name')
            if not node_name:
                logger.error("A node without a 'name' was found in sub-flow.")
                raise ValueError("A node without a 'name' was found in sub-flow.")

            sub_flow_node_names.add(node_name)  # Collect node names in sub-flow

            prefixed_node_name = sub_flow_prefix + node_name
            if prefixed_node_name in self.node_name_set:
                logger.error(f"Duplicate node name '{prefixed_node_name}' detected in sub-flow.")
                raise ValueError(f"Duplicate node name '{prefixed_node_name}' detected in sub-flow.")

            self.node_name_set.add(prefixed_node_name)

            node_data = node_data.copy()
            node_data['name'] = prefixed_node_name

            # Replace parameter references in node params with actual sub_flow_params
            node_data['params'] = self._replace_params(node_data.get('params', {}), sub_flow_params)

            # Adjust any dependencies (in params) to include the sub_flow_prefix
            node_data['params'] = self._adjust_params(node_data.get('params', {}), sub_flow_prefix, sub_flow_node_names)

            adjusted_nodes.append(node_data)

        return adjusted_nodes

    def _replace_params(self, params, sub_flow_params):
        """
        Replaces parameter placeholders in params with the values passed in from the parent flow.

        Args:
            params (dict): The parameters to adjust.
            sub_flow_params (dict): Parameters passed from the parent flow to the sub-flow.

        Returns:
            dict: The adjusted parameters.
        """
        adjusted_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                for param_name, param_value in sub_flow_params.items():
                    pattern = r"\{\{\s*" + re.escape(param_name) + r"\s*\}\}"
                    value = re.sub(pattern, str(param_value), value)
                adjusted_params[key] = value
            else:
                adjusted_params[key] = value
        return adjusted_params

    def _adjust_params(self, params, prefix, sub_flow_node_names):
        """
        Adjusts parameter references to use prefixed node names.

        Args:
            params (dict): The parameters to adjust.
            prefix (str): The prefix for node names.
            sub_flow_node_names (set): Set of node names within the sub-flow.

        Returns:
            dict: The adjusted parameters.
        """
        adjusted_params = {}
        pattern = r"(\{\{\s*)([\w_\.]+)(\s*\}\})"

        for key, value in params.items():
            if isinstance(value, str):
                def replace_func(match):
                    full_match = match.group(0)
                    prefix_match = match.group(1)
                    reference = match.group(2)
                    suffix_match = match.group(3)

                    # Split the reference into parts
                    parts = reference.split('.')
                    first_part = parts[0]
                    rest_parts = parts[1:]

                    # Only adjust if the first part is in sub_flow_node_names
                    if first_part in sub_flow_node_names:
                        adjusted_first_part = prefix + first_part
                    else:
                        adjusted_first_part = first_part

                    adjusted_reference = '.'.join([adjusted_first_part] + rest_parts)

                    return prefix_match + adjusted_reference + suffix_match

                adjusted_value = re.sub(pattern, replace_func, value)
                adjusted_params[key] = adjusted_value
            else:
                adjusted_params[key] = value
        return adjusted_params

    def _adjust_references_in_nodes(self, nodes, yml_flow_node_name, output_mapping):
        """
        Adjusts references in the nodes to replace references to outputs from the yml_flow node.

        Args:
            nodes (list): The list of nodes to adjust.
            yml_flow_node_name (str): The name of the yml_flow node.
            output_mapping (dict): Mapping from yml_flow outputs to sub-flow node outputs.
        """
        for node_data in nodes:
            params = node_data.get('params', {})
            adjusted_params = {}
            for key, value in params.items():
                if isinstance(value, str):
                    for yml_flow_output, sub_flow_output in output_mapping.items():
                        # Replace references like {{ yml_flow_node_name.output_name }}
                        pattern = r"\{\{\s*" + re.escape(yml_flow_node_name + '.' + yml_flow_output.split('.')[-1]) + r"\s*\}\}"
                        value = re.sub(pattern, '{{ ' + sub_flow_output + ' }}', value)
                adjusted_params[key] = value
            node_data['params'] = adjusted_params
