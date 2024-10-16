# main.py

import logging
import traceback


# Set up logging configuration before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../app.log", mode='w'),
        logging.StreamHandler()
    ]
)

#logger = logging.getLogger(__name__)

from gensphere.genflow import GenFlow
logger = logging.getLogger('gensphere')

from gensphere.yaml_utils import YmlCompose
import pprint
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    # Compose the YAML files into one combined data structure
    load_dotenv()
    try:
        composer = YmlCompose('sample.yaml','functions','structured_output_schema')
        combined_yaml_data = composer.compose(save_combined_yaml=True, output_file='combined.yaml')

        # Log the combined YAML data
        logger.info("Combined YAML Data:")
        pprint.pprint(combined_yaml_data)

        # Initialize gensphere with the combined YAML data
        flow = GenFlow('combined.yaml','functions','structured_output_schema')
        flow.parse_yaml()

        # Log the nodes parsed
        logger.info("Nodes parsed:")
        for node_name in flow.nodes:
            logger.info(f"- {node_name}")

        # Run the flow and log outputs
        flow.run()
        logger.info("Flow execution completed. Outputs:")
        for node_name, outputs in flow.outputs.items():
            logger.info(f"Node '{node_name}' outputs: {outputs}")

    except Exception as e:
        logger.error(f"An error occurred during flow execution: {e}")
        logger.error(traceback.format_exc())
