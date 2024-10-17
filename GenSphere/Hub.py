# hub.py

import requests
import os
import logging
from typing import Optional
import yaml
from Gensphere.YmlUtils import validate_yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hub")


class Hub:
    def __init__(
            self,
            yaml_file: Optional[str] = None,
            functions_file: Optional[str] = None,
            schema_file: Optional[str] = None,
            api_base_url: str = "http://127.0.0.1:8001"
    ):
        self.yaml_file = yaml_file
        self.functions_file = functions_file
        self.schema_file = schema_file
        self.api_base_url = api_base_url

    def validate_before_push(self):
        if not self.yaml_file:
            logger.error("No yaml_file provided for validation.")
            raise ValueError("You must provide a yaml_file to validate.")

        # Read and validate the YAML file
        try:
            with open(self.yaml_file, 'r') as f:
                yaml_content_str = f.read()
            yaml_content = yaml.safe_load(yaml_content_str)
            validated, error_msgs, node_outputs = validate_yaml(self.yaml_file)
            if validated:
                logger.info(f"yaml file {self.yaml_file} passed all consistency checks")
            else:
                raise Exception(
                    f"yaml file {self.yaml_file} didn't pass consistency checks.\nError messages: {error_msgs}")
            logger.info("yaml_file is valid.")
        except yaml.YAMLError as exc:
            logger.error(f"YAML parsing error: {str(exc)}")
            raise ValueError("Invalid YAML file.")
        except Exception as e:
            logger.error(f"Error reading yaml_file: {str(e)}")
            raise

    def push(self, push_name: Optional[str] = None):
        if not self.yaml_file:
            logger.error("No yaml_file provided for push.")
            raise ValueError("You must provide at least a yaml_file to push.")

        self.validate_before_push()

        files = {}
        file_handles = []
        try:
            # Open files and add to files dictionary
            f_yaml = open(self.yaml_file, 'rb')
            files['yaml_file'] = ('yaml_file.yaml', f_yaml, 'application/x-yaml')
            file_handles.append(f_yaml)

            if self.functions_file:
                if not self.functions_file.endswith('.py'):
                    logger.error("functions_file must be a .py file.")
                    raise ValueError("functions_file must be a .py file.")
                f_functions = open(self.functions_file, 'rb')
                files['functions_file'] = ('functions.py', f_functions, 'text/x-python')
                file_handles.append(f_functions)

            if self.schema_file:
                if not self.schema_file.endswith('.py'):
                    logger.error("schema_file must be a .py file.")
                    raise ValueError("schema_file must be a .py file.")
                f_schema = open(self.schema_file, 'rb')
                files['schema_file'] = ('structured_output_schema.py', f_schema, 'text/x-python')
                file_handles.append(f_schema)

            data = {}
            if push_name:
                data['push_name'] = push_name

            logger.info("Sending push request to the platform.")
            response = requests.post(f"{self.api_base_url}/push", files=files, data=data)
            logger.info(f"Push response status code: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            logger.info(f"Push successful. push_id: {result['push_id']}")
            return result
        except requests.HTTPError as e:
            logger.error(f"HTTP error during push: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error during push: {str(e)}")
            raise
        finally:
            # Close all file handles
            for f in file_handles:
                f.close()

    def pull(
            self,
            push_id: str,
            save_to_disk: bool = True,
            yaml_filename: Optional[str] = None,
            functions_filename: Optional[str] = None,
            schema_filename: Optional[str] = None,
            download_path: str = "."
    ):
        try:
            logger.info(f"Sending pull request for push_id: {push_id}")
            response = requests.get(f"{self.api_base_url}/pull/{push_id}")
            logger.info(f"Pull response status code: {response.status_code}")
            response.raise_for_status()
            files_content = response.json()

            if save_to_disk:
                # Map platform filenames to custom filenames
                filename_mapping = {
                    'yaml_file.yaml': yaml_filename or 'yaml_file.yaml',
                    'functions.py': functions_filename or 'functions.py',
                    'structured_output_schema.py': schema_filename or 'structured_output_schema.py'
                }
                for original_filename, content in files_content.items():
                    save_filename = filename_mapping.get(original_filename, original_filename)
                    full_path = os.path.join(download_path, save_filename)
                    # Check if file exists and rename if necessary
                    base, extension = os.path.splitext(full_path)
                    counter = 1
                    while os.path.exists(full_path):
                        full_path = f"{base}_{counter}{extension}"
                        counter += 1
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Saved {save_filename} to disk at {full_path}")
            return files_content
        except requests.HTTPError as e:
            logger.error(f"HTTP error during pull: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error during pull: {str(e)}")
            raise

    def inspect_from_platform(self, push_id: str):
        try:
            logger.info(f"Sending inspect request for push_id: {push_id}")
            response = requests.get(f"{self.api_base_url}/inspect/{push_id}")
            logger.info(f"Inspect response status code: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            push_name = result.get('push_name', '')
            files_content = result.get('files', {})
            if push_name:
                logger.info(f"Push Name: {push_name}")
            else:
                logger.info("No push_name associated with this push_id.")
            return {'push_name': push_name, 'files_content': files_content}
        except requests.HTTPError as e:
            logger.error(f"HTTP error during inspect: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error during inspect: {str(e)}")
            raise
