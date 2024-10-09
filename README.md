# GenFlow - the AI agent framework for people who don't like frameworks

**GenFlow** is a lightweight open-source framework that streamlines the process of creating, combining and nesting agentic systems, while keeping low-level control and avoiding unnecessary abstractions.
It provides a minimalistic approach to agentic system development.

---

## Why GenFlow?

GenFlow is built to address four primary needs for AI developers:

### 1. Absence of unnecessary abstractions

AI agent frameworks have become quite complicated, with excessive layers of abstractions which are not always transparent, reliable or easy to use.
GenFlow defines agentic systems as graphs of simple python functions and LLM API calls. No other abstractions needed.  

### 2. Composability

Most real-world applications of AI agents involve building blocks which appear across different use cases. 

For instance, whether you're optimizing a financial portfolio, automating reports of listed companies
or creating a dashboard to visualize the latest trends in an industry, you would greatly benefit from having
pre-built agents that analyze investor relation reports and answer questions about them. Most agentic worflows today are built as big monolithic blocks that encompass functionalities that could be reused across different applications.

**GenFlow makes it easy to build complex agentic workflows by combining reusable and specialized agents.** 
Define each step of your workflow in a YAML file, and GenFlow will handle dependency resolution and node orchestration automatically. Reference outputs from other nodes with a simple, intuitive syntax, allowing for highly modular and reusable code.

### 2. Portability

An agentic system in GenFlow is defined by a set of **YAML files and python functions**. This make it easy to share your workflows with other developers.

### 3. Community collaboration

You can push and pull agentic workflows to **GenSphere, our community hub of AI agents**. Think of HuggingFace
for agents - GenSphere is a repository of agentic workflows defined by YAML files and python functions. 

We will also hold a set of **leaderboards for agentic systems**, such that you can either compete to have the 
best implementation or leverage foundational agents from the hub and avoid working from scratch on problems 
that have already been solved by the community. 

### 4. Compatibility with other AI agent frameworks

You can integrate agents built with other frameworks (crewAI, AutoGen) easily and combine them together with other
blocks written directly with genFlow. You can also use GenFlow to orchestrate multi-agent systems composed of agents from different frameworks.
GenFlow also supports tools from [composio](https://composio.dev/). 


---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart Guide](#quickstart-guide)
- [Examples](#examples)
- [Integrations](#integrations)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Define workflows with simple YAML files**: Create complex execution graphs using simple YAML files.
- **Nest agentic systems easily**: You can reference other YML files as nodes in your workflow, and integrate agentic systems easily.
- **Push and pull agentic systems to our community hub [coming soon]**: Collaborate with others by using agents from *GenSphere*, our repository of agentic systems. You can also publish your agents to the platform and score them against pre-defined benchmarks.
- **Visualize workflows and gain low-level control**: explore agentic systems visually with graph and depedency visualization. You can quickly see which functions are attached to which nodes and have complete control over your workflows.
- **Integrate agents built with other platforms**: Compatible with other agent frameworks like AutoGen and Crews by exposing code as functions.

---

## Installation

```bash
pip install genflow
````
---

## Everything that is to know about GenFlow in under 5 minutes

This quick tutorial shows how to define an agentic system using a yml file, 

### 1. Define Your Workflow in YAML
Create a sample `sample.yaml`. Each nodes holds a function or an LLM service call. You can reference inputs of nodes
as outputs from other using the `{{node_name.node_output}}` syntax.

```yaml
# sample.yaml

nodes:
  - name: read_csv
    type: function_call
    function: read_csv_file
    params:
      file_path: "data.csv"
    outputs:
      - data

  - name: process_data
    type: function_call
    function: process_csv_data
    params:
      data: "{{ read_csv.data }}"
    outputs:
      - processed_data

  - name: analyze_data
    type: llm_service
    service: openai
    model: "gpt-4"
    function_call:
      name: AnalyzeDataFunction
    params:
      prompt: "{{ process_data.processed_data }}"
    outputs:
      - analysis_result

```
### 2. Create a Sub-Flow with a separate YAML file
Create a `data_processing_flow.yaml` file

```yaml
# data_processing_flow.yaml

nodes:
  - name: clean_data
    type: function_call
    function: clean_data_function
    params:
      data: "{{ data }}"
    outputs:
      - cleaned_data

  - name: analyze_cleaned_data
    type: llm_service
    service: openai
    model: "gpt-4"
    function_call:
      name: AnalyzeCleanedDataFunction
    params:
      cleaned_data: "{{ clean_data.cleaned_data }}"
    outputs:
      - cleaned_analysis

```
### 3. Use `yml_compose` to compose flows
Create a `compose_flows.py` file:

```python
# compose_flows.py

from genflow import YmlCompose

# Combine the main flow with the sub-flow
composer = YmlCompose(
    base_flow='sample.yaml',
    sub_flows={'data_processing': 'data_processing_flow.yaml'}
)

# Save the combined flow to a new file
composer.compose(output_file='combined_flow.yaml')

```

### 4. Implement your functions
Create  a `functions.py` file

```python
# functions.py

import pandas as pd

def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    return {'data': data}

def process_csv_data(data):
    # Process your data here
    processed_data = data.describe().to_string()
    return {'processed_data': processed_data}

def clean_data_function(data):
    # Clean your data here
    cleaned_data = data.dropna()
    return {'cleaned_data': cleaned_data}
```
### 5. Run the Combined Workflow
Create a `main.py` file

```python
# main.py

from genflow import GenFlow

if __name__ == '__main__':
    # Load the combined flow
    flow = GenFlow.from_yaml('combined_flow.yaml')
    flow.run()
    print(flow.outputs)
```

### 6. Execute

```bash
python compose_flows.py
python main.py
```
This sequence will first combine `sample.yaml` and `data_processing_flow.yaml` into a single flow (combined_flow.yaml) and then run it.

---
## Examples

[coming soon]

---
## Integrations

### Using Other Agent Frameworks

GenFlow is designed to be interoperable with other agent-building frameworks like AutoGen and Crews. You can integrate these frameworks by exposing their functionalities as functions that can be called within GenFlow nodes.

**Example**:
[coming soon]

---
## License

This project is licensed under the MIT License - see the LICENSE file for details.