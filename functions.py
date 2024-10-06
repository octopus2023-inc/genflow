# functions.py

import pandas as pd
from duckduckgo_search import DDGS
import requests
from typing import List, Dict
import os

def read_csv_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Reads a CSV file and returns the data.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary with the key 'data' containing the DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Read CSV file from {file_path}")
        return {'data': data}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def process_csv_data(data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Processes the CSV data.

    Args:
        data (pd.DataFrame): The DataFrame containing the CSV data.

    Returns:
        dict: A dictionary with the key 'processed_data' containing processed results.
    """
    print("Processing CSV data")
    # Example processing: calculate mean of numeric columns
    processed_data = data.mean(numeric_only=True).to_dict()
    return {'processed_data': processed_data}

def format_analysis_request(processed_data: Dict) -> Dict[str, str]:
    """
    Formats the processed data into a string for analysis.

    Args:
        processed_data (dict): The processed data summary.

    Returns:
        dict: A dictionary with the key 'analysis_prompt' containing the prompt string.
    """
    print("Formatting analysis request")
    prompt = f"Please analyze the following data summary:\n{processed_data}"
    return {'analysis_prompt': prompt}

def get_exchange_rate(base_currency: str, target_currency: str, date: str = "latest") -> Dict[str, float]:
    """
    Get the current exchange rate of a base currency and target currency.

    Args:
        base_currency (str): The base currency code, e.g., USD, EUR.
        target_currency (str): The target currency code, e.g., USD, EUR.
        date (str): Specific date in YYYY-MM-DD format (default is 'latest').

    Returns:
        dict: A dictionary with the key 'exchange_rate' containing the rate.
    """
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/currencies/{base_currency.lower()}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        rate = data.get(target_currency.lower())
        return {"exchange_rate": rate}
    else:
        raise Exception(f"Failed to fetch exchange rate: {response.status_code}")

def search_internet(search_query: str) -> Dict[str, List[str]]:
    """
    Get internet search results for real-time information.

    Args:
        search_query (str): The query to search the web for.

    Returns:
        dict: A dictionary with the key 'search_results' containing a list of results.
    """
    results = [result['title'] for result in DDGS().text(str(search_query), max_results=5)]
    return {"search_results": results}

def spider_cloud_scrape(url):
    from spider import Spider
    # Initialize the Spider with your API key
    spider_api_key=os.getenv(SPIDER_API_KEY)
    app = Spider(api_key=spider_api_key)

    # Crawl a entity
    crawler_params = {
        "limit": 1,
        "proxy_enabled": True,
        "store_data": False,
        "metadata": False,
        "request": "http",
        "return_format": "markdown",
    }

    try:
        scraped_data = app.crawl_url(url, params=crawler_params)
        print("scraped data found")
        markdown = scraped_data[0]["content"]
    except Exception as e:
        print(e)
        markdown = "Error: " + str(e)

    return markdown