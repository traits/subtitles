import os

class Analyzer:
    def connect(self):
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set the DEEPSEEK_API_KEY environment variable.")
        # Placeholder for the actual connection logic to DeepSeek-VL2 model
        print(f"Connected to DeepSeek-VL2 with API key: {api_key}")
