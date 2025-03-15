import time, tiktoken
from openai import OpenAI
import openai
import os, anthropic, json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "mistral": 0.50 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3.5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "mistral": 2.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def retry_with_exponential_backoff(
    func,
    max_retries=5,
    initial_delay=1,
    exponential_base=2,
    error_types=(Exception,)
):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if i == max_retries - 1:  # Last iteration
                    raise
                print(f"Attempt {i + 1} failed with error: {str(e)}")
                time.sleep(delay)
                delay *= exponential_base
    return wrapper

def call_mistral_api_direct(api_key, messages, temp=0.7, max_tokens=1024):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "mistral-medium",
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens
    }
    
    # Create a session with retry strategy
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Direct API call failed: {str(e)}")
        raise

def query_model(model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "mistral":
                # Use much more conservative limits for Mistral
                max_prompt_length = 4000
                
                # Create a simplified system prompt
                simple_system = "You are a helpful AI research assistant."
                if len(system_prompt) > 100:
                    simple_system += " " + system_prompt[:100] + "..."
                
                # Focus on the most recent/relevant part of the user prompt
                simplified_prompt = prompt
                if len(prompt) > max_prompt_length:
                    last_period = prompt[-max_prompt_length:].find('.')
                    if last_period != -1:
                        simplified_prompt = "..." + prompt[-max_prompt_length + last_period:]
                    else:
                        simplified_prompt = "..." + prompt[-max_prompt_length:]
                
                try:
                    messages = [
                        {"role": "system", "content": simple_system},
                        {"role": "user", "content": simplified_prompt}
                    ]
                    answer = call_mistral_api_direct(
                        api_key=openai_api_key,
                        messages=messages,
                        temp=temp if temp is not None else 0.7,
                        max_tokens=1024
                    )
                except Exception as e:
                    print(f"Error with Mistral API: {str(e)}")
                    # Fallback to an even simpler prompt
                    try:
                        very_simple_messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": simplified_prompt[-2000:]}
                        ]
                        answer = call_mistral_api_direct(
                            api_key=openai_api_key,
                            messages=very_simple_messages,
                            temp=0.7,
                            max_tokens=1024
                        )
                    except Exception as e:
                        print(f"Fallback also failed: {str(e)}")
                        raise Exception("Failed to get response from Mistral API")

            if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "mistral"]:
                encoding = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base for Mistral
            elif model_str in ["deepseek-chat"]:
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.encoding_for_model(model_str)
            if model_str not in TOKENS_IN:
                TOKENS_IN[model_str] = 0
                TOKENS_OUT[model_str] = 0
            TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
            TOKENS_OUT[model_str] += len(encoding.encode(answer))
            if print_cost:
                print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))