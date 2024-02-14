import json
import random

with open('/home/graham/ai_ct_assistant/simple_concept/AL_reponses.jsonl', 'r') as json_file:
    json_list = list(json_file)

text_list = []

for json_str in json_list:
    result = json.loads(json_str)
    text_list.append(result['text'])
# print(text_list)

print("Enter a prompt:")
prompt_str = input()
weights = [random.random() for _ in range(len(text_list))]
# print(f"weights: {weights}")
response = random.choices(text_list, weights, k = 2)

def replace_elements(input_string, search_elements, replace_with):
    """
    Replace search_elements in input_string with replace_with.

    Args:
    - input_string (str): The input string to search in.
    - search_elements (list): A list of elements to search for.
    - replace_with (str): The string to replace the search_elements with.

    Returns:
    - str: The modified string.
    """
    for element in search_elements:
        input_string = input_string.replace(element, replace_with)
    return input_string

# Example usage
input_string = response[0]
search_elements = ["...","()"]
replace_with = " " + prompt_str
modified_string = replace_elements(input_string, search_elements, replace_with)

print(modified_string)