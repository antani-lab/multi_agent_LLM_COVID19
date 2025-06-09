# Description: This script is used to generate the json file for the json files to fine-tune the GPT-4o model with instruct training.

import os
import pandas as pd
import json

root = os.path.join(os.getcwd(),'new_5fold')
data = pd.read_csv(os.path.join(root, '5_fold_partition_annotation.csv'))

data1 = data[data['fold'] == 1]
data2 = data[data['fold'] == 2]
data3 = data[data['fold'] == 3]
data4 = data[data['fold'] == 4]
data5 = data[data['fold'] == 5]
data5_part1 = data5.iloc[:len(data5)//2]
data5_part2 = data5.iloc[len(data5)//2:]

def generate_prompts(images_with_prompts, output_file="data.jsonl"):
    """
    Generates instruction-based prompts for each image and writes them to a .jsonl file,
    where each image has its own conversation sequence in a "messages" list.

    Parameters:
    - images_with_prompts (list of tuples): A list of tuples where each tuple contains:
        - image_url (str): The URL of the image.
        - prompt_text (str): The prompt text for the assistant.
    - output_file (str): The name of the .jsonl file to write the prompts to.
    """

    # Open the output file in append mode to write multiple entries

    with open(output_file, "a") as f:
        # Loop through each image and prompt pair
        for image_url, prompt_text in images_with_prompts:
            question, answer = prompt_text
            # Define a conversation for each image
            conversation = [
                # System prompt
                {
                    "role": "system",
                    "content": "You are a radiologist to examinate chest X-ray (CXR) and to provide radiograph diagnosis and assessments."
                },
                # User question prompt
                {
                    "role": "user",
                    "content": question
                },
                # Image URL prompt
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                },
                # Assistant prompt with the response
                {
                    "role": "assistant",
                    "content": answer
                }
            ]

            # Wrap the conversation list in a dictionary with the key "messages"
            data = {
                "messages": conversation
            }

            # Write the conversation as a JSON object to the .jsonl file
            f.write(json.dumps(data) + "\n")

    print(f"Prompts have been written to {output_file}")

# all_data = [data]
# all_data = [data1, data2, data3, data4, data5]
all_data = [data5_part1, data5_part2]
images_with_prompts_list = []

for index, df in enumerate(all_data):
    for idx, row in df.iterrows():
        image_url = row['url']
        question_1 = "Does the image present COVID-19 pneumonia ?"
        answer_1 = row['label']
        prompt = (image_url, (question_1, answer_1))
        images_with_prompts_list.append(prompt)
        question_2 = "what is the mRALE(modified Radiographic Assessment of Lung Edema) score presented on the image ?"
        score = row['mRALE_Score']
        answer_2 = f"The mRALE score is {score}."
        prompt = (image_url, (question_2, answer_2))
        images_with_prompts_list.append(prompt)
        question_3 = "Describe the clinical findings ?"
        answer_3 = row['description']
        prompt = (image_url, (question_3, answer_3))
        images_with_prompts_list.append(prompt)
        question_4 = "rate the pneumonia severity in one of the four levels: low, mild, moderate, and severe"
        answer_4 = row['level']
        prompt = (image_url, (question_4, answer_4))
        images_with_prompts_list.append(prompt)
    json_file = f"data_5_{index+1}.jsonl" 
    # Generate prompts and write to data.jsonl
# json_file = "data.jsonl"
    json_path = os.path.join(root, json_file)
    generate_prompts(images_with_prompts_list, output_file=json_path)

