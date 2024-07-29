"""
    Semi-automatic Video Annotation Pipeline - Step # 4: Generate QA pairs using video descriptions generated in Step # 3 using GPT-3.5-Turbo.

    Copyright 2024 MBZUAI ORYX

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import time
from openai import OpenAI
import os
import json
import ast
import argparse
import warnings
from tqdm import tqdm
from multiprocessing.pool import Pool

# Suppressing all warnings
warnings.filterwarnings('ignore')

class Opt():
    def __init__(self):
        self.root_caption_dir = '/mnt/tnas/AIHUB/238-1.실내(편의점, 매장) 사람 구매행동 데이터/Caption'
        self.ann_video_ids_file = 'unique_video_ids.json'
        self.output_dir = 'video_aihub_qa'
        self.video_caption_dir = 'video_aihub_caption'
        self.api_keys = os.environ.get('OPENAI_API_KEY')
        self.num_tasks = 32

# def parse_args():
#     """
#     Command-line argument parser.
#     """
#     parser = argparse.ArgumentParser(description="Generate QA pairs using video descriptions generated in Step # 3")

#     parser.add_argument("--ann_video_ids_file", required=True,
#                         help="Path to the JSON file with unique video IDs (e.g. path to unique_video_ids.json).")
#     parser.add_argument("--output_dir", required=False, help="Directory to save the annotation JSON files.",
#                         default="video_qa")
#     parser.add_argument("--video_descriptions_path", required=False,
#                         help="Directory containing the generated video descriptions.", default="video_descriptions")
#     parser.add_argument("--gt_caption_file", required=True,
#                         help="Path to the ground truth captions file (e.g. path to activitynet_gt_captions_train.json).")
#     parser.add_argument("--api_keys", required=True, nargs='+', help="List of OpenAI API keys.")
#     parser.add_argument("--num_tasks", type=int, default=32, help="Number of splits.")

#     return parser.parse_args()


def get_summary_qa_prompt(gt_caption):
    system_prompt = (
        "당신은 한국의 편의점 혹은 마트에서 촬영된 비디오 내용에 대한 질문과 답변을 생성하는 AI 어시스턴트입니다. "
        "목표는 제공된 비디오 캡션을 바탕으로 비디오의 전체 내용과 고객 행동에 대한 상세한 정보를 추출하여 비디오 교육 데이터셋을 만드는 것입니다. "
        "##작업:"
        "1. 사용자는 비디오의 실제 캡션을 제공합니다."
        "2. 비디오 전체 내용과 고객 행동에 대한 상세한 설명을 유도하는 세 가지 질문을 생성하세요."
        "------"
        "##지침:"
        "- 각 질문은 비디오 전체 내용과 고객의 행동 순서를 상세히 설명하도록 유도해야 합니다."
        "- 매장 환경, 고객의 외모와 의상, 상호작용 시간, 관심 상품, 구매 상품, 행동 묘사, 구매 관심도 등에 초점을 맞추세요."
        "- 제공된 캡션의 모든 정보를 활용하여 질문과 답변을 생성하세요."
        "- 시각적, 시간적 세부사항에 집중하세요."
        "##샘플 질문:"
        "- 비디오에 등장하는 고객의 특징과 행동을 처음부터 끝까지 상세히 설명해주시겠습니까?"
        "- 고객이 매장에서 어떤 상품들에 관심을 보이고 어떻게 상호작용했는지 시간 순서대로 설명해주시겠습니까?"
        "- 매장 환경과 고객의 구매 관심도를 포함하여 전체 비디오 내용을 자세히 설명해주시겠습니까?"
    )
    
    # system_prompt = (
    #     "You are an AI assistant tasked with generating questions and answers about video content to create a video instruction tuning dataset. "
    #     "Your goal is to extract detailed visual and temporal information from the video, ensuring the explanations are comprehensive enough for someone to understand the entire sequence of events in the video."
    #     "##TASK:"
    #     "1. Users provide a video ground truth caption and a detailed description."
    #     "2. Generate three questions that effectively prompt a detailed description of the entire video content and sequence of events."
    #     "------"
    #     "##INSTRUCTIONS:"
    #     "- Ensure each question targets the goal of generating a detailed description of the entire video from start to end."
    #     "- Avoid questions that focus on small parts, less relevant details, or abstract concepts such as logical reasoning, attention to subtle details, overall aesthetic."
    #     "- Every answer must include all the details from the ground truth caption and integrate additional specifics from the detailed description."
    #     "- Focus on visual and temporal details."
    #     "##SAMPLE QUESTIONS:"
    #     "- Can you describe the entire video in detail from start to finish?"
    #     "- What happens throughout the entire video, including all key actions and events?"
    #     "- Could you provide a detailed walkthrough of the entire video?"
    # )

    user_prompt = (
        f"비디오의 실제 캡션은 다음과 같습니다: {gt_caption} "
        "이 캡션을 바탕으로 비디오의 전체 내용과 고객 행동에 대한 세 가지 질문과 답변을 생성해주세요. "
        "각 질문은 비디오의 전체 내용과 고객 행동의 전체 순서에 대한 포괄적인 설명을 유도해야 합니다. "
        "각 답변은 제공된 캡션의 모든 세부사항을 포함하고 가능한 한 상세하게 작성해야 합니다. "
        "출력 형식은 다음과 같이 JSON 스타일의 딕셔너리 리스트로 작성해주세요: "
        "[{'Q': '첫 번째 질문...', 'A': '첫 번째 답변...'}, "
        "{'Q': '두 번째 질문...', 'A': '두 번째 답변...'}, "
        "{'Q': '세 번째 질문...', 'A': '세 번째 답변...'}]. "
        "가장 중요한 것은 모든 답변이 제공된 캡션의 모든 세부사항을 포함하여 비디오에 대한 완전한 이해를 제공해야 한다는 점입니다."
    )
    
    # user_prompt = (
    #     f"The video ground truth caption is: {gt_caption}. "
    #     "Generate three questions and answers about the entire content and sequence of events in the video. "
    #     "Each question should aim to elicit a comprehensive description of the full sequence of events in the video from start to finish. "
    #     "Each answer must include all the details from the ground truth caption and integrate additional specifics from the detailed description. "
    #     "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
    #     "For example: "
    #     "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
    #     "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
    #     "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
    #     "Most importantly, every answer must provide a full understanding of the video by incorporating ALL the details from the ground truth caption and additional specifics from the detailed description."
    # )

    return system_prompt, user_prompt


def get_generic_qa_prompt(gt_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers based on video descriptions. "
        "Your goal is to extract important information from the video content, ensuring the questions focus on significant aspects and the answers are comprehensive and detailed."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description, and you will generate a set of questions and answers related to the video. "
        "The questions should be designed to extract information directly from the given information, so that the provided information or parts of it can serve as the answers. "
        "Generate THREE different questions and detailed answers based on the given information. Each question should focus on a different aspect such as appearance, motion, trajectory, and reasoning."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the events in the video and focus on significant aspects."
        "- The questions should be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers."
        "- The answers must be detailed and descriptive."
        "- The answers must include details about the setting, objects involved, and any specific techniques or methods used."
        "- Each question should focus on a different key aspect such as appearance, motion, trajectory, and reasoning."
        "- Avoid asking about irrelevant details."
        "##SAMPLE QUESTIONS:"
        "- Describe the entire process the person goes through from start to finish."
        "- Can you provide a detailed description of the appearance and activities of all individuals."
        "- Explain how the main activity in the video is performed step by step."
        "- What are the different stages of the activity shown in the video, and how does the person's approach change at each stage?"
        "- Outline the key moments and interactions between people, objects, and their environment.")

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        "The detailed description is provided as a supplementary source of information. "
        "It may contain additional details about objects or activities mentioned in the video caption, but the main focus should be on the information provided in the video caption. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, the question should focus on a different key aspect such as appearance, action, trajectory, and reasoning."
    )

    return system_prompt, user_prompt


def get_temporal_qa_prompt(gt_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers related to the temporal events in a video. "
        "Your goal is to help users understand the sequence and timing of events in the video by asking and answering questions that focus on when events occur."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description  generated from ordered frames of the video in the correct order of events. "
        "You will generate a set of questions and answers related to the events in the video using approximate time references, by closely analyzing the sequence of sentences in the provided information. "
        "Generate THREE different descriptive questions and detailed answers based on the caption and detailed description."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the events in the video and focus on significant temporal aspects."
        "- Use approximate time references such as the beginning, middle, and end."
        "- The answers must be based on the information provided in the caption and detailed description."
        "- The answers must be detailed and descriptive."
        "- Do not explicitly mention in the answers that it is based on the caption or frames."
        "##SAMPLE QUESTIONS:"
        "- When does the main character start the primary task, and what leads up to it?"
        "- What actions occur after the initial setup, and how do they progress towards the climax?"
        "- What significant events happen midway, and how do they transition from earlier to later scenes?"
        "- Can you outline the key events from beginning to end, highlighting any turning points?"
        "- How do the events unfold in the final part, and what marks the video's conclusion?"
    )
    user_prompt = (
        f"The ground truth caption is: {gt_caption}. "
        "The detailed description provides more detailed explanations of the video content and is in the correct order of events. "
        "Please use the detailed description to extract any relevant additional information, but do not base your questions or answers solely on them. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Emphasize that ALL THREE questions must be designed to extract information DIRECTLY from the given information, focusing on the time and order of events in the video."
    )
    return system_prompt, user_prompt


def get_short_temporal_qa_prompt(gt_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers related to the temporal events in a video. "
        "Your goal is to help users understand the sequence and timing of events in the video by asking and answering questions that focus on when events occur."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description generated from ordered frames of the video in the correct order of events. "
        "You will generate a set of questions and answers related to the events in the video using approximate time references, by closely analyzing the sequence of sentences in the provided information. "
        "Generate THREE different descriptive questions and answers based on the provided caption and detailed description."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the events in the video and focus on significant temporal aspects."
        "- Use approximate time references such as the beginning, middle, and end."
        "- The answers must be based on the information provided in the caption and detailed description."
        "- Do not explicitly mention in the answers that it is based on the caption or frames."
        "##SAMPLE QUESTIONS:"
        "- When does event x happen in the video?"
        "- What happens after event x in the video?"
        "- What happens before event x in the video?"
        "- Can you tell me the sequence of events in the video?"
        "- How do the events in the video progress from beginning to end?"
        "- What do the girls do after visiting the park?"
        "- At which part of the video does the dog play with the ball?"
        "- When does the car hit the motorcycle?"
        "- Why is the woman hunched over in the beginning?"
        "- Why does the boy start crying towards the end of the video?"
        "- When does he shoot at the basket?"
        "- What happens before the boys enter the bus?"
    )
    user_prompt = (
        f"The ground truth caption is: {gt_caption}. "
        "The provided detailed description has more detailed explanations of the video content and is in the correct order of events. "
        "Please use the detailed description to extract any relevant additional information, but do not base your questions or answers solely on them. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Emphasize that ALL THREE questions must be designed to extract information DIRECTLY from the given information, focusing on the time and order of events in the video."
    )
    return system_prompt, user_prompt


def get_spatial_qa_prompt(gt_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers based on video descriptions. "
        "Your goal is to extract important spatial information from the video content, ensuring the questions focus on significant visual details."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description, and you will generate a set of questions and answers related to the video. "
        "The questions should be designed to extract spatial information directly from the given information, so that the provided information or parts of it can serve as the answers. "
        "Generate THREE different questions and detailed answers focusing on different spatial aspects such as colors, outfits, location, and displayed text."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the visual events in the video and focus on significant spatial details."
        "- The questions should be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers."
        "- The answers must include details about the setting, objects involved, and any specific visual features."
        "- Each question should focus on a different key aspect such as colors, attire, displayed texts, or location."
        "- Avoid asking about irrelevant details."
        "##SAMPLE QUESTIONS:"
        "- What is the color of the woman's shirt?"
        "- What is the name of the drink on the bottle?"
        "- Describe the outfit of the dancers."
        "- Explain the setting of the video and the objects in the scene."
        "- What is the goalkeeper wearing in the video?")

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        "The detailed description is provided as a supplementary source of information. "
        "It may contain additional details about objects or activities mentioned in the video caption, but the main focus should be on the visual information provided in the video caption. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, the question should focus on key aspects such as appearance, colors, outfits, location, and displayed text."
    )

    return system_prompt, user_prompt


def get_reasoning_qa_prompt(gt_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and answers based on video descriptions. "
        "Your goal is to extract specific, detailed information from the video content, focusing on observable actions, objects, and settings, ensuring the questions are diverse and cover a range of aspects like the identity of objects, actions of individuals, types or styles of activities, and the reasoning or context for actions."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description, and you will generate a set of questions and answers related to the video. "
        "The questions should be designed to extract specific details directly from the given information, ensuring the provided information or parts of it can serve as the answers. "
        "Generate THREE different questions and concise answers based on the given information. Each question should focus on a different aspect such as actions of individuals, objects involved, and reasoning behind actions."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be specific and based on significant details visible or inferred from the events in the video."
        "- Ensure the questions cover different types such as what, where, why, and how, focusing on individual actions, object details, and context or reasoning."
        "- Answers should be concise, incorporating brief details about the setting, objects involved, and any specific techniques or methods used."
        "- Avoid asking about generic or irrelevant details."
        "##SAMPLE QUESTIONS:"
        "- What is the man in the red shirt doing?"
        "- Where does the woman look after picking up the object?"
        "- Who is John Davis in the video?"
        "- Why did the player throw the ball?"
        "- What action does the coach take after the whistle blows?")

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        "The detailed description is provided as a supplementary source of information. "
        "It may contain additional details about objects or activities mentioned in the video caption, but the main focus should be on the information provided in the video caption. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, each question should explore a different key aspect such as what, where, why, and how, focusing on object identification, specific actions, and contextual or reasoning details."
    )

    return system_prompt, user_prompt


def annotate(caption_files, curr_output_dir, api_key):
    """
    Generate question-answer pairs using caption and using OpenAI GPT-4o.
    """
    summary_qa_pairs = True
    generic_qa_pairs = False
    temporal_qa_pairs = False
    spatial_qa_pairs = False
    reasoning_qa_pairs = False
    short_temporal_qa_pairs = False
    # model = "gpt-3.5-turbo"
    # https://openai.com/api/pricing/
    model = "gpt-4o-mini"
    # model = "gpt-4o"
    client = OpenAI(api_key=api_key)

    for file in tqdm(caption_files):
        annotated_dit = {}
        gt_caption = get_gt_caption(file)
        video_id = os.path.splitext(file.split(os.path.sep)[-1])[0]

        if summary_qa_pairs:
            # Generate QA pairs with OpenAI GPT-3: Summarization
            system_prompt, user_prompt = get_summary_qa_prompt(gt_caption)
            completion_0 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response = completion_0.choices[0].message.content

            print(f"Video ID: {file}, Summary QA pairs: {response}")
            annotated_dit['summary_qa_pairs'] = response

        if generic_qa_pairs:
            # Generate QA pairs with OpenAI GPT-3
            system_prompt, user_prompt = get_generic_qa_prompt(gt_caption)
            completion_1 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response = completion_0.choices[0].message.content
            response_dict_1 = ast.literal_eval(response_message_1)

            annotated_dit['generic_qa_pairs'] = response_dict_1

        if temporal_qa_pairs:
            system_prompt, user_prompt = get_temporal_qa_prompt(gt_caption)
            completion_2 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response = completion_0.choices[0].message.content
            response_dict_2 = ast.literal_eval(response_message_2)

            annotated_dit['temporal_qa_pairs'] = response_dict_2

        if spatial_qa_pairs:
            system_prompt, user_prompt = get_spatial_qa_prompt(gt_caption)
            completion_3 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_3 = completion_3["choices"][0]["message"]["content"]
            response_dict_3 = ast.literal_eval(response_message_3)

            annotated_dit['spatial_qa_pairs'] = response_dict_3

        if reasoning_qa_pairs:
            system_prompt, user_prompt = get_reasoning_qa_prompt(gt_caption)
            completion_4 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_4 = completion_4["choices"][0]["message"]["content"]
            response_dict_4 = ast.literal_eval(response_message_4)

            annotated_dit['reasoning_qa_pairs'] = response_dict_4

        if short_temporal_qa_pairs:
            system_prompt, user_prompt = get_short_temporal_qa_prompt(gt_caption)
            completion_5 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_5 = completion_5["choices"][0]["message"]["content"]
            response_dict_5 = ast.literal_eval(response_message_5)

            annotated_dit['short_temporal_qa_pairs'] = response_dict_5

        # Save the response dictionary into a JSON file
        json_file_path = os.path.join(curr_output_dir, f"{video_id}.json")
        with open(json_file_path, "w", encoding='utf-8') as f:
            json.dump(annotated_dit, f, ensure_ascii=False, indent=4)

    print(f"Completed, Annotations saved in {curr_output_dir}")


# def get_gt_caption(json_data, video_id):
#     video_data = json_data[video_id]
#     gt_captions = video_data['sentences']
#     gt_caption = ''.join(gt_captions)
#     return gt_caption

def get_gt_caption(file):
    # json load
    with open(file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    return json_data['answer']

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    # args = parse_args()
    args = Opt()
    os.makedirs(args.output_dir, exist_ok=True)
    
    caption_files = []
    ''' 
    # os.walk로 특정 json 파일 모두 찾기
    for root, dirs, files in os.walk(args.root_caption_dir):
        for file in files:
            if file.endswith('.json') and '복사본' in file:
                caption_files.append(os.path.join(root, file))
    
    gt_dict = {}
    for caption_file in caption_files:
        with open(caption_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
            video_id_start = content.find('"video_id": "') + 13
            video_id_end = content.find('",', video_id_start)
            video_id = content[video_id_start:video_id_end]
            video_id = os.path.splitext(video_id.split(os.path.sep)[-1])[0]

            # prompt 추출
            prompt_start = content.find('"prompt": "') + 11
            prompt_end = content.find('"answer":', prompt_start)
            prompt = content[prompt_start:prompt_end]

            # answer 추출
            answer_start = content.find('"answer": "') + 11
            answer_end = content.rfind('"\n}')
            answer = content[answer_start:answer_end]
            
            data = {
                'prompt': prompt,
                'answer': answer
            }
            
            # print("Video ID:", video_id)
            # print("\nPrompt (첫 100자):", prompt[:100] + "...")
            # print("\nAnswer (첫 100자):", answer[:100] + "...")
            
            output_path = os.path.join(args.video_caption_dir, f"{video_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    '''
    
    # glob with args.video_caption_dir *.json
    caption_files = [os.path.join(args.video_caption_dir, f) for f in os.listdir(args.video_caption_dir) if f.endswith('.json')]
    
    # List of OpenAI API keys
    api_keys = args.api_keys
    
    num_tasks = args.num_tasks

    if not isinstance(api_keys, list):
        api_keys = [api_keys]
        
    # Main loop: Continues until all question-answer pairs are generated for all captions
    while True:
        try:
            # Files that have already been completed.
            completed_files = os.listdir(args.output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                print("All tasks completed!")
                break

            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            num_tasks = min(len(incomplete_files), num_tasks)
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]

            # Distribute API keys to tasks
            task_args = [
                (part, args.output_dir, api_keys[i % len(api_keys)]) for
                i, part
                in enumerate(all_parts)]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 1 minute...")
            time.sleep(60)  # wait for 1 minute before trying again


if __name__ == "__main__":
    main()
