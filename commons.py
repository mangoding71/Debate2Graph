
import time
from math_parsing import parse_math
from prompt import agent_prompt
from openai import OpenAI


def query_model(client, agent_context, model_name="gpt-3.5-turbo-0125", max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=agent_context,
                n=1
            )
            content = completion.choices[0].message.content
            return content
        except Exception as e:
            retries += 1
            if "rate limit" in str(e).lower():
                wait_time = min(30, 2 ** retries)
                time.sleep(wait_time)
            else:
                wait_time = min(10, retries * 2)
                time.sleep(wait_time)

    raise Exception(f"Request failed after {max_retries} retries")


def parse_question_answer(dataset_name, sample):
    if dataset_name == "mmlu":
        question_raw = sample[0]
        a = sample[1]
        b = sample[2]
        c = sample[3]
        d = sample[4]
        answer = sample[5]
        if type(sample) == list:
            raw_task = tuple(sample)
        else:
            raw_task = tuple(sample.values())
        question = agent_prompt[dataset_name]['question'].format(question_raw, a, b, c, d)
        return question, answer, raw_task

    elif dataset_name == "math":
        question_raw = sample["problem"]
        answer = parse_math(sample["solution"])
        question = agent_prompt[dataset_name]['question'].format(question_raw)
        raw_task = sample
        return question, answer, raw_task

    elif dataset_name == "chess":
        question_raw = sample["input"]
        last_move = sample['input'].split(' ')[-1]
        question = agent_prompt[dataset_name]['question'].format(question_raw, last_move)
        answer = sample["target"]
        raw_task = sample
        return question, answer, raw_task

    elif dataset_name == "mquake":
        question_raw = sample['questions'][0]
        answer = [sample['answer']] + sample['answer_alias']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
        return question, answer, raw_task

    elif dataset_name == "musique":
        question_raw = sample['question']
        answer = [sample['answer']] + sample['answer_aliases']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
        return question, answer, raw_task

    elif dataset_name == "truthfulqa":
        question_raw = sample['question']
        answers_raw = sample['mc1_targets']
        answers = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw)]
        answer = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw) if answers_raw[answer] == 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in answers])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task

    elif dataset_name == "medmcqa":
        question_raw = sample['question']
        answers_letters = ['a', 'b', 'c', 'd']
        answers = [sample['opa'], sample['opb'], sample['opc'], sample['opd']]
        answer = answers_letters[sample['cop'] - 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in zip(answers_letters, answers)])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task

    elif dataset_name == 'scalr':
        question_raw = sample['question']
        answers_letters = ['a', 'b', 'c', 'd', 'e']
        answers = [sample['choice_0'], sample['choice_1'], sample['choice_2'], sample['choice_3'], sample['choice_4']]
        answer = answers_letters[sample['answer']]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in zip(answers_letters, answers)])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")