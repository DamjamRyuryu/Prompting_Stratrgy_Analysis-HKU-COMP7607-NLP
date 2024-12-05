from utils import *
from baseline import get_input_list, write_jsonl, stream_jsonl
from codeT import match_solution_testcases, PROBLEM_CNT
from self_evolve import get_prompt_list_init, next_step_prompts
import os
import sys
import random
from A2_prompt_disturber import disturb_rationale
from A2_prompt_modifier import PromptModifier

EARLY_STOP = False
# the setting determining all the sections of the prompt (using dict key to denote):
#     (">@<"means @ is the default value, using D for a single column will choose randomly for it)
#     sys : system prompt     str:(short/>base</long/D)
#     klh : knowledge header  int:(>0<~5) or str: 'D'
#     kle : knowledge end     str:(>none</package/border/both/D)
#     soh : solution header   int:(>0<~5) or str: 'D'
#     egn : example number    int:(>0<~3) or str: 'D'
#     soe : solution end      int:(>0<~3) or str: 'D'
#     preset : preset         str: (default/complex/simple/!ULTRA_DIVERSE!)
# 'mode' in PROMPT_MODIFIER can be either ('base', 0, 'none', 0, 0, 0) or ('default',)
#       6 params means customized setting, 1 param means using existing presets
PROMPT_MODIFIER = {'enabled': True, 'mode': ('simple',), 'keys': ('sys', 'klh', 'kle', 'soh', 'egn', 'soe')}
DISTURBER = {'enabled': False, 'ratio': 1.0}
BATCH_SIZE = 10
TOP_K = 5
OUTPUTFILE = "A2_test_"
TEST_FILE = "pre_generated_data/shortened_generated_testcase.jsonl"
# following prompts templates are based on the method from this paper:https://arxiv.org/pdf/2306.02907
SELF_REFINEMENT={
    "NO-PAD": "The user will give you their failed Python code problem, help them fix te bug so they can pass the check process.\nBe attention, you should only output the codes without any explanation, comment, natural language and testcode.\nEnvironment: ipython.",
    "system": "The user will give you their failed Python code problem, help them refine the function so they can pass the check process.\nPay attention to their required output format.\nEnvironment: ipython.",
    "padding":"My function failed when going through the tests: %\n\nHelp me refine the code.\nYou should only output the codes without any explanation, comment, natural language and testcode.\nWrap your code with \"'''\""
}

# The process is very time-consuming, create checkpoints at some stages, so you can continue next time without go through previous sections.
# The checkpoints are:
# 0. After randomly sampling the inputs, keep the selected questions fixed when necessary.
# 1. After finishing the knowledge part (reasoning), save the responses from LLM
# 2. After finishing the solution part (output 1), save the responses from LLM
# 3. After finishing the cross verification of the solutions, save the verification results
# (4) Then the code will select top k solution for each problem, concatenate all its failed testcases and request LLM for the last time,
#     which is saved as final output. After that the code finish and you should run 'my_evaluation.py' to get the final pass@k

CHECKPOINTS = {
    'step 0': 'temp_folder/sampled_data.jsonl',
    'step 1': 'temp_folder/combined_1_temp_',
    'step 2': 'temp_folder/combined_2_temp_',
    'step 3': 'temp_folder/combined_3_temp.jsonl',
}

def uni_agreement_afb(_results: list[dict], k: int=3, q_cnt: int = 164) -> dict[str, list | dict]:
    """
        find the top k solutions according to their verification results on testcases
        return a list of integers indicating the line index of top k results in the whole solution list
        AND ALSO:
        record the failed tests of each selected completion and return the information for later refinement.
        """
    assert BATCH_SIZE >= k, 'ERROR: not enough batches'
    _counter = defaultdict(Counter)
    _selected_indices = []
    # count passed cases for single solution
    for line in _results:
        if line['passed']:
            _counter[line['task_id']][line['line_index']] += 1
        if line['task_id'] not in _selected_indices:
            _selected_indices.append(line['task_id'])
    # choose top k solutions
    _indices = []
    for task_id, line_index in _counter.items():
        if len(line_index) >= k:
            top_k = line_index.most_common(k)
            _indices.extend(copy.deepcopy([i[0] for i in top_k]))
        elif len(line_index) > 0:
            _diff = k - len(line_index)
            top_k = line_index.most_common(len(line_index))
            _idx_start = _selected_indices.index(task_id) * BATCH_SIZE
            _idx_group = [_idx_start + i for i in range(BATCH_SIZE)]
            correct_cases = [i[0] for i in top_k]
            left_cases = [i for i in _idx_group if i not in correct_cases]
            _indices.extend(copy.deepcopy(correct_cases + left_cases[0:_diff]))
    # scan for all-failed problems
    for i in range(q_cnt):
        task_id = _selected_indices[i]
        if task_id not in _counter.keys():
            print(f'Notice: task_id: {task_id}, has no correct solution')
            _idx_start = i * BATCH_SIZE
            _append_indices = copy.deepcopy([_idx_start + i for i in range(k)])
            _indices.extend(_append_indices)

    # record failed testcases
    _outputs = sorted(_indices)
    _dict = defaultdict(dict)
    for line in _results:
        line_index = line['line_index']
        if line_index in _outputs:
            if line_index not in _dict['concat_tests'].keys():
                _dict['concat_tests'][line_index] = "" if line['passed'] else line['test'].strip()
                _dict['entry_point'][line_index] = line['entry_point']
            elif not line['passed']:
                _temp = _dict['concat_tests'][line_index]
                _dict['concat_tests'][line_index] += line['test'].strip() if not _temp else line['test'][line['test'].find("\n    assert "):].rstrip()
    return {"indices":_outputs, "wrong_tests": _dict}


def construct_refinement_prompt(_output: list[dict], _completions: list[str], wrong_tests: dict[str, dict]):
    _iter = wrong_tests['concat_tests']
    _entry_points = [line for line in wrong_tests['entry_point'].values()]
    for idx, _str in enumerate(_iter.values()):
        if not _str:
            _output[idx]['prompt'] = _output[idx].pop('sub_prompt_1')
            _output[idx]['output'] = solution_to_completion(_completions[idx])
        else:
            _wrong_completion = solution_to_completion(_completions[idx]) + '\n\n' + _str + '\n\n'
            _prompt = [
                {"role": "system", "content": SELF_REFINEMENT['system']},
                {"role": "user", "content": _wrong_completion + f"check({_entry_points[idx]})".join(SELF_REFINEMENT['padding'].split("%"))}
            ]
            _output[idx]['sub_prompt_2'] = copy.deepcopy(_prompt)

if __name__ == '__main__':
    start_time = time.perf_counter()
    service = LlamaModel(URL, API_KEY, 0.8, 1, 0.8)
    if PROMPT_MODIFIER['enabled']:
        print("Prompt modifier created")
        modifier = PromptModifier()
        if len(PROMPT_MODIFIER['mode']) == 1:
            if PROMPT_MODIFIER['mode'] == '!ULTRA_DIVERSE!':
                print("ULTRA_DIVERSE!!!!!!!!!!!")
            modifier.set_plan(preset=PROMPT_MODIFIER['mode'][0])
        elif len(PROMPT_MODIFIER['mode']) == 6:
            modifier.set_plan(**{PROMPT_MODIFIER['keys'][index]:PROMPT_MODIFIER['mode'][index] for index in range(6)})
        file_tag = '-'.join(f"{item}" for item in PROMPT_MODIFIER['mode'])
    else:
        file_tag = ''
        modifier = None
    # inputs = get_input_list(HUMAN_EVAL)
    if not os.path.exists(CHECKPOINTS['step 0']):
        print('sampling the original data...')
        inputs = random.sample(get_input_list(HUMAN_EVAL), 80) # only use part of the test case to speed up experiment. select randomly
        inputs = sorted(inputs, key= lambda x: int(x['task_id'].split('/')[1]))
        write_jsonl(CHECKPOINTS['step 0'], inputs)
    else:
        print('loading the sampled data...')
        inputs = [item for item in stream_jsonl(CHECKPOINTS['step 0'])]
    prompts = get_prompt_list_init(inputs)
    history = [{'task_id': item['task_id']} for item in inputs]
    concatenate_dict(history, inputs, ['input'], ['prompt'])
    concatenate_dict(history, prompts, ['sub_prompt_0'], ['prompt'])

    # Step 1: request for knowledge
    slow_print('first step start, ask for knowledge...')
    history = get_batch(history, BATCH_SIZE)

    # TODO 1: system & knowledge prompt modification
    if modifier:
        # switch system prompt
        prompts = modifier.change_system_prompt(history)
    if modifier:
        # switch knowledge prompt
        prompts = modifier.change_knowledge_prompt(history)

    step = 0
    if not os.path.exists(CHECKPOINTS['step 1'] + file_tag + '.jsonl'):
        res_1 = service.request_response([line[f'sub_prompt_{step}'] for line in history])
        write_jsonl(CHECKPOINTS['step 1']  + file_tag + '.jsonl', [{'response': item} for item in res_1])
    else:
        res_1 = [item['response'] for item in stream_jsonl(CHECKPOINTS['step 1'] + file_tag + '.jsonl')]

    # TODO 2: disturb the rationale!
    if DISTURBER['enabled']:
        res = disturb_rationale(res_1, BATCH_SIZE, DISTURBER['ratio']) if DISTURBER['enabled'] else res_1
        file_tag = f'ds{DISTURBER['ratio']}.jsonl'
        slow_print(f'disturber on, ratio: {DISTURBER['ratio']}')
    else:
        res = res_1
        file_tag = f'no_ds.jsonl'
        slow_print('disturber off')

    # Step 2: request for solution
    slow_print('knowledge get, ask for solution...')
    step = next_step_prompts(history, res, step)

    # TODO 3: system & knowledge prompt modification
    if modifier:
        # switch system prompt
        prompts = modifier.change_system_prompt(history)
    if modifier:
        # switch system prompt
        prompts = modifier.change_knowledge_prompt(history)

    if not os.path.exists(CHECKPOINTS['step 2'] + file_tag):
        res = service.request_response([line[f'sub_prompt_{step}'] for line in history])  # the response is solution now
        write_jsonl(CHECKPOINTS['step 2'] + file_tag, [{'response': item} for item in res])
    else:
        res = [item['response'] for item in stream_jsonl(CHECKPOINTS['step 2'] + file_tag)]

    if EARLY_STOP:
        concatenate_str(history, res, 'output', processing=True)
        write_jsonl(OUTPUTFILE + file_tag, history)
        sys.exit('Early stop activated.')

    # Step 3: load pre-generated testcases and verify the samples
    slow_print('Solution get. Loading testcases...')
    if not os.path.exists(TEST_FILE):
        sys.exit('Testcases file NOT FOUND.')
    else:
        testcase_list = [case for case in stream_jsonl(TEST_FILE)]

    slow_print("Verifying solutions...")
    if not os.path.exists(CHECKPOINTS['step 3']):
        sample_list = match_solution_testcases(history, res, testcase_list)
        test_results = check_testcase(sample_list, inputs, verify=True, n_workers=24)
        write_jsonl(CHECKPOINTS['step 3'], test_results)
    else:
        test_results = [item for item in stream_jsonl(CHECKPOINTS['step 3'])]

    # Step 4: go over uni-agreement and request LLM for refinement
    feedbacks = uni_agreement_afb(test_results, TOP_K, len(inputs))  # find top k solution
    output = [history[idx] for idx in feedbacks['indices']]
    res = [res[idx] for idx in feedbacks['indices']]
    construct_refinement_prompt(output, res, feedbacks['wrong_tests'])
    slow_print("Requesting LLM for refinement...")
    sub_list = [{'index': index, f'sub_prompt_2': item[f'sub_prompt_2']}
                for index, item in enumerate(output) if f'sub_prompt_2' in item]
    res = service.request_response([item['sub_prompt_2'] for item in sub_list])
    concatenate_str(sub_list, res, 'output', processing=True)
    concatenate_dict(output, sub_list, ['output'], ['output'], has_indices=True)
    write_jsonl(OUTPUTFILE +'refined_'+ file_tag, output)
    end_time = time.perf_counter()
    print(f'DONE,wall_clock time:{end_time-start_time}')