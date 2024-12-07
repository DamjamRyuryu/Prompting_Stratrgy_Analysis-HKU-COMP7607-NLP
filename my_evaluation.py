import argparse
from utils import HUMAN_EVAL
from baseline import stream_jsonl, write_jsonl
from evaluation import evaluate_functional_correctness
import os


ZEROSHOT_FILE = f'baseline_evals/zeroshot.baseline_top_p1.0.jsonl'
SELF_EVOLVE_FILE = f'method_SelfEvolve.jsonl'
def construct_completion(ansset_file: str = ZEROSHOT_FILE):
    id_completion_pairs = [{'task_id':output['task_id'], 'completion':output['output']} for output in stream_jsonl(ansset_file)]
    return id_completion_pairs

def entry_point(
    sample_file: str,
    k: str = "1,3,10",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_file", type=str, required=True, help="problem prompt")
    parser.add_argument("--output_file", type=str, required=True, help="jsonl file including the output")
    _args = parser.parse_args(args=["--problem_file", 'temp_folder/sampled_questions.jsonl', "--output_file", 'A2_test_!ULTRA_DIVERSE!_no_ds_refined.jsonl'])
    return _args

if __name__ == '__main__':
    args = args_parse()
    answers = construct_completion(args.output_file)

    temp_file = args.output_file[:args.output_file.rfind('.')]+ f'_temp.jsonl'
    write_jsonl(temp_file, answers)

    if not os.path.exists(args.problem_file):
        # pickup related problems only
        problems = [item for item in stream_jsonl(HUMAN_EVAL)]
        lst = []
        for item in answers:
            if item['task_id'] not in lst:
                lst.append(item['task_id'])
        idx = 0
        shortened = []
        for q in problems:
            if q['task_id'] == lst[idx]:
                idx += 1
                shortened.append(q)
            if idx == len(lst):
                break
        write_jsonl(args.problem_file, shortened)

    entry_point(
        problem_file= args.problem_file,
        sample_file= temp_file,
        n_workers=24
    )
    os.remove(temp_file)

    print('DONE')

