The pipeline of this project is summarized as follows:

<font size =4>1. Result generation (Not applicable)</font>

Adjust the settings directly in A2_combined.py and then run it.

Then you get the output of the method, which has already been included in the .zip file

<font size=1> This is not applicable because the API key of the LLM is removed. Provide your own key in "utils.py" if you want to generate results.</font>

<font size = 1 color="yellow"> This project only samples parts of the HumanEval dataset for processing due to time constraint.
Modification is needed if you want to test the whole dataset. </font>

<font size=4>2. Test the results </font>

Go to my_evaluation.py edit the "--output_file" to specify the result you want to examine.

<font size=4>File structure</font>
    
    #######################     folders
    --pre_generated_data/           the original dataset and generated testcases
        --generated_testcase.jsonl
        --HumanEval.jsonl
        --shortened_generated_testcase.jsonl
    --logged_results/               all the results generated during experiments
    --temp_folder/                  randomly sampled questions from HumanEval for all experiments
    #######################     New/modified scripts for assignment 2
    --A2_combined.py
    --A2_prompt_disturber.py
    --A2_prompt_modifier.py
    #######################     integration of useful functions needed for multiple methods
    --utils.py
    #######################     testing scripts of HumanEval
    --evaluation.py
    --my_evaluation.py          (modified for running in PyCharm IDE)
    --evaluate_functional_correctness.py
    --modified_execution.py     (slightly different from original script)
    #######################     previous methods
    --baseline.py               (some functions are imported by combined methods)
    --self_evolve.py
    --codeT.py
    