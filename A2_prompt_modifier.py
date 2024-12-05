import random
from baseline import stream_jsonl

class PromptModifier:
    """
    this prompt modifier is relative to:
        the 2nd (complexity),
        3rd (demonstration),
        and 4th (diversity)
        parts of the analytical experiment
    """
    def __init__(self, max_testcase: int = 3):
        self._testcases_path = "pre_generated_data/shortened_generated_testcase.jsonl"
        self._max_testcase = max_testcase
        self.testcases_subset = []
        # 3 versions for system prompt (complexity)
        self._system_prompts={
            "short" : "Environment: ipython",
            "base" : "The user will ask you about Python code problem, follow their instructions. Pay attention to their required output format. Environment: ipython.",
            "long" : "You are an expert in coding. The user is having difficulties with Python coding questions. Each of the questions is an unimplemented Python function. "
                    +"The name of the function is given, and there are some requirements or hints given in the comments. "
                    +"Help them complete the functions. Be attention, the user will ask you to teach them how to complete the function. Do not give the code in this step."
                    +"tell them how to solve the problem step by step instead. After the reasoning, the user will ask you for the code solution."
                    +"This time you should answer with complete code implementation based on the previous reasoning. Do not include any natural language, comment, test case in your code."
                    +"Environment: ipython."
        }
        # 6 paraphrased versions of the reasoning padding's header (diversity)
        self._knowledge_header={
            0: "\nFor the above question, could you briefly teach me how to solve it step by step in natural language?",
            1: "\nCould you tell me how to complete the previous function step by step in natural language?",
            2: "\nCould you explain how to complete the function in detail, using simple language and a step-by-step approach?",
            3: "\nCould you use natural language to explain the solution to the previous problem in simple, clear steps?",
            4: "\nWould you walk me through how to solve the question above, using natural language?",
            5: "\nPlease break down the solution to the question above in an easy-to-follow manner and tell me in natural language."
        }
        # end of reasoning and some extra crucial points (complexity and diversity)
        self._knowledge_ending={
            'end': "Don't write the code in this step.",
            'package': "Pay attention to whatever package the problem include.",
            'border': "Is there any common pitfall like boundary issue that I have to pay attention to? If such pitfall exists, how to fix it?",
        }
        # 6 paraphrased versions of the solution enquiry header (diversity)
        self._solution_header={
            0: "Based on the above idea, help me complete the function.",
            1: "Now implement the function code mentioned in the coding problem according to the concept described above.",
            2: "Using the approach mentioned above, provide only the pure function code.",
            3: "Write the bare code for this function following the previous logic.",
            4: "Show me just the function implementation based on the concept discussed.",
            5: "Provide the raw code that fulfills the previous coding problem."
        }
        # template for constructing extra example cases (demonstration)
        self._few_shot_template="Input {%}: {%}\nExpected Output: {%}\n"
        # 5 paraphrased versions of the solution enquiry ending (diversity)
        self._solution_ending = {
            0: "\nBe attention, you should only output the codes without any explanation, comment, natural language and testcode.\nWarp your code with \"'''\"",
            1: "\nPlease write with no comments or explanations and warp your code with \"'''\"",
            2: "\nPlease exclude comments and test cases.\nWarp your code with \"'''\"",
            3: "\nYou should give the code without any supplementary text and warp your code with \"'''\"",
            4: "\nOmit any explanations or testing code. Keep the pure function only and warp it with \"'''\""
        }
        self._corpus={
            'sys': self._system_prompts,
            'klh': self._knowledge_header,
            'kle': self._knowledge_ending,
            'soh': self._solution_header,
            'egn': self._few_shot_template,
            'soe': self._solution_ending,
        }
        # the setting determining all the sections of the prompt (using dict key to denote):
        #     (">@<"means @ is the default value, using D for a single column will choose randomly for it)
        #     sys : system prompt     str:(short/>base</long/D)
        #     klh : knowledge header  int:(>0<~5) or str: 'D'
        #     kle : knowledge end     str:(>none</package/border/both/D)
        #     soh : solution header   int:(>0<~5) or str: 'D'
        #     egn : example number    int:(>0<~self._max_testcase) or str: 'D'
        #     soe : solution end      int:(>0<~3) or str: 'D'
        self._modify_plan={
            'sys' : 'base',
            'klh' : 0,
            'kle' : 'none',
            'soh' : 0,
            'egn' : 0,
            'soe' : 0,
        }
        # some presets, use tuple for simpler presentation
        self.preset_options={
            'default': ('base', 0, 'none', 0, 0, 0),
            'complex': ('long', 2, 'both', 1, 0, 0),
            'simple' : ('short', 4, 'none', 0, 0, 2),
            '!ULTRA_DIVERSE!' : ('D',) * 6,
        }

    def set_plan(self,**kwargs):
        """
        set the config of the modifier, use key-value pair to send modification
        [KEY] : [DEFINITION]    >>  [POSSIBLE VALUES]
        sys : system prompt     >>  str: (short/>base</long/D)
        klh : knowledge header  >>  int: (>0<~5) or str: 'D'
        kle : knowledge end     >>  str: (>none</package/border/both/D)
        soh : solution header   >>  int: (>0<~5) or str: 'D'
        egn : example number    >>  int: (>0<~self._max_testcase) or str: 'D'
        soe : solution end      >>  int: (>0<~3) or str: 'D'
        preset : preset         >>  str: (default/complex/simple/!ULTRA_DIVERSE!)
        """
        _keys = ('sys', 'klh', 'kle', 'soh', 'egn', 'soe')
        if not kwargs:
            print('No value received')
        elif 'preset' in kwargs:
            if kwargs['preset'] in ('default', 'complex', 'simple', '!ULTRA_DIVERSE!'):
                # assume the dict is not ordered, add robustness
                for i in range(len(self._modify_plan)):
                    self._modify_plan[_keys[i]] = self.preset_options[kwargs['preset']][i]
                print(f'using preset : {kwargs["preset"]}')
            else:
                print(f"{kwargs['preset']} is not a valid preset.")
        else:
            # customize set
            for k, v in kwargs.items():
                if k in ('sys', 'klh', 'kle', 'soh', 'egn', 'soe'):
                    # make sure the input key is valid
                    if v in self._corpus[k] or v == 'D':
                        self._modify_plan[k] = v
                        print(f"key {k} = {v}.")
                    elif k == 'kle' and v in ('none', 'both'):
                        self._modify_plan[k] = v
                        print(f"key {k} = {v}.")
                    elif k == 'egn' and v in range(4):
                        self._modify_plan[k] = v
                        print(f"key {k} = {v}.")
                    else:
                        print(f"{v} is not a valid value for key {k}.")
                else:
                    print(f"{k} is not a valid key.")

    def change_system_prompt(self, inputs):
        """
        change the system prompt according to modify plan
        """
        if self._modify_plan['sys'] not in ('base', 'D') :
            print(f"modifying {len(inputs)} samples, change the system prompt to '{self._modify_plan['sys']}' version")
            for line in inputs:
                line['sub_prompt_0'][0]['content'] = self._corpus['sys'][self._modify_plan['sys']]
        elif self._modify_plan['sys'] == 'D':
            print(f"sys prompt !!DIVERSE!!")
            for line in inputs:
                line['sub_prompt_0'][0]['content'] = random.choice(list(self._corpus['sys'].values()))
        return inputs

    def _create_knowledge_ending(self, method):
        """
        create knowledge ending according to given method(modify plan)
        """
        if method == 'none':
            return '\n' + self._corpus['kle']['end']
        elif method == 'package':
            return '\nrequirements:\n1.' + self._corpus['kle']['package'] + '\n2.' + self._corpus['kle']['end']
        elif method == 'border':
            return '\nrequirements:\n1.' + self._corpus['kle']['border'] + '\n2.' + self._corpus['kle']['end']
        elif method == 'both':
            return '\nrequirements:\n1.' + self._corpus['kle']['package'] + '\n2.' + self._corpus['kle']['border'] + '\n3.' + self._corpus['kle']['end']
        else:
            raise KeyError(f"{method} is not in the corpus.")

    def change_knowledge_prompt(self, inputs):
        """
        change the knowledge (rationale) prompt padding
        """
        # default setting
        if (self._modify_plan['klh'], self._modify_plan['kle']) == (0, 'none'):
            print(f"using default settings for knowledge part.")
            return inputs
        # erase the padding
        for line in inputs:
            _index2 = line['sub_prompt_0'][1]['content'].rfind('\n', 0, line['sub_prompt_0'][1]['content'].rfind('\n'))
            if _index2 == -1:
                raise IndexError("Too few '\\n' in knowledge prompt, there must be something wrong with the original prompt.")
            line['sub_prompt_0'][1]['content'] = line['sub_prompt_0'][1]['content'][:_index2]

        # add new header
        if self._modify_plan['klh'] == 'D':
            print(f"knowledge header !!DIVERSE!!")
            for line in inputs:
                line['sub_prompt_0'][1]['content'] += random.choice(list(self._corpus['klh'].values()))
        else:
            print(f"modifying {len(inputs)} samples, change the knowledge header to '{self._modify_plan['klh']}' version")
            for line in inputs:
                line['sub_prompt_0'][1]['content'] += self._corpus['klh'][self._modify_plan['klh']]

        # add new ending
        if self._modify_plan['kle'] == 'D':
            print(f"knowledge ending !!DIVERSE!!")
            for line in inputs:
                line['sub_prompt_0'][1]['content'] += self._create_knowledge_ending(random.choice(('none', 'package', 'border', 'both')))
        else:
            print(f"modifying {len(inputs)} samples, change the knowledge ending to '{self._modify_plan['kle']}' version")
            for line in inputs:
                line['sub_prompt_0'][1]['content'] += self._create_knowledge_ending(self._modify_plan['kle'])

        return inputs

    def _load_necessary_testcases(self, inputs):
        """
        load pre-generated testcases, keep relevant subset only
        """
        _lst = []
        for _item in inputs:
            if _item['task_id'] not in _lst:
                _lst.append(_item['task_id'])
        _test_cases = [{
                'task_id': _item['task_id'],
                'test':_item['test'].split('\n    assert ')[1]
            }
            for _item in stream_jsonl(self._testcases_path) if _item['task_id'] in _lst]
        self.testcases_subset = _test_cases.copy()

    def _get_test_case(self, task_id, num_case):
        _lst = [_line for _line in self.testcases_subset if _line['task_id'] == task_id]
        assert num_case <= len(_lst), f"too few test cases for {task_id}"
        _iter_cap = 10
        _unit = self._corpus['egn'].split('{%}')
        _string = ''
        _selected_indices = []
        for i in range(num_case):
            for _ in range(_iter_cap):
                try:
                    _not_picked_indices = [j for j in range(len(_lst)) if j not in _selected_indices]
                    _idx = random.choice(_not_picked_indices)
                    if " == " in _lst[_idx]['test']:
                        _temp = _lst[_idx]['test'].split(" == ")
                        _input = _temp[0][_temp[0].find('(') + 1:_temp[0].rfind(')')]
                        _output = _temp[1].strip()
                        if _lst[_idx]['test'].startswith('not '):
                            if _output in ('True', 'False'):
                                _output = {'True': False, 'False': True}[_output]
                            else:
                                continue
                        elif _output in ('True', 'False'):
                            _output = {'True': True, 'False': False}[_output]
                        else:
                            # numerical result
                            if _output[0] in '0123456789':
                                j = 0
                                while _output[j] in '0123456789':
                                    j += 1
                                _output = _output[:j]
                            # string result
                            elif _output[0] in ('"', "'"):

                                raise NotImplementedError
                    else:
                        raise NotImplementedError(f'bad test case for {task_id}, try another.')
                except Exception as e:
                    print(str(e))
                    continue
            _string += _unit[0] + str(i) + _unit[1]
        raise NotImplementedError

    def _create_demonstration(self, test_cases):
        raise NotImplementedError

    def change_solution_prompt(self, inputs):
        if (self._modify_plan['soh'], self._modify_plan['egn'], self._modify_plan['soe']) == (0,0,0):
            print(f"using default settings for solution part.")
            return inputs

        # replace the original prompt with header
        if self._modify_plan['soh'] == 'D':
            print(f"solution header !!DIVERSE!!")
            for line in inputs:
                line['sub_prompt_1'][3]['content'] = random.choice(list(self._corpus['soh'].values()))
        else:
            print(f"modifying {len(inputs)} samples, change the solution header to '{self._modify_plan['soh']}' version")
            for line in inputs:
                line['sub_prompt_1'][3]['content'] = self._corpus['soh'][self._modify_plan['soh']]

        # add demonstrations
        if self._modify_plan['egn'] == 'D':
            print(f"demonstrations !!DIVERSE!!")
            for line in inputs:
                _cases = random.randint(0, self._max_testcase)
                if _cases > 0:
                    line['sub_prompt_1'][3]['content'] += '\n\nHere are some example test cases:\n'
                    line['sub_prompt_1'][3]['content'] += self._create_demonstration(self._get_test_case(line['task_id'], _cases))
            raise NotImplementedError
        elif self._modify_plan['egn'] > 0:
            print(f"modifying {len(inputs)} samples, adding '{self._modify_plan['egn']}' demonstrations")
            self._load_necessary_testcases(inputs)
            for line in inputs:
                line['sub_prompt_1'][3]['content'] += '\n\nHere are some example test cases:\n'
                line['sub_prompt_1'][3]['content'] += self._create_demonstration(self._get_test_case(line['task_id'], self._modify_plan['egn']))
            raise NotImplementedError
        else:
            print(f"no extra demonstrations added")

        # add solution ending
        if self._modify_plan['soe'] == 'D':
            print(f"solution header !!DIVERSE!!")
            for line in inputs:
                line['sub_prompt_1'][3]['content'] += random.choice(list(self._corpus['soe'].values()))
        else:
            print(
                f"modifying {len(inputs)} samples, change the solution ending to '{self._modify_plan['soe']}' version")
            for line in inputs:
                line['sub_prompt_1'][3]['content'] += self._corpus['soe'][self._modify_plan['soe']]
        return inputs