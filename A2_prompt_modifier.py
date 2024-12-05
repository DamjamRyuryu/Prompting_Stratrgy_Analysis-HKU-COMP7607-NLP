import random

class PromptModifier:
    """
    this prompt modifier is relative to:
        the 2nd (complexity),
        3rd (demonstration),
        and 4th (diversity)
        parts of the analytical experiment
    """
    def __init__(self):
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
            0: "For the above question, could you briefly teach me how to solve it step by step in natural language?",
            1: "Could you tell me how to complete the previous function step by step in natural language?",
            2: "Could you explain how to complete the function in detail, using simple language and a step-by-step approach?",
            3: "Could you use natural language to explain the solution to the previous problem in simple, clear steps?",
            4: "Would you walk me through how to solve the question above, using natural language?",
            5: "Please break down the solution to the question above in an easy-to-follow manner and tell me in natural language."
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
            2: "\nRemember to exclude comments and test cases.\nWarp your code with \"'''\"",
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
        #     egn : example number    int:(>0<~3) or str: 'D'
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
        egn : example number    >>  int: (>0<~3) or str: 'D'
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

    def change_system_prompt(self, prompts):
        """
        change the system prompt according to modify plan
        """
        if self._modify_plan['sys'] != 'base':
            print(f"modifying {len(prompts)} samples, change the system prompt to '{self._modify_plan['sys']}' version")
            for line in prompts:
                raise NotImplementedError
        return prompts
