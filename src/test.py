import pickle, random, collections
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

random.seed(4)

DATAs = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
model = 'qwen2.5-7b'
N=5
R=5

TYPE = ''
# TYPE = 'SPARSE'

MOMENTUM = 0.4

for DATA in DATAs :
    print(DATA)
    for (R, TYPE) in [(2,''),(3,''),(5,''),(2,'SPARSE'),(3,'SPARSE'),(5,'SPARSE')]:
        name = f'{DATA}__{model}_N={N}_R=5'
        if TYPE != '' : name += f'_{TYPE}'

        with open(f'out/debate/{name}.pkl', 'rb') as f:
            data = pickle.load(f)


        orig_corr = 0
        random_corr = 0
        argmax_corr = 0
        momentum_corr = 0
        oracle_corr = 0
        for sample in data :
            answer = sample[0]['answer']
            orig_corr += int(sample[R]['debate_answer_iscorr'])

            random_resps, argmax_resps, momentum_resps, oracle_resps = [], [], [], []
            for i in range(N):
                self_answers = []
                for j in range(R):
                    self_answers.append(sample[j+1]['final_answers'][i])
                
                # 1.
                for _ in range(1000):
                    random_ans = random.choice(self_answers)
                    if random_ans != '' :
                        break
                random_resps.append(random_ans)

                # 2.
                counter = collections.Counter(self_answers)
                max_count = max(counter.values())
                most_common = [key for key, value in counter.items() if value == max_count]
                argmax_ans = random.choice(most_common) # if there is a tie, will choose randomly
                argmax_resps.append(random_ans)

                # 3.
                for i, ans in enumerate(self_answers) :
                    if i == 0 :
                        curr_ans = ans
                    else :
                        if random.random() < MOMENTUM :
                            curr_ans = curr_ans
                        else :
                            curr_ans = ans
                    momentum_resps.append(curr_ans)

                # 4. ORACLE
                for i, ans in enumerate(self_answers) :
                    if ans == answer :
                        oracle_resps.append(ans)



            
            counter = collections.Counter([x for x in random_resps if x != ""])
            try :
                max_count = max(counter.values())
                most_common = [key for key, value in counter.items() if value == max_count]
            except :
                most_common = ['']
            random_ans = random.choice(most_common) # if there is a tie, will choose randomly

            counter = collections.Counter([x for x in argmax_resps if x != ""])
            try :
                max_count = max(counter.values())
                most_common = [key for key, value in counter.items() if value == max_count]
            except :
                most_common = ['']
            argmax_ans = random.choice(most_common) # if there is a tie, will choose randomly

            counter = collections.Counter([x for x in momentum_resps if x != ""])
            try :
                max_count = max(counter.values())
                most_common = [key for key, value in counter.items() if value == max_count]
            except :
                most_common = ['']
            momentum_ans = random.choice(most_common) # if there is a tie, will choose randomly

            counter = collections.Counter([x for x in oracle_resps if x != ""])
            try :
                max_count = max(counter.values())
                most_common = [key for key, value in counter.items() if value == max_count]
            except :
                most_common = ['']
            oracle_ans = random.choice(most_common) # if there is a tie, will choose randomly

            random_corr += int(random_ans == answer)
            argmax_corr += int(argmax_ans == answer)
            momentum_corr += int(momentum_ans == answer)
            oracle_corr += int(oracle_ans == answer)
            
        # print(orig_corr / len(data))
        # print(oracle_corr / len(data))
        # # print(random_corr / len(data))
        # print(argmax_corr / len(data))
        print(momentum_corr / len(data))
    
