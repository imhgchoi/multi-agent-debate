
DATASETS = ['arithmetics_100','gsm8k_300','pro_medicine_0','formal_logic_0','hellaswag_300','csqa_300','hh_rlhf_300']
DATANAME = ['Arithmetics','GSM8K','Pro. Medicine','Formal Logic','HellaSwag', 'CSQA', 'HH-RLHF']
model = 'qwen2.5-7b'
N=5
R=5

TYPE = ''
# TYPE = 'SPARSE'


dictionary = {}
for DATA in DATASETS:
    name = f'{DATA}__{model}_N={N}_R=5'
    if TYPE != '' : name += f'_{TYPE}'

    with open(f'out/debate/{name}.pkl', 'rb') as f:
        data = pickle.load(f)

    