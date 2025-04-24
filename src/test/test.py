



import pickle 
import matplotlib.pyplot as plt
import numpy as np

DATA = 'gsm8k_100'
AGENTS = ['L8','L3','M','Q','P']
AGENTS = ['Q'] * 5
# TYPE = 'SPARSE'
TYPE = ''

name = f'{DATA}__{"__".join(AGENTS)}' if TYPE == "" else f'{DATA}_{TYPE}__{"__".join(AGENTS)}'

with open(f'out/debate/{name}.pkl', 'rb') as f:
    data = pickle.load(f)


for sample in data :
    text = ""
    for r in range(3):
        _text = f" -> {str(sample[r]['final_answers'])}"
        text += _text
    text = f"{sample[r]['answer']}:" + text[3:]
    print(text)


corr_to_wrong = [0] * 5
wrong_to_corr = [0] * 5
wrong_to_wrong = [0] * 5
corr_to_corr = [0] * 5
for sample in data :
    for i, (pre, post) in enumerate(zip(sample[0]['final_answer_iscorr'], sample[2]['final_answer_iscorr'])):
        if not pre and post :
            wrong_to_corr[i] += 1 / len(data)
        elif pre and not post :
            corr_to_wrong[i] += 1 / len(data)
        elif pre and post :
            corr_to_corr[i] += 1 / len(data)
        else :
            wrong_to_wrong[i] += 1 / len(data)

x = np.arange(len(AGENTS))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, corr_to_wrong, width, label='Correct → Wrong')
bars2 = ax.bar(x + width/2, wrong_to_corr, width, label='Wrong → Correct')

ax.set_title(f'{name}')
ax.set_xticks(x)
ax.set_xticklabels(AGENTS)
ax.legend()

# Optional: Add bar labels
ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)

plt.tight_layout()
plt.savefig(f'out/answer_change/{name}.png')
plt.close()