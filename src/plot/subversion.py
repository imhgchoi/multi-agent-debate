import pickle, random, collections
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

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

    init_right_num, init_wrong_num = 0, 0
    prev_right_num, prev_wrong_num = 0, 0
    round_subversion = {'total':[0] * R, 'subvert':[0] * R, 'correct':[0] * R}
    for sample in tqdm(data):
        
        # initial correctness
        init_corr = sample[0]['debate_answer_iscorr']
        if init_corr :
            init_right_num += 1
            prev_right_num += 1
        else :
            init_wrong_num += 1
            prev_wrong_num += 1

        for i in range(1,6):
            round_corr = sample[i]['debate_answer_iscorr']
            round_subversion['total'][i-1] = round_subversion['total'][i-1] + 1
            if init_corr and not round_corr :
                round_subversion['subvert'][i-1] = round_subversion['subvert'][i-1] + 1
            elif not init_corr and round_corr :
                round_subversion['correct'][i-1] = round_subversion['correct'][i-1] + 1


        # for j in range(N):
        #     # initial correctness
        #     init_ans = sample[0]['final_answers'][j]
        #     init_corr = init_ans == sample[0]['answer']
        #     if init_corr :
        #         init_right_num += 1
        #         prev_right_num += 1
        #     else :
        #         init_wrong_num += 1
        #         prev_wrong_num += 1

        #     for i in range(1,6):
        #         round_corr = sample[i]['final_answers'][j] == sample[i]['answer']
        #         round_subversion['total'][i-1] = round_subversion['total'][i-1] + 1
        #         if init_corr and not round_corr :
        #             round_subversion['subvert'][i-1] = round_subversion['subvert'][i-1] + 1
        #         elif not init_corr and round_corr :
        #             round_subversion['correct'][i-1] = round_subversion['correct'][i-1] + 1




    # RATES: LINE PLOT 
    prev_corr, prev_wrong, total, subvert, correct = [0]*R, [0]*R, [0]*R, [0]*R, [0]*R
    for sample in tqdm(data):
        answer = sample[0]['answer']
        for i in range(1,R+1):
            # prev_ans = sample[i-1]['debate_answer']
            # curr_ans = sample[i]['debate_answer']
            # total[i-1] = total[i-1] + 1
            # if prev_ans == answer :
            #     prev_corr[i-1] = prev_corr[i-1] + 1
            # if not prev_ans == answer:
            #     prev_wrong[i-1] = prev_wrong[i-1] + 1
            # if prev_ans == answer and curr_ans != answer :
            #     subvert[i-1] = subvert[i-1] + 1
            # elif prev_ans != answer and curr_ans == answer :
            #     correct[i-1] = correct[i-1] + 1

            for j in range(N):
                prev_ans = sample[i-1]['final_answers'][j]
                curr_ans = sample[i]['final_answers'][j]
                if prev_ans == answer :
                    prev_corr[i-1] = prev_corr[i-1] + 1
                if not prev_ans == answer:
                    prev_wrong[i-1] = prev_wrong[i-1] + 1
                total[i-1] = total[i-1] + 1
                if prev_ans == answer and curr_ans != answer :
                    subvert[i-1] = subvert[i-1] + 1
                elif prev_ans != answer and curr_ans == answer :
                    correct[i-1] = correct[i-1] + 1
                    
    subversion_rates, correction_rates = [], []
    for i in range(R):
        subversion_rates.append(subvert[i] / prev_corr[i])
        correction_rates.append(correct[i] / prev_wrong[i])

    # subversion = [x / init_right_num for x in round_subversion['subvert']]
    # correction = [x / init_wrong_num for x in round_subversion['correct']]
    subversion = [x / len(data) / 5 for x in round_subversion['subvert']]
    correction = [x / len(data) / 5 for x in round_subversion['correct']]

    dictionary[DATA] = (correction, subversion, correction_rates, subversion_rates)
import pdb;pdb.set_trace()
fig, ax = plt.subplots(figsize=(25, 10))

# ax_line = ax.twinx()
# ax_line.set_ylim(-0.5, 0.5)
# ax.set_ylim((-50,50))
# ax_line.set_ylabel('Marginal Rates (%)', fontsize=25)
# ax_line.tick_params(axis='y', labelsize=20)

# Settings
num_rounds = 5
bar_width = 0.05
inner_spacing = 1.05
effective_bar_width = bar_width * inner_spacing
group_width = effective_bar_width * (num_rounds - 1)
x_ticks = []
round_tick_positions = []
round_tick_labels = []

# Plot data
for i, (key, (corrections, subversions, correction_rates, subversion_rates)) in enumerate(dictionary.items()):
    group_start = i * (group_width + bar_width) * 1.2
    round_positions = group_start + np.arange(num_rounds) * effective_bar_width
    x_ticks.append(round_positions.mean())

    # Draw correction and subversion bars
    ax.bar(round_positions, [v * 100 for v in corrections], width=bar_width, color='#f5dc73', label='Accumulative Correction' if i == 0 else "")
    ax.bar(round_positions, [-v * 100 for v in subversions], width=bar_width, color='#1f3b75', hatch='//', edgecolor='white', label='Accumuulative Subversion' if i == 0 else "")

    # # Overlay red line for correction rates (per dataset)
    # ax.plot(round_positions, [v * 100 for v in correction_rates], color='red', linewidth=2, marker='o', label='Marginal Correction' if i == 0 else "")

    # # Optional: overlay blue line for subversion rates (per dataset)
    # ax.plot(round_positions, [-v * 100 for v in subversion_rates], color='blue', linewidth=2, marker='o', label='Marginal Subversion' if i == 0 else "")
    # Overlay red line for correction rates (per dataset)

    # ax_line.plot(round_positions, [v for v in correction_rates],
    #             color='red', linewidth=2, marker='o', label='Marginal Correction' if i == 0 else "")

    # # Overlay blue line for subversion rates (per dataset)
    # ax_line.plot(round_positions, [-v for v in subversion_rates],
    #             color='blue', linewidth=2, marker='o', label='Marginal Subversion' if i == 0 else "")

    round_tick_positions.extend(round_positions)
    round_tick_labels.extend([str(r + 1) for r in range(num_rounds)])

# for i, (key, (corrections, subversions)) in enumerate(dictionary.items()):
#     group_start = i * (group_width + bar_width) * 1.2
#     round_positions = group_start + np.arange(num_rounds) * effective_bar_width
#     x_ticks.append(round_positions.mean())

#     ax.bar(round_positions, [v * 100 for v in corrections], width=bar_width, color='#f5dc73', label='Correction' if i == 0 else "")
#     ax.bar(round_positions, [-v * 100 for v in subversions], width=bar_width, color='#1f3b75', hatch='//', edgecolor='white', label='Subversion' if i == 0 else "")

#     round_tick_positions.extend(round_positions)
#     round_tick_labels.extend([str(r + 1) for r in range(num_rounds)])

# Set main x-tick labels (datasets)
ax.set_xticks(x_ticks)
ax.set_xticklabels(DATANAME, fontsize=25)
# ax.set_xlabel('Dataset', fontsize=30)

# Add a secondary x-axis for debate round numbers with larger font
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(round_tick_positions)
ax2.set_xticklabels(round_tick_labels, fontsize=20)
ax2.tick_params(axis='x', pad=35)
ax2.set_xlabel("Debate Rounds", fontsize=20, labelpad=40)

# Y-axis labels and formatting
ax.set_ylabel('% of Questions', fontsize=30)
ax.tick_params(axis='y', labelsize=25)
ax.axhline(0, color='black', linewidth=0.8)
ax.legend(fontsize=25)

# lines_labels = [ax.get_legend_handles_labels(), ax_line.get_legend_handles_labels()]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# ax.legend(lines, labels, fontsize=20, loc='upper center', ncol=4)

# Gridlines
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('out/subversion.png')
plt.savefig('out/subversion.pdf', format='pdf', bbox_inches='tight')
