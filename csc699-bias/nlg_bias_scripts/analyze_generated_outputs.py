"""Script to analyze model's generated outputs."""


import argparse
import numpy as np
import pandas as pd
import os
import random
import time
from constants import *
from util import format_score_sentence_output
from collections import Counter
from collections import OrderedDict
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.font_manager as font_manager
import warnings
warnings.simplefilter("ignore")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
warnings.simplefilter("default")


def calc_sample_scores(files, first_period=True, score_type='vader'):
    """Calculate/format scores for samples."""
    scores = []
    lines = []

    for fi_idx, fi in enumerate(files):
        with open(fi, 'r') as f:
            for line in f:
                line = line.strip()
                sample = line.split('\t')[-1]
                if first_period:
                    # Cut off the line when we see the first period.
                    if '.' in sample:
                        period_idx = sample.index('.')
                    else:
                        period_idx = len(sample)
                    sample_end = min(period_idx + 1, len(sample))
                else:
                    sample_end = len(sample)
                sample = sample[:sample_end]
                lines.append(sample)

    if score_type == 'textblob':
        for line_idx, line in enumerate(lines):
            blob = TextBlob(line)
            o_score = blob.sentences[0].sentiment.polarity
            scores.append(o_score)
    elif score_type == 'vader':
        def sentiment_analyzer_scores(sent):
            vader_score = analyzer.polarity_scores(sent)
            return vader_score
        analyzer = SentimentIntensityAnalyzer()
        for line_idx, line in enumerate(lines):
            score = sentiment_analyzer_scores(line)
            c = score['compound']
            if c >= 0.05:
                scores.append(1)
            elif c <= -0.05:
                scores.append(-1)
            else:
                scores.append(0)
    elif score_type == 'bert':
        for fi in files:  # Analyze the classifier-labeled samples.
            with open(fi) as f:
                for line in f:
                    line = line.strip()
                    line_split = line.split('\t')
                    score = int(line_split[0])
                    scores.append(score)
    else:
        raise NotImplementedError('score_type = textblob, vader, bert')

    assert(len(scores) == len(lines))

    return list(zip(lines, scores))


def plot_scores(bias_dim, labeled_file, score_list, label_list, ratio=False):
    """Plot sentiment"""
    width = 0.25
    ind = np.arange(3)
    for score_idx in range(len(score_list)):
        scores = score_list[score_idx]
        label = label_list[score_idx]
        score_counts = Counter()
        for s in scores:
            if s >= 0.05:
                score_counts['+'] += 1
            elif s <= -0.05:
                score_counts['-'] += 1
            else:
                score_counts['0'] += 1
        if ratio:
            if len(scores):
                score_len = float(len(scores))
                score_counts['+'] /= score_len
                score_counts['-'] /= score_len
                score_counts['0'] /= score_len
        ordered_score_counts = [round(score_counts['-'], 3), round(score_counts['0'], 3),
                                    round(score_counts['+'], 3)]
        print('Demographic: %s, # samples: %s, [neg, neu, pos] ratio: %s' % (label, len(scores), ordered_score_counts))

        plt.bar(ind + (score_idx * width), ordered_score_counts, width=width, align='edge',
                label=label)
    plt.xticks(ind + width * 3, ['negative', 'neutral', 'positive'])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, framealpha=0.9)
    n = str(time.time()).replace(".", "")
    p = os.path.dirname(labeled_file)
    pth = os.path.join(p, n + f"_{bias_dim}.png")
    plt.savefig(pth, transparent=True)
    #plt.show()
    
def plot_scores2(base_name, bias_dim, labeled_file, score_list, label_list, ratio=False):
    fig, ax = plt.subplots(figsize=(4.1, 4.5))
    width = .25
    x = np.arange(len(label_list)/2)
    ld = {'neg': [], 'neu': [], 'pos': []}
    for score_idx in range(len(score_list)):
        scores = score_list[score_idx]
        label = label_list[score_idx]
        score_counts = Counter()
        for s in scores:
            if s >= 0.05:
                score_counts['+'] += 1
            elif s <= -0.05:
                score_counts['-'] += 1
            else:
                score_counts['0'] += 1
        if ratio:
            if len(scores):
                score_len = float(len(scores))
                score_counts['+'] /= score_len
                score_counts['-'] /= score_len
                score_counts['0'] /= score_len
        ld['neg'].append(score_counts['-'])
        ld['neu'].append(score_counts['0'])
        ld['pos'].append(score_counts['+'])
    # Black/, Man/, Straight/
    neg = ax.bar(x - width/2, [ld['neg'][0], ld['neg'][2], ld['neg'][4]], width/1.05, 
                 label='negative', color='black')
    neu = ax.bar(x - width/2, [ld['neu'][0], ld['neu'][2], ld['neu'][4]], width/1.05, 
                        bottom=[ld['neg'][0], ld['neg'][2], ld['neg'][4]],
                        label="neutral", color='white', 
                        hatch='...', edgecolor='black')
    pos = ax.bar(x - width/2, 
                        [ld['pos'][0], ld['pos'][2], ld['pos'][4]], 
                        width/1.05, 
                        bottom=[neu + neg for neu, neg in zip([ld['neu'][0], ld['neu'][2], ld['neu'][4]], [ld['neg'][0], ld['neg'][2], ld['neg'][4]])], 
                        label='positive', color='gray')
    # /White, /Woman, /Gay

    neg2 = ax.bar(x + width/2, [ld['neg'][1], ld['neg'][3], ld['neg'][5]], width/1.05, 
                 label='negative', color='black')
    neu2 = ax.bar(x + width/2, [ld['neu'][1], ld['neu'][3], ld['neu'][5]], width/1.05, 
                        bottom=[ld['neg'][1], ld['neg'][3], ld['neg'][5]],
                        label="neutral", color='white', 
                        hatch='...', edgecolor='black')
    pos2 = ax.bar(x + width/2, 
                        [ld['pos'][1], ld['pos'][3], ld['pos'][5]], width/1.05, 
                        bottom=[neu + neg for neu, neg in zip([ld['neu'][1], ld['neu'][3], ld['neu'][5]], [ld['neg'][1], ld['neg'][3], ld['neg'][5]])], 
                        label='positive', color='gray')
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylabel('Regard', family='serif')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='y', direction='in')
    x1 = np.arange(len(label_list)/2)
    x2 = x1 - width/2
    x3 = x1 + width/2
    x4 = x2.tolist()
    x4.extend(x3.tolist())
    ax.set_xticks(np.array(x4))
    xl = [label_list[0], label_list[2], label_list[4]]
    xl2 = [label_list[1], label_list[3], label_list[5]]
    xl.extend(xl2)
    ax.set_xticklabels(family='serif', labels=xl, rotation=25, ha='right')
    ax.set_xticks(x4)
    if "lm1b" in base_name:
        base_name = "LM1B"
    elif "gpt" in base_name:
        base_name = "GPT2"
    else:
        base_name = base_name.capitalize()
    ax.set_title(f"{bias_dim.capitalize()} context(s), {base_name.replace('.tsv', '')} dataset.\n")
    p = matplotlib.patches.Patch(facecolor='white', 
                                 edgecolor='black', 
                                 hatch=r'...', 
                                 label='neutral')
    font = font_manager.FontProperties(family='serif')
    ax.legend(bbox_to_anchor=(0.5, 1.15), 
                       ncol=3, 
                       prop=font, 
                       handles=[neg, p, pos], 
                       loc='lower center', edgecolor='black')

    #plt.tight_layout()
    n = str(time.time()).replace(".", "")
    p = os.path.dirname(labeled_file)
    pth = os.path.join(p, f'{base_name.replace(".tsv", "")}_{n}_{bias_dim}.png')
    plt.savefig(pth, transparent=True)

def plot_scores3(base_name, bias_dim, labeled_file, score_list, label_list, ratio=False):
    fig, ax = plt.subplots(figsize=(4.1, 4.5))
    width = .25
    x = np.arange(len(label_list)/2)
    ld = {'neg': [], 'neu': [], 'pos': []}
    for score_idx in range(len(score_list)):
        scores = score_list[score_idx]
        label = label_list[score_idx]
        score_counts = Counter()
        for s in scores:
            if s >= 0.05:
                score_counts['+'] += 1
            elif s <= -0.05:
                score_counts['-'] += 1
            else:
                score_counts['0'] += 1
        if ratio:
            if len(scores):
                score_len = float(len(scores))
                score_counts['+'] /= score_len
                score_counts['-'] /= score_len
                score_counts['0'] /= score_len
        ld['neg'].append(score_counts['-'])
        ld['neu'].append(score_counts['0'])
        ld['pos'].append(score_counts['+'])
    # Black/, Man/, Straight/ 
    # BW/
    neg = ax.bar(x - width/2, [ld['neg'][0], ld['neg'][2], ld['neg'][4], ld['neg'][6]], width/1.05, 
                 label='negative', color='black')
    neu = ax.bar(x - width/2, [ld['neu'][0], ld['neu'][2], ld['neu'][4], ld['neu'][6]], width/1.05, 
                        bottom=[ld['neg'][0], ld['neg'][2], ld['neg'][4],  ld['neg'][6]],
                        label="neutral", color='white', 
                        hatch='...', edgecolor='black')
    pos = ax.bar(x - width/2, 
                        [ld['pos'][0], ld['pos'][2], ld['pos'][4],  ld['pos'][6]], 
                        width/1.05, 
                        bottom=[neu + neg for neu, neg in zip([ld['neu'][0], ld['neu'][2], ld['neu'][4],  ld['neu'][6]], 
                                                                                              [ld['neg'][0], ld['neg'][2], ld['neg'][4],  ld['neg'][6]])], 
                        label='positive', color='gray')
    # /White, /Woman, /Gay
    # /WW
    neg2 = ax.bar(x + width/2, [ld['neg'][1], ld['neg'][3], ld['neg'][5],  ld['neg'][7]], width/1.05, 
                 label='negative', color='black')
    neu2 = ax.bar(x + width/2, [ld['neu'][1], ld['neu'][3], ld['neu'][5],  ld['neu'][7]], width/1.05, 
                        bottom=[ld['neg'][1], ld['neg'][3], ld['neg'][5],  ld['neg'][7]],
                        label="neutral", color='white', 
                        hatch='...', edgecolor='black')
    pos2 = ax.bar(x + width/2, 
                        [ld['pos'][1], ld['pos'][3], ld['pos'][5],  ld['pos'][7]], width/1.05, 
                        bottom=[neu + neg for neu, neg in zip([ld['neu'][1], ld['neu'][3], ld['neu'][5],  ld['neu'][7]], 
                                                                                              [ld['neg'][1], ld['neg'][3], ld['neg'][5],  ld['neg'][7]])], 
                        label='positive', color='gray')
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_ylabel('Regard', family='serif')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='y', direction='in')
    x1 = np.arange(len(label_list)/2)
    x2 = x1 - width/2
    x3 = x1 + width/2
    x4 = x2.tolist()
    x4.extend(x3.tolist())
    x5 = x1 - width/2
    x4.extend(x5.tolist())
    ax.set_xticks(np.array(x4))
    xl = [label_list[0], label_list[2], label_list[4], label_list[6]]
    xl2 = [label_list[1], label_list[3], label_list[5], label_list[7]]
    xl.extend(xl2)
    ax.set_xticklabels(family='serif', labels=xl, rotation=25, ha='right')
    ax.set_xticks(x4)
    if "lm1b" in base_name:
        base_name = "LM1B"
    elif "gpt" in base_name:
        base_name = "GPT2"
    else:
        base_name = base_name.capitalize()
    ax.set_title(f"{bias_dim.capitalize()} context(s), {base_name.replace('.tsv', '')} dataset.\n")
    p = matplotlib.patches.Patch(facecolor='white', 
                                 edgecolor='black', 
                                 hatch=r'...', 
                                 label='neutral')
    font = font_manager.FontProperties(family='serif')
    ax.legend(bbox_to_anchor=(0.5, 1.15), ncol=3, prop=font,handles=[neg, p, pos], loc='lower center', edgecolor='black')

    #plt.tight_layout()
    n = str(time.time()).replace(".", "")
    p = os.path.dirname(labeled_file)
    pth = os.path.join(p, f'{base_name.replace(".tsv", "")}_{n}_{bias_dim}.png')
    plt.tight_layout()
    plt.savefig(pth, transparent=True)
    
    
def respect_in_string(s):
    """Helper fn to check if respect context is in string s."""
    return KNOWN in s or DESCRIBED in s or REGARDED in s or THOUGHT in s or WELL_KNOWN in s


def respect_start_idx_in_string(s):
    """Helper fn to get leftmost respect context start idx in s."""
    start_idx = 9999
    context_list = [KNOWN, DESCRIBED, REGARDED, THOUGHT, WELL_KNOWN]
    for c in context_list:
        if c in s:
            start_idx = min(start_idx, s.index(c))
    return start_idx


def occupation_in_string(s):
    """Helper fn to check if occupation context is in string s."""
    return WORKED in s or JOB in s or PART_TIME in s or EARNED in s or STARTED_WORKING in s


def occupation_start_idx_in_string(s):
    """Helper fn to get leftmost occupation context start idx in s."""
    start_idx = 9999
    context_list = [WORKED, JOB, PART_TIME, EARNED, STARTED_WORKING]
    for c in context_list:
        if c in s:
            start_idx = min(start_idx, s.index(c))
    return start_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_tsv_file',
                        required=False,
                        default='../data/all_generated_output.tsv',
                        help='TSV file to evaluate.')
    parser.add_argument('--bias_dim',
                        required=False,
                        default='respect',
                        help='Options are `respect`, `occupation` or `all`.')
    parser.add_argument('--first_period',
                        required=False,
                        default=1,
                        help='Whether to cut samples off after first period.')
    parser.add_argument('--model_type',
                        required=False,
                        default='regard2',
                        help='`regard2`, `sentiment2`, `regard1` or `sentiment1`.')
    params = parser.parse_args()

    params.first_period = int(params.first_period) == 1

    print('params', params)

    # Format BERT outputs.
    dir_name = os.path.dirname(params.full_tsv_file)
    base_name = os.path.basename(params.full_tsv_file)
    pred_file = os.path.join(dir_name, params.model_type + '_' + base_name + '_preds.tsv')
    new_lines = format_score_sentence_output(params.full_tsv_file, pred_file)
    labeled_file = os.path.join(dir_name, params.model_type + '_' + base_name + '_labeled.tsv')
    with open(labeled_file, 'w') as o:
        o.write('\n'.join(new_lines))

    sample_to_score = calc_sample_scores([labeled_file],
                                         first_period=params.first_period,
                                         score_type='bert')

    #scores = OrderedDict({BLACK: [], WHITE: [], MAN: [], WOMAN: [], STRAIGHT: [], GAY: [], BW: [], WW: []})
    scores = OrderedDict({BLACK: [], WHITE: [], MAN: [], WOMAN: [], STRAIGHT: [], GAY: []})
    for l, val in sample_to_score:
        occ_idx = occupation_start_idx_in_string(l)
        respect_idx = respect_start_idx_in_string(l)
        if params.bias_dim == 'occupation':
            if occupation_in_string(l) and occ_idx < respect_idx :
                if l.startswith('The Black person'):
                    scores[BLACK].append(val)
                elif l.startswith('The White person'):
                    scores[WHITE].append(val)
                elif l.startswith('The man'):
                    scores[MAN].append(val)
                elif l.startswith('The woman'):
                    scores[WOMAN].append(val)
                elif l.startswith('The straight person'):
                    scores[STRAIGHT].append(val)
                elif l.startswith('The gay person'):
                    scores[GAY].append(val)
                #elif l.startswith('The black woman'):
                    #scores[BW].append(val)
                #elif l.startswith('The white woman'):
                    #scores[WW].append(val)
                else:
                    raise NotImplementedError('Unidentified demographic: %s' % l)
        elif params.bias_dim == 'respect':
            if respect_in_string(l) and respect_idx < occ_idx:
                if l.startswith('The Black person'):
                    scores[BLACK].append(val)
                elif l.startswith('The White person'):
                    scores[WHITE].append(val)
                elif l.startswith('The man'):
                    scores[MAN].append(val)
                elif l.startswith('The woman'):
                    scores[WOMAN].append(val)
                elif l.startswith('The straight person'):
                    scores[STRAIGHT].append(val)
                elif l.startswith('The gay person'):
                    scores[GAY].append(val)
            #elif l.startswith('The black woman'):
                #scores[BW].append(val)
            #elif l.startswith('The white woman'):
                #scores[WW].append(val)
                else:
                    raise NotImplementedError('Unidentified demographic: %s' % l)
        elif params.bias_dim == 'all':
            if l.startswith('The Black person'):
                scores[BLACK].append(val)
            elif l.startswith('The White person'):
                scores[WHITE].append(val)
            elif l.startswith('The man'):
                scores[MAN].append(val)
            elif l.startswith('The woman'):
                scores[WOMAN].append(val)
            elif l.startswith('The straight person'):
                scores[STRAIGHT].append(val)
            elif l.startswith('The gay person'):
                scores[GAY].append(val)
            #elif l.startswith('The black woman'):
                #scores[BW].append(val)
            #elif l.startswith('The white woman'):
                #scores[WW].append(val)
            else:
                raise NotImplementedError('Unidentified demographic: %s' % l)
    
    
    with open(labeled_file + "_scores.txt", "w") as sfile:
            for k, v in scores.items():
                sfile.write(f"{k}: {v}\n")
                
    scores = list(scores.values())

    plot_scores2(base_name, params.bias_dim, labeled_file, scores, [BLACK, WHITE, MAN, WOMAN, STRAIGHT, GAY], ratio=True)
    
    #plot_scores3(base_name, params.bias_dim, labeled_file, scores, [BLACK, WHITE, MAN, WOMAN, STRAIGHT, GAY, BW, WW], ratio=True)


if __name__ == '__main__':
    main()
