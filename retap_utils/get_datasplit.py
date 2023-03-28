"""
Find stratified and balanced data splitting
of development and hold-out test data

balanced for centers and distribution of
updrs tap scores
"""

# import public functions
import numpy as np


def find_dev_holdout_split(
    feats,
    holdout_split = .2,
    accept_perc_range = 3,
    choose_random_split=None,
    centers = ['BER', 'DUS'],
    splits = ['dev', 'hout'],
    to_print=False,
    subs_excl=[],
    traces_excl=[],
    EXCL_4s = False,
):
    """
    Main script to run to get balanced data splitting
    of development and hold out data sets

    Input:
        - choose_random_split: predefine which datasplit
            is used to reproduce data split and results
    """
    excl_states = [111,]

    print('SPLITTING DATA IN DEV AND HOLD-OUT')
    # get unique subs per center, get score-distribution in full data set
    subs_dict, og_score_distr, og_score_perc = get_population_distribution(
        feats=feats,
        holdout_split=holdout_split,
        centers=centers,
        splits=splits,
        to_print=to_print,
        subs_excl=subs_excl,
        traces_excl=traces_excl,
        EXCL_4s=EXCL_4s,
    )
    # define n samples per center for development set
    n_dev = [len(subs_dict[c]) - int(len(subs_dict[c]) * holdout_split) for c in subs_dict]
    n_dev = min(n_dev) - 1  # -1 to ensure large enough hold out sample

    print(f'Original score distribution: {og_score_distr}')
    print(f'Original score %: {og_score_perc}')
    
    # loop over random seeds to test different splitting samples
    for rand_state in np.arange(300):
        # if exact random_split is defined, skip all others
        if choose_random_split:
            if rand_state != choose_random_split: continue
        if rand_state in excl_states: continue

        # get random split of subs per center and their scores
        if EXCL_4s:
            subset_subs, subset_updrs, excl_fours = get_random_split_and_scores(
                feats=feats, subs_dict=subs_dict,
                n_dev=n_dev, rand_state=rand_state,
                EXCL_4s=EXCL_4s,
            )
        else:
            subset_subs, subset_updrs = get_random_split_and_scores(
                feats=feats, subs_dict=subs_dict,
                n_dev=n_dev, rand_state=rand_state,
                EXCL_4s=EXCL_4s,
            )

        # determine score-distribution of applied splitting
        split_scores = get_split_score_distr(subset_updrs)

        # test whether distribution is within accepted range
        accept_split = test_split_distr(
            split_scores, og_score_perc, accept_perc_range
        )

        # if all distribution counts of split are within accepted ranges 
        if accept_split:
            print(f'Accepted Split: random state {rand_state}')
            # show resulting scores distribution
            print(f'\nResulting distributions in splitted data sets:')
            print()
            data_splits = {}
            for split in split_scores.keys():
                data_splits[split] = []  # empty list to store the subs per split
                for cen in subset_subs.keys():
                    data_splits[split].extend(subset_subs[cen][split])  # add subs to final dict
                # print results as feedback
                n_split = sum(split_scores[split].values())
                print(f'\t{split} data set (n = {n_split}):')
                scores = split_scores[split].keys()

                for s in scores:
                    c = split_scores[split][s]
                    print(f'score {s}: # {c} ({round(c / n_split * 100)} %)')
            
            if EXCL_4s: 
                print(f'\tTraces excl as FOURS: {excl_fours}')
                return data_splits, excl_fours
            else: return data_splits
        
        else:
            continue
    
    print('no accepted split found')

    return None


def get_population_distribution(
    feats,
    holdout_split=.2,
    centers=['BER', 'DUS'],
    splits=['dev', 'hout'],
    to_print=True,
    subs_excl=[],
    traces_excl=[],
    EXCL_4s=False,
):
    """
    
    Input:
        - feats: feature classes per trace
    """

    subs = []
    all_updrs = []
    total_score_distr, total_score_perc = {}, {}

    for trace in feats.incl_traces:

        if getattr(feats, trace).sub in subs_excl: continue
        if trace in traces_excl: continue
        if EXCL_4s:
            if getattr(feats, trace).tap_score == 4: continue

        subs.append(getattr(feats, trace).sub)
        all_updrs.append(getattr(feats, trace).tap_score)

    n_total_traces = len(subs)
    if to_print: print(f'# of total traces included: {n_total_traces}')

    scores, counts = np.unique(all_updrs, return_counts=True)
    for score, count in zip(scores, counts):
        if to_print: print(f'\tscore {score}: # {count}, '
                        f'{round(count / len(all_updrs) * 100)} %')
        total_score_distr[score] = count
        total_score_perc[score] = round(count / len(all_updrs) * 100, 1)

    subs = list(set(subs))
    if to_print: print(f'# of unique subjects included {len(subs)}')

    subs_dict = {}
    for c in centers:
        subs_dict[c] = [s for s in subs if s[:3] == c]
        if to_print: print(f'\t# {c} subs: {len(subs_dict[c])}')

    dev_split = 1 - holdout_split

    if to_print: print(f'Aimed numbers: # dev traces: {n_total_traces * dev_split}'
          f', # of hold-out traces: {n_total_traces * holdout_split}')

    return subs_dict, total_score_distr, total_score_perc


def get_random_split_and_scores(
    feats, subs_dict, n_dev, rand_state,
    EXCL_4s=False,
):
    np.random.seed(rand_state)

    subset_subs, subset_updrs = {}, {}
    excluded_fours = []
    
    for cen in subs_dict.keys():
        subset_subs[cen], subset_updrs[cen] = {}, {}

        # subset_subs[cen]['dev'] = random.sample(subs_dict[cen], k=n_dev)
        subset_subs[cen]['dev'] = np.random.choice(
            sorted(subs_dict[cen]), n_dev, replace=False
        )
        subset_subs[cen]['hout'] = [
            s for s in subs_dict[cen]
            if s not in subset_subs[cen]['dev']
        ]

        for split in ['dev', 'hout']:
            subset_updrs[cen][split] = []

            for trace in feats.incl_traces:
                if getattr(feats, trace).sub in subset_subs[cen][split]:
                    if EXCL_4s:
                        if getattr(feats, trace).tap_score == 4:
                            excluded_fours.append(trace)
                            continue
                    
                    subset_updrs[cen][split].append(
                        getattr(feats, trace).tap_score
                    )
    if EXCL_4s: return subset_subs, subset_updrs, excluded_fours
    else: return subset_subs, subset_updrs


def get_split_score_distr(subset_updrs):

    split_scores = {}  # create empty dict to store score counts (total together, not per center)
    
    for split in ['dev', 'hout']:
        split_scores[split] = {}
        for s in range(5): split_scores[split][s] = 0
    
    # count scores and fill distribution
    for cen in subset_updrs.keys():
        for split in subset_updrs[cen].keys():
            scores, counts = np.unique(subset_updrs[cen][split], return_counts=True)
    
            # add scores from both centers to same counting dict (dev and holdout seperately)
            for score, count in zip(scores, counts):
                split_scores[split][score] = split_scores[split][score] + count
    
    return split_scores


def test_split_distr(
    split_scores, og_score_perc, accept_perc_range
):
    accept_split = True  # at beginning of every iteration
    for split in split_scores.keys():

        n_split = sum(split_scores[split].values())
        
        if split == 'hout':
            if n_split < 73: accept_split = False
        
        for score in range(4):
        
            if abs(
                (split_scores[split][score] / n_split * 100) - 
                og_score_perc[score]
            ) > accept_perc_range:

                accept_split = False
        
    
    return accept_split

