import sqlite3
import pandas as pd
import pickle
import numpy as np
import scipy
from math import sqrt
from statsmodels.stats.power import TTestIndPower
import pdb
import pingouin as pg
import sys, os
import csv
from datetime import datetime
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(cur_dir + '/../simple_game_test/app/augmented_taxi/policy_summarization'))
# import policy_summarization.BEC_helpers as BEC_helpers


# ---------------------------------- Global Variables ---------------------------------- #
CONFIDENCE_MAPPING = {0: '0 - Strongly Disagree', 1: '1 - Disagree', 2: '2 - Neutral', 3: '3 - Agree',
                      4: '4 - Strongly Agree'}
FRUSTRATION_MAPPING = {0: '0 - Extremely Frustrating', 1: '1 - Somewhat Frustrating',
                       2: '2 - Neither Frustrating nor Pleasant', 3: '3 - Somewhat Pleasant',
                       4: '4 - Extremely Pleasant'}

ONLINE_INTERACTION_TYPES = ['no_feedback', 'showing', 'preference', 'binary_combined', 'credit_assignment']

LOOP_MAPPING = {'cl': 'Full', 'pl': 'Partial', 'open': 'Open', 'wt': 'Direct reward', 'wtcl': "Joint"}

alpha = 0.05

skip_users = []
# remove users who are missing data (the first two are missing reward weight estimates and one of the final tests,
# and the final person is a pl participant who was not able to provide reward weight estimates)
skip_users.extend(['58aca85e0da7f10001de92d4', '602fb33786b077cdbad2d5bb', '54420ec6faa52483c585496', '5dc5da21d999de45a504651b'])
# remove users who did not complete the study
skip_users.extend(['614b55e22ff3944a165736bb', '5f0214e58782120a8c970fd6', '649cbefc9ac085131a41f626', '56cb314508d7300005fd312a'])
# remove users who restarted the study after a partial first attempt (weight only condition)
skip_users.extend(['5cf7796a99ccba000193a8f1'])
# remove users who ended up being able to attempt the study twice in a different condition (though their data isn't correct and/or they returned their submission)
# note that some of these users originally belonged to another condition (and thus should be careful not to remove them from the original condition)
skip_users.extend(['60bdecc23648063db2bb828a', '5c92f797803bff0017fef8dd', '652985d179fb620c91ac14a1', '6511e688100fd33b378fa688',
'58d3bc1642648b0001f21ca4', '5dec6a96da522335df62a1b5', '653e69acd1d1340f02fa4cbc', '5dd05d80b2204f1487f7812c', '59ccd702faf42e00012b68f5'])

# the rationale for removing the users below are explained in the paper
# remove users who perfectly went through the closed-loop teaching framework (and thus did not see any remedial instruction)
skip_users.extend(['59dbbdce5de9b000017ebf19'])
# remove users who underwent more than 3 std dev more interactions than the average in the closed-loop teaching framework
skip_users.extend(['5ebd5fa512c47d04402403da'])

# ---------------------------------- Data Processing ---------------------------------- #

def safe_unpickle(x, default_value):
    try:
        return pickle.loads(x)
    except Exception:
        return default_value
def get_data():
    conn = sqlite3.connect('app.db')

    # Set up Users db
    df_users = pd.read_sql_query('SELECT * FROM user', conn)
    df_users['user_id'] = df_users['id'].astype(np.int64)
    df_users = df_users[df_users['study_completed'] != 0]
    df_users = df_users[df_users['username'].map(lambda d: d not in skip_users)]
    df_users['username'] = df_users.username.str.strip()
    df_users['unpickled_study_type'] = df_users['study_type'].apply(pickle.loads)
    df_users['unpickled_ethnicity'] = df_users['ethnicity'].apply(pickle.loads)
    df_users['unpickled_final_feedback'] = df_users['final_feedback'].apply(pickle.loads)
    df_users['education'] = df_users['education'].apply(lambda x: min(x, 5))

    # these people accidentally entered in their username incorrectly
    df_users.loc[
        df_users[df_users.username == 'https://app.prolific.com/studies/654fde536d706cb9de5b40be?source=pa'].index[
            0], 'username'] = '5fff0b0bcbc37a5f5928ed26'
    df_users.loc[
        df_users[df_users.username == 'https://machine-teaching-study.harp.ri.cmu.edu/login?next=%2F'].index[
            0], 'username'] = '6511f1f1490c9df9f8c7890e'
    df_users.loc[
        df_users[df_users.username == '616ebc0e14bbc010f11935b3@email.prolific.com'].index[
            0], 'username'] = '616ebc0e14bbc010f11935b3'

    # only parse the columns that I want
    df_users = df_users[['username', 'consent', 'age', 'gender', 'unpickled_ethnicity', 'education', 'unpickled_final_feedback', 'loop_condition', 'domain_1', 'domain_2', 'domain_3', 'user_id']]

    # Set up Trials db
    df_trials = pd.read_sql_query('SELECT * FROM trial', conn)
    df_trials['user_id'] = df_trials['user_id'].astype(np.int64)
    df_trials = df_trials[df_trials['user_id'].map(lambda d: d in df_users.user_id.unique())] # remove users who didn't complete the study
    df_trials['unpickled_human_model_pf_weights'] = df_trials['human_model_pf_weights'].apply(pickle.loads)
    df_trials['unpickled_human_model_pf_pos'] = df_trials['human_model_pf_pos'].apply(pickle.loads)
    df_trials['unpickled_mdp_parameters'] = df_trials['mdp_parameters'].apply(pickle.loads)
    df_trials['unpickled_reward_ft_weights'] = df_trials['reward_ft_weights'].apply(pickle.loads)
    df_trials['unpickled_improvement_short_answer'] = df_trials['improvement_short_answer'].apply(
        lambda x: safe_unpickle(x, '')) # some data wasn't stored as bytes
    df_trials['unpickled_moves'] = df_trials['moves'].apply(pickle.loads)
    df_trials['test_difficulty'] = df_trials['unpickled_mdp_parameters'].apply(lambda x: x['test_difficulty'])
    df_trials['tag'] = df_trials['unpickled_mdp_parameters'].apply(lambda x: x['tag'])
    df_trials['opt_traj_reward'] = df_trials['unpickled_mdp_parameters'].apply(lambda x: x['opt_traj_reward'])


    # Set up Domain db (which contains the training likert scales)
    df_domain = pd.read_sql_query('SELECT * FROM domain', conn)
    df_domain['user_id'] = df_domain['user_id'].astype(np.int64)
    df_domain = df_domain[df_domain['user_id'].map(lambda d: d in df_users.user_id.unique())] # remove users who didn't complete the study
    df_domain['unpickled_engagement_short_answer'] = df_domain['engagement_short_answer'].apply(
        lambda x: safe_unpickle(x, '')) # some data wasn't stored as bytes

    df_domain = df_domain[['user_id', 'domain', 'attn1', 'attn2', 'attn3', 'use1', 'use2', 'use3', 'understanding', 'unpickled_engagement_short_answer']]
    # reverse the codes for use1, use2, use3
    df_domain['use1'] = df_domain['use1'].apply(lambda x: 6 - x)
    df_domain['use2'] = df_domain['use2'].apply(lambda x: 6 - x)
    df_domain['use3'] = df_domain['use3'].apply(lambda x: 6 - x)

    df_trials = df_trials[df_trials.user_id.isin(df_users.user_id)]
    df_domain = df_domain[df_domain.user_id.isin(df_users.user_id)]

    df_trials = pd.merge(df_trials, df_users, on='user_id')
    df_domain = pd.merge(df_domain, df_users, on='user_id')

    # individual data clean up
    # user '6176b584db27cdcecc3896c8' (user id 299) wt only condition has a redundant high difficulty test recorded
    df_trials = df_trials.drop(df_trials[(df_trials.username == '6176b584db27cdcecc3896c8') & (df_trials.test_difficulty == 'high')].index[4])
    # user '64699c39810132db3b4250b1' (user id 303) wt only condition has a dud low difficulty test recorded
    df_trials = df_trials.drop(df_trials[(df_trials.username == '64699c39810132db3b4250b1') & (df_trials.test_difficulty == 'low')].index[2])
    # user '62b44f66a16d45783569fad6' (user id 155) has a duplicate row
    df_domain = df_domain.drop(df_domain[df_domain.user_id == 155][:1].index)

    # these people went through the study more than once (for curiosity's sake I'm guessing). remove the second time (all in the weight only condition)
    repeat_usernames = ['5d5fc0ad29b1d80001430de0', '6527e1e86d48566d81c6557c', '6176b584db27cdcecc3896c8', '64699c39810132db3b4250b1', '6511e688100fd33b378fa688']
    drop_indices_trials = []
    for username in repeat_usernames:
        drop_indices_trials.extend(
            df_trials[(df_trials.username == username) & (df_trials.interaction_type == 'final test')].index[12:])
    df_trials = df_trials.drop(drop_indices_trials)

    drop_indices_domain = []
    for username in repeat_usernames:
        drop_indices_domain.extend(df_domain[(df_domain.username == username)].index[2:])
    df_domain = df_domain.drop(drop_indices_domain)

    # this person's data didn't get properly saved but they responded [-2, 1, -1] for the reward weights in a separate message on Prolific
    df_trials = df_trials.copy()
    missing_idx = (df_trials[df_trials.username == '5d49d17b3dad1f0001e2aba1'].interaction_type == 'final test').index[-1]
    df_trials.iloc[missing_idx].unpickled_reward_ft_weights.append('-2')
    df_trials.iloc[missing_idx].unpickled_reward_ft_weights.append('1')

    # swap or remove mistakes in condition labels
    df_trials_open = df_trials[df_trials.loop_condition == 'open']

    # open loop participants should not see diagnostic tests nor diagnostic feedback
    swap_list_open = np.union1d(df_trials_open[(df_trials_open.interaction_type == 'diagnostic test')].username.unique(),
                 df_trials_open[(df_trials_open.interaction_type == 'diagnostic feedback')].username.unique())
    print('Number of incorectly labeled open loop participants: {}'.format(len(swap_list_open)))
    # print(swap_list_open)

    # pl participants should see demos and diagnostic tests
    df_trials_pl = df_trials[df_trials.loop_condition == 'pl']
    swap_list_pl = np.setdiff1d(df_trials_pl[(df_trials_pl.interaction_type == 'demo')].username.unique(),
                                  df_trials_pl[(df_trials_pl.interaction_type == 'diagnostic test')].username.unique())
    print('Number of incorectly labeled pl loop participants: {}'.format(len(swap_list_pl)))

    # cl participants should see demos and diagnostic tests
    df_trials_cl = df_trials[(df_trials.loop_condition == 'cl')]
    swap_list_cl = np.setdiff1d(df_trials_cl[(df_trials_cl.interaction_type == 'demo')].username.unique(),
                                  df_trials_cl[(df_trials_cl.interaction_type == 'diagnostic test')].username.unique())
    print('Number of incorrectly labeled cl participants: {}'.format(len(swap_list_cl)))

    # wt participants should see final tests and diagnostic tests
    df_trials_wt = df_trials[(df_trials.loop_condition == 'wt')]
    swap_list_wt = df_trials_wt[(df_trials_wt.interaction_type == 'demo')].username.unique()
    print('Number of incorrectly labeled wt participants: {}'.format(len(swap_list_wt)))

    # wtcl participants should see final tests and diagnostic tests
    df_trials_wtcl = df_trials[(df_trials.loop_condition == 'wtcl')]
    swap_list_wtcl = np.setdiff1d(df_trials_wtcl[(df_trials_wtcl.interaction_type == 'final test')].username.unique(),
                                df_trials_wtcl[(df_trials_wtcl.interaction_type == 'diagnostic test')].username.unique())
    print('Number of incorrectly labeled wtcl participants: {}'.format(len(swap_list_wtcl)))

    # sanity checks (check for consistency)
    # for username in swap_list_pl:
    #     print(len(df_trials[(df_trials.interaction_type == 'demo') & (df_trials.username == username)]))
    #
    # for username in swap_list_open:
    #     print(len(df_trials[(df_trials.interaction_type == 'diagnostic test') & (df_trials.username == username)]))

    # remove incorrectly labeled conditions
    df_trials = df_trials[~df_trials['username'].isin(swap_list_open)]
    df_trials = df_trials[~df_trials['username'].isin(swap_list_pl)]
    df_trials = df_trials[~df_trials['username'].isin(swap_list_cl)]
    df_trials = df_trials[~df_trials['username'].isin(swap_list_wtcl)]
    df_trials = df_trials[~df_trials['username'].isin(swap_list_wt)]

    df_users = df_users[~df_users['username'].isin(swap_list_open)]
    df_users = df_users[~df_users['username'].isin(swap_list_pl)]
    df_users = df_users[~df_users['username'].isin(swap_list_cl)]
    df_users = df_users[~df_users['username'].isin(swap_list_wtcl)]
    df_users = df_users[~df_users['username'].isin(swap_list_wt)]

    df_domain = df_domain[~df_domain['username'].isin(swap_list_open)]
    df_domain = df_domain[~df_domain['username'].isin(swap_list_pl)]
    df_domain = df_domain[~df_domain['username'].isin(swap_list_cl)]
    df_domain = df_domain[~df_domain['username'].isin(swap_list_wtcl)]
    df_domain = df_domain[~df_domain['username'].isin(swap_list_wt)]

    # string versions of the loop conditions
    df_trials['loop_condition_string'] = df_trials['loop_condition'].apply(lambda x: LOOP_MAPPING[x])
    df_domain['loop_condition_string'] = df_domain['loop_condition'].apply(lambda x: LOOP_MAPPING[x])

    # save the data into a csv for inspection
    df_trials_subset = df_trials[
        ['user_id', 'interaction_type', 'loop_condition', 'is_opt_response', 'unpickled_moves', 'domain',
         'unpickled_mdp_parameters', 'opt_traj_reward', 'test_difficulty', 'tag', 'age', 'gender',
         'unpickled_ethnicity', 'education']]
    df_trials_subset.to_csv('data_dump_corrected.csv', index=False)

    # join the databases (below has the correct data for cl, wt, wtcl)
    with open('dfs_f23_processed_correct_cl_wt_wtcl.pickle', 'rb') as f:
        df_users_correct_cl_wt_wtcl, df_trials_correct_cl_wt_wtcl, df_domain_correct_cl_wt_wtcl = pickle.load(f)

    df_users_correct_cl_wt_wtcl = df_users_correct_cl_wt_wtcl[df_users_correct_cl_wt_wtcl.loop_condition.isin(['cl', 'wt', 'wtcl'])]
    df_trials_correct_cl_wt_wtcl = df_trials_correct_cl_wt_wtcl[df_trials_correct_cl_wt_wtcl.loop_condition.isin(['cl', 'wt', 'wtcl'])]
    df_domain_correct_cl_wt_wtcl = df_domain_correct_cl_wt_wtcl[df_domain_correct_cl_wt_wtcl.loop_condition.isin(['cl', 'wt', 'wtcl'])]

    df_users = df_users[df_users.loop_condition.isin(['pl', 'open'])]
    df_trials = df_trials[df_trials.loop_condition.isin(['pl', 'open'])]
    df_domain = df_domain[df_domain.loop_condition.isin(['pl', 'open'])]

    # sanity check after joining the databases
    overlap = list(set(df_users.username.unique()) & set(df_users_correct_cl_wt_wtcl.username.unique()))
    if len(overlap) > 0:
        print("Overlap between the two datasets: ")
        print(overlap)
    else:
        print("No overlap")

    df_users = pd.concat([df_users, df_users_correct_cl_wt_wtcl]).reset_index()
    df_trials = pd.concat([df_trials, df_trials_correct_cl_wt_wtcl]).reset_index()
    df_domain = pd.concat([df_domain, df_domain_correct_cl_wt_wtcl]).reset_index()

    # sanity check after joining the databases
    # open loop participants should not see diagnostic tests nor diagnostic feedback
    df_trials_open = df_trials[df_trials.loop_condition == 'open']
    swap_list_open = np.union1d(df_trials_open[(df_trials_open.interaction_type == 'diagnostic test')].username.unique(),
                 df_trials_open[(df_trials_open.interaction_type == 'diagnostic feedback')].username.unique())
    print('Number of incorectly labeled open loop participants: {}'.format(len(swap_list_open)))

    # pl participants should see demos and diagnostic tests
    df_trials_pl = df_trials[df_trials.loop_condition == 'pl']
    swap_list_pl = np.setdiff1d(df_trials_pl[(df_trials_pl.interaction_type == 'demo')].username.unique(),
                                  df_trials_pl[(df_trials_pl.interaction_type == 'diagnostic test')].username.unique())
    print('Number of incorectly labeled pl loop participants: {}'.format(len(swap_list_pl)))

    # cl participants should see demos and diagnostic tests
    df_trials_cl = df_trials[(df_trials.loop_condition == 'cl')]
    swap_list_cl = np.setdiff1d(df_trials_cl[(df_trials_cl.interaction_type == 'demo')].username.unique(),
                                  df_trials_cl[(df_trials_cl.interaction_type == 'diagnostic test')].username.unique())
    print('Number of incorrectly labeled cl participants: {}'.format(len(swap_list_cl)))

    # wt participants should see final tests and diagnostic tests
    df_trials_wt = df_trials[(df_trials.loop_condition == 'wt')]
    swap_list_wt = df_trials_wt[(df_trials_wt.interaction_type == 'demo')].username.unique()
    print('Number of incorrectly labeled wt participants: {}'.format(len(swap_list_wt)))

    # wtcl participants should see final tests and diagnostic tests
    df_trials_wtcl = df_trials[(df_trials.loop_condition == 'wtcl')]
    swap_list_wtcl = np.setdiff1d(df_trials_wtcl[(df_trials_wtcl.interaction_type == 'final test')].username.unique(),
                                df_trials_wtcl[(df_trials_wtcl.interaction_type == 'diagnostic test')].username.unique())
    print('Number of incorrectly labeled wtcl participants: {}'.format(len(swap_list_wtcl)))

    conn.close()

    return df_users, df_trials, df_domain

# ---------------------------------- Helper Functions ---------------------------------- #
def id_to_confidence(id):
    return CONFIDENCE_MAPPING[int(id)]

def id_to_frustration(id):
    return FRUSTRATION_MAPPING[int(id)]

def print_demographics(df_users):
    print("\n========== DEMOGRAPHICS ==========")

    print("Conditions Represented (want 68 of each): ")
    print(df_users.loop_condition.value_counts())
    print("Conditions Represented (want 34 of each): ")
    print("cl: ")
    print(df_users[df_users.loop_condition == 'cl'].domain_1.value_counts())
    print("pl: ")
    print(df_users[df_users.loop_condition == 'pl'].domain_1.value_counts())
    print("open: ")
    print(df_users[df_users.loop_condition == 'open'].domain_1.value_counts())
    print("wt: ")
    print(df_users[df_users.loop_condition == 'wt'].domain_1.value_counts())
    print("wtcl: ")
    print(df_users[df_users.loop_condition == 'wtcl'].domain_1.value_counts())

    print("Ages (description)")
    ages = pd.to_numeric(df_users.age)
    print(ages.describe())

    print("Genders: ")
    gender_vals = [0, 0, 0, 0]
    mapping = {0: 'Male', 1: 'Female', 2: 'Non-binary', 3: 'Prefer not to disclose'}
    answers = df_users.gender
    for answer in answers:
        gender_vals[int(answer)] += 1

    for idx, num in enumerate(gender_vals):
        print(mapping[idx] + " : " + str(num) + " (" + str(num / (np.sum(gender_vals))) + "%)")

def validate_submissions(df_users, df_trials, df_domain, use_csv=False):
    if use_csv:
        usernames = []

        # reading csv file
        with open('prolific.csv', 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)

            # extracting each data row one by one
            for row in csvreader:
                usernames.append(row[1])

        # reward_weight_data = df_trials[df_trials['unpickled_reward_ft_weights'].map(lambda d: len(d) > 0)].unpickled_reward_ft_weights
        # improvement_likert_data = df_trials[df_trials.likert > 0]

        # first entry is the column title
        usernames = usernames[1:]
    else:
        usernames = df_domain.username.unique()

    reward_weight_data = df_trials[df_trials['unpickled_reward_ft_weights'].map(lambda d: len(d) > 0)].unpickled_reward_ft_weights
    improvement_likert_data = df_trials[df_trials.likert > 0]

    for j, username in enumerate(usernames):
        flag = False

        if sum(df_trials[df_trials.username == username].interaction_type == 'final test') != 12:
            print("Has more or fewer than 12 tests")
            flag = True

        if len(df_domain[(df_domain.username == username)]) != 2:
            print("Has more or fewer teaching surveys")
            flag = True
        try:
            if df_trials[df_trials.username == username].loop_condition.iloc[0] in ['cl', 'pl', 'open', 'wtcl']:
                if df_trials[df_trials.username == username].loop_condition.iloc[0] in ['cl', 'pl', 'open']:
                    if len(reward_weight_data[df_trials.username == username]) != 2:
                        print("Didn't complete all 2 reward weight tests")
                        flag = True
        except:
            print("Wasn't able to check reward weight data")
            flag = True

            if len(improvement_likert_data[improvement_likert_data.username == username].likert) < 23:
                print("Not enough improvement likert scales")
                flag = True

        if flag:
            print("{} didn't pass the validation".format(username))
        else:
            print("{} passed the validation".format(username))


def print_means(data, dv, iv):
    for i in data[iv].unique():
        print("{} on {} mean: {}".format(i, dv, data[data[iv] == i][dv].mean()))

def perform_mixed_anova(dv, within, subject, between, data):
    aov = pg.mixed_anova(dv=dv, within=within, subject=subject, between=between,
                      data=data, correction=True)

    try: print(aov['p-GG-corr'])
    except: print(aov['p-unc'])
    pg.print_table(aov)

    df_between = data.groupby('username').agg({
        between: 'first',  # get the first loop_condition for each username
        dv: 'mean'  # get the mean of reward_diff for each username
    }).reset_index()

    df_between_within = data.groupby(['username', within]).agg({
        between: 'first',  # get the first loop_condition for each username
        dv: 'mean'  # get the mean of reward_diff for each username
    }).reset_index()

    post_hoc(aov, 0, dv, between, df_between)
    post_hoc(aov, 1, dv, within, df_between_within)

    if within == 'domain':
        data_at = data[data.domain == 'at']
        data_loop_at = data_at.groupby('username').agg({
            between: 'first',  # get the first loop_condition for each username
            dv: 'mean'  # get the mean of reward_diff for each username
        }).reset_index()

        data_sb = data[data.domain == 'sb']
        data_loop_sb = data_sb.groupby('username').agg({
            between: 'first',  # get the first loop_condition for each username
            dv: 'mean'  # get the mean of reward_diff for each username
        }).reset_index()

        print("at tukey")
        post_hoc(aov, 2, dv, between, data_loop_at)

        print("sb tukey")
        post_hoc(aov, 2, dv, between, data_loop_sb)
    else:
        data_low = data[data.test_difficulty == 'low']
        data_loop_low = data_low.groupby('username').agg({
            between: 'first',  # get the first loop_condition for each username
            dv: 'mean'  # get the mean of reward_diff for each username
        }).reset_index()
        data_medium = data[data.test_difficulty == 'medium']
        data_loop_medium = data_medium.groupby('username').agg({
            between: 'first',  # get the first loop_condition for each username
            dv: 'mean'  # get the mean of reward_diff for each username
        }).reset_index()
        data_high = data[data.test_difficulty == 'high']
        data_loop_high = data_high.groupby('username').agg({
            between: 'first',  # get the first loop_condition for each username
            dv: 'mean'  # get the mean of reward_diff for each username
        }).reset_index()

        print("low tukey")
        post_hoc(aov, 2, dv, between, data_loop_low)

        print("medium tukey")
        post_hoc(aov, 2, dv, between, data_loop_medium)

        print("high tukey")
        post_hoc(aov, 2, dv, between, data_loop_high)


# ---------------------------------- Subjective Metrics ---------------------------------- #


# ----------------------------------  Objective Metrics ---------------------------------- #
def post_hoc(aov, location, dv, between, data):
    if ('p-GG-corr' in aov and aov['p-GG-corr'].iloc[location] < alpha) or ('p-GG-corr' not in aov and aov['p-unc'].iloc[location] < alpha):
        print('Reject H0: different distributions across {}. Perform post-hoc Tukey HSD.'.format(aov['Source'][location]))
    else:
        print('Accept H0: Same distributions across {}.'.format(aov['Source'][location]))
    try:
        print("Corrected p-val: {}, DOF effect: {}, DOF error: {}, F: {}".format(aov['p-GG-corr'][location], aov['DF1'][location], aov['DF2'][location], aov['F'][location]))
    except:
        print("Uncorrected p-val: {}, DOF effect: {}, DOF error: {}, F: {}".format(aov['p-unc'][location], aov['DF1'][location], aov['DF2'][location], aov['F'][location]))

    if len(data[between].unique()) > 2:
        print("Tukey HSD")

        pt = pg.pairwise_tukey(dv=dv, between=between, data=data)
        pg.print_table(pt)
    else:
        print("T-test")

        vars = data[between].unique()
        pt = pg.ttest(data[data[between] == vars[0]][dv], data[data[between] == vars[1]][dv])
        pg.print_table(pt)

        for var in vars:
            print("{} mean: {}".format(var, data[data[between] == var][dv].mean()))

def compare_feedback_domain_on_performance(df_trials, dv='reward_diff', within='domain', plot=False):
    print("\n========== ANOVA: TEST DEMONSTRATION PERFORMANCE ==========")

    data = df_trials[df_trials.interaction_type == 'final test']
    perform_mixed_anova(dv, within, 'username', 'loop_condition', data)

    if plot:
        data = df_trials[df_trials.interaction_type == 'final test']
        ax = sns.barplot(data, x='loop_condition_string', y=dv, errorbar='ci',
                         order=["Open", "Partial", "Full"])
        ax.set(xlabel='Feedback Loop', ylabel='Average Reward Gap of Human Test Responses')
        ax.set(title='Effect of Feedback Loop on Reward Gap of Human Test Responses')
        # ax.set_ylim(0, 0.55) # for reward gap
        ax.set_ylim(0, 1)      # for scaled reward
        plt.show()


def compare_feedback_domain_on_engagement(df_domain, within='domain', plot=False):
    df_domain_engagement = pd.DataFrame(
        columns=['username', within, 'loop_condition', 'loop_condition_string', 'engagement', 'attn', 'use'])

    for username in np.unique(df_domain.username):
        for domain in np.unique(df_domain.domain):
            engagement = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 6

            # attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 3
            attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 2

            use = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 3

            loop_condition = df_domain[df_domain.username == username].loop_condition.values[0]
            loop_condition_string = df_domain[df_domain.username == username].loop_condition_string.values[0]

            df_domain_engagement = pd.concat((df_domain_engagement, pd.DataFrame({'username': username, 'loop_condition': loop_condition, 'domain': domain,
                               'loop_condition_string': loop_condition_string, 'engagement': engagement, 'attn': attn, 'use': use})), axis=0, ignore_index=True)

    # print("\n============================== ANOVA: ENGAGEMENT ==============================")
    # print_means(df_domain_engagement, 'engagement', 'loop_condition')
    # perform_mixed_anova(dv='engagement', within=within, subject='username', between='loop_condition',
    #                     data=df_domain_engagement)

    print("\n============================== ANOVA: ATTENTION ==============================")
    print_means(df_domain_engagement, 'attn', 'loop_condition')
    perform_mixed_anova(dv='attn', within=within, subject='username', between='loop_condition',
                        data=df_domain_engagement)

    df_domain_attn = df_domain[['attn1', 'attn2', 'attn3']]
    print("Overall cronbach's alpha: {}".format(pg.cronbach_alpha(df_domain_attn)))
    for key in df_domain_attn.keys():
        df_domain_attn_temp = df_domain_attn.drop(columns=key)
        print("Cronbach's alpha with {} dropped: {}".format(key, pg.cronbach_alpha(df_domain_attn_temp)))

    print("\n============================== ANOVA: USE ==============================")
    print_means(df_domain_engagement, 'use', 'loop_condition')
    perform_mixed_anova(dv='use', within=within, subject='username', between='loop_condition',
                        data=df_domain_engagement)

    df_domain_use = df_domain[['use1', 'use2', 'use3']]
    print("Overall cronbach's alpha: {}".format(pg.cronbach_alpha(df_domain_use)))
    for key in df_domain_use.keys():
        df_domain_use_temp = df_domain_use.drop(columns=key)
        print("Cronbach's alpha with {} dropped: {}".format(key, pg.cronbach_alpha(df_domain_use_temp)))

    if plot:
        ax = sns.barplot(df_domain_engagement[df_domain_engagement.domain == 'at'], x='loop_condition_string', y='use',
                         errorbar='ci',
                         order=["Open", "Partial", "Full"])
        ax.set(xlabel='Feedback Loop', ylabel='Perceived Usability')
        ax.set(title='Effect of Feedback Loop on Perceived Usability (Delivery Domain)')
        ax.set_ylim(-3.5, 0)
        plt.show()

def plot_joint(df_trials, df_domain, within='domain', between='reward_diff'):
    df_domain_engagement = pd.DataFrame(
        columns=['username', within, 'loop_condition', 'loop_condition_string', 'engagement', 'attn', 'use', between])

    for username in np.unique(df_domain.username):
        for domain in np.unique(df_domain.domain):
            # engagement = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 - \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 6

            # attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 3
            attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 2

            use = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 3

            loop_condition = df_domain[df_domain.username == username].loop_condition.values[0]
            loop_condition_string = df_domain[df_domain.username == username].loop_condition_string.values[0]

            if between == 'reward_diff':
                between_val = np.mean(
                    df_trials[(df_trials.username == username) & (df_trials.domain == domain)].reward_diff)
            elif between == 'regret_norm':
                between_val = np.mean(
                    df_trials[(df_trials.username == username) & (df_trials.domain == domain)].regret_norm)
            else:
                raise ValueError("Invalid between-subjects variable")

            df_domain_engagement = pd.concat((df_domain_engagement, pd.DataFrame({'username': username, 'loop_condition': loop_condition, 'domain': domain,
                               'loop_condition_string': loop_condition_string, 'attn': attn, 'use': use, between: between_val})), axis=0, ignore_index=True)


    # Increase the font size for all text elements
    plt.rcParams.update({'font.size': 14})
    bar_width = 0.7
    subtitle_font_size = 14
    fig, axs = plt.subplots(ncols=2, figsize=(7, 6))  # Adjust the figure size

    df_trials = df_trials[df_trials.interaction_type == 'final test']
    sns.barplot(data=df_trials, x='loop_condition_string', y=between, ax=axs[0], errorbar='ci',
                order=["Open", "Partial", "Full"], width=bar_width)  # Adjust the bar width
    axs[0].set(xlabel='Feedback Loop', ylabel='Normalized Regret of Human Test Responses')
    axs[0].set_title('Feedback Loop on Regret \n of Human Test Responses', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, 0.4) # use for regret norm
    axs[0].set_ylim(0, 0.55) # use for reward diff

    df_domain_engagement = df_domain_engagement[df_domain_engagement.domain == 'sb']
    sns.barplot(data=df_domain_engagement, x='loop_condition_string', y='use', ax=axs[1],
                errorbar='ci', order=["Open", "Partial", "Full"], width=bar_width)  # Adjust the bar width
    axs[1].set(xlabel='Feedback Loop', ylabel='Perceived Usability')
    axs[1].set_title('Feedback Loop on Perceived \n Usability (Skateboard Domain)', fontsize=subtitle_font_size-1)
    axs[1].set_ylim(0, 4.5)

    # plt.suptitle('Effect of Feedback Loop', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_joint_attn_improvement(df_trials, df_domain, within='domain', between='reward_diff'):
    df_domain_engagement = pd.DataFrame(
        columns=['username', within, 'loop_condition', 'loop_condition_string', 'engagement', 'attn', 'use', between])

    for username in np.unique(df_domain.username):
        for domain in np.unique(df_domain.domain):
            # engagement = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 - \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 6

            # attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 3
            attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 2

            use = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 3

            loop_condition = df_domain[df_domain.username == username].loop_condition.values[0]
            loop_condition_string = df_domain[df_domain.username == username].loop_condition_string.values[0]

            if between == 'reward_diff':
                between_val = np.mean(
                    df_trials[(df_trials.username == username) & (df_trials.domain == domain)].reward_diff)
            elif between == 'regret_norm':
                between_val = np.mean(
                    df_trials[(df_trials.username == username) & (df_trials.domain == domain)].regret_norm)
            else:
                raise ValueError("Invalid between-subjects variable")

            df_domain_engagement = pd.concat((df_domain_engagement, pd.DataFrame({'username': username, 'loop_condition': loop_condition, 'domain': domain,
                               'loop_condition_string': loop_condition_string, 'attn': attn, 'use': use, between: between_val})), axis=0, ignore_index=True)


    # Increase the font size for all text elements
    plt.rcParams.update({'font.size': 17})
    bar_width = 0.7
    subtitle_font_size = 19
    fig, axs = plt.subplots(ncols=3, figsize=(14, 9))  # Adjust the figure size

    # for plotting subjective results
    ylim = 5
    # ignore the first column (only keep to have the second two columns the same size as the first)
    # data = df_domain_engagement
    # sns.barplot(data=data, x='loop_condition_string', y='use', ax=axs[0],
    #             errorbar='ci', order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    # axs[0].set(xlabel='Explanation Type', ylabel='Perceived Usability')
    # axs[0].set_title('Main Effect (across both domains)', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, ylim)

    # for plotting objective results
    final_test_data = df_trials[df_trials.interaction_type == 'final test']
    sns.barplot(data=final_test_data, x='loop_condition_string', y=between, ax=axs[0], errorbar='ci',
                order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    axs[0].set(xlabel='Explanation Type', ylabel='Regret of Human Test Responses')
    axs[0].set_title('Regret of Human Test Responses', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, 0.4) # use for regret norm
    axs[0].set_ylim(0, 0.95) # use for reward diff

    # attn
    data = df_domain_engagement
    sns.barplot(data=data, x='loop_condition_string', y='attn', ax=axs[1],
                errorbar='ci', order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    axs[1].set(xlabel='Explanation Type', ylabel='Focused Attention Rating')
    axs[1].set_title('Focused Attention Rating', fontsize=subtitle_font_size)
    axs[1].set_ylim(1, ylim)

    # Get the current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Create a new color cycle that starts with the second color
    new_colors = colors[1:] + colors[:1]
    # Set the color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_colors)

    # improvement
    data = df_trials[df_trials.likert > 0]
    sns.barplot(data=data, x='loop_condition_string', y='likert', ax=axs[2],
                errorbar='ci', order=["Full", "Joint"], width=bar_width)  # Adjust the bar width
    axs[2].set(xlabel='Explanation Type', ylabel='Improvement Rating')
    axs[2].set_title('Improvement Rating', fontsize=subtitle_font_size)
    axs[2].set_ylim(1, ylim)

    # plt.suptitle('Explanation Type on Perceived Usability', fontsize=20)

    plt.tight_layout()
    plt.show()
    a = 2

def plot_custom_followup(df_trials, df_domain, within='domain', between='reward_diff'):
    df_domain_engagement = pd.DataFrame(
        columns=['username', within, 'loop_condition', 'loop_condition_string', 'engagement', 'attn', 'use', between])

    for username in np.unique(df_domain.username):
        for domain in np.unique(df_domain.domain):
            # engagement = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 - \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 6

            # attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
            #             df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 3
            attn = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3) / 2

            use = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 3

            loop_condition = df_domain[df_domain.username == username].loop_condition.values[0]
            loop_condition_string = df_domain[df_domain.username == username].loop_condition_string.values[0]

            if between == 'reward_diff':
                between_val = np.mean(
                    df_trials[(df_trials.username == username) & (df_trials.domain == domain)].reward_diff)
            elif between == 'regret_norm':
                between_val = np.mean(
                    df_trials[(df_trials.username == username) & (df_trials.domain == domain)].regret_norm)
            else:
                raise ValueError("Invalid between-subjects variable")

            df_domain_engagement = pd.concat((df_domain_engagement, pd.DataFrame({'username': username, 'loop_condition': loop_condition, 'domain': domain,
                               'loop_condition_string': loop_condition_string, 'attn': attn, 'use': use, between: between_val})), axis=0, ignore_index=True)


    # Increase the font size for all text elements
    plt.rcParams.update({'font.size': 17})
    bar_width = 0.7
    subtitle_font_size = 17
    fig, axs = plt.subplots(ncols=3, figsize=(14, 9))  # Adjust the figure size

    # for plotting objective results
    final_test_data = df_trials[df_trials.interaction_type == 'final test']
    sns.barplot(data=final_test_data, x='loop_condition_string', y=between, ax=axs[0], errorbar='ci',
                order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    axs[0].set(xlabel='Explanation Type', ylabel='Regret of Human Test Responses')
    axs[0].set_title('Main Effect (across both domains)', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, 0.4) # use for regret norm
    axs[0].set_ylim(0, 1.5) # use for reward diff

    data = final_test_data[final_test_data.domain == 'at']
    sns.barplot(data=data, x='loop_condition_string', y=between, ax=axs[1], errorbar='ci',
                order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    axs[1].set(xlabel='Explanation Type', ylabel='Regret of Human Test Responses')
    axs[1].set_title('Delivery Domain', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, 0.4) # use for regret norm
    axs[1].set_ylim(0, 1.5) # use for reward diff

    data = final_test_data[final_test_data.domain == 'sb']
    sns.barplot(data=data, x='loop_condition_string', y=between, ax=axs[2], errorbar='ci',
                order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    axs[2].set(xlabel='Explanation Type', ylabel='Regret of Human Test Responses')
    axs[2].set_title('Skateboard Domain', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, 0.4) # use for regret norm
    axs[2].set_ylim(0, 1.5)  # use for reward diff
    plt.suptitle('Explanation Type on Regret of Human Test Responses', fontsize=20)

    # # for plotting subjective results
    # ylim = 4.55
    # data = df_domain_engagement
    # sns.barplot(data=data, x='loop_condition_string', y='use', ax=axs[0],
    #             errorbar='ci', order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    # axs[0].set(xlabel='Explanation Type', ylabel='Perceived Usability')
    # axs[0].set_title('Main Effect (across both domains)', fontsize=subtitle_font_size)
    # axs[0].set_ylim(0, ylim)
    #
    # data = df_domain_engagement[df_domain_engagement.domain == 'at']
    # sns.barplot(data=data, x='loop_condition_string', y='use', ax=axs[1],
    #             errorbar='ci', order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    # axs[1].set(xlabel='Explanation Type', ylabel='Perceived Usability')
    # axs[1].set_title('Delivery Domain', fontsize=subtitle_font_size)
    # axs[1].set_ylim(0, ylim)
    #
    # data = df_domain_engagement[df_domain_engagement.domain == 'sb']
    # sns.barplot(data=data, x='loop_condition_string', y='use', ax=axs[2],
    #             errorbar='ci', order=["Direct reward", "Full", "Joint"], width=bar_width)  # Adjust the bar width
    # axs[2].set(xlabel='Explanation Type', ylabel='Perceived Usability')
    # axs[2].set_title('Skateboard Domain', fontsize=subtitle_font_size)
    # axs[2].set_ylim(0, ylim)
    # plt.suptitle('Explanation Type on Perceived Usability', fontsize=20)

    plt.tight_layout()
    plt.show()

def compare_feedback_domain_on_understanding(df_domain, within='domain'):

    df_between = df_domain.groupby('username').agg({
        'loop_condition': 'first',  # get the first loop_condition for each username
        'understanding': 'median'  # get the mean of reward_diff for each username
    }).reset_index()

    kruskal = pg.kruskal(dv='understanding', between='loop_condition', data=df_between)
    pg.print_table(kruskal)

    if kruskal['p-unc'][0] < alpha:
        print("There is a significant difference of feedback loop on understanding!")
        print("mean understanding for open: {}".format(
            np.mean(df_between[df_between.loop_condition == 'open'].understanding)))
        print("mean understanding for pl: {}".format(
            np.mean(df_between[df_between.loop_condition == 'pl'].understanding)))
        print("mean understanding for cl: {}".format(
            np.mean(df_between[df_between.loop_condition == 'cl'].understanding)))
        print("mean understanding for wt: {}".format(
            np.mean(df_between[df_between.loop_condition == 'wt'].understanding)))
        print("mean understanding for wtcl: {}".format(
            np.mean(df_between[df_between.loop_condition == 'wtcl'].understanding)))

        for domain in df_domain[within].unique():
            print(domain)
            print("mean understanding for open: {}".format(
                np.mean(df_domain[(df_domain.loop_condition == 'open') & (df_domain.domain == domain)].understanding)))
            print("mean understanding for pl: {}".format(
                np.mean(df_domain[(df_domain.loop_condition == 'pl') & (df_domain.domain == domain)].understanding)))
            print("mean understanding for cl: {}".format(
                np.mean(df_domain[(df_domain.loop_condition == 'cl') & (df_domain.domain == domain)].understanding)))
            print("mean understanding for wt: {}".format(
                np.mean(df_domain[(df_domain.loop_condition == 'wt') & (df_domain.domain == domain)].understanding)))
            print("mean understanding for wtcl: {}".format(
                np.mean(df_domain[(df_domain.loop_condition == 'wtcl') & (df_domain.domain == domain)].understanding)))

        if ('wt' in df_between.loop_condition.unique() and 'cl' in df_between.loop_condition.unique() and 'wtcl' in df_between.loop_condition.unique()):
            print("cl vs wt")
            pg_wilcoxon = pg.wilcoxon(df_between[df_between.loop_condition == 'cl'].understanding,
                                      df_between[df_between.loop_condition == 'wt'].understanding)
            pg.print_table(pg_wilcoxon)

            print("wtcl vs wt")
            pg_wilcoxon = pg.wilcoxon(df_between[df_between.loop_condition == 'wtcl'].understanding,
                                      df_between[df_between.loop_condition == 'wt'].understanding)
            pg.print_table(pg_wilcoxon)

            print("cl vs wtcl")
            pg_wilcoxon = pg.wilcoxon(df_between[df_between.loop_condition == 'cl'].understanding,
                                      df_between[df_between.loop_condition == 'wtcl'].understanding)
            pg.print_table(pg_wilcoxon)
        else:
            raise Exception("Post-hoc analyses haven't been implemented for these conditions.")

    else:
        print("There is no significant difference of feedback loop on understanding!")

    print("domain on understanding")
    pg_wilcoxon = pg.wilcoxon(df_domain[df_domain.domain == 'at'].understanding, df_domain[df_domain.domain == 'sb'].understanding)
    pg.print_table(pg_wilcoxon)

    scipy_wilcoxon = wilcoxon(df_domain[df_domain.domain == 'at'].understanding, df_domain[df_domain.domain == 'sb'].understanding, method='approx')
    print("z-statistic: {}".format(scipy_wilcoxon.zstatistic))


    if pg_wilcoxon['p-val'][0] < alpha:
        print("There is a significant difference of domain on understanding!")
        print("mean understanding for taxi: {}".format(
            np.mean(df_domain[df_domain.domain == 'at'].understanding)))
        print("mean understanding for skateboard: {}".format(
            np.mean(df_domain[df_domain.domain == 'sb'].understanding)))

        print("median understanding for taxi: {}".format(
            np.median(df_domain[df_domain.domain == 'at'].understanding)))
        print("median understanding for skateboard: {}".format(
            np.median(df_domain[df_domain.domain == 'sb'].understanding)))
    else:
        print("There is no significant difference of domain on understanding!")




def compare_feedback_on_improvement(df_trials, within='domain'):
    print("\n========== KRUSKAL: IMPROVEMENT SUBJECTIVE RATING ==========")

    # check if feedback affects ratings on improvement
    data = df_trials[df_trials.likert > 0]

    print_means(data, 'likert', 'loop_condition')

    df_trials_improvement = pd.DataFrame(
        columns=['username', within, 'loop_condition', 'median_improvement'])

    # todo: I think I decided against using the median improvement (and instead opted for parametric analysis since we've obtained many measurements) -- verify and remove this code if so
    for username in np.unique(data.username):
        for domain in np.unique(data.domain):
            median_improvement = float(np.median(data[(data.username == username) & (data.domain == domain)].likert))

            loop_condition = data[data.username == username].loop_condition.values[0]

            df_trials_improvement = pd.concat([df_trials_improvement, pd.DataFrame([{'username': username, 'loop_condition': loop_condition, 'domain': domain,
                                   'median_improvement': median_improvement}])], ignore_index=True)

    perform_mixed_anova('likert', within, 'username', 'loop_condition', data)


def calculate_median_num_interactions(df_trials, condition='cl'):
    print("\n========== MEDIAN NUMBER OF INTERACTIONS ==========")

    exclude = ['diagnostic feedback', 'remedial feedback', 'final test']

    # obtain the number of times each person interacted with Chip in each domain
    num_interactions_at = []
    num_interactions_sb = []
    perfect_training = []
    perfect_interactions_at_usernames = []
    for username in df_trials[df_trials.loop_condition == condition].username.unique():
        num_interactions_at_user = sum(~df_trials[(df_trials.domain == 'at') & (
                df_trials.loop_condition == condition) & (df_trials.username == username)].interaction_type.isin(exclude))
        num_interactions_sb_user = sum(~df_trials[(df_trials.domain == 'sb') & (
                df_trials.loop_condition == condition) & (df_trials.username == username)].interaction_type.isin(exclude))
        num_interactions_at.append(num_interactions_at_user)
        num_interactions_sb.append(num_interactions_sb_user)

        if num_interactions_at_user == 9 and num_interactions_sb_user == 14:
            perfect_training.append(username)
        if num_interactions_at_user == 9:
            perfect_interactions_at_usernames.append(username)

    median_interactions_at = np.median(num_interactions_at)
    median_interactions_sb = np.median(num_interactions_sb)
    print("Median number of closed-loop interactions for taxi: {}".format(median_interactions_at))
    print("Median number of closed-loop interactions for skateboard: {}".format(median_interactions_sb))

    # did anyone go through the training perfectly?
    print("Number of people who perfectly underwent taxi training: {}".format(np.sum(np.array(num_interactions_at) == 9)))
    print("Number of people who perfectly underwent skateboard training: {}".format(np.sum(np.array(num_interactions_sb) == 14)))
    print("People who perfectly went through both training: {}".format(perfect_training))

    # obtain the participants who saw the median number of interactions
    median_interactions_at_users = np.array(df_trials[df_trials.loop_condition == condition].username.unique())[np.array(num_interactions_at) == median_interactions_at]
    median_interactions_sb_users = np.array(df_trials[df_trials.loop_condition == condition].username.unique())[np.array(num_interactions_sb) == median_interactions_sb]
    median_training = []
    for at_user in median_interactions_at_users:
        if at_user in median_interactions_sb_users:
            median_training.append(at_user)
    print("People who went through both median training: {}".format(perfect_training))


def compare_education_domain_on_performance(df_trials, dv='is_opt_response'):
    data = df_trials[df_trials.interaction_type == 'final test']
    perform_mixed_anova(dv, 'domain', 'username', 'education', data)

def composition_closed_loop(df_trials, summation=True):
    '''Calculate the composition of the closed-loop condition (i.e., how many users did the demo, test, feedback)'''

    # 'demo', 'final test', 'diagnostic test', 'diagnostic feedback',
    #        'remedial demo', 'remedial test', 'remedial feedback'

    interaction_types = ['demo', 'final test', 'diagnostic test', 'diagnostic feedback',
    'remedial demo', 'remedial test', 'remedial feedback']

    if summation:
        for domain in df_trials.domain.unique():
            for interaction_type in interaction_types:
                avg_interaction_count = np.sum(df_trials[(df_trials.domain == domain) & (df_trials.interaction_type == interaction_type)].loop_condition == 'cl') / len(df_trials.username.unique())
                print("Avg # of {} interactions for {}: {}".format(interaction_type, domain, avg_interaction_count))
    else:
        for username in df_trials.username.unique():
            print(username)
            for domain in df_trials.domain.unique():
                print(domain)
                if domain == 'at':
                    print(df_trials[(df_trials.username == username) & (df_trials.domain == domain)].interaction_type)

def individual_reward_weight_predictions(df_trials):
    data = df_trials[df_trials['unpickled_reward_ft_weights'].map(lambda d: len(d) > 0)].unpickled_reward_ft_weights
    for username in df_trials.username.unique():
        for domain in df_trials.domain.unique():
            if len(df_trials[(df_trials.username == username) & (df_trials.domain == domain)]) > 0:

                print("Estimation for {} on {}: {}".format(username, domain, data[
                    (df_trials.username == username) & (df_trials.domain == domain)].iloc[0]))

def print_qualitative_feedback(df_trials, df_users, df_domain):
    print("\n========== QUALITATIVE FEEDBACK ==========")

    data = df_trials[df_trials['unpickled_improvement_short_answer'].map(
        lambda d: len(d) > 0)]
    data_domain = df_domain[df_domain['unpickled_engagement_short_answer'].map(
        lambda d: len(d) > 0)]
    data_users = df_users[df_users['unpickled_final_feedback'].map(
        lambda d: len(d) > 0)]

    data.style.set_properties(**{'text-align': 'left'})
    data_domain.style.set_properties(**{'text-align': 'left'})
    data_users.style.set_properties(**{'text-align': 'left'})

    pd.set_option('display.max_colwidth', None)

    width = max(data['unpickled_improvement_short_answer'].str.len().max(), data_domain['unpickled_engagement_short_answer'].str.len().max(), data_users['unpickled_final_feedback'].str.len().max())
    data2 = data.copy()
    data2['unpickled_improvement_short_answer'] = data['unpickled_improvement_short_answer'].str.ljust(width)
    data_domain2 = data_domain.copy()
    data_domain2['unpickled_engagement_short_answer'] = data_domain['unpickled_engagement_short_answer'].str.ljust(width)
    data_users2 = data_users.copy()
    data_users2['unpickled_final_feedback'] = data_users['unpickled_final_feedback'].str.ljust(width)

    for loop_condition in df_trials.loop_condition.unique():
        print("===================================== Loop condition: {} =====================================".format(
            loop_condition))
        if len(data_users2[data_users2.loop_condition == loop_condition].unpickled_final_feedback) > 0:
            print(data_users2[data_users2.loop_condition == loop_condition].unpickled_final_feedback)
        for domain in df_trials.domain.unique():
            print(
                "===================================== Domain condition: {} =====================================".format(
                    domain))

            if len(data2[(data2.loop_condition == loop_condition) & (data2.domain == domain)].unpickled_improvement_short_answer) > 0:
                print(data2[(data2.loop_condition == loop_condition) & (data2.domain == domain)].unpickled_improvement_short_answer)
            if len(data_domain2[(data_domain2.loop_condition == loop_condition) & (data_domain2.domain == domain)].unpickled_engagement_short_answer) > 0:
                print(data_domain2[(data_domain2.loop_condition == loop_condition) & (data_domain2.domain == domain)].unpickled_engagement_short_answer)

    # # by username
    # for username in df_trials.username.unique():
    #     print("Username: {}".format(username))
    #     if len(data2[data.username == username].unpickled_improvement_short_answer) > 0:
    #         print(data2[data.username == username].unpickled_improvement_short_answer)
    #     if len(data_domain[data_domain.username == username].unpickled_engagement_short_answer) > 0:
    #         print(data_domain[data_domain.username == username].unpickled_engagement_short_answer)
    #     if len(data_users[data_users.username == username].unpickled_final_feedback) > 0:
    #         print(data_users[data_users.username == username].unpickled_final_feedback)


def analyze_time_spent(df_trials):
    np.mean(df_trials[df_trials.interaction_type == 'final test'].duration_ms) / 1000

def analyze_reward_weights(df_trials):
    df_trials_reward_weights = df_trials[df_trials['unpickled_reward_ft_weights'].map(lambda d: (len(d) > 0) and len(d[1]) > 0)].copy()
    def normalize_reward_weights(weights):
        normalized_weights = np.array([[float(weights[0]), float(weights[1]), -1]])

        return normalized_weights / np.linalg.norm(normalized_weights[0, :], ord=2)

    df_trials_reward_weights['scaled_reward_ft_weights'] = df_trials_reward_weights['unpickled_reward_ft_weights'].apply(normalize_reward_weights)

    sb_gt_weights = np.array([[0.59565914, 0.3519804, -0.72201107]])
    at_gt_weights = np.array([[-0.63599873, 0.74199852, -0.21199958]])

    def correct_sign(data, iv, gt_weights):
        gt_sign = np.sign(gt_weights)
        for condition in np.unique(data[iv]):
            print(condition)
            incorrect_sign_ct, correct_sign_ct = 0, 0
            data_condition = data[data[iv] == condition]
            for i in range(len(data_condition)):
                if (np.sign(data_condition.iloc[i].scaled_reward_ft_weights) == gt_sign).all():
                    correct_sign_ct += 1
                else:
                    incorrect_sign_ct += 1

            print('correct sign: ', correct_sign_ct)
            print('incorrect sign: ', incorrect_sign_ct)

    print('at')
    correct_sign(df_trials_reward_weights[df_trials_reward_weights.domain == 'at'], 'loop_condition', at_gt_weights)
    print('sb')
    correct_sign(df_trials_reward_weights[df_trials_reward_weights.domain == 'sb'], 'loop_condition', sb_gt_weights)

    # if I want to see how many estimates belong in the BEC area
    # BEC_helpers.sample_human_models_uniform([np.array([[0, 0, -1]])], 50)
    at_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[-1, 0, 2]]), np.array([[0, -1, -4]])]
    sb_BEC_constraints = [np.array([[5, 2, 5]]), np.array([[-6,  4, -3]]), np.array([[ 3, -3,  1]])]


# ---------------------------------- Running the Analysis ---------------------------------- #

if __name__ == '__main__':
    get_remote = False

    if get_remote:
        df_users, df_trials, df_domain = get_data()
        with open('dfs_f23.pickle', 'wb') as f:
            pickle.dump((df_users, df_trials, df_domain), f)
    else:
        with open('dfs_f23_processed.pickle', 'rb') as f:
            df_users, df_trials, df_domain = pickle.load(f)

    # ------------------------ helper functions ------------------------#
    # validate_submissions(df_users, df_trials, df_domain)

    # calculate_median_num_interactions(df_trials, condition='cl')

    # ------------------------ data selection ------------------------#
    # if I'm interested in only considering a subset of the full dataset
    # original user study conditions (cl, pl, open)
    df_users = df_users[(df_users.loop_condition == 'cl') | (df_users.loop_condition == 'pl') | (df_users.loop_condition == 'open')]
    df_trials = df_trials[(df_trials.loop_condition == 'cl') | (df_trials.loop_condition == 'pl') | (df_trials.loop_condition == 'open')]
    df_domain = df_domain[(df_domain.loop_condition == 'cl') | (df_domain.loop_condition == 'pl') | (df_domain.loop_condition == 'open')]

    # follow-up user study on direct reward explanations (cl, wt, wtcl)
    # df_users = df_users[(df_users.loop_condition == 'cl') | (df_users.loop_condition == 'wt') | (df_users.loop_condition == 'wtcl')]
    # df_trials = df_trials[(df_trials.loop_condition == 'cl') | (df_trials.loop_condition == 'wt') | (df_trials.loop_condition == 'wtcl')]
    # df_domain = df_domain[(df_domain.loop_condition == 'cl') | (df_domain.loop_condition == 'wt') | (df_domain.loop_condition == 'wtcl')]

    #------------------------ descriptive statistics ------------------------#
    # individual_performances(df_trials, 'reward_diff')
    # individual_reward_weight_predictions(df_trials)
    # print_qualitative_feedback(df_trials, df_users, df_domain)

    # print_demographics(df_users)

    # ------------------------ primary analyses ------------------------#

    # compare effect of feedback and domain on performance (H1)
    compare_feedback_domain_on_performance(df_trials)

    # compare effect of feedback and domain on focused attn and perceived usability (H2)
    # compare_feedback_domain_on_engagement(df_domain, plot=False)

    # compare effect of feedback and domain on improvement (H3)
    # compare_feedback_on_improvement(df_trials)

    # compare effect of feedback and domain on understanding (H4)
    # compare_feedback_domain_on_understanding(df_domain)

    # custom plots for original user study conditions (cl, pl, open)
    # plot regret and usability (latter only in the delivery domain)
    # plot_joint(df_trials, df_domain, between='reward_diff')
    # plot_joint_attn_improvement(df_trials, df_domain)

    # custom plots for follow-up user study conditions (cl, wt, wtcl)
    # plot_custom_followup(df_trials, df_domain, between='reward_diff')

    #------------------------ secondary analyses ------------------------#

    # compare effect of education on performance
    # compare_education_domain_on_performance(df_trials, 'regret_norm')

    # compare effect of feedback and domain on performance
    # compare_feedback_domain_on_performance(df_trials, 'regret_norm', within='test_difficulty')

    # composition of closed-loop per domain
    # composition_closed_loop(df_trials, summation=False)

    # time spent
    # analyze_time_spent(df_trials)

    # estimating reward weights (IRL)
    # analyze_reward_weights(df_trials)

    # todo: other potential analyses
    #  1. some people take a long time and some people take a short time (can compare the duration -and the number of pages total). is there a difference in performance?
    #  2. how many diagnostic tests were answered correctly but rated as unhelpful for improving their understanding of the game strategy?
    #  3. how many people said that they understood the game strategy but did poorly on the final tests?