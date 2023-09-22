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

cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(cur_dir + '/../simple_game_test/app/augmented_taxi/policy_summarization'))
# import policy_summarization.BEC_helpers as BEC_helpers


# Pallavi To-do Items:
# [ ] Users never get the same rule 2x
# [ ] Disallow clicking on a card that has already been clicked
# [ ] Make "which of the two robots did you prefer" a radio response option, with a textbox for explanation afterwards.
# [ ] df_users['unpickled_feedback_counts'] = df_users['feedback_counts'].apply(pickle.loads) THIS IS A HOLDOVER FROM ROSHNI, DELETE FROM MODEL.


# ---------------------------------- Global Variables ---------------------------------- #
CONFIDENCE_MAPPING = {0: '0 - Strongly Disagree', 1: '1 - Disagree', 2: '2 - Neutral', 3: '3 - Agree',
                      4: '4 - Strongly Agree'}
FRUSTRATION_MAPPING = {0: '0 - Extremely Frustrating', 1: '1 - Somewhat Frustrating',
                       2: '2 - Neither Frustrating nor Pleasant', 3: '3 - Somewhat Pleasant',
                       4: '4 - Extremely Pleasant'}

ONLINE_INTERACTION_TYPES = ['no_feedback', 'showing', 'preference', 'binary_combined', 'credit_assignment']

alpha = 0.05

# ---------------------------------- Data Processing ---------------------------------- #
def get_data():
    conn = sqlite3.connect('app.db')

    # Set up Users db
    df_users = pd.read_sql_query('SELECT * FROM user', conn)
    df_users['user_id'] = df_users['id'].astype(np.int64)
    df_users = df_users[df_users['study_completed'] != 0]
    # df_users = df_users[df_users['username'].map(lambda d: d not in ['mt1', 'mt3'])]  # remove users who didn't complete the study
    df_users['unpickled_study_type'] = df_users['study_type'].apply(pickle.loads)
    df_users['unpickled_ethnicity'] = df_users['ethnicity'].apply(pickle.loads)
    df_users['unpickled_final_feedback'] = df_users['final_feedback'].apply(pickle.loads)

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
    df_trials['unpickled_improvement_short_answer'] = df_trials['improvement_short_answer'].apply(pickle.loads)
    df_trials['test_difficulty'] = df_trials['unpickled_mdp_parameters'].apply(lambda x: x['test_difficulty'])


    # Set up Domain db (which contains the training likert scales)
    df_domain = pd.read_sql_query('SELECT * FROM domain', conn)
    df_domain['user_id'] = df_domain['user_id'].astype(np.int64)
    df_domain = df_domain[df_domain['user_id'].map(lambda d: d in df_users.user_id.unique())] # remove users who didn't complete the study
    df_domain['unpickled_engagement_short_answer'] = df_domain['engagement_short_answer'].apply(pickle.loads)
    df_domain = df_domain[['user_id', 'domain', 'attn1', 'attn2', 'attn3', 'use1', 'use2', 'use3', 'understanding', 'unpickled_engagement_short_answer']]

    df_trials = df_trials[df_trials.user_id.isin(df_users.user_id)]
    df_domain = df_domain[df_domain.user_id.isin(df_users.user_id)]

    df_trials = pd.merge(df_trials, df_users, on='user_id')
    df_domain = pd.merge(df_domain, df_users, on='user_id')

    conn.close()

    return df_users, df_trials, df_domain


def online_sanity_check(df):
    df2 = df[df['fb_type'] != 'no_feedback'][['user_id', 'fb_type']].groupby('user_id', as_index=False).count()
    bad_ids = df2[df2['fb_type'] != 1]['user_id'].values.tolist()
    df2 = df[df['fb_type'] == 'no_feedback'][['user_id', 'fb_type']].groupby('user_id', as_index=False).count()
    bad_ids.extend(df2[df2['fb_type'] != 1]['user_id'].values.tolist())
    if bad_ids:
        print(f'User IDs with incorrect online feedback conditions: {bad_ids}')
        pdb.set_trace()
    else:
        print('Sanity check passed')


# ---------------------------------- Helper Functions ---------------------------------- #
def bonus_msg_to_value(msg):
    if '$' not in msg:
        return 0.0
    return float(msg.split('$')[1][:4])


def id_to_confidence(id):
    return CONFIDENCE_MAPPING[int(id)]


def id_to_frustration(id):
    return FRUSTRATION_MAPPING[int(id)]


# ---------------------------------- Power Analysis ---------------------------------- #
def compute_means(data, column):
    results = {}

    df_no_feedback = df[df['fb_type'] == 'no_feedback']
    for interaction in ONLINE_INTERACTION_TYPES[1:]:
        df_curr = df[df['fb_type'] == interaction]
        user_ids = df_curr['user_id'].unique()
        df_no_feedback_filtered = df_no_feedback[df_no_feedback['user_id'].isin(user_ids)]

        # Could do mode instead, if I wanted
        feedback_tmp = df_curr[column].apply(lambda x: np.mean(x))
        no_feedback_tmp = df_no_feedback_filtered[column].apply(lambda x: np.mean(x))

        feedback = f'Feedback: {feedback_tmp.mean()} +- {feedback_tmp.std()}'
        no_feedback = f'No_Feedback: {no_feedback_tmp.mean()} +- {no_feedback_tmp.std()}'

        results[interaction] = (feedback, no_feedback)
    return results


def compute_power(df, x, y, column, alpha=0.05, power=0.8, alternative='two-sided'):
    df_x = df[df['fb_type'] == x]
    user_ids_x = df_x['user_id'].unique()

    df_y = df[df['fb_type'] == y]
    user_ids_y = df_y['user_id'].unique()

    users = np.intersect1d(user_ids_x, user_ids_y)

    df_x = df_x[df_x['user_id'].isin(users)]
    df_y = df_y[df_y['user_id'].isin(users)]

    # averages
    x_data = df_x[column].str[0]
    y_data = df_y[column].str[0]

    # print(df_x[column])
    # print(df_y[column])
    # x_data = df_x[column].apply(lambda x: np.mean(x))
    # y_data = df_y[column].apply(lambda x: np.mean(x))

    # calculation of effect size
    # size of samples in pilot study
    n1, n2 = len(x_data), len(y_data)

    # variance of samples in pilot study
    s1, s2 = x_data.var(), y_data.var()

    # calculate the pooled standard deviation
    # (Cohen's d)
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    # means of the samples
    u1, u2 = x_data.mean(), y_data.mean()

    # calculate the effect size
    d = (u1 - u2) / s
    print(f'Effect size: {d}')

    # perform power analysis to find sample size
    # for given effect
    obj = TTestIndPower()
    n = obj.solve_power(effect_size=d, alpha=alpha, power=power,
                        ratio=1, alternative=alternative)

    print('Sample size/Number needed in each group: {:.3f}'.format(n))

    return


# ---------------------------------- Subjective Metrics ---------------------------------- #
def compare_termination_confidence(data, version):
    results = {}
    if version == 'online':
        # Wilcoxon Signed-Rank Test for Online Participants
        df_no_feedback = df[df['fb_type'] == 'no_feedback']
        for interaction in ONLINE_INTERACTION_TYPES[1:]:
            # Get the set of users with the interaction condition, and their corresponding no_feedbacks
            df_curr = df[df['fb_type'] == interaction]
            user_ids = df_curr['user_id'].unique()
            df_no_feedback_filtered = df_no_feedback[df_no_feedback['user_id'].isin(user_ids)]

            # Take the average of termination confidences (in case someone terminated multiple times in one game)
            feedback_tmp = df_curr['unpickled_terminate_confidences'].apply(lambda x: np.mean(x))
            no_feedback_tmp = df_no_feedback_filtered['unpickled_terminate_confidences'].apply(lambda x: np.mean(x))

            feedback_confidences = np.array(feedback_tmp)
            no_feedback_confidences = np.array(no_feedback_tmp)

            # null hypothesis is that no_feedback_confidence >= feedback_confidences, hence the alternative is 'less'
            results[interaction] = scipy.stats.wilcoxon(no_feedback_confidences, feedback_confidences,
                                                        alternative='less')

    return results


def compare_frustration(data, version, comparison='within'):
    results = {}
    if version == 'online':
        if comparison == 'within':
            # Wilcoxon Signed-Rank Test for Online Participants
            df_no_feedback = df[df['fb_type'] == 'no_feedback']
            for interaction in ONLINE_INTERACTION_TYPES[1:]:
                # Get the set of users with the interaction condition, and their corresponding no_feedbacks
                df_curr = df[df['fb_type'] == interaction]
                user_ids = df_curr['user_id'].unique()
                df_no_feedback_filtered = df_no_feedback[df_no_feedback['user_id'].isin(user_ids)]

                feedback_tmp = df_curr['frustration']
                no_feedback_tmp = df_no_feedback_filtered['frustration']

                feedback_frustration = np.array(feedback_tmp)
                no_feedback_frustration = np.array(no_feedback_tmp)

                # null hypothesis is that no_feedback_frustration >= feedback_confidences, hence the alternative is 'less'
                # this is because the scale is 1 - strongly disagree to 5 - strongly agree
                results[interaction] = scipy.stats.wilcoxon(no_feedback_frustration, feedback_frustration,
                                                            alternative='less')
        elif comparison == 'between':
            return "Not sure how to do this yet."
    return results


def compare_ease_of_teaching(data, version):
    results = {}
    if version == 'online':
        # Wilcoxon Signed-Rank Test for Online Participants
        df_no_feedback = df[df['fb_type'] == 'no_feedback']
        for interaction in ONLINE_INTERACTION_TYPES[1:]:
            # Get the set of users with the interaction condition, and their corresponding no_feedbacks
            df_curr = df[df['fb_type'] == interaction]
            user_ids = df_curr['user_id'].unique()
            df_no_feedback_filtered = df_no_feedback[df_no_feedback['user_id'].isin(user_ids)]

            feedback_tmp = df_curr['ease_of_teaching']
            no_feedback_tmp = df_no_feedback_filtered['ease_of_teaching']

            feedback_ease = np.array(feedback_tmp)
            no_feedback_ease = np.array(no_feedback_tmp)

            # null hypothesis is that no_feedback_ease >= feedback_confidences, hence the alternative is 'less'
            # this is because the scale is 1 - strongly disagree to 5 - strongly agree
            results[interaction] = scipy.stats.wilcoxon(no_feedback_ease, feedback_ease, alternative='less')
    return results


def compare_utility_of_feedback(version):
    return "TODO"


def print_demographics(version):
    return "TODO"


# ----------------------------------  Objective Metrics ---------------------------------- #
def compare_num_cards_until_convergence(data, version='online'):
    results = {}
    if version == 'online':
        # Wilcoxon Signed-Rank Test for Online Participants
        df_no_feedback = df[df['fb_type'] == 'no_feedback']
        for interaction in ONLINE_INTERACTION_TYPES[1:]:
            # Get the set of users with the interaction condition, and their corresponding no_feedbacks
            df_curr = df[df['fb_type'] == interaction]
            user_ids = df_curr['user_id'].unique()
            df_no_feedback_filtered = df_no_feedback[df_no_feedback['user_id'].isin(user_ids)]

            feedback_tmp = df_curr['n_cards_to_learn_rule']
            no_feedback_tmp = df_no_feedback_filtered['n_cards_to_learn_rule']

            feedback_num = np.array(feedback_tmp)
            no_feedback_num = np.array(no_feedback_tmp)

            # null hypothesis is that no_feedback_num <= feedback_num, hence the alternative is 'greater'
            # this is because we anticipate feedback will lead to fewer cards
            results[interaction] = scipy.stats.wilcoxon(no_feedback_num, feedback_num, alternative='greater')
    return results


def compare_num_cards_between_true_and_believed_convergence(version, mode):
    return "TODO"


def compare_card_selection_duration(version):
    return "TODO"


def compare_session_duration(version):
    return "TODO"


def compare_hypothesis_rate_of_change(version):
    return "TODO"


def compare_rule_similarity_over_time(version):
    return "TODO"


def compare_feedback_domain_on_performance(df_trials):
    # todo: could consider analyzing the scaled rewards of answers rather than binary correctness

    print("\n========== ANOVA: TEST DEMONSTRATION PERFORMANCE ==========")
    aov = pg.mixed_anova(dv='is_opt_response', within='domain', subject='username', between='loop_condition',
                      data=df_trials, correction=True)

    if ('p-GG-corr' in aov and aov['p-GG-corr'].iloc[0] < alpha) or ('p-GG-corr' not in aov and aov['p-unc'].iloc[0] < alpha):
        print('Reject H0: different distributions. Perform post-hoc Tukey HSD.')
        try:
            print("Corrected p-val: {}, DOF effect: {}, DOF error: {}, F: {}".format(aov['p-GG-corr'][0], aov['ddof1'][0], aov['ddof2'][0], aov['F'][0]))
        except:
            print("Uncorrected p-val: {}, DOF effect: {}, DOF error: {}, F: {}".format(aov['p-unc'][0], aov['ddof1'][0], aov['ddof2'][0], aov['F'][0]))

        print("\n========== TUKEY: TEST DEMONSTRATION PERFORMANCE ==========")
        pt = pg.pairwise_tukey(dv='is_opt_response', between='loop_condition', data=df_trials)
        print(pt)
    else:
        print('Accept H0: Same distributions.')


def compare_feedback_domain_on_engagement(df_domain):
    a = 2

    df_domain_engagement = pd.DataFrame(
        columns=['username', 'domain', 'loop_condition', 'engagement'])
    for username in np.unique(df_domain.username):
        for domain in np.unique(df_domain.domain):
            engagement = (df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn1 + df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn2 + \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].attn3 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use1 - \
                        df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use2 - df_domain[(df_domain.username == username) & (df_domain.domain == domain)].use3) / 6

            loop_condition = df_domain[df_domain.username == username].loop_condition.values[0]

            df_domain_engagement = pd.concat((df_domain_engagement, pd.DataFrame({'username': username, 'loop_condition': loop_condition, 'domain': domain,
                               'engagement': engagement})), axis=0, ignore_index=True)



    print("\n========== ANOVA: ENGAGEMENT ==========")
    aov = pg.mixed_anova(dv='engagement', within='domain', subject='username', between='loop_condition',
                      data=df_domain_engagement, correction=True)

def compare_feedback_on_improvement(df_trials):
    # check if feedback affects ratings on improvement
    data = df_trials[df_trials.likert > 0]

    df_trials_improvement = pd.DataFrame(
        columns=['username', 'domain', 'loop_condition', 'median_improvement'])

    for username in np.unique(data.username):
        for domain in np.unique(data.domain):
            median_improvement = float(np.median(data[(data.username == username) & (data.domain == domain)].likert))

            loop_condition = data[data.username == username].loop_condition.values[0]

            df_trials_improvement = pd.concat([df_trials_improvement, pd.DataFrame([{'username': username, 'loop_condition': loop_condition, 'domain': domain,
                                   'median_improvement': median_improvement}])], ignore_index=True)

    kruskal = pg.kruskal(dv='median_improvement', between='loop_condition',
                         data=df_trials_improvement[df_trials_improvement.domain == 'at'])


    # if the anova is significant, run a post-hoc test (per domain)
    mwu = pg.mwu(df_trials_improvement[df_trials_improvement.loop_condition == 'cl'].median_improvement,
                 df_trials_improvement[df_trials_improvement.loop_condition == 'open'].median_improvement)
    print(mwu)



    kruskal = pg.kruskal(dv='median_improvement', between='loop_condition',
                         data=df_trials_improvement[df_trials_improvement.domain == 'sb'])

def calculate_avg_num_interactions(df_trials):
    # calculate the average number of interactions for the closed-loop condition
    avg_interactions_at = np.sum(df_trials[(df_trials.domain == 'at') & (df_trials.loop_condition == 'cl')].interaction_type != 'final test') / len(
        df_trials.username.unique())
    avg_interactions_sb = np.sum(df_trials[(df_trials.domain == 'sb') & (df_trials.loop_condition == 'cl')].interaction_type != 'final test') / len(
        df_trials.username.unique())

    print("Median number of closed-loop interactions for taxi: {}".format(avg_interactions_at))
    print("Median number of closed-loop interactions for skateboard: {}".format(avg_interactions_sb))

def compare_education_domain_on_performance(df_trials):
    print("\n========== ANOVA: TEST DEMONSTRATION PERFORMANCE ==========")
    aov = pg.mixed_anova(dv='is_opt_response', within='domain', subject='username', between='education',
                         data=df_trials, correction=True)

def composition_closed_loop(df_trials):
    # 'demo', 'final test', 'diagnostic test', 'diagnostic feedback',
    #        'remedial demo', 'remedial test', 'remedial feedback'

    interaction_types = ['demo', 'final test', 'diagnostic test', 'diagnostic feedback',
    'remedial demo', 'remedial test', 'remedial feedback']

    for domain in df_trials.domain.unique():
        for interaction_type in interaction_types:
            avg_interaction_count = np.sum(df_trials[(df_trials.domain == domain) & (df_trials.interaction_type == interaction_type)].loop_condition == 'cl') / len(df_trials.username.unique())
            print("Avg # of {} interactions for {}: {}".format(interaction_type, domain, avg_interaction_count))

def analyze_time_spent(df_trials):
    np.mean(df_trials[df_trials.interaction_type == 'final test'].duration_ms) / 1000

# ---------------------------------- Running the Analysis ---------------------------------- #

if __name__ == '__main__':
    # If we are getting the data from theorem, we should process it this way
    # if sys.argv[1] == 'remote':
    #     FLAG_LOCAL_VERSION = False
    #     data = get_data()
    df_users, df_trials, df_domain = get_data()

    calculate_avg_num_interactions(df_trials)

    # ------------------------ primary analyses ------------------------#

    # compare effect of feedback and domain on performance
    compare_feedback_domain_on_performance(df_trials)

    # compare effect of feedback and domain on engagement
    compare_feedback_domain_on_engagement(df_domain)

    # compare effect of feedback and domain on improvement
    compare_feedback_on_improvement(df_trials)


    #------------------------ secondary analyses ------------------------#

    # compare effect of education on performance
    compare_education_domain_on_performance(df_trials)

    # composition of closed-loop per domain
    composition_closed_loop(df_trials)

    # time spent
    analyze_time_spent(df_trials)

    # estimating reward weights (IRL)
    df_trials[df_trials['unpickled_reward_ft_weights'].map(lambda d: len(d) > 0)].unpickled_reward_ft_weights









    #
    # data = df_trials
    # df_trials_reward_weights = pd.DataFrame(
    #     columns=['username', 'domain', 'loop_condition', 'median_improvement'])
    #
    # for username in np.unique(data.username):
    #     for domain in np.unique(data.domain):
    #
    #         median_improvement = float(np.median(data[(data.username == username) & (data.domain == domain)].likert))
    #
    #         loop_condition = data[data.username == username].loop_condition.values[0]
    #
    #         df_trials_reward_weights = pd.concat([df_trials_reward_weights, pd.DataFrame(
    #             [{'username': username, 'loop_condition': loop_condition, 'domain': domain,
    #               'median_improvement': median_improvement}])], ignore_index=True)
    #
    #
    #
    df_trials_reward_weights = df_trials[df_trials['unpickled_reward_ft_weights'].map(lambda d: len(d) > 0)].copy()
    def normalize_reward_weights(weights):
        normalized_weights = np.array([[float(weights[0]), float(weights[1]), -1]])

        return normalized_weights / np.linalg.norm(normalized_weights[0, :], ord=2)

    df_trials_reward_weights['scaled_reward_ft_weights'] = df_trials_reward_weights['unpickled_reward_ft_weights'].apply(normalize_reward_weights)


    # BEC_helpers.sample_human_models_uniform([np.array([[0, 0, -1]])], 50)
    at_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[-1, 0, 2]]), np.array([[0, -1, -4]])]
    sb_BEC_constraints = [np.array([[5, 2, 5]]), np.array([[-6,  4, -3]]), np.array([[ 3, -3,  1]])]







    print("\n==================== H1 ====================")
    print("\n========== ANOVA: TEST DEMONSTRATION PERFORMANCE ==========")
    # aov = pg.rm_anova(dv='scaled_human_reward', within=['test_difficulty'], subject='uniqueid',
    #                   data=df, correction=True)
    aov = pg.rm_anova(dv='is_opt_response', within=['domain'], subject='username',
                      data=df_trials, correction=True)

    df_trials[df_trials['interaction_type'] == 'final test'].moves

