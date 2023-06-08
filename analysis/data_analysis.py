import sqlite3
import pandas as pd
import pickle
import numpy as np
import scipy
from math import sqrt
from statsmodels.stats.power import TTestIndPower
import pdb

# Pallavi To-do Items:
# [ ] Users never get the same rule 2x
# [ ] Disallow clicking on a card that has already been clicked 
# [ ] Make "which of the two robots did you prefer" a radio response option, with a textbox for explanation afterwards. 
# [ ] df_users['unpickled_feedback_counts'] = df_users['feedback_counts'].apply(pickle.loads) THIS IS A HOLDOVER FROM ROSHNI, DELETE FROM MODEL.

# ---------------------------------- Global Variables ---------------------------------- # 
CONFIDENCE_MAPPING = {0: '0 - Strongly Disagree', 1: '1 - Disagree', 2: '2 - Neutral', 3: '3 - Agree', 4: '4 - Strongly Agree'}
FRUSTRATION_MAPPING = {0: '0 - Extremely Frustrating', 1: '1 - Somewhat Frustrating', 2: '2 - Neither Frustrating nor Pleasant', 3: '3 - Somewhat Pleasant', 4: '4 - Extremely Pleasant'}

ONLINE_INTERACTION_TYPES = ['no_feedback', 'showing', 'preference', 'binary_combined', 'credit_assignment']
# ---------------------------------- Data Processing ---------------------------------- #
def get_data(): 
	conn = sqlite3.connect('app.db')

	# Set up Users db
	df_users = pd.read_sql_query('SELECT * FROM user', conn)
	df_users['user_id'] = df_users['id'].astype(np.int64)
	df_users = df_users[df_users['study_completed'] != 0]
	df_users['unpickled_study_type'] = df_users['study_type'].apply(pickle.loads)
	df_users['unpickled_ethnicity'] = df_users['ethnicity'].apply(pickle.loads)
	df_users['unpickled_final_feedback'] = df_users['final_feedback'].apply(pickle.loads)

	# Set up Trials db
	df_trials = pd.read_sql_query('SELECT * FROM trial', conn)
	df_trials['user_id'] = df_trials['user_id'].astype(np.int64)
	# df_trials = df_trials[df_trials['study_completed'] != 0]
	df_trials['unpickled_cards_played'] = df_trials['cards_played'].apply(pickle.loads)
	df_trials['unpickled_card_select_times'] = df_trials['card_select_times'].apply(pickle.loads)
	df_trials['unpickled_n_hypotheses_remaining'] = df_trials['n_hypotheses_remaining'].apply(pickle.loads)
	df_trials['unpickled_terminate_confidences'] = df_trials['terminate_confidences'].apply(pickle.loads)
	df_trials['unpickled_feedback_confidences'] = df_trials['feedback_confidences'].apply(pickle.loads)
	df_trials['unpickled_terminate_record'] = df_trials['terminate_record'].apply(pickle.loads)


	# Set up Survey db
	df_survey = pd.read_sql_query('SELECT * FROM survey', conn)
	df_survey['user_id'] = df_survey['user_id'].astype(np.int64)
	df_survey['trial_num'] = df_survey['round_num']

	df = pd.merge(df_trials, df_survey, on=['user_id', 'trial_num'])
	df = pd.merge(df, df_users, on='user_id')

	# Set up OnlineCondition db
	# Don't need this info for analysis, just need to understand what conditions each user had
	df_online = pd.read_sql_query('SELECT * FROM online_condition', conn)
	df_online['unpickled_trials'] = df_online['trials'].apply(pickle.loads)
	df_online['unpickled_feedback_type'] = df_online['feedback_type'].apply(pickle.loads)
	df_online['unpickled_no_feedback_trial'] = df_online['no_feedback_trial'].apply(pickle.loads)
	df_online['unpickled_feedback_trial'] = df_online['feedback_trial'].apply(pickle.loads)

	df['bonus'] = [bonus_msg_to_value(msg) for msg in df['bonus_value']]

	conn.close()

	return df

def online_sanity_check(df):
	df2 = df[df['fb_type'] != 'no_feedback'][['user_id','fb_type']].groupby('user_id', as_index=False).count()
	bad_ids = df2[df2['fb_type'] != 1]['user_id'].values.tolist()
	df2 = df[df['fb_type'] == 'no_feedback'][['user_id','fb_type']].groupby('user_id', as_index=False).count()
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

	#calculation of effect size
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
			results[interaction] = scipy.stats.wilcoxon(no_feedback_confidences, feedback_confidences, alternative='less')

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
				results[interaction] = scipy.stats.wilcoxon(no_feedback_frustration, feedback_frustration, alternative='less')
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

# ---------------------------------- Running the Analysis ---------------------------------- #

if __name__ == '__main__':
	# If we are getting the data from theorem, we should process it this way
    # if sys.argv[1] == 'remote':
    #     FLAG_LOCAL_VERSION = False
    #     data = get_data()
    df = get_data()
    online_sanity_check(df)
    bonus_df = df.groupby('username', as_index=False)['bonus'].sum()

    # print(df[df['fb_type'] != 'no_feedback']['username'])
    #results = compare_num_cards_until_convergence(df)
    # print(results)
    # print(len(df['user_id'].unique()))

    # which users had which conditions
    # print(df[['user_id', 'username', 'online_condition_id', 'study_completed']])

    # print(compute_means(df, 'unpickled_terminate_confidences'))
    compute_power(df, 'no_feedback', 'credit_assignment', 'unpickled_terminate_confidences', alternative='two-sided')
    # print(df['unpickled_final_feedback'].values)
    # else:
    #     filename = sys.argv[2]
    #     data = get_data(sys.argv[2])
    #     # Set global table values
    #     table_setup(data)
    #     flagged_ids = flag_ids()
    #     edit_tables(data)

    #     if sys.argv[3] == 'compare_termination_confidence' or sys.argv[3] == 'all':
    #         compare_termination_confidence()
