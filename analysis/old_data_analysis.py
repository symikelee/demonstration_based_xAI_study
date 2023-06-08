from sqlalchemy import create_engine, MetaData, Table
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as ols
import statsmodels
import itertools
import sys
import statistics
import pingouin as pg
import altair as alt
from altair_saver import save
from ast import literal_eval
import dbkeys

'''
CONSTANTS and FLAGS
'''
ACCURACY_OPTION = 1
RETURN_LIST_OF_AVERAGES = False #if you want the list of all the trials' averages (or total list of reversals) rather than the overall averages
UNSUPPORTED_BROWSER_ERROR = False #for the pilot study, some people used unsupported browsers. If that's the case, this will trigger and adapt the rest of it

FLAG_EXPORT = True
FLAG_LOCAL_VERSION = True
COMPUTE_EFFECT_SIZE = False
DOMAIN_MAPPING = {0: 'SDM', 1: 'Classif.'}
#INTERACTION_MAPPING = {0: 'Demonstrations', 1: 'Preferences', 2: 'Labels', 3: 'Critiques'}
INTERACTION_MAPPING = {0: 'Showing', 1: 'Sorting', 2: 'Characterizing', 3: 'Evaluating'}
CONFIDENCE_MAPPING = {0: '0 - Very Unconfident', 1: '1 - Unconfident', 2: '2 - Confident', 3: '3 - Very Confident'}

df_calibration = []
df_trial = []
df_interim_survey = []
df_post_survey = []
df_misc = []

flagged_ids = []
ids_to_modify = []

# Corrections to make include removing my own testing code, and adjusting participants' responses where explicitly asked
testing_ids = dbkeys.testing_ids
cond_correction = dbkeys.cond_correction
rankings_to_correct = dbkeys.rankings_to_correct
corrections = dbkeys.corrections

'''
 START: BOILER PLATE SET UP
'''
def get_data(filename=''):
    if not FLAG_LOCAL_VERSION:
        db_url = dbkeys.db_url
        table_name = dbkeys.table_name
        data_column_name = dbkeys.data_column_name
        # boilerplace sqlalchemy setup
        engine = create_engine(db_url)
        metadata = MetaData()
        metadata.bind = engine
        table = Table(table_name, metadata, autoload=True)
        # make a query and loop through
        s = table.select()
        rows = s.execute()


        data = []

        #status codes of subjects who completed experiment
        statuses = [3,4,5,7]
        # if you have workers you wish to exclude, add them here 
        # excluding my own test
        exclude = dbkeys.exclude
        for row in rows:
            # only use subjects who completed experiment and aren't excluded
            if row['status'] in statuses and row['uniqueid'] not in exclude:
                data.append(row[data_column_name])
            
        #For use when not excluding participants:
        #for row in rows:
        #        data.append(row[data_column_name])
        #        #ids.append((row['assignmentID'], condition))
                

        # Now we have all participant datastrings in a list.
        # Let's make it a bit easier to work with:

        # parse each participant's datastring as json object
        # and take the 'data' sub-object
        data = [json.loads(part)['data'] for part in data]
        # insert uniqueid field into trialdata in case it wasn't added
        # in experiment:
        for part in data:
            for record in part:
                record['trialdata']['uniqueid'] = record['uniqueid']
            

            
        # flatten nested list so we just have a list of the trialdata recorded
        # each time psiturk.recordTrialData(trialdata) was called.
        data = [record['trialdata'] for part in data for record in part]

        # Put all subjects' trial data into a dataframe object from the
        # 'pandas' python library: one option among many for analysis
        df = pd.DataFrame(data)
        result = df.to_json(orient="table")
        print(result)

    else:
        data = json.load(open(filename))
        first_trial_data = json.load(open(dbkeys.first_set))
        df = pd.DataFrame(data["data"])
        append_df = pd.DataFrame(first_trial_data["data"])

        df = pd.concat([df, append_df], axis=0, join='outer', ignore_index=True)
        
        # Remove these from the data outright, as they were only for testing purposes
        for id in testing_ids:
            df = df[df.uniqueid != id]
        df = df.reset_index(drop=True)

        return df

'''
 END: BOILER PLATE SET UP
'''

'''
 START: TABLE DEFINITIONS
'''
def edit_tables(df): 

    # Need to remove flagged ids from analysis
    for id in flagged_ids:
        df = df[df.uniqueid != id]

    # Have one extra participant, so need to scrap that data to ensure complete counterbalancing
    find_id = df[df['domain'] == cond_correction[0]]
    find_id = find_id[find_id['condition'] == cond_correction[1]]
    correction_id = find_id['uniqueid'].iloc[0]
    df = df[df.uniqueid != correction_id]

    # Two participants indicated that they want to change their orderings
    for idx, comment in enumerate(rankings_to_correct):
        df_correct = df_post_survey[df_post_survey["primary_answer"].str.contains(comment)]
        participant_id = df_correct['uniqueid'].iloc[0]
        curr_answer = df_correct['primary_answer'].iloc[0]
        split = curr_answer.split('@')
        index = df_post_survey[df_post_survey['uniqueid'] == participant_id].index.item()
        for change in range(len(corrections[idx])):
            split[change] = str(corrections[idx][change])
        df.at[index, 'primary_answer'] = "@".join(split)
    table_setup(df, True)

def flag_ids():
    participants = df_interim_survey.uniqueid.unique()
    multiple_responses = []
    missing_attn = []
    incorrect_response = []
    no_secondaries = []

    for participant in participants:
        ## Find participants who completed the survey more than once
        calibs = df_calibration[df_calibration['uniqueid'] == participant]
        if len(calibs['primary_answer']) > 1:
            multiple_responses.append(participant)
            continue
        ## Find participants who missed attention checks
        participant_info = df_interim_survey[df_interim_survey['uniqueid'] == participant]
        num_missed = 0
        num_incorrects = 0
        for answers in participant_info['attn_checks']:
            num_missed += len(answers)
        if num_missed == 2:
            missing_attn.append(participant)
            continue
        participant_responses = df_trial[df_trial['uniqueid'] == participant]

        ## Find participants who missed ALL secondary checks
        participant_secondaries = 0
        for idx, row in participant_responses.iterrows():
            if (row['secondary_rt'] != 'null'):
                participant_secondaries += 1

        if participant_secondaries == 0:
            no_secondaries.append(participant)
            continue

        ## Find participants who selected bogus answers
        if participant_info['domain'].iloc[0] == 0:
            # Preferences checks
            check = participant_responses[participant_responses['interaction_type'] == 1]
            response = check[check['question_id'] == 1]['primary_answer'].item()
            if int(response) == 3:
                # print(participant + " incorrectly responded in (0, 1, 1)")
                num_incorrects += 1

            # Labels checks
            check = participant_responses[participant_responses['interaction_type'] == 2]
            response = check[check['question_id'] == 0]['primary_answer'].item()
            if int(response) == 1:
                # print(participant + " incorrectly responded in (0, 2, 0)")
                num_incorrects += 1

            response = check[check['question_id'] == 1]['primary_answer'].item()
            if int(response) == 0:
                # print(participant + " incorrectly responded in (0, 2, 1)")
                num_incorrects += 1
            if num_incorrects >= 2:
                incorrect_response.append(participant)
                continue

        else:
            # Preferences checks
            check = participant_responses[participant_responses['interaction_type'] == 1]
            response = check[check['question_id'] == 1]['primary_answer']
            if int(response) == 1:
                # print(participant + " incorrectly responded in (1, 1, 1)")
                num_incorrects += 1

            # response = check[check['question_id'] == 3]['primary_answer'].item()
            # if int(response) == 1:
            #     # print(participant + " incorrectly responded in (1, 1, 3)")
            #     num_incorrects += 1

            # Labels checks
            check = participant_responses[participant_responses['interaction_type'] == 2]
            response = check[check['question_id'] == 1]['primary_answer'].item()
            if int(response) == 1:
                # print(participant + " incorrectly responded in (1, 2, 1)")
                num_incorrects += 1

            if num_incorrects >= 2:
                incorrect_response.append(participant)
                continue

    print("Flagging " + str(multiple_responses) + " for completing part/all of the study multiple times.")
    print("Flagging " + str(missing_attn) + " for explicit attention_checks.")
    # print("Flagging " + str(incorrect_response) + " for incorrect responses.")  # Not worth throwing the data away
    # print("Flagging " + str(no_secondaries) + " for no secondary responses.").  # Just have as the max amt of time

    flagged_ids =  multiple_responses + missing_attn 
    # + no_secondaries (it's not 1, it's 5 people -- run this by Henny in the morning)
    # + incorrect_response (vetoed by Henny)
    print("Flagged " + str(len(flagged_ids)) + " total.")

    # Code to look up responses from a particular uniqueid
    # check = df_post_survey[df_post_survey['uniqueid'] == 'debugnihVw:debugpSQIt']
    # for feedback in check['primary_answer']:
    #     print(feedback)
    return flagged_ids

def split_questions_into_columns(df_to, df_from, start):
    for i in range(len(df_from.columns)):
        qname = "Q" + str(i)
        df_to.insert(i + start, qname, df_from[i], True)
    return

def table_setup(df, convert_likert=False):
    global df_calibration;
    global df_confirmation;
    global df_trial;
    global df_interim_survey;
    global df_post_survey;

    # Calibrations
    df_calibration = df[['uniqueid', 'question_type', 'condition', 'domain', 'primary_answer', 'attn_checks', 'vid_curr_time', 'video_loops']]
    df_calibration = df_calibration[df_calibration['question_type'] == 'calibration']

    df_misc = df[['uniqueid', 'question_type', 'condition', 'domain', 'attn_checks']]
    df_misc = df_misc[df_misc['question_type'] == 'interim_survey']

    # Interaction Questions
    df_trial = df[['uniqueid', 'question_type',  'domain', 'interaction_type', 'question_id', 'primary_answer', 'primary_rt', 'primary_misc', 'secondary_rt', 'vid_curr_time', 'video_loops', 'likert_response']] 
    df_trial = df_trial[df_trial['question_type'] == 'interaction']
    # Parse Likert responses

    # Here Q0: The confidence response
    if convert_likert:
        df_trial['likert_response'] = df_trial['likert_response'].apply(likert_to_int, idx='Q0')

    # Interim Surveys
    df_interim_survey = df[['uniqueid', 'question_type', 'domain', 'condition', 'interaction_type', 'primary_answer', 'attn_checks']]
    df_interim_survey = df_interim_survey[df_interim_survey['question_type'] == 'interim_survey']

    # Parse Likert responses
    # Here Q0: mental effort, Q1: insecure/stressed, Q2: unnecess. compelx, Q3: easy to use, Q4: confident, Q5: attn
    df_interim_survey['primary_answer'] = df_interim_survey['primary_answer'].apply(likert_series_to_json)
    df_split = df_interim_survey['primary_answer'].apply(lambda x: pd.Series(int(num) for num in x.split('@')))
    if convert_likert:
        split_questions_into_columns(df_interim_survey, df_split, 3)

    # Post-Study Survey
    df_post_survey = df[['uniqueid', 'question_type', 'condition', 'domain', 'primary_answer']]
    df_post_survey = df_post_survey[df_post_survey['question_type'] == 'post_survey']

    # Parse Likert responses
    # Here, Q0: First choice, Q1: Second Choice, Q2: Third Choice, Q3: Fourth Choice, Q4: Feedback, Q5: Age, Q6: Gender
    df_post_survey['primary_answer'] = df_post_survey['primary_answer'].apply(likert_series_to_json)
    df_split = df_post_survey['primary_answer'].apply(lambda x: pd.Series(x.split('@')))
    if convert_likert:
        split_questions_into_columns(df_post_survey, df_split, 4)

    return df_calibration, df_trial, df_interim_survey, df_post_survey
'''
 END: TABLE DEFINITIONS
'''

''' START: HELPER FNS '''
def id_to_confidence(id):
    return CONFIDENCE_MAPPING[int(id)]

def id_to_domain_name(id):
    return DOMAIN_MAPPING[int(id)]

def id_to_interaction_name(id):
    return INTERACTION_MAPPING[int(id)]

def likert_series_to_json(likert, convert_to_str=True):
    if '@' in likert:
        return likert
    parsed = json.loads(likert)
    ret = ''
    for idx, key in enumerate(parsed.keys()):
        ret += str(parsed[key])
        if idx != len(parsed.keys()) - 1:
            ret += "@"
    return ret

def likert_to_int(likert, idx):
    if '@' in likert:
        return likert
    parsed = json.loads(likert)[idx]
    return int(parsed)

def process_secondary(domain):
    df_trial_domain = df_trial[df_trial['domain'] == domain]
    df_secondaries = pd.DataFrame(columns = ['uniqueid', 'interaction_type', 'secondary_rt'])
    df_incorrects = pd.DataFrame(columns = ['uniqueid', 'interaction_type', 'secondary_rt'])
    df_duplicates = pd.DataFrame(columns = ['uniqueid', 'interaction_type', 'secondary_rt'])

    participants = df_trial_domain.uniqueid.unique()
    # For each question, find the list of corresponding video times
    intervals = json.load(open("pink_intervals.json"))
    n = [intervals['n1'], intervals['n2'], intervals['n3'], intervals['n4'], intervals['n5']]
    q = [intervals['q1'], intervals['q2'], intervals['q3'], intervals['q4'], intervals['q5']]

    avgs = np.empty((0, len(INTERACTION_MAPPING)), int)

    # We need a group of four scores (averaged reactions) per participant
    for participant in participants:
        responses = df_trial_domain[df_trial_domain['uniqueid'] == participant]
        for i in range(len(INTERACTION_MAPPING)):
            responses_i = responses[responses['interaction_type'] == i]

            iter_adjusted_responses = []
            iter_duplicate_responses = 0
            iter_incorrect_responses = 0

            for idx, row in responses_i.iterrows():
                # Get ground-truth intervals for the question
                if (domain == 0 and i != 2) or (domain == 1 and i == 3):
                    curr_intervals = n[int(row['question_id'])]
                else:
                    curr_intervals = q[int(row['question_id'])]

                # Pull relevant data from row
                loops = row['video_loops']
                end_time = row['vid_curr_time']

                adj_start = []
                for loop in range(int(loops) + 1):
                    for interval in curr_intervals:
                        if loop < loops + 1 or interval[0] < end_time:
                            adj_start.append(interval[0])

                if (row['secondary_rt'] != 'null'):
                    curr_responses = literal_eval(row['secondary_rt'])
                else:
                    # print ("Participant " + row['uniqueid'] + " did not have a secondary response for interaction type \
                    #     " + INTERACTION_MAPPING[row['interaction_type']] + " question " + str(row['question_id']))
                    curr_responses = []

                adjusted_responses = []
                duplicate_responses = []
                accounted_for = []
                already_found = [False] * len(adj_start)

                for s_idx, start in enumerate(adj_start):
                    for response in curr_responses:
                        if response in accounted_for:    # this response has already been mapped to a ground truth interval
                            continue

                        diff = response - start
                        if diff <= 2.0 and diff >= 0.0:
                            if not already_found[s_idx]:
                                already_found[s_idx] = True
                                adjusted_responses.append(diff)
                                accounted_for.append(response)
                                continue
                            else:
                                duplicate_responses.append(response)
                                accounted_for.append(response)
                                continue

                incorrect_responses = list(set(curr_responses) - set(accounted_for))
                for found in already_found:
                    if not found:
                        adjusted_responses.append(2)  # the penalty (max time duration) of a pink interval
                iter_incorrect_responses += len(incorrect_responses)
                iter_duplicate_responses += len(duplicate_responses)
                iter_adjusted_responses.extend(adjusted_responses)
            # I think this is all off .... 
            # TODO(pkoppol): This will cause issues if a question if adj_start is []
            avg = np.mean(iter_adjusted_responses)
            inc_len = float(iter_incorrect_responses)
            dup_len = float(iter_duplicate_responses)

            df_secondaries = df_secondaries.append({'uniqueid' : participant, 'interaction_type': i, 'secondary_rt': avg}, ignore_index = True) 
            df_incorrects = df_incorrects.append({'uniqueid' : participant, 'interaction_type': i, 'num_incorrects':  inc_len}, ignore_index = True) 
            df_duplicates = df_duplicates.append({'uniqueid' : participant, 'interaction_type': i, 'num_duplicates': dup_len}, ignore_index = True)
    return df_secondaries, df_incorrects, df_duplicates

''' END: HELPER FNS '''

'''
 START: INFERENTIAL STATISTICS
'''
# TODO(pkoppol): Can alternatively use mode, IQR
# TODO(pkoppol): Should maybe combine w the one below
# https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
def compare_confidence(domain, mode='median'):
    print("\n========== SELF-REPORTED CONFIDENCE IN DOMAIN " + DOMAIN_MAPPING[domain] + " ==========")
    df_confidences = df_trial[df_trial['domain'] == domain]
    participants = df_confidences.uniqueid.unique()

    all_scores = np.empty((0, len(INTERACTION_MAPPING)), int)
    for participant in participants:
        participant_scores = df_confidences[df_confidences['uniqueid'] == participant]
        participant_vals = np.zeros(len(INTERACTION_MAPPING))

        for interaction in range(len(INTERACTION_MAPPING)):
            curr_interactions = participant_scores[participant_scores['interaction_type'] == interaction]
            curr_scores = curr_interactions['likert_response'].to_list()

            if mode == 'median':
                participant_vals[interaction] = statistics.median(curr_scores)
            elif mode == 'mode':
                participant_vals[interaction] = statistics.mode(curr_scores)

        participant_vals = participant_vals.reshape((1, 4))
        all_scores = np.append(all_scores, participant_vals, axis=0)

    alpha = 0.05
    res = pg.friedman(data=df_confidences, dv='likert_response', within='interaction_type', subject='uniqueid')
    print(res)

    combinations = list(itertools.combinations(INTERACTION_MAPPING.keys(), 2))
    bonferroni_adj_alpha = alpha / len(combinations)
    print("Bonferonni alpha is : " + str(bonferroni_adj_alpha))
    for combination in combinations:
        # get difference x - y
        diff = all_scores[:, combination[0]] - all_scores[:, combination[1]]
        less = pg.wilcoxon(all_scores[:, combination[0]], all_scores[:, combination[1]], tail="less")
        if less['p-val'].iloc[0] < bonferroni_adj_alpha:
                print("less : " + str(combination) + " where (Z=" + str(less['W-val'].iloc[0]) + "with p=" + str(less['p-val'].iloc[0]))
                continue
        great = pg.wilcoxon(all_scores[:, combination[0]], all_scores[:, combination[1]], tail="greater")
        if great['p-val'].iloc[0] < bonferroni_adj_alpha:
                print("greater : " + str(combination) +  "where (Z=" + str(great['W-val'].iloc[0]) + " with p=" + str(great['p-val'].iloc[0]))
                continue
        print("less (none): " + str(combination) + " where (Z=" + str(less['W-val'].iloc[0]) + ", p < " + str(less['p-val'].iloc[0]) + ")")

    return

# TODO(pkoppol): Can alternatively use mode, IQR
# TODO(pkoppol): Chrombach's Alpha to see how correlated the questions are .. then can group together possibly, but also 5 isn't that many
# https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
def compare_interim(domain, qid='Q0'):
    print("\n========== INTERIM SURVEYS IN DOMAIN " + DOMAIN_MAPPING[domain] + ", " + qid + " ==========")
    df_interim = df_interim_survey[df_interim_survey['domain'] == domain]
    participants = df_interim.uniqueid.unique()
    q = np.empty((0, len(INTERACTION_MAPPING)), int)

    for participant in participants:
        participant_scores = df_interim[df_interim['uniqueid'] == participant]
        pq = np.zeros(len(INTERACTION_MAPPING))
        for interaction in range(len(INTERACTION_MAPPING)):
            curr_interactions = participant_scores[participant_scores['interaction_type'] == interaction]
            pq[interaction] = curr_interactions[qid].item()
        pq = pq.reshape(1, 4)
        q = np.append(q, pq, axis = 0)

    stat, p = stats.friedmanchisquare(*[q[x, :] for x in np.arange(q.shape[0])])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print(p)
        print('Different distributions (reject H0):')
        combinations = list(itertools.combinations(INTERACTION_MAPPING.keys(), 2))
        bonferroni_adj_alpha = alpha / len(combinations)
        print("Bonferonni alpha is: " + str(bonferroni_adj_alpha))
        for combination in combinations:
            # get difference x - y
            diff = q[:, combination[0]] - q[:, combination[1]]

            # Doesn't work when diff = 0; why would this happen?
            nonzero = False
            for element in diff:
                if element != 0:
                    nonzero = True
            if not nonzero:
                continue

            stat, p_less = stats.wilcoxon(diff, alternative="less")
            stat1, p_great = stats.wilcoxon(diff, alternative="greater")
            if p_less < bonferroni_adj_alpha:
                print("less : " + str(combination) + " where (Z=" + str(stat) + ", p < " + str(p_less) + ")")
                continue
            if p_great < bonferroni_adj_alpha:
                print("greater : " + str(combination) + " where (Z=" + str(stat1) + ", p <" + str(p_great) + ")")
                continue
            print("less (none): " + str(combination) + " where (Z=" + str(stat) + ", p < " + str(p_less) + ")")

def compare_primary_rt(domain, alpha = 0.05):
    print("\n========== PRIMARY RT IN " + DOMAIN_MAPPING[domain] + " ==========")
    df_trial_domain = df_trial[df_trial['domain'] == domain]
    participants = df_trial_domain.uniqueid.unique()
    aov = pg.rm_anova(dv='primary_rt', within='interaction_type', subject='uniqueid', data=df_trial_domain, detailed=True, effsize="np2")
    print(aov)

    if ('p-GG-corr' in aov and aov['p-GG-corr'].iloc[0] < alpha) or ('p-GG-corr' not in aov and aov['p-unc'].iloc[0] < alpha):
        print('Different distributions (reject H0, and perform post-hoc analyses w/ bonferonni adjustment)')
        combinations = list(itertools.combinations(INTERACTION_MAPPING.keys(), 2))
        bonferroni_adj_alpha = alpha / len(combinations)

        ph = pg.pairwise_tukey(data=df_trial_domain, dv='primary_rt', between='interaction_type')
        print(ph)
        for idx, row in ph.iterrows():
            if (row['p-tukey'] < alpha):
                print("(" + INTERACTION_MAPPING[row['A']] + ", " + INTERACTION_MAPPING[row['B']] + ") with p=" + str(row['p-tukey']))

def compare_secondary_rt(domain, alpha = 0.05):
    print("\n========== SECONDARY RT IN " + DOMAIN_MAPPING[domain] + " ==========")
    df_secondaries, df_incorrects, df_duplicates = process_secondary(domain)
    
    aovs = []
    aov_mapping = {0: 'secondary_rt', 1: 'num_incorrects', 2: 'num_duplicates'}
    aovs.append(pg.rm_anova(dv='secondary_rt', within='interaction_type', subject='uniqueid', data=df_secondaries, detailed=True, effsize="np2"))
    aovs.append(pg.rm_anova(dv='num_incorrects', within='interaction_type', subject='uniqueid', data=df_incorrects, detailed=True, effsize="np2"))
    aovs.append(pg.rm_anova(dv='num_duplicates', within='interaction_type', subject='uniqueid', data=df_duplicates, detailed=True, effsize="np2"))
    for idx, aov in enumerate(aovs):
        if ('p-GG-corr' in aov and aov['p-GG-corr'].iloc[0] < alpha) or ('p-GG-corr' not in aov and aov['p-unc'].iloc[0] < alpha):
            # print(aov['p-GG-corr'].iloc[0])
            print(aov)
            print(aov['p-unc'].iloc[0])
            print(aov_mapping[idx] + ': Different distributions (reject H0, and perform post-hoc analyses)')
            combinations = list(itertools.combinations(INTERACTION_MAPPING.keys(), 2))
            bonferroni_adj_alpha = alpha / len(combinations)
            bonferroni_adj_alpha = 0.05
            ph = pg.pairwise_tukey(data=df_secondaries, dv=aov_mapping[idx], between='interaction_type')
            print(ph)
            for idx, row in ph.iterrows():
                if (row['p-tukey'] < bonferroni_adj_alpha):
                    print("(" + INTERACTION_MAPPING[row['A']] + ", " + INTERACTION_MAPPING[row['B']] + ") with p=" + str(row['p-tukey']))
    return 

def compare_variance(domain):
    print("\n========== RESPONSE VARIANCE IN DOMAIN " + DOMAIN_MAPPING[domain] + " ==========")
    df_curr_domain = df_trial[df_trial['domain'] == domain]
    interaction_types = [1, 2]
    for interaction in interaction_types:
        df_curr_type = df_curr_domain[df_curr_domain['interaction_type'] == interaction]
        df_curr_type['primary_answer'] = df_curr_type['primary_answer'].astype(int)
        print(df_curr_type.groupby('question_id')['primary_answer'].var())
    return

'''
 END: INFERENTIAL STATISTICS
'''

'''
 START: DESCRIPTIVE STATISTICS
'''
def describe_confidence(domain):
    print("\n========== (DESCRIPTIVE) SELF-REPORTED CONFIDENCE IN DOMAIN " + DOMAIN_MAPPING[domain] + " ==========")

    df_confidences_0 = df_trial[df_trial['domain'] == 0]
    df_confidences_0['interaction_type'] = df_confidences_0['interaction_type'].apply(id_to_interaction_name)
    df_confidences_0['likert_response'] = df_confidences_0['likert_response'].apply(id_to_confidence)

    df_confidences_1 = df_trial[df_trial['domain'] == 1]
    df_confidences_1['interaction_type'] = df_confidences_1['interaction_type'].apply(id_to_interaction_name)
    df_confidences_1['likert_response'] = df_confidences_1['likert_response'].apply(id_to_confidence)

    d0 = alt.Chart(df_confidences_0).mark_bar().encode(
        x= alt.X('count(likert_response):Q', axis=alt.Axis(title='Number of Responses per Scale Item')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
        color=alt.Color('likert_response:O', scale=alt.Scale(scheme='purplegreen'), legend=alt.Legend(title="SDM responses",orient='top')),
        order=alt.Order('likert_response:O', sort='ascending')
    ).properties(
        title='Per-trial Confidence',
    )

    d1 = alt.Chart(df_confidences_1).mark_bar().encode(
        x= alt.X('count(likert_response):Q', axis=alt.Axis(title='Number of Responses per Scale Item')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
        color=alt.Color('likert_response:O', scale=alt.Scale(scheme='blueorange'), legend=alt.Legend(title="Classification responses",orient='top')),
        order=alt.Order('likert_response:O', sort='ascending')
    ).properties(
        title='Per-trial Confidence',
    )

    final_chart = alt.vconcat(d0, d1).resolve_scale(color='independent')
    final_chart.show()
    return

def describe_interim_survey(qid, title):
    print("\n========== (DESCRIPTIVE) INTERIM SURVEYS  ==========")
    df_interim_0 = df_interim_survey[df_interim_survey['domain'] == 0]
    df_interim_1 = df_interim_survey[df_interim_survey['domain'] == 1]
    df_interim_0['interaction_type'] = df_interim_0['interaction_type'].apply(id_to_interaction_name)
    df_interim_1['interaction_type'] = df_interim_1['interaction_type'].apply(id_to_interaction_name)


    d0 = alt.Chart(df_interim_0).mark_bar().encode(
        x= alt.X('count(' + qid + '):Q', axis=alt.Axis(title='Number of Responses per Scale Item')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
        color=alt.Color(qid + ':O', scale=alt.Scale(scheme='purples'), legend=alt.Legend(title="SDM responses",orient='top')),
        order=alt.Order(qid + ':O', sort='ascending')
    ).properties(
        title=title,
    )

    d1 = alt.Chart(df_interim_1).mark_bar().encode(
        x= alt.X('count(' + qid + '):Q', axis=alt.Axis(title='Number of Responses per Scale Item')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
        color=alt.Color(qid + ':O', scale=alt.Scale(scheme='oranges'), legend=alt.Legend(title="Classification responses",orient='top')),
        order=alt.Order(qid + ':O', sort='ascending'),
    )

    final_chart = alt.vconcat(d0, d1).resolve_scale(color='independent')
    final_chart.show()
    return

def describe_performance():
    print("\n========== (DESCRIPTIVE) PERFORMANCE (RT) ==========")
    df_perf_0 = df_trial[df_trial['domain'] == 0]
    df_perf_0['interaction_type'] = df_perf_0['interaction_type'].apply(id_to_interaction_name)
    # print(df_perf.groupby('interaction_type')['primary_rt'].describe())

    df_perf_1 = df_trial[df_trial['domain'] == 1]
    df_perf_1['interaction_type'] = df_perf_1['interaction_type'].apply(id_to_interaction_name)

    d0_base = alt.Chart(df_perf_0).mark_boxplot(color='forestgreen').encode(
        x= alt.X('primary_rt:Q', axis=alt.Axis(title='Response Time (ms)')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
    ).properties(
        title="Primary Task Reaction Time",
    )
    d0_points = alt.Chart(df_perf_0).mark_point(opacity=0.1, color='darkgreen').encode(
        y='interaction_type:O',
        x='primary_rt:Q',
    )
    d0_mean = alt.Chart(df_perf_0).mark_point(opacity=0.9, color='darkgreen').encode(
        y='interaction_type:O',
        x='mean(primary_rt):Q',
    )

    d1_base = alt.Chart(df_perf_1).mark_boxplot(color='dodgerblue').encode(
        x= alt.X('primary_rt:Q', axis=alt.Axis(title='Response Time (ms)')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
    )

    d1_points = alt.Chart(df_perf_1).mark_point(opacity=0.1, color='cornflowerblue').encode(
        y='interaction_type:O',
        x='primary_rt:Q',
    )
    d1_mean = alt.Chart(df_perf_1).mark_point(opacity=0.9, color='cornflowerblue').encode(
        y='interaction_type:O',
        x='mean(primary_rt):Q',
    )

    d0 = d0_points + d0_base + d0_mean
    d1 = d1_points + d1_base +  d1_mean

    final_chart = alt.vconcat(d0, d1).resolve_scale(color='independent')
    final_chart.show()
    return

def describe_secondary(domain):
    print("\n========== DESCRIBE SECONDARY RT ==========")
    df_secondaries_0, _, _ = process_secondary(0)
    df_secondaries_0['interaction_type'] = df_secondaries_0['interaction_type'].apply(id_to_interaction_name)

    df_secondaries_1, _, _ = process_secondary(1)
    df_secondaries_1['interaction_type'] = df_secondaries_1['interaction_type'].apply(id_to_interaction_name)

    print(df_secondaries_0.groupby('interaction_type').mean())

    d0_base = alt.Chart(df_secondaries_0).mark_boxplot(color='forestgreen').encode(
        x= alt.X('secondary_rt:Q', axis=alt.Axis(title='Response Time (s)')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
    ).properties(
        title="Secondary Task Reaction Time",
    )
    d0_points = alt.Chart(df_secondaries_0).mark_point(opacity=0.2, color='darkgreen').encode(
        y='interaction_type:O',
        x='secondary_rt:Q',
    )
    d0_mean = alt.Chart(df_secondaries_0).mark_point(opacity=0.9, color='darkgreen').encode(
        y='interaction_type:O',
        x='mean(secondary_rt):Q',
    )

    d1_base = alt.Chart(df_secondaries_1).mark_boxplot(color='dodgerblue').encode(
        x= alt.X('secondary_rt:Q', axis=alt.Axis(title='Response Time (s)')),
        y= alt.Y('interaction_type:N', axis=alt.Axis(title='Interaction Type')),
    )

    d1_points = alt.Chart(df_secondaries_1).mark_point(opacity=0.2, color='cornflowerblue').encode(
        y='interaction_type:O',
        x='secondary_rt:Q',
    )
    d1_mean = alt.Chart(df_secondaries_1).mark_point(opacity=0.9, color='cornflowerblue').encode(
        y='interaction_type:O',
        x='mean(secondary_rt):Q',
    )

    d0 = d0_base + d0_points + d0_mean
    d1 = d1_base + d1_points + d1_mean

    final_chart = alt.vconcat(d0, d1).resolve_scale(color='independent')
    final_chart.show()
    return

# TODO(pkoppol): Need to implement this properly (in the comparison as well)
# def compare_variance(domain):
#     print("\n========== RESPONSE VARIANCE IN DOMAIN " + DOMAIN_MAPPING[domain] + " ==========")
#     df_curr_domain = df_trial[df_trial['domain'] == domain]

#     interaction_types = [1, 2]
#     for interaction in interaction_types:
#         df_curr_type = df_curr_domain[df_curr_domain['interaction_type'] == interaction]
#         print(df_curr_type['primary_answer'])
#     return

'''
 END: DESCRIPTIVE STATISTICS
'''
def compute_ranked_victor(domain, model='plurality'):
    print("\n========== RANKED VICTOR IN DOMAIN " + DOMAIN_MAPPING[domain] + " ==========")
    counts = get_victor(domain, model)
    ranked = {}

    for i in range(len(counts)):
        curr_max = np.argmax(counts)
        ranked[i] = INTERACTION_MAPPING[curr_max]
        counts[curr_max] = -1
    print("The final ordering is therefore: " + str(ranked))
    return

def describe_victor(model='plurality'):
    df_counts = pd.DataFrame(columns = ['interaction_type', 'domain', 'votes'])
    counts = [get_victor(0, model), get_victor(1, model)]
    for i, count in enumerate(counts):
        for j in range(len(count)):
            for k in range(int(count[j])):
                df_counts = df_counts.append({'interaction_type': j, 'domain': i, 'votes': k}, ignore_index = True)

    df_counts['interaction_type'] = df_counts['interaction_type'].apply(id_to_interaction_name)
    df_counts['domain'] = df_counts['domain'].apply(id_to_domain_name)

    d0 = alt.Chart(df_counts).mark_bar().encode(
        y= alt.Y('count(votes):Q', axis=alt.Axis(title='Number of Votes')),
        x= alt.X('domain:N', axis=None),
        column=alt.Column('interaction_type:N', header=alt.Header(title='Preferred Interaction Type (' + model + ')')),
        color=alt.Color('domain:N', sort=['SDM'])
        #color=alt.Color('interaction_type:N', scale=alt.Scale(scheme='purples'), legend=alt.Legend(title="SDM responses",orient='top')),
        # order=alt.Order('votes:O', sort='ascending')
    ).properties(
        title=''
    )
    d0.show()
    return

def get_victor(domain, model='plurality'):
    counts = np.zeros(4)
    df_forced_ranking = df_post_survey[df_post_survey['domain'] == domain]
    for ids in ids_to_modify:
        df_forced_ranking = df_forced_ranking[df_forced_ranking.uniqueid != ids]
    if model == 'plurality':
        for choice in df_forced_ranking['Q0']:
            counts[int(choice)] += 1

    if model == 'borda':
        for choice in df_forced_ranking['Q0']:
            counts[int(choice)] += 3

        for choice in df_forced_ranking['Q1']:
            counts[int(choice)] += 2

        for choice in df_forced_ranking['Q2']:
            counts[int(choice)] += 1

    if model == 'condorcet':
        candidates = {0, 1, 2, 3}
        scores = dict()

        # https://codereview.stackexchange.com/questions/42359/condorcet-voting-method-in-oop-python
        # Builds pairwise scores
        for idx, row in df_forced_ranking.iterrows():
            rank = [int(row['Q0']), int(row['Q1']), int(row['Q2']), int(row['Q3'])]
            for pair in list(itertools.permutations(rank, 2)):
                if pair not in scores:
                    scores[pair] = 0
                if rank.index(pair[0]) < rank.index(pair[1]):
                    scores[pair] += 1

        # get winner of each matchup
        results = dict()
        for match in list(itertools.combinations(candidates, 2)):
            reverse = tuple(reversed(match))
            if scores[match] > scores[reverse]:
                results[match] = match[0]
            else:
                results[match] = match[1]
            if scores[match] == scores[reverse]:
                print(str(match) + " results in a tie with " + str(scores[match]) + " each")

        for candidate in candidates:
            for result in results:
                if candidate in result and results[result] == candidate:
                    counts[candidate] += 1

    print("The final scores for each interaction type are: " + str(counts))
    return counts

# TODO(pkoppol): Can run summary statistics on this information, and also find average age by gender, for example
def print_demographics(): 
    print("\n========== DEMOGRAPHIC INFORMATION ==========")
    participants = df_post_survey.uniqueid.unique()

    # 6 are from my own tests, so we discount those entirely
    print("Flagged " + str(len(flagged_ids)) + " total.")
    print("Number of participants: " + str(len(participants)))

    df_0 = df_post_survey[df_post_survey['domain'] == 0]
    df_1 = df_post_survey[df_post_survey['domain'] == 1]
    participants_0 = len(df_0.uniqueid.unique())
    participants_1 = len(df_1.uniqueid.unique())
    print("Number of participants (" + DOMAIN_MAPPING[0] + "): " + str(participants_0) + "(" + str(participants_0/len(participants)) + " %)")
    print("Number of participants (" + DOMAIN_MAPPING[1] + "): " + str(participants_1) + "(" + str(participants_1/len(participants)) + " %)")

    print("Conditions Represented: ")
    conds = dict()
    for participant in participants:
        participant_info = df_post_survey[df_post_survey['uniqueid'] == participant]
        if not participant_info.empty:
            curr_cond = (participant_info['domain'].iloc[0], participant_info['condition'].iloc[0])
            if curr_cond not in conds:
                conds[curr_cond] = 1
            else:
                conds[curr_cond] += 1
        else:
            print("No post study data recorded for " + participant)

    for cond in sorted(conds):
        if conds[cond] > 3:
            print(" Domain/Ordering: " + str(cond) + " : " + str(conds[cond]))

    print("Ages (description)")
    ages = pd.to_numeric(df_post_survey.Q5)
    print(ages.describe())
    print("Max: " + str(ages.max()))
    print("Min: " + str(ages.min()))
    print("Mean: " + str(ages.mean()))
    # print("SE: " + str(ages.sem()))
    print("Genders: ")
    gender_vals = [0, 0, 0, 0]
    mapping = {0: 'Male', 1: 'Female', 2: 'Non-binary', 3: 'Prefer not to disclose'}
    answers = df_post_survey.Q6
    for answer in answers:
        gender_vals[int(answer)] += 1

    for idx, num in enumerate(gender_vals):
        print(mapping[idx] + " : " + str(num) + "(" + str(num/(np.sum(gender_vals))) +"%)")

# We need to print this, since it's a textbox format and therefore non-standardized.
def print_feedback(): 
    # Get feedback for domain 0
    print("\n========== SEQUENTIAL DECISION MAKING DOMAIN ==========")
    df_post_survey_0 = df_post_survey[df_post_survey['domain'] == 0]
    for feedback in df_post_survey_0['primary_answer']:
        responses = json.loads(feedback)
        print (responses['Q4'])
        print("--")

    # Get feedback on domain 1
    print("\n========== CLASSIFICATION DOMAIN ==========")
    df_post_survey_1 = df_post_survey[df_post_survey['domain'] == 1]
    for feedback in df_post_survey_1['primary_answer']:
        responses = json.loads(feedback)
        print (responses['Q4'])
        print("--")


# TODO: Data Analysis. 
    #Look here: https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
    #and here: https://reneshbedre.github.io/blog/anova.html

if __name__ == '__main__':
    if sys.argv[1] == 'remote':
        FLAG_LOCAL_VERSION = False
        data = get_data()

    else:
        filename = sys.argv[2]
        data = get_data(sys.argv[2])
        # Set global table values
        table_setup(data)
        flagged_ids = flag_ids()
        edit_tables(data)

        if sys.argv[3] == 'print_demographics' or sys.argv[3] == 'all':
            print_demographics()

        if sys.argv[3] == 'describe_performance' or sys.argv[3] == 'all':
            describe_performance()

        if sys.argv[3] == 'get_performance' or sys.argv[3] == 'all':
            compare_primary_rt(0)
            compare_primary_rt(1)

        if sys.argv[3] == 'get_secondary' or sys.argv[3] == 'all':
            compare_secondary_rt(0)
            compare_secondary_rt(1)

        if sys.argv[3] == 'describe_secondary' or sys.argv[3] == 'all':
            describe_secondary(0)
            # describe_secondary(1)

        if sys.argv[3] == 'get_variance' or sys.argv[3] == 'all':
            compare_variance(0)
            compare_variance(1)

        if sys.argv[3] == 'describe_interim' or sys.argv[3] == 'all':
            qs = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']
            titles = ['M1: Cognitive Load', 'M5: Frustration', 'M6: Unnecessary Complexity', 'M7: Ease of Use', 'M8: Overall Confidence']
            for t, q in enumerate(qs):
                describe_interim_survey(q, titles[t])

        if sys.argv[3] == 'get_interim' or sys.argv[3] == 'all':
            qs = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']
            for q in qs:
                compare_interim(0, q)
                compare_interim(1, q)
            #compare_interim(1, 'median')

        if sys.argv[3] == 'describe_confidence' or sys.argv[3] == 'all':
            describe_confidence(0)

        if sys.argv[3] == 'get_confidence' or sys.argv[3] == 'all':
            compare_confidence(0, 'median')
            compare_confidence(1, 'median')

        if sys.argv[3] == 'get_ranked_victor' or sys.argv[3] == 'all':
            compute_ranked_victor(0, 'plurality')
            compute_ranked_victor(1, 'plurality')

        if sys.argv[3] == 'describe_victor' or sys.argv[3] == 'all':
            describe_victor('condorcet')
        if sys.argv[3] == 'print_feedback' or sys.argv[3] == 'all':
            print_feedback()


