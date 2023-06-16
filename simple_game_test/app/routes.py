from flask import render_template, flash, redirect, url_for, request, session
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm, TrialForm, DemoForm, ConsentForm, AttentionCheckForm, FinalForm, TrainingForm, FeedbackSurveyForm, NoFeedbackSurveyForm
from app.models import User, Trial, Demo, OnlineCondition, InPersonCondition, Survey
from app.params import *
from utils import rules_to_str, str_to_rules
import numpy as np
import random as rand
from datetime import datetime
from generate_rules import generate_rule, generate_hard_rule_constrained
from environment import Environment
from learner import Learner

# rule_str = None
# TODO need a proper solution instead of global variables, i.e. per-user environment
# https://stackoverflow.com/questions/27611216/how-to-pass-a-variable-between-flask-pages
learners = {}
MODE = 'hard'
IS_IN_PERSON = False

CARD_ID_TO_FEATURES = [
    [color, fill, shape, number] for color in ['red', 'green', 'purple'] for fill in ['hollow', 'striped', 'solid'] for shape in ['diamond', 'ellipse', 'squiggle'] for number in ['one', 'two', 'three']
]


# HOW TO PREVENT RULE / STATE CHANGE ON RELOAD???

@app.route("/test", methods=["GET", "POST"])
@login_required
def test():
    # global rule_str
        # if rule_str is None:
    global learners
    start_time = datetime.now().utcnow().isoformat()
    # num_completed_trials = db.session.query(Trial).filter_by(user_id=current_user.id, round_num=round).count()
    # condition_id = current_user.condition_id
    # current_condition = db.session.query(Condition).get(condition_id)

    # if num_completed_trials == len(cards):
    # flash("You have seen all the trials in this round!")
    #     return redirect(url_for("survey", round=round))

    # if check_previous_demos < len(RULE_PROPS[check_rule_name]['demo_cards']):
    #     return redirect(url_for("consent"))
    print("trial id")
    print(current_user.id)

    bins, rule_str = generate_rule(MODE)
    env = Environment(bins, MODE)
    learners[current_user.username] = Learner(MODE)
    session['rule_str'] = rule_str
    session['rule_str_as_bin'] = bins
    session['env'] = env.to_dict()
    session['start_time_s'] = datetime.now().timestamp()
    # session['learner'] = learner.to_dict()
    db.session.commit()

    if IS_IN_PERSON:
        in_person_condition_id = current_user.in_person_condition_id
        current_condition = db.session.query(InPersonCondition).get(in_person_condition_id)
    else:
        online_condition_id = current_user.online_condition_id
        current_condition = db.session.query(OnlineCondition).get(online_condition_id)

    if current_user.num_trials_completed == len(current_condition.trials):
        return redirect(url_for("survey"))

    if current_user.training != 1:
        tutorial_mode = True
        feedback_condition = 'no_feedback'
        round_num = "0 (Tutorial)"
    else: 
        tutorial_mode = False
        feedback_condition = current_condition.trials[current_user.num_trials_completed]
        round_num = current_user.num_trials_completed + 1
    learners[current_user.username].set_feedback_type(feedback_condition)
    return render_template("test.html",
        title="Robot Sorting Game, Round ",
        round_num=round_num,
        rule_str=rule_str,
        tutorial_mode=tutorial_mode,
        feedback_condition=feedback_condition,
        vid_name="neutral_long_1",
        debug=True)

@app.route("/place_card", methods=["GET", "POST"])
def place_card():
    global learners
    learner = learners[current_user.username]
    if 'env' in session:
        env = Environment.from_dict(session['env'])
    else:
        print('No ENV found in session')
        return {}
    # if 'learner' in session:
    #     learner = Learner.from_dict(session['learner'])
    # else:
    #     print(session)
    #     print('No LEARNER found in session')
    #     return {}
    card_id_str = request.args.get("card_id")
    card_id = int(card_id_str.split("-")[-1])
    card_features = CARD_ID_TO_FEATURES[card_id]
    print('Placing card with features: ', card_features)
    bin, outcome_str = env.place_card(card_features)
    print(outcome_str)
    if 'rule_str' in session:
        print(session['rule_str'])
    else:
        print('No RULE_STR found in session')
    # Update learner observation
    if bin != -1:
        learner.observe(card_features, bin)
        print(f'Learner\'s belief now contains {learner.get_n_valid_rules()} valid rule(s)')
    # fb_str = learner.get_feedback()
    # Update session variables
    session['env'] = env.to_dict()
    learned = 1 if learner.get_n_valid_rules() == 1 else 0
    # session['learner'] = learner.to_dict()
    return {'bin': bin+1, 'outcome_str': outcome_str, 'card_id': card_id, 'status': learned}

@app.route("/get_feedback", methods=["GET", "POST"])
def get_feedback():
    global learners
    learner = learners[current_user.username]
    fb_str = learner.get_feedback()
    print(f'User {current_user.username} got feedback: {fb_str}')
    return {'fb_str': fb_str}

def jsonStrToList(str):
    if len(str) > 2:
        return [int(val) for val in str[1:-1].split(',')]
    else:
        return []

@app.route("/trial_completed", methods=["GET", "POST"])
def trial_completed():
    global learners
    learner = learners[current_user.username]

    if IS_IN_PERSON:
        in_person_condition_id = current_user.in_person_condition_id
        current_condition = db.session.query(InPersonCondition).get(in_person_condition_id)
    else:
        online_condition_id = current_user.online_condition_id
        current_condition = db.session.query(OnlineCondition).get(online_condition_id)
    current_user.set_browser(request.args.get("browser"))
    trial_duration_ms = 1000 * (datetime.now().timestamp() - session['start_time_s'])
    feedback_confidences = jsonStrToList(request.args.get("feedback_confidences"))
    terminate_confidences = jsonStrToList(request.args.get("termination_confidences"))
    terminate_record = jsonStrToList(request.args.get("termination_record"))
    card_select_times = jsonStrToList(request.args.get("card_select_times"))
    feedback_strings = request.args.get("feedback_strings")
    if len(feedback_strings) > 2:
        feedback_strings = feedback_strings.split('","')
        feedback_strings[-1] = feedback_strings[-1][:-2]
        feedback_strings[0] = feedback_strings[0][2:]
    else:
        feedback_strings = []
    trial = Trial(
        user_id=current_user.id,
        trial_num=current_user.num_trials_completed + 1,
        duration_ms=trial_duration_ms,
        rule_str=session['rule_str'],
        fb_type=learner._fb_type,
        cards_played=learner._observations,
        n_cards=learner._n_observations,
        n_cards_to_learn_rule=learner._n_observations_to_learn_rule,
        card_select_times=card_select_times,
        n_hypotheses_remaining=learner._n_hypotheses_remaining,
        n_failed_terminations=request.args.get("n_failed_terminations"),
        terminate_confidences=terminate_confidences,
        feedback_confidences=feedback_confidences,
        terminate_record=terminate_record,
        bonus_value=request.args.get("bonus_value"),
        feedback_strings=feedback_strings
    )
    db.session.add(trial)
    db.session.commit()
    # if updated_num_trials_completed < len(current_condition.trials):
    #     return {"url":url_for("test", round=current_user.num_trials_completed)}

    # Remove learner from global dictionary
    print(learners.keys())
    del learners[current_user.username]
    print(learners.keys())
    return {"url":url_for("survey")}

@app.route("/check_termination_condition", methods=["GET", "POST"])
def check_termination_condition(): 
    global learners
    learner = learners[current_user.username]
    print(learner.get_n_valid_rules)
    has_learned = 1 if learner.get_n_valid_rules() == 1 else 0
    bonus = learner.get_bonus()
    print(learner.get_n_valid_rules)
    print(bonus)
    return {'has_learned': has_learned, 'bonus': bonus}


@app.route("/hover_card", methods=["GET", "POST"])
def hover_card():
    if 'env' in session:
        env = Environment.from_dict(session['env'])
    else:
        print('No ENV found in session')
        return {}
    card_id_str = request.args.get("card_id")
    card_id = int(card_id_str.split("-")[-1])
    card_features = CARD_ID_TO_FEATURES[card_id]
    bin = env.get_bin_for_card(card_features)
    return {"bin": bin+1}

@app.route("/terminate_learning", methods=["GET", "POST"])
def terminate_learning():
    global learners
    learner = learners[current_user.username]
    if 'env' in session:
        env = Environment.from_dict(session['env'])
    else:
        print('No ENV found in session')
        return {}
    # if 'learner' in session:
    #     learner = Learner.from_dict(session['learner'])
    # else:
    #     print('No LEARNER found in session')
    #     return {}

    debug = (request.args.get("debug") == "True")
    if not debug:
        return {'dialog_str': ''}
        return {'redirect_str': '/survey/999'}
    print('Terminating Learning.')
    dialog_str = "Learning terminated."
    if learner.get_n_valid_rules() == 1:
        dialog_str += f' Learner successfully identified the rule:\n\n{learner.get_most_confident_rule_str()}'
    else:
        dialog_str += f' Learner\'s belief contains {learner.get_n_valid_rules()} valid rules.'
    dialog_str += f'\n\nMetrics:'
    
    learner_metrics = learner.get_metrics(env.bins)
    for key in learner_metrics:
        dialog_str += f'\n- {key}: {learner_metrics[key]}'
    
    return {'dialog_str': dialog_str}

@app.route("/set_fb_type", methods=["GET", "POST"])
def set_fb_type():
    global learners
    learner = learners[current_user.username]
    # if 'learner' in session:
    #     learner = Learner.from_dict(session['learner'])
    # else:
    #     print('No LEARNER found in session')
    #     return "nothing"

    fb_type = request.args.get("fb_type")
    learner.set_feedback_type(fb_type)

    # session['learner'] = learner.to_dict()
    return "nothing"

@app.route("/attn_check", methods=["GET", "POST"])
@login_required
def attn_check():
    form = AttentionCheckForm()

    true_rule = session['rule_str']
    print(f"the true rule is {true_rule}")

    # because exception_val is hard to identify, lets consider (primary class, exception class)
    # the true rule is (TP, TP)
    rules = [true_rule]
    # so we generate (TP, FP)
    rules.append(generate_hard_rule_constrained(session['rule_str_as_bin'], True, False))
    # as well as (FP, TP)
    rules.append(generate_hard_rule_constrained(session['rule_str_as_bin'], False, True))
    # and, finally, (FP, FP)
    rules.append(generate_hard_rule_constrained(session['rule_str_as_bin'], False, False))
    rand.shuffle(rules)

    if 'attention_check_rules' in session:
        print(f"The attention check rules already exist, and are: {session['attention_check_rules']}")

    else:
        print(f"Defining attention check rules to be {rules}")
        session['attention_check_rules'] = rules

    # print (session['attention_check_rules'])
    if form.validate_on_submit():
        attention_check_val = 1 if (int(form.prev_rule.data) == session['attention_check_rules'].index(session['rule_str'])) else 0
        print(f"Attention Check value is: {attention_check_val}")
        print(f"Form.prev_rule.data was {form.prev_rule.data}")
        print(f"Rules is {session['attention_check_rules']}")
        print(f"And the location we indexed into was {session['attention_check_rules'].index(true_rule)}")

        current_user.set_attention_check(attention_check_val)

        # Clear this variable -- for some reason was not clearing on its own. 
        session.pop('attention_check_rules', None)

        db.session.commit()
        redirect(url_for("test"))

    online_condition_id = current_user.online_condition_id
    current_condition = db.session.query(OnlineCondition).get(online_condition_id)

    # We only provide an attention check after the first survey
    completed_survey = db.session.query(Survey).filter_by(user_id=current_user.id, round_num=1).count()

    # Make sure previous step (survey after first game) has been completed
    # if current_user.num_trials_completed == len(current_condition.trials) and current_user.study_completed == 0:
    #     return redirect(url_for("survey"))
    if completed_survey == 0:
        return redirect(url_for("survey"))

    # Only complete the attention check once
    if current_user.attention_check != -1:
        if current_user.num_trials_completed < len(current_condition.trials):
            return redirect(url_for("test"))
        else:
            return redirect(url_for("survey"))

    return render_template("attention_check.html",
                            title="Attention Check",
                            rule_1=session['attention_check_rules'][0],
                            rule_2=session['attention_check_rules'][1],
                            rule_3=session['attention_check_rules'][2],
                            rule_4=session['attention_check_rules'][3],
                            form=form)

@app.route("/training/<string:page>", methods=["GET", "POST"])
@login_required
def training(page):
    form = TrainingForm()
    if form.validate_on_submit():
        current_user.training = 1
        db.session.commit()
        redirect(url_for("test"))
    if current_user.training:
        # flash("Training already completed!")
        return redirect(url_for("test"))
    elif not current_user.consent:
        # flash("Consent not yet completed!")
        return redirect(url_for("consent"))
    else:
        if not IS_IN_PERSON:
            study_format = "Let's go over the format of the study. You will be teaching the robot in two separate games, with two separate rules. In one of these games, you will not receive any feedback from the robot at all and in the other, you will receive feedback from the robot after every card you play."
            compensation_info = "Then, you will receive a random code to enter into Prolific so that you can be compensated for completing this study."
        else:
            study_format = "Let's go over the format of the study. You will be teaching the robot in five separate games, with five separate rules. In all of these games, you will receive some form of feedback from the robot after every card you play."
            compensation_info = "At the conclusion of the study, you will be compensated for your time."
        url = "/training/" + page + ".html"
        return render_template(url, title="Training", form=form, study_format=study_format, compensation_info=compensation_info)

# @app.route("/", methods=["GET", "POST"])
# @app.route("/index/", methods=["GET", "POST"])
# def index():
#     start_time = datetime.now().utcnow().isoformat()
#     form = TrialForm(start_time=start_time)
    
#     num_completed_trials = 0 #db.session.query(Trial).filter_by(user_id=current_user.id, round_num=round).count()
    
#     # condition_id = current_user.condition_id
#     # current_condition = db.session.query(Condition).get(condition_id)
#     round = 0
#     rule_name = 'EASY' # current_condition.difficulty[round]
#     rule = RULE_PROPS[rule_name]['rule']
#     demo_cards = RULE_PROPS[rule_name]['demo_cards']
#     cards = RULE_PROPS[rule_name]['cards']
#     answers = RULE_PROPS[rule_name]['answers']
#     demo_answers = RULE_PROPS[rule_name]['demo_answers']

#     previous_cards = [[], []]
#     for ii in range(len(demo_cards)):
#         if demo_answers[ii] == 0:
#             previous_cards[0].append(demo_cards[ii])
#         if demo_answers[ii] == 1:
#             previous_cards[1].append(demo_cards[ii])

#     for ii in range(num_completed_trials):
#         if answers[ii] == 0:
#             previous_cards[0].append(cards[ii])
#         if answers[ii] == 1:
#             previous_cards[1].append(cards[ii])

#     if form.validate_on_submit():
#         chosen_bin = int(form.chosen_bin.data[3])
#         feedback_chosen = form.feedback_chosen.data
#         trial = Trial(author=current_user,
#                       trial_num=num_completed_trials + 1,
#                       card_num=cards[num_completed_trials],
#                       round_num=round,
#                       correct_bin=answers[num_completed_trials],
#                       chosen_bin=chosen_bin,
#                       feedback=feedback_chosen,
#                       rule_set=rule,
#                       switches=int(form.switches.data))
#         db.session.add(trial)

#         feedback_counts = current_user.feedback_counts
#         new_feedback_counts = {}
#         for new_vid_name in VIDEO_LIST:
#             if new_vid_name == feedback_chosen:
#                 new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name] + 1
#             else:
#                 new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name]

#         current_user.feedback_counts = new_feedback_counts
#         db.session.commit()
#         return redirect(url_for('trials', round=round))

#     if num_completed_trials == len(cards):
#         # flash("You have seen all the trials in this round!")
#         return redirect(url_for("survey", round=round))
    
#     #Check if previous thing is done, previous demos must be done
#     check_rule_name = rule_name# current_condition.difficulty[round]
#     check_previous_demos = db.session.query(Demo).filter_by(user_id=current_user.id, round_num=round).count()
#     if check_previous_demos < len(RULE_PROPS[check_rule_name]['demo_cards']):
#         return redirect(url_for("consent"))

#     #Pick the video to play
#     feedback_counts = current_user.feedback_counts
#     cur_names = []
#     cur_counts = []
#     for vid_name in NEUTRAL:
#         cur_names.append(vid_name)
#         cur_counts.append(feedback_counts[vid_name])
#     cur_counts = np.array(cur_counts)
#     if np.sum(cur_counts) == 0:
#         p = np.ones_like(cur_counts)/np.sum(np.ones_like(cur_counts))
#     else:
#         p= 1 - cur_counts/np.sum(cur_counts)
#     for ii in range(len(p)):
#         if p[ii] < 0.05:
#             p[ii] = 0.05
#     p = p / np.sum(p)
#     vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p= p)
#     neutral_vid_name = cur_names[vid_choice]
    
#     new_feedback_counts = {}
#     for new_vid_name in VIDEO_LIST:
#         if new_vid_name == vid_name:
#             new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name] + 1
#         else:
#             new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name]

#     current_user.feedback_counts = new_feedback_counts
#     db.session.commit()

#     if answers[num_completed_trials] == 0:
#         correct_bin = 'bin0'
#     else:
#         correct_bin = 'bin1'


#     #Choose correct video
#     current_nonverbal = 'NONVERBAL' # current_condition.nonverbal[round]
#     cur_names = []
#     cur_counts = []
#     if correct_bin == 'bin0':
#         for vid_name in FEEDBACK[current_nonverbal]['CORRECT-LEFT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     else:
#         for vid_name in FEEDBACK[current_nonverbal]['CORRECT-RIGHT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     cur_counts = np.array(cur_counts)
#     if np.sum(cur_counts) == 0:
#         cur_counts = np.ones_like(cur_counts)
#     vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p=cur_counts/np.sum(cur_counts))
#     correct_vid_name = cur_names[vid_choice]

#     #Choose incorrect video
#     cur_names = []
#     cur_counts = []
#     if correct_bin == 'bin0':
#         for vid_name in FEEDBACK[current_nonverbal]['INCORRECT-LEFT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     else:
#         for vid_name in FEEDBACK[current_nonverbal]['INCORRECT-RIGHT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     cur_counts = np.array(cur_counts)
#     if np.sum(cur_counts) == 0:
#         cur_counts = np.ones_like(cur_counts)
#     vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p=cur_counts/np.sum(cur_counts))
#     incorrect_vid_name = cur_names[vid_choice]

    # return render_template("trials.html",
    #     title="Trials",
    #     form=form,
    #     num_bins=len(rule),
    #     card=cards[num_completed_trials],
    #     correct_bin=correct_bin,
    #     num_completed_trials=num_completed_trials + 1,
    #     num_trials=len(cards),
    #     previous_cards=previous_cards,
    #     round=round,
    #     vid_name=neutral_vid_name,
    #     correct_vid_name = correct_vid_name,
    #     incorrect_vid_name = incorrect_vid_name)

@app.route("/", methods=["GET", "POST"])
# @app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    # condition_id = current_user.condition_id
    # print(condition_id)
    # current_condition = db.session.query(OnlineCondition).get(condition_id)
    # num_rounds = len(current_condition.difficulty)

    # completed = []
    # for round in range(num_rounds):
    #     completed.append([])
    #     rule_name = current_condition.difficulty[round]
        
    #     demo_cards = RULE_PROPS[rule_name]['demo_cards']
    #     num_completed_demos = db.session.query(Demo).filter_by(user_id=current_user.id, round_num=round).count()
    #     if num_completed_demos < len(demo_cards):
    #         completed[round].append(False)
    #     else:
    #         completed[round].append(True)
        
    #     cards = RULE_PROPS[rule_name]['cards']
    #     num_completed_trials = db.session.query(Trial).filter_by(user_id=current_user.id, round_num=round).count()
    #     if num_completed_trials < len(cards):
    #         completed[round].append(False)
    #     else:
    #         completed[round].append(True)
        
    #     num_completed_surveys = db.session.query(Survey).filter_by(user_id=current_user.id, round_num=round).count()
    #     if num_completed_surveys < 1:
    #         completed[round].append(False)
    #     else:
    #         completed[round].append(True)
    online_condition_id = current_user.online_condition_id
    current_condition = db.session.query(OnlineCondition).get(online_condition_id)

    completed = True if current_user.study_completed == 1 else False

    return render_template("index.html",
                           title="Home Page",
                           completed=completed,
                           code=current_user.code)

@app.route("/sign_consent", methods=["GET", "POST"])
@login_required
def sign_consent():
    current_user.consent = 1
    db.session.commit() 
    return {"url":url_for("training", page=1)}       


@app.route("/intro", methods=["GET", "POST"])
@login_required
def intro():
    # form = LoginForm()
    # if form.validate_on_submit():
    #     user = User.query.filter_by(username=form.username.data).first()
    #
    #     if user is None:
    #         user = User(username=form.username.data)
    #         user.set_num_trials_completed(0)
    #         user.set_completion(0)
    #         user.set_attention_check(-1)
    #
    #         # Change depending on the study type.
    #         cond = user.set_condition("in_person" if IS_IN_PERSON else "online")
    #         code = user.set_code()
    #
    #         db.session.add(user)
    #
    #         cond.users.append(user)
    #         cond.count += 1
    #
    #         db.session.commit()
    #
    #     login_user(user)
    #     next_page = request.args.get("next")
    #     if not next_page or url_parse(next_page).netloc != "":
    #         next_page = url_for("index")
    #     return redirect(next_page)
    #
    # render_template("login.html", title="Sign In", form=form)

    # just testing out my code
    # return render_template("intro.html")
    return render_template("augmented_taxi2.html")


@app.route("/consent", methods=["GET", "POST"])
@login_required
def consent():
    form = ConsentForm()
    if current_user.consent:
        # flash("Consent completed!")
        online_condition_id = current_user.online_condition_id
        current_condition = db.session.query(OnlineCondition).get(online_condition_id)

        if current_user.num_trials_completed < (len(current_condition.trials)):
            return redirect(url_for("intro")) # verifying url_for and displaying training/testing simulations
            # return redirect(url_for("test"))
        return redirect(url_for("survey"))

    else:
        if IS_IN_PERSON:
            procedure = "This study may take up to 90 minutes, and audio/screen recordings will be collected."
        else:
            procedure = "This study may take up to 30 minutes."
        return render_template("consent.html", title="Consent", form=form, procedure=procedure)

# @app.route("/training", methods=["GET", "POST"])
# @login_required
# def training():
#     form = TrainingForm()
#     if form.validate_on_submit():
#         current_user.training = 1
#         db.session.commit()
#         redirect(url_for("demos", round=0))
#     if current_user.training:
#         # flash("Training already completed!")
#         return redirect(url_for("demos", round=0))
#     elif not current_user.consent:
#         # flash("Consent not yet completed!")
#         return redirect(url_for("consent"))
#     else:
#         return render_template("training.html", title="Training", form=form)

# @app.route("/demos/<int:round>", methods=["GET", "POST"])
# @login_required
# def demos(round):
#     # form = DemoForm()
    
#     # num_completed_demos = db.session.query(Demo).filter_by(user_id=current_user.id, round_num=round).count()
    
#     # condition_id = current_user.condition_id
#     # current_condition = db.session.query(Condition).get(condition_id)
#     # rule_name = current_condition.difficulty[round]
#     # rule = RULE_PROPS[rule_name]['rule']
#     # demo_cards = RULE_PROPS[rule_name]['demo_cards']
#     # demo_answers = RULE_PROPS[rule_name]['demo_answers']

#     # previous_cards = [[], []]
#     # for ii in range(num_completed_demos):
#     #     if demo_answers[ii] == 0:
#     #         previous_cards[0].append(demo_cards[ii])
#     #     if demo_answers[ii] == 1:
#     #         previous_cards[1].append(demo_cards[ii])

#     # if form.validate_on_submit():
#     #     demo = Demo(author=current_user,
#     #                 demo_num=num_completed_demos + 1,
#     #                 round_num=round,
#     #                 card_num=demo_cards[num_completed_demos],
#     #                 correct_bin=demo_answers[num_completed_demos],
#     #                 rule_set=rule)
#     #     db.session.add(demo)
#     #     db.session.commit()
#     #     return redirect(url_for('demos', round=round))

#     # if num_completed_demos == len(demo_cards):
#     #     return redirect(url_for("trials",round=round))
    
#     #Check if previous thing is done

#     #If first round, training must be done
#     # if (round == 0) and (not current_user.training):
#         return redirect(url_for("consent"))
    
#     #If not first round, previous survey must be done
#     # if round > 0:
#     #     check_previous_surveys = db.session.query(Survey).filter_by(user_id=current_user.id, round_num=round-1).count()
#     #     if check_previous_surveys < 1:
#     #         return redirect(url_for("consent"))

#     #Pick the video to play
#     # feedback_counts = current_user.feedback_counts
#     # cur_names = []
#     # cur_counts = []
#     # for vid_name in NEUTRAL:
#     #     cur_names.append(vid_name)
#     #     cur_counts.append(feedback_counts[vid_name])
#     # cur_counts = np.array(cur_counts)
#     # if np.sum(cur_counts) == 0:
#     #     p = np.ones_like(cur_counts)/np.sum(np.ones_like(cur_counts))
#     # else:
#     #     p= 1 - cur_counts/np.sum(cur_counts)
#     # for ii in range(len(p)):
#     #     if p[ii] < 0.05:
#     #         p[ii] = 0.05
#     # p = p / np.sum(p)
#     # vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p= p)
#     # vid_name = cur_names[vid_choice]
    
#     # new_feedback_counts = {}
#     # for new_vid_name in VIDEO_LIST:
#     #     if new_vid_name == vid_name:
#     #         new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name] + 1
#     #     else:
#     #         new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name]

#     # current_user.feedback_counts = new_feedback_counts
#     # db.session.commit()

#     #Render the next demonstration
#     return render_template("demos.html",
#         title="Demonstrations",
#         form=form,
#         num_bins=len(rule),
#         card=demo_cards[num_completed_demos],
#         correct_bin=demo_answers[num_completed_demos],
#         num_completed_demos=num_completed_demos + 1,
#         num_demos=len(demo_cards),
#         previous_cards=previous_cards,
#         round=round,
#         vid_name=vid_name)

# @app.route("/trials/<int:round>", methods=["GET", "POST"])
# @login_required
# def trials(round):
#     start_time = datetime.now().utcnow().isoformat()
#     form = TrialForm(start_time=start_time)
    
#     num_completed_trials = db.session.query(Trial).filter_by(user_id=current_user.id, round_num=round).count()
    
#     condition_id = current_user.condition_id
#     current_condition = db.session.query(Condition).get(condition_id)
#     rule_name = current_condition.difficulty[round]
#     rule = RULE_PROPS[rule_name]['rule']
#     demo_cards = RULE_PROPS[rule_name]['demo_cards']
#     cards = RULE_PROPS[rule_name]['cards']
#     answers = RULE_PROPS[rule_name]['answers']
#     demo_answers = RULE_PROPS[rule_name]['demo_answers']

#     previous_cards = [[], []]
#     for ii in range(len(demo_cards)):
#         if demo_answers[ii] == 0:
#             previous_cards[0].append(demo_cards[ii])
#         if demo_answers[ii] == 1:
#             previous_cards[1].append(demo_cards[ii])

#     for ii in range(num_completed_trials):
#         if answers[ii] == 0:
#             previous_cards[0].append(cards[ii])
#         if answers[ii] == 1:
#             previous_cards[1].append(cards[ii])

#     if form.validate_on_submit():
#         chosen_bin = int(form.chosen_bin.data[3])
#         feedback_chosen = form.feedback_chosen.data
#         trial = Trial(author=current_user,
#                       trial_num=num_completed_trials + 1,
#                       card_num=cards[num_completed_trials],
#                       round_num=round,
#                       correct_bin=answers[num_completed_trials],
#                       chosen_bin=chosen_bin,
#                       feedback=feedback_chosen,
#                       rule_set=rule,
#                       switches=int(form.switches.data))
#         db.session.add(trial)

#         feedback_counts = current_user.feedback_counts
#         new_feedback_counts = {}
#         for new_vid_name in VIDEO_LIST:
#             if new_vid_name == feedback_chosen:
#                 new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name] + 1
#             else:
#                 new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name]

#         current_user.feedback_counts = new_feedback_counts
#         db.session.commit()
#         return redirect(url_for('trials', round=round))

#     if num_completed_trials == len(cards):
#         # flash("You have seen all the trials in this round!")
#         return redirect(url_for("survey", round=round))
    
#     #Check if previous thing is done, previous demos must be done
#     check_rule_name = current_condition.difficulty[round]
#     check_previous_demos = db.session.query(Demo).filter_by(user_id=current_user.id, round_num=round).count()
#     if check_previous_demos < len(RULE_PROPS[check_rule_name]['demo_cards']):
#         return redirect(url_for("consent"))

#     #Pick the video to play
#     feedback_counts = current_user.feedback_counts
#     cur_names = []
#     cur_counts = []
#     for vid_name in NEUTRAL:
#         cur_names.append(vid_name)
#         cur_counts.append(feedback_counts[vid_name])
#     cur_counts = np.array(cur_counts)
#     if np.sum(cur_counts) == 0:
#         p = np.ones_like(cur_counts)/np.sum(np.ones_like(cur_counts))
#     else:
#         p= 1 - cur_counts/np.sum(cur_counts)
#     for ii in range(len(p)):
#         if p[ii] < 0.05:
#             p[ii] = 0.05
#     p = p / np.sum(p)
#     vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p= p)
#     neutral_vid_name = cur_names[vid_choice]
    
#     new_feedback_counts = {}
#     for new_vid_name in VIDEO_LIST:
#         if new_vid_name == vid_name:
#             new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name] + 1
#         else:
#             new_feedback_counts[new_vid_name] = feedback_counts[new_vid_name]

#     current_user.feedback_counts = new_feedback_counts
#     db.session.commit()

#     if answers[num_completed_trials] == 0:
#         correct_bin = 'bin0'
#     else:
#         correct_bin = 'bin1'


#     #Choose correct video
#     current_nonverbal = current_condition.nonverbal[round]
#     cur_names = []
#     cur_counts = []
#     if correct_bin == 'bin0':
#         for vid_name in FEEDBACK[current_nonverbal]['CORRECT-LEFT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     else:
#         for vid_name in FEEDBACK[current_nonverbal]['CORRECT-RIGHT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     cur_counts = np.array(cur_counts)
#     if np.sum(cur_counts) == 0:
#         cur_counts = np.ones_like(cur_counts)
#     vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p=cur_counts/np.sum(cur_counts))
#     correct_vid_name = cur_names[vid_choice]

#     #Choose incorrect video
#     cur_names = []
#     cur_counts = []
#     if correct_bin == 'bin0':
#         for vid_name in FEEDBACK[current_nonverbal]['INCORRECT-LEFT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     else:
#         for vid_name in FEEDBACK[current_nonverbal]['INCORRECT-RIGHT']:
#             cur_names.append(vid_name)
#             cur_counts.append(feedback_counts[vid_name])
#     cur_counts = np.array(cur_counts)
#     if np.sum(cur_counts) == 0:
#         cur_counts = np.ones_like(cur_counts)
#     vid_choice = np.random.choice(np.arange(cur_counts.shape[0]), p=cur_counts/np.sum(cur_counts))
#     incorrect_vid_name = cur_names[vid_choice]

#     return render_template("trials.html",
#         title="Trials",
#         form=form,
#         num_bins=len(rule),
#         card=cards[num_completed_trials],
#         correct_bin=correct_bin,
#         num_completed_trials=num_completed_trials + 1,
#         num_trials=len(cards),
#         previous_cards=previous_cards,
#         round=round,
#         vid_name=neutral_vid_name,
#         correct_vid_name = correct_vid_name,
#         incorrect_vid_name = incorrect_vid_name)

@app.route("/survey", methods=["GET", "POST"])
@login_required
def survey():
    if IS_IN_PERSON:
        in_person_condition_id = current_user.in_person_condition_id
        current_condition = db.session.query(InPersonCondition).get(in_person_condition_id)
    else:
        online_condition_id = current_user.online_condition_id
        current_condition = db.session.query(OnlineCondition).get(online_condition_id)

    round = (current_user.num_trials_completed + 1)
    if current_user.num_trials_completed >= len(current_condition.trials):
        # If they already completed the final form, they are done.
        check_final_survey_completion = db.session.query(Survey).filter_by(user_id=current_user.id, round_num=round).count()
        if check_final_survey_completion == 1:
            return redirect(url_for("index"))
        (form, template) = (FinalForm(), "final_survey.html")

    else:
        (form, template) = (NoFeedbackSurveyForm(), "survey.html") if current_condition.trials[current_user.num_trials_completed] == 'no_feedback' else (FeedbackSurveyForm(), "feedback_survey.html")


    if form.validate_on_submit():
        if current_user.num_trials_completed >= len(current_condition.trials):
            current_user.age = form.age.data
            current_user.gender = form.gender.data
            current_user.ethnicity = form.ethnicity.data
            current_user.education = form.education.data
            current_user.robot = form.robot.data
            current_user.final_robot_choice = form.robot_choice.data
            current_user.final_feedback = form.opt_text.data
            current_user.set_completion(1)

            print(form.ethnicity.data)
            db.session.commit()      

        elif current_condition.trials[current_user.num_trials_completed] == 'no_feedback':
            survey = Survey(author=current_user,
                            round_num=round,
                            frustration = form.frustration.data,
                            ease_of_teaching = form.ease_of_teaching.data,
                            opt_text = form.opt_text.data)
            db.session.add(survey)

        else:
            survey = Survey(author=current_user,
                            round_num=round,
                            frustration = form.frustration.data,
                            utility_of_feedback = form.utility_of_feedback.data,
                            ease_of_teaching = form.ease_of_teaching.data,
                            opt_text = form.opt_text.data)
            db.session.add(survey)

        updated_num_trials_completed = current_user.num_trials_completed + 1
        current_user.set_num_trials_completed(updated_num_trials_completed)

        db.session.commit()

        # If the user submits the form, and they stil have more trials to do, we take them to the next game
        if current_user.num_trials_completed < len(current_condition.trials):
            if current_user.study_type == 'online' and current_user.num_trials_completed == 1:
                # with an attention check intermission if they are online participants
                return redirect(url_for("attn_check"))
            return redirect(url_for("test"))
        # If they have completed all the trials, but haven't completed the study, they still have the final survey to do
        elif current_user.num_trials_completed == len(current_condition.trials) and current_user.study_completed == 0:
            return redirect(url_for("survey"))
        # Otherwise, they are complete and can receive their payment code
        else:
            return redirect(url_for("index"))

    # If the survey corresponding to the current trial number has already been completed, they either need to
    # complete the next game or move on to the final round
    check_previous_surveys = db.session.query(Survey).filter_by(user_id=current_user.id, round_num=round).count()
    if check_previous_surveys == 1:
        if current_user.num_trials_completed < len(current_condition.trials):
            print("or this?")
            return redirect(url_for("test"))
        elif current_user.study_completed == 0:
            return redirect(url_for("survey"))
        else:
            return redirect(url_for("index"))

    # If they have not completed the previous game, they must do that 
    completed_game = db.session.query(Trial).filter_by(user_id=current_user.id).count()
    print(current_user.id)
    print(completed_game)
    print(db.session.query(Trial).filter_by(user_id=current_user.id))
    if completed_game == 0 and round <= len(current_condition.trials):
        print("is this where we are?")
        print(round)
        print(completed_game)
        return redirect(url_for("test"))
    # if current_user.num_trials_completed < round - 1:
    #     return redirect(url_for("consent"))
    # TODO add an appropriate check back in
    # check_rule_name = current_condition.difficulty[round]
    # check_previous_trials = db.session.query(Trial).filter_by(user_id=current_user.id, round_num=round).count()
    # if check_previous_trials < len(RULE_PROPS[check_rule_name]['cards']):
    #     return redirect(url_for("consent"))    
    return render_template(template, 
                            methods=["GET", "POST"], 
                            form=form, 
                            round=round)

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user is None:
            user = User(username=form.username.data)
            user.set_num_trials_completed(0)
            user.set_completion(0)
            user.set_attention_check(-1)

            # Change depending on the study type. 
            cond = user.set_condition("in_person" if IS_IN_PERSON else "online")
            code = user.set_code()

            db.session.add(user)
        
            cond.users.append(user)
            cond.count += 1

            db.session.commit()

        login_user(user)
        next_page = request.args.get("next")
        if not next_page or url_parse(next_page).netloc != "":
            next_page = url_for("index")
        return redirect(next_page)

    return render_template("login.html", title="Sign In", form=form)


@app.route("/logout")
def logout():
    # Clear this variable -- for some reason was not clearing on its own. 
    session.pop("attention_check_rules", None)
    logout_user()
    return {"url":url_for("index")}
    # return redirect(url_for("index"))

# @app.route("/register", methods=["GET", "POST"])
# def register():
#     if current_user.is_authenticated:
#         # return redirect(url_for("test"))
#         return redirect(url_for("index"))
#     form = RegistrationForm()
#     if form.validate_on_submit():
#         user = User(username=form.username.data)
#         user.set_password(form.password.data)
#         user.set_num_trials_completed(0)
#         user.set_completion(0)
#         user.set_attention_check(-1)

#         # Change depending on the study type. 
#         cond = user.set_condition("online")
#         code = user.set_code()


#         db.session.add(user)
        
#         cond.users.append(user)
#         cond.count += 1

#         feedback = {}
#         for vid in VIDEO_LIST:
#             feedback[vid] = 0
        
#         user.feedback_counts = feedback

#         db.session.commit()
#         flash("Congratulations, you are now a registered user!")
#         return redirect(url_for("login"))
#     return render_template("register.html", title="Register", form=form)