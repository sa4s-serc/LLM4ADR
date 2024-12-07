from streamlit import session_state as state
import sys
from datetime import datetime
from constants import total_runs
import streamlit as st

log_file = "logs.txt"

@st.cache_data(ttl=0, persist=False)
def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_new_user(uid):
    text = f"[{get_time()}] [{uid}] User registered successfully\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_decision_generated(uid):
    decisions = state.decisions
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = f"[{curr_time}] [{uid}:{run_num}] Decision generated\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Context: {state.context_input}\n"
    for i, decision in enumerate(decisions):
        text += f"[{get_time()}] [{uid}:{run_num}] Decision {i + 1}: {decision.text}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_decision_submitted(uid):
    decisions = state.decisions
    run_num = total_runs - state.runs_left + 1

    text = f"[{get_time()}] [{uid}:{run_num}] Rating submitted\n"
    for i, decision in enumerate(decisions):
        text += f"[{get_time()}] [{uid}:{run_num}] {state.mapping[i + 1]}:{decision.rating}|{decision.notes}|{decision.text}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_custom_decision_saved(uid, context, decision):
    text = f"[{get_time()}] [{uid}] Custom decision saved\n"
    text += f"[{get_time()}] [{uid}] Context: {context}\n"
    text += f"[{get_time()}] [{uid}] Decision: {decision}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_study_completed(uid):
    text = f"[{get_time()}] [{uid}] Study completed\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_approach_one(uid, context, decision, gen_time):
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = "**********\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Approach 1\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Context: {context}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Decision: {decision}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Generation time: {gen_time}\n"
    text += "**********\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_approach_two(uid, context, decision, matched_ids, gen_time):
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = "**********\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Approach 2\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Context: {context}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Decision: {decision}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Matched IDs: {matched_ids}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Generation time: {gen_time}\n"
    text += "**********\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_approach_three(uid, context, decision, gen_time):
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = "**********\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Approach 3\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Context: {context}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Decision: {decision}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Generation time: {gen_time}\n"
    text += "**********\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_approach_four(uid, context, decision, matched_ids, gen_time):
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = "**********\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Approach 4\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Context: {context}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Decision: {decision}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Matched IDs: {matched_ids}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Generation time: {gen_time}\n"
    text += "**********\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_approach_five(uid, context, decision, matched_ids, gen_time):
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = "**********\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Approach 5\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Context: {context}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Decision: {decision}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Matched IDs: {matched_ids}\n"
    text += f"[{curr_time}] [{uid}:{run_num}] Generation time: {gen_time}\n"
    text += "**********\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def log_error(error):
    uid = state.uid
    run_num = total_runs - state.runs_left + 1
    curr_time = get_time()
    text = f"[{curr_time}] [{uid}:{run_num}] Error: {error}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, file=sys.stderr)


def logger(type, **kwargs):
    """
    type
    1: User registered successfully 
    2: Decision generated
    3: Decision submitted
    4: Custom decision saved
    5: Study completed
    """
    if type == 1:
        log_new_user(state.uid)

    if type == 2:
        log_decision_generated(state.uid)

    if type == 3:
        log_decision_submitted(state.uid)

    if type == 4:
        log_custom_decision_saved(state.uid, kwargs.get('context', ""), kwargs.get('decision', ""))

    if type == 5:
        log_study_completed(state.uid)

    if type == 'appr1':
        log_approach_one(state.uid, kwargs.get('context', ""), kwargs.get('decision', ""), kwargs.get('gen_time', ""))

    if type == 'appr2':
        log_approach_two(state.uid, kwargs.get('context', ""), kwargs.get('decision', ""), kwargs.get('matched_ids', ""), kwargs.get('gen_time', ""))

    if type == 'appr3':
        log_approach_three(state.uid, kwargs.get('context', ""), kwargs.get('decision', ""), kwargs.get('gen_time', ""))

    if type == 'appr4':
        log_approach_four(state.uid, kwargs.get('context', ""), kwargs.get('decision', ""), kwargs.get('matched_ids', ""), kwargs.get('gen_time', ""))

    if type == 'appr5':
        log_approach_five(state.uid, kwargs.get('context', ""), kwargs.get('decision', ""), kwargs.get('matched_ids', ""), kwargs.get('gen_time', ""))

    if type == 'error':
        log_error(kwargs.get('error', ""))
