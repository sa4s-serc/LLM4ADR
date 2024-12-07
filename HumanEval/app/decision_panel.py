import streamlit as st
from streamlit import session_state as state
import time

from decision_viewer import DecisionViewer
from db import ExperimentRun

from decision_model import DecisionModel

from logger import logger
from datetime import datetime


def init_state():
    # Initialize session state if it doesn't exist
    if "decisions" not in state:
        state.decisions = [
            DecisionModel(1, ""),
            DecisionModel(2, ""),
        ]
        state.generate_decision_clicked = False
        state.context_input = ""
        state.selected_context_id = None
        # state.context_library_open = False

        # rerun()


class DecisionPanel():

    def __init__(self, root):
        dec1, dec2 = root.columns(2)
        self.uid = state.uid
        # self.cookie_manager = cookie_manager

        with dec1:
            DecisionViewer(0)
        with dec2:
            DecisionViewer(1)

        for decision in state.decisions:
            if decision.rating == 0:
                st.warning(f"Please rate Decision {decision.id} before submitting")
                break

        btn_disabled = any(decision.rating == 0 for decision in state.decisions)

        if not btn_disabled:
            st.button(
                "Submit ratings",
                key="right_button",
                on_click=self.handle_submit_rating,
                disabled=btn_disabled,
                help="Rate each decision first" if btn_disabled else "Click to submit ratings"
            )

    def handle_submit_rating(self):
        decisions = []
        for i, decision in enumerate(state.decisions):
            decisions.append(
                {
                    "text": decision.text,
                    "rating": decision.rating,
                    "notes": decision.notes,
                    "approach": int(str(state.mapping[i+1]).replace("Approach ", ""))
                }
            )

        if decisions[1]["approach"] < decisions[0]["approach"]:
            decisions[0], decisions[1] = decisions[1], decisions[0]

        exp = ExperimentRun(
            user_id=self.uid,
            context=state.context_input,
            timestamp=datetime.now(),

            decision1=decisions[0]["text"],
            decision1_rating=decisions[0]["rating"],
            decision1_note=decisions[0]["notes"],
            decision1_approach=decisions[0]["approach"],

            decision2=decisions[1]["text"],
            decision2_rating=decisions[1]["rating"],
            decision2_note=decisions[1]["notes"],
            decision2_approach=decisions[1]["approach"],
        )

        run_id = state.db.insert_experiment_run(exp)

        if run_id is None:
            st.error("An error occurred while submitting ratings!")
            return

        success = st.success("Ratings submitted successfully!")
        logger(3)
        state.runs_left -= 1
        time.sleep(1)

        # self.cookie_manager.set(
            # "runs_left", self.cookie_manager.get("runs_left") - 1)

        # if state.runs_left == 0:
        # # if self.cookie_manager.get("runs_left") == 0:
        #     logger(4)

        del state.decisions
        success.empty()
