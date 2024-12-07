from logger import logger
import streamlit as st
from streamlit import session_state as state
from approaches import generating_decision
context_placeholder_text = """
Example:
## Context
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
...
"""

decision_placeholder_text = """
## Decision
...
"""


def gen_decision(context):
    if context == "":
        st.warning("Please enter some text")
        return
    generating_decision(context)
    state.generate_decision_clicked = True

    logger(2)


def save_custom_decision():
    decision = state.decision_input
    context = state.context_input
    res = state.db.add_custom_decision(context, decision, state.uid)
    logger(4, context=context, decision=decision)
    state.custom_decision_saved = res


class ContextEditor():
    def __init__(self, button=True, decision_writer=False):
        with st.container():
            if decision_writer:
                st.write(
                    "Write the decision you would make in the given context in the text box below.")
                st.text_area(
                    "Enter your decision here in markdown format",
                    placeholder=decision_placeholder_text,
                    height=400, key="decision_input",
                    disabled=False
                )
                if state.decision_input == "":
                    st.error("Please write your decision")
                
                if state.decision_input != "":
                    st.button(
                        "Submit Decision",
                        key="right_submit",
                        on_click=lambda: save_custom_decision()
                    )
                return

            st.text_area(
                "Enter your context here in markdown format",
                placeholder=context_placeholder_text,
                height=400, key="context_input",
                disabled=state.generate_decision_clicked
            )

            if not button:
                return

            st.button(
                "Generate Decision",
                key="left_submit",
                disabled=(state.generate_decision_clicked or state.context_input == "" or not state.user_instr or len(state.context_input.split()) > 500),
                on_click=lambda: gen_decision(
                    state.context_input)
            )

            if state.context_input == "":
                st.warning("Please enter some text")

            if len(state.context_input.split()) > 500:
                st.warning("Maximum word limit exceeded")
