import streamlit as st
from streamlit import session_state as state
from db import User
from streamlit import rerun
from logger import logger
from constants import total_runs
from datetime import datetime

def validate_form(name, email, age, done_se, industry_exp, industry_org_name, rate_yourself, check):
    if not name:
        return False, "Name is required"
    if email and (not email.count("@") == 1 or not email.count(".")) > 0:
        return False, "Please enter a valid email"
    if not age or age == 18:
        return False, "Age is required"
    if not done_se:
        return False, "Please select if you have done software engineering courses before"
    if industry_exp == -1:
        return False, "Industry experience is required"
    if industry_exp > 0 and not industry_org_name:
        return False, "Please enter the name of the organization you work for"
    if rate_yourself == 0:
        return False, "Rate yourself is required"
    if not check:
        return False, "Please consent your participation in the study"
    return True, None


def register():
    st.markdown("# LLM4ADR")

    st.markdown(
        """
        > Note: You are expected to complete the study in one sitting. \\
        If you do not have the time to complete the study now, please come back later. \\
        Please do not close the browser tab or refresh the page during the study. 


        ### Welcome to the study!

        This study is to understand how effective Large Language Models are in generating Architectural Decision Records (ADRs). 
        An ADR is a document that captures an architectural decision made along with its context and consequences. 
        In this study, you will provide a context and you will be asked to rate two different decisions generated by a language model. 
        You can also use existing context from the context library.

        """
    )

    # st.write("This is an example of an ADR:")
    # st.text("\u00ad    Context: We need to decide on a database for our application.")
    # st.text("\u00ad    Decision: We will use PostgreSQL because it is open-source and has good community support.")

    st.markdown(
        """
        Please fill in the following details to register for the study. 
        Your information will be kept confidential and will not be shared with anyone.
        """
    )

    with st.form(key="register-form"):
        st.write("Please enter your name")
        name = st.text_input("Name *")

        st.write("Please enter your email. This will be used to contact you for further studies (if any)")
        email = st.text_input("Email (optional)")

        # st.write("Please enter your age")
        # age = st.number_input("Age", 18, 100, 18)
        age = 25

        st.write("Have you done software engineering courses before?")
        done_se = st.radio("Select", ["Yes", "No"])

        st.write("How many years of industry experience do you have?")
        industry_exp = st.number_input("Industry experience (in years)", 0, 30, 0)

        st.write("Please enter the name(s) of the organization you work/had worked for (if you have industry experience)")
        industry_org_name = st.text_input("Organization name")

        st.write("Rate your knowledge of software engineering on a scale of 1 to 5 *")
        rate_yourself = st.number_input("Rate yourself", 0, 5, 0)

        check = st.checkbox("I agree to participate in the study")

        submit = st.form_submit_button("Register")

        # Validate the form after the submit button is clicked
        if submit:
            is_valid, error = validate_form(
                name, email, age, done_se, industry_exp, industry_org_name, rate_yourself, check)
            if not is_valid:
                st.error(error)
                return

            user = User(
                name=name, email=email, age=age, done_se=done_se,
                industry_exp=industry_exp, industry_org_name=industry_org_name, rate_yourself=rate_yourself,
                created_at=datetime.now()
            )

            uid = state.db.insert_user(user)
            # cookies = {
            #     "uid": uid,
            #     "runs_left": 5
            # }
            state.uid = uid
            state.runs_left = total_runs
            state.user_instr = False
            state.custom_decision_saved = False
            state.selected_ids = []
            logger(1)
            st.success("User registered successfully")

            rerun()
