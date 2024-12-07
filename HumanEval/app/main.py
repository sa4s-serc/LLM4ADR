# from setup import cookie_manager
import streamlit as st
import json

from streamlit import session_state as state

from experiment import exp_main
from register import register
from db import Database


def footer():
    st.markdown(
        """
        <style>
        .footer {
            position: relative;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: grey;
        }
        </style>
        <div class="footer">
            <hr>
            <p>Copyright © 2024 LLM4ADR. Open Source Software.</p>
            <p><a style="color: grey; text-decoration: none;" href="https://sa4s-serc.github.io">SA4S Group</a>, Software Engineering Research Center, IIIT Hyderabad</p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.set_page_config(
    page_title="LLM4ADR",
    page_icon="🧊",
    layout="wide",
)

st.markdown(
    r"""
    <style>
    .stDeployButton {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

config = json.load(open("config.json"))

if config['switched'] == False:
    st.markdown(
        config['message'],
    )
    
else:
    if not state.get("uid"):
        state.db = Database()
        state.db.start()

        register()
    else:
        exp_main()

footer()
