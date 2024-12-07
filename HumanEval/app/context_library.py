import streamlit as st
from functools import partial
from streamlit import session_state as state
import random

class ContextLibraryUI():
    def __init__(self):
        data = {}
        resp = state.db.get_context_library()
        for row in resp:
            data[row[1]] = row[0]
        for id_ in state.selected_ids:
            if id_ in data:
                del data[id_]
        self._contexts = data

    def select_context(self, id):
        state.context_input = self._contexts[id]
        state.selected_context_id = id
        state.selected_ids.append(id)

    def display(self):
        for id, cont in self._contexts.items():
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])

                with col1:
                    st.markdown(cont.replace("\\n", " <br/> "),
                                unsafe_allow_html=True)
                col2.button("Select", id, on_click=partial(self.select_context, id))


def get_final_context():
    data = {}
    resp = state.db.get_context_library()
    for row in resp:
        data[row[1]] = row[0]
    for id_ in state.selected_ids:
        if id_ in data:
            del data[id_]
    if len(data) == 0:
        return "## Context and Problem Statement\nAll libraries use their own query syntax for advanced search options. To increase usability, users should be able to formulate their (abstract) search queries in a query syntax that can be mapped to the library specific search queries. To achieve this, the query has to be parsed into an AST.\nWhich query syntax should be used for the abstract queries?\nWhich features should the syntax support?\n"
    ind = random.choice(list(data.keys()))
    return data[ind]
