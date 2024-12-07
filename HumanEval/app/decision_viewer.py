import streamlit as st
from streamlit import session_state as state


class DecisionViewer():

    def __init__(self, dec_id):
        decision = state.decisions[dec_id]
        with st.container():
            # st.write(f"## Decision {decision.id}")
            # state.decisions[dec_id].root = st.empty()
            st.markdown(decision.text)

            new_rating = st.slider(
                "Rating",
                min_value=0,
                max_value=5,
                value=decision.rating,
                key=f"rating_{decision.id}",
                step=1
            )
            state.decisions[dec_id].rating = new_rating

            notes = st.text_area(
                "Any feedback about the generated decision?",
                value=decision.notes,
                placeholder="Write about what you like or dislike about the decision.",
                key=f"notes_{decision.id}"
            )
            state.decisions[dec_id].notes = notes
