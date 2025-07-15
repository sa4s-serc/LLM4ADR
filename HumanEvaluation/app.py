# app.py
import streamlit as st
import random
from db_operations import (
    get_user_by_email, create_user,
    get_next_unrated_question, get_responses_for_question,
    save_rating
)

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'shuffled_responses' not in st.session_state:
    st.session_state.shuffled_responses = []
if 'collected_ratings' not in st.session_state:
    st.session_state.collected_ratings = {} # To temporarily hold ratings before submission

# --- Helper Functions ---
def fetch_and_display_question():
    """Fetches the next unrated question and its responses."""
    question = get_next_unrated_question(st.session_state.user_id)
    if question:
        st.session_state.current_question = question
        responses = get_responses_for_question(question['question_id'])
        # Shuffle responses to anonymize
        st.session_state.shuffled_responses = random.sample(responses, len(responses)) # Use random.sample to get a shuffled copy
        st.session_state.collected_ratings = {} # Reset ratings for new question
        st.rerun() # Rerun to update the display with the new question/responses
    else:
        st.session_state.current_question = None
        st.success("🎉 You have rated all available questions! Thank you for your contributions.")
        st.session_state.shuffled_responses = []
        st.session_state.collected_ratings = {}


def submit_all_ratings():
    """Saves all collected ratings for the current question to the database."""
    if not st.session_state.current_question:
        st.error("No question loaded to submit ratings for.")
        return

    question_id = st.session_state.current_question['question_id']
    user_id = st.session_state.user_id

    # Check if all 5 responses have ratings collected
    if len(st.session_state.collected_ratings) != 5:
        st.warning("Please provide ratings for all five responses.")
        return

    try:
        for response_id, ratings_data in st.session_state.collected_ratings.items():
            save_rating(
                user_id,
                question_id,
                response_id,
                ratings_data['relevance'],
                ratings_data['coherence'],
                ratings_data['completeness'],
                ratings_data['conciseness'],
                ratings_data['notes']
            )
        st.success("Ratings submitted successfully! Loading next question...")
        fetch_and_display_question() # Load next question after submission
    except Exception as e:
        st.error(f"An error occurred while saving ratings: {e}")

# --- Login Page ---
def login_page():
    st.title("Evaluator Login")

    with st.form("login_form"):
        name = st.text_input("Name", key="login_name").strip()
        email = st.text_input("Email", key="login_email").strip().lower()
        submitted = st.form_submit_button("Login / Register")

        if submitted:
            if not name or not email:
                st.error("Please enter both your name and email.")
                return

            user = get_user_by_email(email)
            if user:
                st.session_state.user_id = user['user_id']
                st.session_state.user_name = user['name']
                st.session_state.logged_in = True
                st.success(f"Welcome back, {st.session_state.user_name}!")
            else:
                new_user_id = create_user(name, email)
                if new_user_id:
                    st.session_state.user_id = new_user_id
                    st.session_state.user_name = name
                    st.session_state.logged_in = True
                    st.success(f"Successfully registered and logged in as {st.session_state.user_name}!")
                else:
                    st.error("Failed to register. Email might already be in use.")
            st.rerun() # Rerun to switch to the main app page

# --- Main Application Page ---
def main_app():
    st.sidebar.title(f"Welcome, {st.session_state.user_name}!")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.title("Finding Evaluation Platform")

    if st.session_state.current_question is None:
        fetch_and_display_question()
        return # Exit to wait for rerun if question was fetched

    if st.session_state.current_question:
        st.header("Question to Evaluate:")
        st.write(st.session_state.current_question['question_text'])

        st.markdown("---") # Visual separator

        # Temporary dictionary to hold ratings for current display cycle
        temp_ratings_for_this_display = {}

        # Use st.form to group all ratings for a single submission
        with st.form("ratings_form"):
            for i, response_data in enumerate(st.session_state.shuffled_responses):
                response_id = response_data['response_id']
                # Pre-fill collected_ratings if available (e.g., from a rerun after minor change)
                initial_relevance = st.session_state.collected_ratings.get(response_id, {}).get('relevance', 3)
                initial_coherence = st.session_state.collected_ratings.get(response_id, {}).get('coherence', 3)
                initial_completeness = st.session_state.collected_ratings.get(response_id, {}).get('completeness', 3)
                initial_conciseness = st.session_state.collected_ratings.get(response_id, {}).get('conciseness', 3)
                initial_notes = st.session_state.collected_ratings.get(response_id, {}).get('notes', "")


                st.subheader(f"Response {i+1}") # Anonymized label

                with st.container(border=True): # Each response gets its own bordered box
                    st.write(response_data['response_text']) # The actual content of the response

                    st.markdown("---") # Separator between response text and rating widgets

                    # Grouping rating sliders horizontally using columns
                    col_rel, col_coh = st.columns(2)
                    with col_rel:
                        relevance = st.slider(
                            "Relevance", 1, 5, value=initial_relevance, key=f"rel_{response_id}"
                        )
                    with col_coh:
                        coherence = st.slider(
                            "Coherence", 1, 5, value=initial_coherence, key=f"coh_{response_id}"
                        )

                    col_com, col_con = st.columns(2)
                    with col_com:
                        completeness = st.slider(
                            "Completeness", 1, 5, value=initial_completeness, key=f"com_{response_id}"
                        )
                    with col_con:
                        conciseness = st.slider(
                            "Conciseness", 1, 5, value=initial_conciseness, key=f"con_{response_id}"
                        )

                    notes = st.text_area(
                        "Optional Notes", value=initial_notes, key=f"notes_{response_id}"
                    )

                    # Store these collected values temporarily for submission
                    temp_ratings_for_this_display[response_id] = {
                        'relevance': relevance,
                        'coherence': coherence,
                        'completeness': completeness,
                        'conciseness': conciseness,
                        'notes': notes
                    }

            st.markdown("---") # Separator before the submit button

            submitted_button = st.form_submit_button("Submit All Ratings and Next Question")

            if submitted_button:
                # Update the main collected_ratings from temp for submission
                st.session_state.collected_ratings = temp_ratings_for_this_display
                submit_all_ratings()

    else:
        # This case is handled by fetch_and_display_question showing success message
        pass


# --- Main Application Logic ---
if st.session_state.logged_in:
    main_app()
else:
    login_page()