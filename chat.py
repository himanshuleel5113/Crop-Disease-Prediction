import streamlit as st
from google import genai

# --- Configuration & API Setup ---

# Set your page configuration
st.set_page_config(page_title="Custom Gemini Chatbot", layout="wide")
st.title("✨ My Streamlit Chatbot")

# You need an API key to run the model.
# Streamlit provides a secure way to handle secrets: st.secrets
# Alternatively, for local testing, you can set it as an environment variable.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (AttributeError, KeyError):
    # Fallback for local testing if not using st.secrets
    API_KEY = "YOUR_GEMINI_API_KEY"  # <-- Replace with your actual key

# Initialize the Gemini client and model
try:
    genai.configure(api_key=API_KEY)
    # Use a powerful model for chat, e.g., gemini-2.5-flash
    MODEL_NAME = "gemini-2.5-flash"
    client = genai.Client()
except Exception as e:
    st.error(f"Error initializing the Gemini API client: {e}")
    st.stop()


# --- Chat History Management ---

# Initialize chat history in Streamlit's session state
if "chat_history" not in st.session_state:
    # A list of dictionaries to store messages
    st.session_state.chat_history = []

# Initialize the generative model chat session
if "chat_session" not in st.session_state:
    # The system instruction/prompt gives your bot its personality and rules!
    system_instruction = (
        "You are a helpful and friendly AI assistant. "
        "Your goal is to provide concise, accurate, and insightful responses. "
        "Use Markdown formatting to make your answers easy to read."
    )
    
    # Pre-populate the model's history for context, if needed
    history_for_model = [
        {"role": "user", "parts": "Hello!"},
        {"role": "model", "parts": "Hello! I am your custom AI assistant. How can I help you today?"}
    ]
    
    st.session_state.chat_session = client.chats.create(
        model=MODEL_NAME,
        system_instruction=system_instruction,
        history=history_for_model
    )
    # Add the initial history to the displayed chat_history
    st.session_state.chat_history.extend(history_for_model)


# --- Display Chat Messages ---

# Display all messages from the chat history
for message in st.session_state.chat_history:
    # Use Streamlit's native chat elements for a modern look
    with st.chat_message(message["role"]):
        st.markdown(message["parts"])

# --- User Input & Response Generation ---

# Accept user input at the bottom of the page
if prompt := st.chat_input("Ask me anything..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to session history
    st.session_state.chat_history.append({"role": "user", "parts": prompt})
    
    # 2. Get Response from LLM
    with st.chat_message("model"):
        # Use st.spinner for a loading state while waiting for the API
        with st.spinner("Thinking..."):
            # Send the message to the model's chat session to maintain history
            response = st.session_state.chat_session.send_message(prompt)
            full_response = response.text
            st.markdown(full_response)
    
    # 3. Add Assistant Message to Session History
    st.session_state.chat_history.append({"role": "model", "parts": full_response})