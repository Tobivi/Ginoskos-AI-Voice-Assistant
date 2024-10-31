import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pyttsx3
import speech_recognition as sr
import os
import pickle
from dotenv import load_dotenv, find_dotenv
import threading
import time

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# Default settings
defaults = {
    "api_key": os.getenv("OPENAI_API_KEY"),  # API key is loaded directly from environment variable
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "voice": "com.apple.eloquence.en-US.Grandpa",
    "volume": 1.0,
    "rate": 200,
    "session_id": "abc123",
    "ability": "Psychology",
    "base_url": "https://api.openai.com/v1",
}

# Initialize speech recognition
r = sr.Recognizer()

# Speak function without caching
def speak(text):
    engine = pyttsx3.init()  # Initialize engine on each call
    engine.setProperty("volume", defaults["volume"])
    engine.setProperty("rate", defaults["rate"])
    engine.say(text)
    engine.runAndWait()
    engine.stop()  # Ensure the engine stops completely after each call

# Function to listen to user input
def listen():
    with sr.Microphone() as source:
        audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

# Function to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

# Main function
def main():
    st.title("🤖 Ginoskos AI Voice Assistant")
    
    # Sidebar settings
    st.sidebar.title("Settings")
    
    # API Settings
    st.sidebar.subheader("API Configuration")

    model = st.sidebar.selectbox("Model", 
                                ["gpt-3.5-turbo", "gpt-4"], 
                                index=0)
    temperature = st.sidebar.slider("Temperature", 
                                  min_value=0.0, 
                                  max_value=1.0, 
                                  value=defaults["temperature"])
    base_url = st.sidebar.text_input("Base URL", 
                                    value=defaults["base_url"])
    
    # Assistant Settings
    st.sidebar.subheader("Assistant Configuration")
    ability = st.sidebar.text_input("Assistant Ability", 
                                   value=defaults["ability"])
    session_id = st.sidebar.text_input("Session ID", 
                                      value=defaults["session_id"])
    
    # Voice Settings
    st.sidebar.subheader("Voice Configuration")
    volume = st.sidebar.slider("Volume", 
                              min_value=0.0, 
                              max_value=1.0, 
                              value=defaults["volume"])
    rate = st.sidebar.slider("Rate", 
                            min_value=20, 
                            max_value=500, 
                            value=defaults["rate"])

    # Initialize LangChain components
    llm = ChatOpenAI(
    temperature=defaults["temperature"],
    model=defaults["model"],
    base_url=defaults["base_url"],
    api_key=defaults["api_key"]  # Use API key from environment, not from user input
)


    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an assistant who's good at {ability}. Respond in 20 words or fewer"),
        ("ai", "Hello, I am Jarvis. How can I help you today?"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    runnable = prompt | llm
    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Voice interaction
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Start Recording", disabled=st.session_state.is_recording):
            st.session_state.is_recording = True
            st.info("Listening...")
            user_input = listen()  # Listen for user input
            st.session_state.is_recording = False  # Stop recording after listening

            if user_input:
                st.session_state.chat_history.append(("user", user_input))  # Log user input
                
                # Generate response
                response = with_message_history.invoke(
                    {"ability": ability, "input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                
                # Log the assistant's response
                st.session_state.chat_history.append(("assistant", response.content))
                
                # Speak the response immediately after generating it
                speak(response.content)

            # Indicate that response has been spoken
            st.success("Response has been spoken. You can ask another question.")

    with col2:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            if session_id in st.session_state:
                del st.session_state[session_id]

    # Text input as alternative to voice
    user_text = st.text_input("Or type your message here:", key="user_text")
    if st.button("Send"):
        if user_text:
            st.session_state.chat_history.append(("user", user_text))
            
            # Generate response
            response = with_message_history.invoke(
                {"ability": ability, "input": user_text},
                config={"configurable": {"session_id": session_id}},
            )
            
            # Log the assistant's response
            st.session_state.chat_history.append(("assistant", response.content))
            
            # Speak the response
            speak(response.content)
            
            # Clear the text input
            st.session_state.user_text = ""

    # Display chat history
    st.subheader("Chat History")
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Jarvis:** {message}")

if __name__ == "__main__":
    main()
