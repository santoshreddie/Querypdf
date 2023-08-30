import streamlit as st
# import chat_function
from chat_function import chat_with_pdf

from streamlit_chat import message

st.markdown("# chat 🎉")
st.sidebar.markdown("# chat 🎉")
    
# Chat component to chat with your uploaded PDF
st.header("Chat with your PDF...")
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.chat_input("Enter your query to retrieve answer from my knowledge...")
    return input_text

user_input = get_text()

if user_input:
    output = chat_with_pdf(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
