import streamlit as st
from functools import partial
from loguru import logger
from rag.helper import process_prompt
st.set_page_config(
    page_title="finance_AId",
    page_icon=":material/account_balance:",
    layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        "About": "https://t.me/ohsapfear",
    },
)


def state_filler(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def _init_state():
    state_filler("output", "Output for your prompt will be here..." )
    state_filler("rag_response", " " )
    state_filler("prompted",  False  )


_init_state()


def send_prompt():
    prompt = st.session_state["prompt_input"]
    out = st.toast("Started processing... :time:")
    logger.debug(prompt)
    st.session_state['rag_response'] = partial(process_prompt, prompt)

    st.session_state["prompted"] = True
    logger.debug(prompt)


st.text_input(label="Try your prompt here: ", 
              key="prompt_input", 
              on_change=send_prompt,
              value=st.session_state['output'])

logger.debug( st.session_state['prompted'])

if st.session_state['prompted'] == True:
    st.write_stream(st.session_state["rag_response"]())
