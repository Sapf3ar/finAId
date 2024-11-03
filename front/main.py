import streamlit as st
from functools import partial
import time
from rag.helper import process_prompt, get_rag, RagQuery
from agents.agent import AgentManager, LLMClient
import json

st.set_page_config(
    page_title="finance AId",
    page_icon=":material/account_balance:",
    layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        "About": "https://t.me/ohsapfear",
    },
)
# st.title( "FinAId: ваша суперсила в анализе документов")
st.markdown(
    "<h2 style='text-align: center; color: black;'>FinAId: ваша суперсила в анализе документов</h1>",
    unsafe_allow_html=True,
)
def state_filler(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_resource
def init_agent():
    if "agent_manager" not in st.session_state:
        with open('prompts/prompts.json', 'r') as f:
            prompts = json.load(f)
        with open('data/filled_metric_database.json', 'r') as f:
            filled_metric_database = json.load(f)
    
        st.session_state["agent_manager"] = AgentManager(prompt_dict=prompts, financials=filled_metric_database['2015'], llm_client=LLMClient)

init_agent()

def colorize_multiselect_options(colors: list[str]) -> None:
    rules = ""
    n_colors = len(colors)

    for i, color in enumerate(colors):
        rules += f""".stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"]:nth-child({n_colors}n+{i}){{background-color: {color};}}"""

    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)





year_color = "#FFA07A"  # Light salmon for years
metric_color = "#87CEEB"  # Sky blue for metrics
plot_type_color = "#90EE90"  # Light green for plot types
colors = [year_color, metric_color, plot_type_color]
colorize_multiselect_options(colors)
options = st.multiselect(
    "**Вы...**",
    [
        "Инвестор в Фонде",
        "Индивидуальный инвестор",
        "Кредитный аналитик",
        "Финансист в корпорации",
    ],
    key="profile",
)


def _init_state():
    state_filler("output", "Type your query here...")
    state_filler("rag_response", " ")
    state_filler("prompted", False)
    state_filler("profile", [])


_init_state()


def send_prompt():
    prompt = st.session_state["prompt_input"]
    prompt = get_rag(RagQuery(prompt=prompt, profiles=st.session_state["profile"]), agent_manager = st.session_state["agent_manager"])
    st.session_state["rag_response"] = prompt
    st.session_state["base_rag_response"] = partial(process_prompt, prompt)

    st.session_state["prompted"] = True


def process_add_metrics(query: RagQuery):
    for plot in query.plots:
        yield plot


st.text_input(
    label="**Type your query here**",
    key="prompt_input",
    on_change=send_prompt,
    value=" ",
)
st.divider()
with st.container(border=True) as cont:
    if st.session_state["prompted"] is True:
        # st.write_stream(st.session_state["base_rag_response"]())

        response = [i for i in st.session_state["base_rag_response"]()]
        message_placeholder = st.empty()
        full_response = "##### "
        for chunk in response:
            full_response += chunk + " "
            time.sleep(0.1)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        # years = ["2021", "2022", "2023"]
        years = st.session_state["rag_response"].data.year.unique().tolist()
        metrics = st.session_state["rag_response"].base_metrics
        comps = st.session_state["rag_response"].companies
        plot_types = ["line"]

        year_color = "#FFA07A"  # Light salmon for years
        metric_color = "#87CEEB"  # Sky blue for metrics
        plot_type_color = "#90EE90"  # Light green for plot types
        comp = "#90EE00"  # Light green for plot types

        def create_tag(tag, color):
            return f'<span style="background-color:{color}; padding:4px; border-radius:4px; color:black; font-weight:bold;">{tag}</span>'

        tag_markdown = ""  # title for the tags section

        tag_markdown += (
            "**Years:** "
            + " ".join([create_tag(year, year_color) for year in years])
            + "&nbsp;&nbsp;&nbsp;&nbsp;"
        )
        tag_markdown += (
            "**Companies:** "
            + " ".join([create_tag(c, comp) for c in comps])
            + "&nbsp;&nbsp;&nbsp;&nbsp;"
        )
        tag_markdown += (
            "**Metrics:** "
            + " ".join([create_tag(metric, metric_color) for metric in metrics])
            + "&nbsp;&nbsp;&nbsp;&nbsp;"
        )
        tag_markdown += (
            "**Plot types:** "
            + " ".join([create_tag(plot, plot_type_color) for plot in plot_types])
            + "&nbsp;&nbsp;&nbsp;&nbsp;"
        )
        with st.expander("### **Found tags**", expanded=True):
            st.markdown(tag_markdown, unsafe_allow_html=True)
with st.container() as cont1:
    if st.session_state["prompted"] is True:
        tab1, tab2 = st.tabs(["### **Extracted plot**", "###  **Suggested plots**"])
        col1, col2 = st.columns([0.6, 0.4], gap="medium")
        with tab1:
            for plot in st.session_state["rag_response"].base_plots:
                st.plotly_chart(plot, use_container_width=True, theme=None)
                time.sleep(0.2)
        with tab2:
            for group in st.session_state["rag_response"].plots:
                for group_name, plot in group.items():
                    with st.expander(group_name):
                        st.plotly_chart(plot, use_container_width=True)

                    # process_add_metrics(query=st.session_state["rag_response"])
