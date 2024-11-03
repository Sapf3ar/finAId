import streamlit as st

pages = {
    "Main Pages": [
        st.Page("front/main.py", title="Query best llm fact checker"),
        st.Page("front/explore_db.py", title="Explore build vector database"),
    ],
}

pg = st.navigation(pages=pages, expanded=False)
pg.run()
