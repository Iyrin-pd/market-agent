import streamlit as st
from agent import run

st.set_page_config(page_title="Market Research Agent", layout="wide")
st.title("Market Research Agent")
st.caption("Ask anything about your market in plain language.")

q = st.text_input("Your question")
context = st.text_area(
    "Business context (this makes the report specific)",
    placeholder="Company, product, target customer, country, price range, and the decision you need to make. "
                
)

if st.button("Research") and q:
    with st.status("Agent working...", expanded=True) as status:
        result = run(q, context=context, log=lambda m: st.write(m))
        status.update(label="Done", state="complete")

    st.markdown(result["report"])

    if result.get("table") is not None:
        st.subheader("Benchmark table")
        st.dataframe(result["table"])

    with st.expander("Sources"):
        for s in result["sources"]:
            st.markdown(f"**[{s['id']}]** [{s['title']}]({s['url']})")

    if result.get("facts"):
        with st.expander("Extracted facts (what the report is based on)"):
            st.dataframe(result["facts"])