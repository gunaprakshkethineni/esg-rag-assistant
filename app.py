"""
My Streamlit front end.

I built five tabs: Chat, Compare Companies, Emissions Dashboard, DEGREE and
Evaluation Summary. The last three just read the CSV/JSON that my scripts in src/
and eval/ generate, so I have to run those first or the tabs tell me to.

    streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.config import COMPANIES, PROJECT_ROOT
from src.rag_chain import ask, detect_company

st.set_page_config(page_title="ESG Scope 1/2/3 RAG Assistant", layout="wide")

st.title("ESG Scope 1/2/3 RAG Assistant")
st.caption(
    "Retrieval-augmented Q&A over 7 corporate sustainability reports "
    "(Infosys, Microsoft, Siemens, Siemens Energy, Siemens Brazil) "
    "with local ChromaDB retrieval, Gemini generation, and a citation-consistency check."
)

tab_chat, tab_compare, tab_emissions, tab_degree, tab_eval = st.tabs(
    ["Chat", "Compare Companies", "Emissions Dashboard", "DEGREE / Target Verification", "Evaluation Summary"]
)


def render_answer(result, show_company_in_sources=True, excerpt_chars=500):
    """Draws one answer + its badges + the sources expander.

    Pulled this out so the Chat and Compare tabs look the same.
    """
    st.markdown(f"**Answer:** {result['answer']}")

    citation_badge = "✅ consistent" if result["citation_consistent"] else "⚠️ unverified citation"
    grounding_badge = "✅ grounded" if result["numerically_grounded"] else "⚠️ unverified numbers"
    st.caption(
        f"Latency: {result['latency_seconds']:.2f}s | Citation check: {citation_badge} | "
        f"Numeric grounding: {grounding_badge}"
    )
    if not result["numerically_grounded"]:
        st.warning(
            f"These numbers in the answer were not found verbatim in the retrieved excerpts "
            f"below -- treat with caution: {', '.join(result['ungrounded_numbers'])}"
        )

    if result["sources"]:
        with st.expander(f"Sources ({len(result['sources'])} retrieved chunks)"):
            for s in result["sources"]:
                dist_label = f"{s['distance']:.3f}" if s["distance"] is not None else "neighbor chunk"
                company_prefix = f"{s['company']} — " if show_company_in_sources else ""
                st.markdown(
                    f"**{company_prefix}{s['report_title']} ({s['report_year']}), p.{s['page_number']}** "
                    f"(similarity distance: {dist_label})"
                )
                st.text(s["text"][:excerpt_chars])
                st.divider()


with tab_chat:
    col1, col2 = st.columns([3, 1])
    with col2:
        company_filter = st.selectbox("Filter by company (optional)", ["All"] + COMPANIES)
        company_arg = None if company_filter == "All" else company_filter

    with col1:
        query = st.text_input("Ask a question about Scope 1/2/3 emissions, targets, or ESG data:")
        if query:
            with st.spinner("Retrieving and generating..."):
                result = ask(query, company=company_arg)

            # ask() guesses the company from the question even on "All", so show
            # which one it picked
            if company_arg is None:
                auto = detect_company(query)
                if auto:
                    st.caption(f"Auto-detected company from your question: **{auto}**")

            render_answer(result)

with tab_compare:
    st.subheader("Compare Two Companies")
    st.caption(
        "Runs the *same* question through the RAG pipeline twice, once per company filter, so "
        "each answer only ever sees that one company's own report(s) -- avoiding the cross-company "
        "retrieval mixing that a single unfiltered query can suffer from."
    )

    col_a_select, col_b_select = st.columns(2)
    with col_a_select:
        company_a = st.selectbox("Company A", COMPANIES, index=0, key="compare_company_a")
    with col_b_select:
        default_b = 1 if len(COMPANIES) > 1 else 0
        company_b = st.selectbox("Company B", COMPANIES, index=default_b, key="compare_company_b")

    compare_query = st.text_input(
        "Question to ask both companies:",
        key="compare_query",
        placeholder="e.g. What is your net-zero or carbon-neutral target year?",
    )

    if compare_query:
        if company_a == company_b:
            st.warning("Pick two different companies to compare.")
        else:
            col_a_result, col_b_result = st.columns(2)
            for col, company in ((col_a_result, company_a), (col_b_result, company_b)):
                with col:
                    st.markdown(f"### {company}")
                    with st.spinner(f"Asking {company}..."):
                        result = ask(compare_query, company=company)
                    render_answer(result, show_company_in_sources=False, excerpt_chars=350)

with tab_emissions:
    st.subheader("Extracted Scope 1/2/3 Emissions Summary")
    csv_path = PROJECT_ROOT / "data" / "emissions_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)

        numeric_df = df.copy()
        numeric_df["value_num"] = pd.to_numeric(
            numeric_df["value"].astype(str).str.replace(",", ""), errors="coerce"
        )
        chart_df = numeric_df.dropna(subset=["value_num"])
        if not chart_df.empty:
            st.bar_chart(chart_df, x="company", y="value_num", color="scope")
        st.caption(
            "Values and units are as extracted by the RAG pipeline from each report; see the "
            "'found' and 'citation_consistent' columns for reliability, and 'source_pages' for "
            "the exact page(s) the model cited."
        )
    else:
        st.info("Run `python -m src.emissions_extractor` first to generate data/emissions_summary.csv.")

with tab_degree:
    st.subheader("DEGREE Framework & Headline Target Verification")
    st.caption(
        "DEGREE is Siemens' own sustainability framework (Sustainability Report 2024, p.8-9). "
        "For Siemens, each field of action's self-reported progress is re-verified against the "
        "RAG pipeline. Other companies don't use DEGREE, so their own headline climate "
        "commitments are verified instead."
    )
    scorecard_path = PROJECT_ROOT / "data" / "degree_scorecard.json"
    if scorecard_path.exists():
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        st.markdown(f"*{scorecard.get('framework_source', '')}*")

        st.markdown("### Siemens DEGREE framework")
        for letter, row in scorecard.get("siemens_degree", {}).items():
            with st.expander(f"{letter} — {row['name']}"):
                st.markdown(f"**Description:** {row['description']}")
                st.markdown(f"**Stated target:** {row['stated_target']}")
                st.markdown(f"**RAG-verified progress:** {row['rag_verified_progress']}")
                st.caption(
                    f"Source pages: {row['source_pages']} | "
                    f"Citation consistent: {row['citation_consistent']} | "
                    f"Latency: {row['latency_seconds']}s"
                )

        st.markdown("### Other companies' headline targets")
        for company, row in scorecard.get("other_companies", {}).items():
            with st.expander(company):
                st.markdown(f"**Stated target:** {row['stated_target']}")
                st.markdown(f"**RAG-verified progress:** {row['rag_verified_progress']}")
                st.caption(
                    f"Source pages: {row['source_pages']} | "
                    f"Citation consistent: {row['citation_consistent']} | "
                    f"Latency: {row['latency_seconds']}s"
                )
    else:
        st.info("Run `python -m src.degree_scorer` first to generate data/degree_scorecard.json.")

with tab_eval:
    st.subheader("Evaluation Results (real measured numbers)")
    eval_path = PROJECT_ROOT / "eval" / "eval_results.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        summary = eval_data["summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval accuracy", f"{summary['retrieval_accuracy']*100:.1f}%")
        c2.metric("Answer accuracy", f"{summary['answer_accuracy']*100:.1f}%")
        c3.metric("Citation consistency", f"{summary['citation_consistency_rate']*100:.1f}%")
        c4.metric("Mean latency", f"{summary['mean_latency_seconds']:.2f}s")

        c5, c6, c7 = st.columns(3)
        c5.metric("Refusal accuracy (anti-hallucination)", f"{summary['refusal_accuracy']*100:.1f}%")
        c6.metric("Numeric grounding rate", f"{summary.get('numeric_grounding_rate', 0)*100:.1f}%")
        c7.metric("Share of responses under 2s", f"{summary['under_2s_rate']*100:.1f}%")

        st.markdown("### Per-question results")
        st.dataframe(pd.DataFrame(eval_data["rows"]), use_container_width=True)
    else:
        st.info("Run `python -m eval.run_eval` first to generate eval/eval_results.json.")
