import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import KNNBasic, Dataset, Reader
from pydantic import BaseModel, Field
from google import genai

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="BookMatch", page_icon="📚", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');

body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a1628 0%, #0d2145 50%, #0a1628 100%) !important;
    color: #e8eef7 !important;
}
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #ffffff !important; }
p, label, div { font-family: 'Inter', sans-serif !important; color: #c8d8f0 !important; }

.book-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(79,163,224,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.book-card:hover { border-color: rgba(79,163,224,0.5); }
.book-rank { font-size: 0.75rem; font-weight: 600; color: #4fa3e0 !important; text-transform: uppercase; letter-spacing: 0.08em; }
.book-title { font-family: 'Playfair Display', serif !important; font-size: 1rem; color: #ffffff !important; margin: 0.2rem 0; }
.book-author { font-size: 0.8rem; color: #8aa4c8 !important; }
.book-reason { font-size: 0.82rem; color: #a0bcdc !important; margin-top: 0.5rem; font-style: italic; border-left: 2px solid #4fa3e0; padding-left: 0.6rem; }
.section-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #4fa3e0 !important; margin-bottom: 0.8rem; }

.stButton > button {
    background: linear-gradient(90deg, #1a5fa8, #4fa3e0) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 500 !important; width: 100% !important;
}
.stButton > button:hover { background: linear-gradient(90deg, #4fa3e0, #a78bfa) !important; }
.stTextInput > div > div > input {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid rgba(79,163,224,0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
            .stSelectbox > div > div {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid rgba(79,163,224,0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
.stSelectbox > div > div > div {
    color: #ffffff !important;
}
[data-baseweb="select"] * {
    background-color: #0d2145 !important;
    color: #ffffff !important;
}

[data-baseweb="popover"] * {
    background-color: #0d2145 !important;
    color: #ffffff !important;
}

li[role="option"] {
    background-color: #0d2145 !important;
    color: #ffffff !important;
}

li[role="option"]:hover {
    background-color: #1a5fa8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    books = pd.read_csv('Books.csv', low_memory=False)
    ratings = pd.read_csv('Ratings.csv')
    books['language_code'] = books['language_code'].fillna('unknown')
    median_year = books['original_publication_year'].median()
    books['original_publication_year'] = books['original_publication_year'].fillna(median_year).astype(int)
    books = books.drop(columns=['isbn'])
    return books, ratings

books, ratings = load_data()
book_lookup = books.set_index('book_id')[['title', 'authors', 'small_image_url']].to_dict('index')
counts = ratings['book_id'].value_counts()
popular_books = set(counts[counts >= 20].index)

# ── Train Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
    full_trainset = data.build_full_trainset()
    # k=50 chosen for serving (more neighbors → smoother scores at inference); evaluation uses k=10
    model = KNNBasic(k=50, sim_options={"name": "pearson", "user_based": True}, verbose=False)
    model.fit(full_trainset)
    return model

# ── Top-N Function ───────────────────────────────────────────────────────────
def top_n_for_user(model, user_id, top_n=5):
    seen = set(ratings.loc[ratings['user_id'] == user_id, 'book_id'])
    scored = [
        (book_lookup[b]['title'], book_lookup[b]['authors'], model.predict(user_id, b).est, book_lookup[b]['small_image_url'])
        for b in books['book_id']
        if b not in seen and b in popular_books
    ]
    return sorted(scored, key=lambda x: -x[2])[:top_n]

# ── Gemini Re-ranking ────────────────────────────────────────────────────────
class BookPick(BaseModel):
    title: str = Field(description="Exact book title from the candidate list.")
    authors: str = Field(description="Book author(s) as listed in the candidate list.")
    reason: str = Field(description="One sentence on why this book fits the preference.")

def rerank_with_gemini(candidates, preference, api_key):
    client = genai.Client(api_key=api_key)
    catalog = "\n".join(
        f"{i+1}. {title} by {authors} (CF score: {score:.2f})"
        for i, (title, authors, score, img) in enumerate(candidates)
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Viewer preference: {preference}\n\nCandidate books:\n{catalog}",
        config={
            "system_instruction": (
                    "You are a book concierge. You must ONLY use books from the candidate list provided. "
                    "Do NOT add any new books or remove any books. Re-rank ALL books from the candidate list "
                    "based on the viewer's preference, returning all of them in a new order with a short reason. "
                     "Use the exact titles and author names from the candidate list. "
                     "Your response must contain exactly the same number of books as the candidate list."

            ),
            "response_mime_type": "application/json",
            "response_schema": list[BookPick],
        },
    )
    return response.parsed

# ── Session State ────────────────────────────────────────────────────────────
for key in ['page', 'candidates', 'picks', 'preference', 'user_id']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.page is None:
    st.session_state.page = 'home'

def hero():
    st.markdown("""
    <div style='text-align:center; padding: 2rem 2rem 1rem;'>
        <h1 style='font-size:3rem; background: linear-gradient(90deg, #4fa3e0, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>📚 BookMatch</h1>
    </div>
    """, unsafe_allow_html=True)

# ── Home Page ────────────────────────────────────────────────────────────────
if st.session_state.page == 'home':
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem 2rem;'>
        <h1 style='font-size:3.5rem; background: linear-gradient(90deg, #4fa3e0, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>📚 BookMatch</h1>
        <p style='color:#8aa4c8; font-size:1.1rem;'>Collaborative filtering meets AI — discover books tailored to your taste and mood.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Get Started →"):
            st.session_state.page = 'select_user'
            st.rerun()

# ── User Selection Page ──────────────────────────────────────────────────────
elif st.session_state.page == 'select_user':
    hero()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Select your User ID")
        user_ids = sorted(ratings['user_id'].unique())
        user_id = st.selectbox("User ID", user_ids)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Find My Books →"):
            with st.spinner("Finding your recommendations..."):
                model = train_model()
                st.session_state.candidates = top_n_for_user(model, user_id, top_n=5)
                st.session_state.user_id = user_id
            st.session_state.page = 'recommendations'
            st.rerun()

# ── Recommendations Page ─────────────────────────────────────────────────────
elif st.session_state.page == 'recommendations':
    hero()
    st.markdown(f"<p style='text-align:center;color:#8aa4c8'>User {st.session_state.user_id}</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Your Top 5 — Collaborative Filtering</div>", unsafe_allow_html=True)
    for i, (title, authors, score, img) in enumerate(st.session_state.candidates[:5], 1):
        st.markdown(f"""
        <div class='book-card' style='display:flex; gap:1rem; align-items:center;'>
            <img src='{img}' style='width:50px; height:75px; border-radius:4px; object-fit:cover;'>
            <div>
                <div class='book-rank'>#{i}</div>
                <div class='book-title'>{title}</div>
                <div class='book-author'>by {authors}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖 Personalize with AI")
    preference = st.text_input("What are you in the mood for?", placeholder="e.g. a dark psychological thriller")
    api_key = st.secrets["GEMINI_API_KEY"]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Re-rank with AI ✨"):
            if not preference:
                st.warning("Please enter a preference!")
            else:
                with st.spinner("Gemini is personalizing your picks..."):
                    picks = rerank_with_gemini(st.session_state.candidates, preference, api_key)
                st.session_state.picks = picks
                st.session_state.preference = preference
                st.session_state.page = 'results'
                st.rerun()
    with col2:
        if st.button("← Start Over"):
            st.session_state.page = 'home'
            st.rerun()

# ── Results Page ─────────────────────────────────────────────────────────────
elif st.session_state.page == 'results':
    hero()
    st.markdown(f"<p style='text-align:center;color:#8aa4c8'>Mood: <em>\"{st.session_state.preference}\"</em></p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_cf, col_ai = st.columns(2)

    with col_cf:
        st.markdown("<div class='section-label'>Collaborative Filtering</div>", unsafe_allow_html=True)
        for i, (title, authors, score, img) in enumerate(st.session_state.candidates[:5], 1):
            st.markdown(f"""
            <div class='book-card' style='display:flex; gap:1rem; align-items:center;'>
                <img src='{img}' style='width:50px; height:75px; border-radius:4px; object-fit:cover;'>
                <div>
                    <div class='book-rank'>#{i}</div>
                    <div class='book-title'>{title}</div>
                    <div class='book-author'>by {authors}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with col_ai:
        st.markdown("<div class='section-label'>AI Re-ranked by Gemini ✨</div>", unsafe_allow_html=True)
        img_lookup = {title: img for title, _, _, img in st.session_state.candidates}
        unmatched = [pick.title for pick in st.session_state.picks[:5] if pick.title not in img_lookup]
        if unmatched:
            st.caption(f"Note: {len(unmatched)} title(s) from AI response didn't match candidates exactly.")
        for i, pick in enumerate(st.session_state.picks[:5], 1):
            img = img_lookup.get(pick.title, '')
            st.markdown(f"""
            <div class='book-card' style='display:flex; gap:1rem; align-items:flex-start;'>
                <img src='{img}' style='width:50px; height:75px; border-radius:4px; object-fit:cover;'>
                <div>
                    <div class='book-rank'>#{i}</div>
                    <div class='book-title'>{pick.title}</div>
                    <div class='book-author'>by {pick.authors}</div>
                    <div class='book-reason'>{pick.reason}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Try Different Preference"):
            st.session_state.page = 'recommendations'
            st.rerun()
    with col2:
        if st.button("Start Over"):
            st.session_state.page = 'home'
            st.rerun()