import streamlit as st
import pyrebase
import os
import time
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# ─── Firebase Configuration ────────────────────────────────────────────────────
# Load credentials securely
config = st.secrets["firebase"]
firebase = pyrebase.initialize_app({
    "apiKey":            config["apiKey"],
    "authDomain":        config["authDomain"],
    "databaseURL":       config["databaseURL"],
    "projectId":         config["projectId"],
    "storageBucket":     config["storageBucket"],
    "messagingSenderId": config["messagingSenderId"],
    "appId":             config["appId"],
    "measurementId":     config["measurementId"]
})
auth = firebase.auth()
db = firebase.database()

# ─── Load API Keys & Initialize LLM ───────────────────────────────────────────
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ─── Prompt Templates & Chains ───────────────────────────────────────────────
question_template = PromptTemplate(
    input_variables=["experience", "tech_stack"],
    template="""\
You are a supportive and encouraging technical interviewer conducting an interview with a candidate who has {experience} years of experience in {tech_stack}.

Given that the candidate is early in their career:
1. Focus on fundamental concepts and principles
2. Ask about their self-learning journey and projects
3. Gauge their understanding of core concepts
4. Assess their problem-solving approach
5. Keep questions beginner-friendly but meaningful

Generate 5 technical questions based on their tech stack. The question should:
- Be clear and specific
- Focus on foundational understanding
- Include a brief context or scenario
- Encourage explanation of thought process
- Be professionally formatted and concise

Only return the questions, each on a new line.
"""
)
answer_analysis_template = PromptTemplate(
    input_variables=["answer", "question"],
    template="""\
You are a supportive technical interviewer providing feedback to an early-career candidate.
Question: {question}
Answer: {answer}

Provide constructive feedback that:
1. Acknowledges the candidate's effort
2. Points out correct concepts they've mentioned
3. Identifies areas for improvement
4. Offers a gentle suggestion for further learning
5. Maintains an encouraging tone

Keep the feedback concise and positive while being honest.
"""
)
scoring_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="""\
You are an objective technical interviewer.
Question:
{question}

Candidate’s Answer:
{answer}

On a scale from 0 to 100, how correct and complete is this answer?
Only output the integer score (0–100).
"""
)

question_chain = LLMChain(llm=llm, prompt=question_template)
answer_analysis_chain = LLMChain(llm=llm, prompt=answer_analysis_template)
scoring_chain = LLMChain(llm=llm, prompt=scoring_template)

# ─── Global CSS ───────────────────────────────────────────────────────────────
# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Global dark theme */
        html, body, [class*="stApp"] {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] .stNumberInput input {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
        }
        [data-testid="stSidebar"] .stSelectbox>div>div {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stButton>button,
        [data-testid="stSidebar"] .stFormSubmitButton>button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
        }
        [data-testid="stSidebar"] .stButton>button:hover,
        [data-testid="stSidebar"] .stFormSubmitButton>button:hover {
            background-color: #1a1a1a !important;
            border-color: #cccccc !important;
        }
        /* Form elements */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
            border: 1px solid #333333 !important;
        }
        /* Question & feedback boxes */
        .question-box, .analysis-box {
            background-color: #1a1a1a !important;
            padding: 15px; margin: 10px 0;
            border-radius: 8px; border: 1px solid #333333 !important;
            color: #ffffff !important;
        }
        /* Buttons */
        .stButton > button, .stFormSubmitButton > button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
            border-radius: 4px; padding: 10px 20px !important;
            text-transform: uppercase; font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background-color: #1a1a1a !important;
            border-color: #cccccc !important;
        }
        .stButton > button:focus, .stFormSubmitButton > button:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(255,255,255,0.2) !important;
        }
        /* Progress bar */
        .stProgress > div > div > div {
            background-color: #00c853 !important;
        }
        /* Labels & markdown */
        .stMarkdown, label, .stTextInput label, .stTextArea label {
            color: #ffffff !important;
        }
        /* Success & spinner */
        .stSuccess {
            background-color: #1a1a1a !important;
            color: #00c853 !important;
        }
        .stSpinner > div {
            border-color: #ffffff !important;
            border-right-color: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = ""
if "user" not in st.session_state:
    st.session_state.user = {}
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.candidate_info = {}
    st.session_state.questions_asked = []
    st.session_state.responses = []
    st.session_state.feedbacks = []
    st.session_state.scores = []
    st.session_state.question_count = 0
    st.session_state.saved = False

# ─── Utility Functions ───────────────────────────────────────────────────────
def reset_interview():
    st.session_state.step = 0
    st.session_state.candidate_info = {}
    st.session_state.questions_asked = []
    st.session_state.responses = []
    st.session_state.feedbacks = []
    st.session_state.scores = []
    st.session_state.question_count = 0
    st.session_state.saved = False

# ─── Sidebar Authentication ──────────────────────────────────────────────────
with st.sidebar:
    st.title("Authentication")
    if st.session_state.logged_in and st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = ""
        st.session_state.user = {}
        st.rerun()

    auth_option = st.selectbox("Choose Option", [
        "Candidate Login", "Candidate Sign Up", "Admin Login", "Admin Sign Up"
    ])

    if not st.session_state.logged_in:

        # ── Candidate Login ─────────────────────────────────────────────
        if auth_option == "Candidate Login":
            st.subheader("Candidate Login")
            with st.form("cand_login"):
                email = st.text_input("Email", key="cand_login_email")
                pwd = st.text_input("Password", type="password", key="cand_login_pwd")
                if st.form_submit_button("Login"):
                    # try/except ONLY around the firebase call
                    try:
                        u = auth.sign_in_with_email_and_password(email, pwd)
                        login_success = True
                    except Exception:
                        st.error("Login failed. Check credentials.")
                        login_success = False

                    if login_success:
                        st.success("Logged in successfully as Candidate!")
                        st.session_state.logged_in = True
                        st.session_state.role = "candidate"
                        st.session_state.user = u
                        st.rerun()

        # ── Candidate Sign Up ───────────────────────────────────────────
        elif auth_option == "Candidate Sign Up":
            st.subheader("Candidate Sign Up")
            with st.form("cand_signup"):
                name = st.text_input("Full Name", key="cand_signup_name")
                email = st.text_input("Email", key="cand_signup_email")
                pwd = st.text_input("Password", type="password", key="cand_signup_pwd")
                if st.form_submit_button("Sign Up"):
                    try:
                        u = auth.create_user_with_email_and_password(email, pwd)
                        db.child("users").child(u["localId"])\
                          .set({"name": name, "email": email, "role": "candidate"})
                        st.success("Sign up successful! Please log in.")
                    except Exception:
                        st.error("Sign up failed.")

        # ── Admin Login ─────────────────────────────────────────────────
        elif auth_option == "Admin Login":
            st.subheader("Admin Login")
            with st.form("admin_login"):
                email = st.text_input("Admin Email", key="admin_login_email")
                pwd = st.text_input("Password", type="password", key="admin_login_pwd")
                if st.form_submit_button("Login"):
                    # only wrap the sign-in call
                    try:
                        u = auth.sign_in_with_email_and_password(email, pwd)
                        login_success = True
                    except Exception:
                        st.error("Login failed. Check credentials.")
                        login_success = False

                    if login_success:
                        # check role in the database
                        data = db.child("users").child(u["localId"]).get().val()
                        if data and data.get("role") == "admin":
                            st.success("Logged in as Admin!")
                            st.session_state.logged_in = True
                            st.session_state.role = "admin"
                            st.session_state.user = u
                            st.rerun()
                        else:
                            st.error("Not authorized as admin.")

        # ── Admin Sign Up ───────────────────────────────────────────────
        else:  # Admin Sign Up
            st.subheader("Admin Sign Up")
            with st.form("admin_signup"):
                name = st.text_input("Full Name", key="admin_signup_name")
                email = st.text_input("Admin Email", key="admin_signup_email")
                pwd = st.text_input("Password", type="password", key="admin_signup_pwd")
                if st.form_submit_button("Sign Up"):
                    try:
                        u = auth.create_user_with_email_and_password(email, pwd)
                        db.child("users").child(u["localId"])\
                          .set({"name": name, "email": email, "role": "admin"})
                        st.success("Admin sign up successful! Please log in.")
                    except Exception:
                        st.error("Sign up failed.")

# ─── Main App ────────────────────────────────────────────────────────────────
if st.session_state.logged_in:
    if st.session_state.role == "candidate":
        # ── Candidate Interview Flow (unchanged) ────────────────────────
        if st.session_state.step == 0:
            st.title("🧠 TalentScout: AI Hiring Assistant")
            st.subheader("👋 Welcome!")
            st.markdown("I’ll guide you through a short technical interview tailored to your experience.")
            if st.button("Start"):
                st.session_state.step = 1

        elif st.session_state.step == 1:
            st.subheader("📋 Candidate Details")
            with st.form("info_form"):
                ci = st.session_state.candidate_info
                name       = st.text_input("Full Name",      ci.get("name", ""))
                email      = st.text_input("Email Address", ci.get("email", ""))
                phone      = st.text_input("Phone Number",  ci.get("phone", ""))
                experience = st.text_input("Years of Experience", ci.get("experience", ""))
                position   = st.text_input("Desired Position",    ci.get("position", ""))
                location   = st.text_input("Current Location",    ci.get("location", ""))
                tech_stack = st.text_area("Tech Stack",            ci.get("tech_stack", ""),
                                          placeholder="e.g., Python, Django, React")
                if st.form_submit_button("Start Interview"):
                    st.session_state.candidate_info = {
                        "name": name, "email": email, "phone": phone,
                        "experience": experience, "position": position,
                        "location": location, "tech_stack": tech_stack
                    }
                    with st.spinner("Generating questions..."):
                        qs = question_chain.predict(experience=experience, tech_stack=tech_stack)
                    st.session_state.questions_asked = [
                        q.strip("• ").strip() for q in qs.splitlines() if q.strip()
                    ][:5]
                    st.session_state.step = 2

        elif st.session_state.step == 2:
            idx = st.session_state.question_count
            if idx < len(st.session_state.questions_asked):
                qnum = idx + 1
                curr_q = st.session_state.questions_asked[idx]
                st.progress(idx / 5)
                st.markdown(f"### Question {qnum} of 5")
                st.markdown(f"<div class='question-box'>{curr_q}</div>", unsafe_allow_html=True)

                with st.form(f"answer_form_{qnum}"):
                    ans = st.text_area("Your Answer", key=f"answer_{qnum}")
                    if st.form_submit_button("Submit Answer"):
                        st.session_state.responses.append(ans)
                        fb = answer_analysis_chain.predict(answer=ans, question=curr_q)
                        st.session_state.feedbacks.append(fb)
                        raw = scoring_chain.predict(answer=ans, question=curr_q)
                        try:
                            sc = int("".join(filter(str.isdigit, raw)))
                        except:
                            sc = 0
                        st.session_state.scores.append(sc)

                        st.session_state.question_count += 1
                        if st.session_state.question_count >= 5:
                            st.session_state.step = 3
                        st.rerun()
            else:
                st.session_state.step = 3

        else:  # step == 3
            st.success("🎉 Interview Complete!")
            st.markdown("Here’s your feedback and scores:")
            if not st.session_state.saved:
                db.child("interviews").push({
                    "candidate_info": st.session_state.candidate_info,
                    "questions":       st.session_state.questions_asked,
                    "responses":       st.session_state.responses,
                    "feedbacks":       st.session_state.feedbacks,
                    "scores":          st.session_state.scores,
                    "timestamp":       time.time()
                })
                st.session_state.saved = True

            for i, (q, a, fb, sc) in enumerate(zip(
                st.session_state.questions_asked,
                st.session_state.responses,
                st.session_state.feedbacks,
                st.session_state.scores
            )):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**Your Answer:** {a}")
                st.markdown(f"<div class='analysis-box'>{fb}</div>", unsafe_allow_html=True)
                st.markdown(f"**Score:** {sc}/100")
                st.markdown("---")

            if st.button("🎯 Start New Interview"):
                reset_interview()
                st.rerun()

    else:
        # ── Admin Dashboard ─────────────────────────────────────────────
        st.title("Admin Dashboard")
        st.subheader("User Interview Performance Overview")
        st.markdown("Below is the list of candidate interviews and their responses:")

        interviews = db.child("interviews").get().val() or {}
        if not interviews:
            st.info("No interviews recorded yet.")
        else:
            records = []
            for _id, rec in interviews.items():
                ci     = rec.get("candidate_info", {})
                scores = rec.get("scores", [])
                avg_sc = round(sum(scores)/len(scores)) if scores else 0
                records.append({
                    "Name": ci.get("name", "Unknown"),
                    "Email": ci.get("email", ""),
                    "Position": ci.get("position", ""),
                    "Experience": ci.get("experience", ""),
                    "AvgScore": avg_sc
                })
                st.markdown(f"### {ci.get('name','Unknown')} — Avg. Score: **{avg_sc}**")
                for i, (q, a, sc) in enumerate(zip(
                    rec.get("questions", []),
                    rec.get("responses", []),
                    scores
                )):
                    st.markdown(f"**Q{i+1}: {q}**")
                    st.markdown(f"**Answer:** {a}")
                    st.markdown(f"**Score:** {sc}/100")
                st.markdown("---")

            df = pd.DataFrame(records)
            st.subheader("Candidate Interview Data Table")
            st.dataframe(df)
            chart = px.bar(
                df, x="Name", y="AvgScore",
                hover_data=["Email","Position","Experience"],
                labels={"AvgScore":"Avg. Score","Name":"Candidate"},
                title="Candidate Performance Overview"
            )
            st.subheader("Candidate Performance Overview")
            st.plotly_chart(chart, use_container_width=True)
