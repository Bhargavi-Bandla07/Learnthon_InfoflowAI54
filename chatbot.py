import streamlit as st
import mysql.connector
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="InfoFlow AI", layout="centered")
st.title("🤖 InfoFlow AI Chatbot")

# -----------------------------
# 🔎 Load Embeddings + Vector DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# -----------------------------
# 🧹 Prompt Cleaner
# -----------------------------
def clean_prompt(text):
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # remove extra spaces
    return text.strip()

# -----------------------------
# 🗄️ DB-Driven Name Detection
# -----------------------------
def extract_employee_from_prompt(prompt):

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="32939",
        database="company_db"
    )

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM employees")
    employees = cursor.fetchall()
    conn.close()

    prompt_clean = clean_prompt(prompt).lower()

    for (name,) in employees:
        if name.lower() in prompt_clean:
            return name

    return None

# -----------------------------
# 🗄️ SQL Functions
# -----------------------------
def get_employee_details(prompt):

    name = extract_employee_from_prompt(prompt)

    if not name:
        return "Employee not found"

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="32939",
        database="company_db"
    )

    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM employees WHERE name=%s", (name,))
    emp = cursor.fetchone()

    cursor.execute(
        "SELECT project_name FROM projects WHERE employee_name=%s",
        (name,)
    )
    projects_data = cursor.fetchall()

    conn.close()

    projects = ", ".join([p["project_name"] for p in projects_data]) if projects_data else "No project assigned"
    designation = "Team Lead" if emp["is_team_lead"] else "Team Member"

    return f"""
👤 Name: {emp['name']}
📌 Role: {emp['role']}
📞 Contact: {emp['contact']}
📧 Email: {emp['email']}
👥 Team: {emp['team']}
🏷️ Designation: {designation}
🚀 Project(s): {projects}
"""

def get_all_employees():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="32939",
        database="company_db"
    )

    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    conn.close()

    if not employees:
        return "No employees found"

    output = "👥 **Employee Directory**\n\n"

    for emp in employees:
        designation = "Team Lead" if emp["is_team_lead"] else "Team Member"

        output += f"""
👤 **{emp['name']}**
📌 Role: {emp['role']}
📞 Contact: {emp['contact']}
📧 Email: {emp['email']}
👥 Team: {emp['team']}
🏷️ Designation: {designation}

"""

    return output

# -----------------------------
# 💬 Chat Memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 Ask me anything!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# 💬 User Input
# -----------------------------
prompt = st.chat_input("Ask your question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        prompt_lower = prompt.lower()
        employee_name = extract_employee_from_prompt(prompt)

        # -----------------------------
        # 🎯 ROUTING LOGIC
        # -----------------------------
        if "leave" in prompt_lower:
            source = "TXT"
            search_query = "leave policy " + prompt

        elif "onboard" in prompt_lower or "training" in prompt_lower:
            source = "TXT"
            search_query = "onboarding policy " + prompt

        elif "all employees" in prompt_lower or "show employees" in prompt_lower:
            source = "SQL_ALL"

        elif employee_name:
            source = "SQL"

        elif "employee" in prompt_lower or "details" in prompt_lower:
            source = "SQL"

        else:
            source = "TXT"
            search_query = prompt

        # -----------------------------
        # 📄 TXT Retrieval (RAG)
        # -----------------------------
        if source == "TXT":
            relevant_docs = retriever.get_relevant_documents(search_query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            full_reply = context if context.strip() else "Not available in knowledge base"
            source_label = "📄 Policy Documents"

        # -----------------------------
        # 🗄️ SQL Retrieval
        # -----------------------------
        elif source == "SQL":
            full_reply = get_employee_details(prompt)
            source_label = "🗄️ Employee Database"

        elif source == "SQL_ALL":
            full_reply = get_all_employees()
            source_label = "🗄️ Employee Database"

        # -----------------------------
        # 💬 Display Response
        # -----------------------------
        message_placeholder.write(full_reply)
        st.caption(f"Source: {source_label}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )