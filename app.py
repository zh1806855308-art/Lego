import streamlit as st
import pandas as pd
import os
import ast
from PIL import Image
from openai import OpenAI
import base64
import hashlib
import io



# =========================
# Page & global config
# =========================
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

CSV_FILE = "lego_subtasks.csv"
SURVEY_FILE = "survey_responses.csv"

# 管理员密码（可以用环境变量覆盖）
ADMIN_PASSWORD = os.getenv("INSTRUCTOR_PASSWORD", "lego-admin-2024")

# =========================
# OpenAI client
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set your OPENAI_API_KEY environment variable!")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# Load subtasks CSV
# =========================
if not os.path.exists(CSV_FILE):
    st.error(f"CSV file '{CSV_FILE}' not found in the app directory.")
    st.stop()

df = pd.read_csv(CSV_FILE)
df["Subassembly"] = df["Subassembly"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df["Final Assembly"] = df["Final Assembly"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])


# =========================
# Utility functions
# =========================
@st.cache_data
def get_encoded_image(image_path: str):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


def show_image(image_path: str, caption: str = ""):
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")


def show_gpt_response(answer: str):
    st.markdown(
        f"""
        <div style='text-align: left; padding: 10px; background-color: #e8f0fe; border-left: 5px solid #4285f4; border-radius: 8px; margin-bottom: 1em;'>
            🧠 <strong>AGEMT says:</strong><br>{answer}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_question_hash(question: str, context: dict) -> str:
    hash_input = question + str(context)
    return hashlib.md5(hash_input.encode()).hexdigest()


def format_task_sequence(df_: pd.DataFrame) -> str:
    lines = []
    for i, row in df_.iterrows():
        line = (
            f"{i + 1}. Subtask: {row['Subtask Name']} | Team: {row['Student Team']} | "
            f"Bag: {row['Bag']} | Subassembly: {row['Subassembly']} | Final Assembly: {row['Final Assembly']}"
        )
        lines.append(line)
    return "\n".join(lines)


def call_chatgpt(user_question: str, context: dict) -> str:
    image_messages = []
    for page in context.get("subassembly", []) + context.get("final_assembly", []):
        img_path = f"manuals/page_{page}.png"
        image_content = get_encoded_image(img_path)
        if image_content:
            image_messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_content}",
                        "detail": "high",
                    },
                }
            )

    team_number = context.get("team_number", "Unknown")

    system_prompt = (
        "You are a helpful assistant helping a student with a physical LEGO assembly task. "
        "The student belongs to a team and is working on a specific subtask. "
        "You also have access to the entire task sequence across all teams to reason about handovers, "
        "task order, and dependencies."
    )

    user_prompt = f"""
Current subtask: {context['subtask_name']}
Team Number: {team_number}
Bag: {context['bag']}
Subassembly Pages: {context['subassembly']}
Final Assembly Pages: {context['final_assembly']}
Previous Step: {context.get('previous_step', 'None')}

Student's question:
\"{user_question}\"

Here is the full task sequence across all teams:
{context['task_sequence_text']}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}] + image_messages},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# =========================
# User info gate
# =========================
if (
    "group_name" not in st.session_state
    or "student_name" not in st.session_state
    or "team_number" not in st.session_state
):
    st.header("Welcome to the Assembly Task")

    group_name_input = st.selectbox("Which group are you in?", ["Red", "Yellow", "Blue", "Green"])
    team_number_input = st.selectbox("Which team number are you in?", [1, 2, 3, 4, 5])
    student_name_input = st.text_input("Enter your name:")

    if st.button("Submit"):
        if student_name_input.strip():
            st.session_state.group_name = group_name_input
            st.session_state.team_number = team_number_input
            st.session_state.student_name = student_name_input.strip()
            st.success("Information saved. You can proceed.")
            st.rerun()
        else:
            st.warning("Please enter your name before submitting.")
    st.stop()

# 初始化 AI 对话历史
if "chat_history" not in st.session_state:
    # 每条记录：{"step": int, "subtask": str, "question": str, "answer": str}
    st.session_state.chat_history = []


# =========================
# Sidebar: Progress + AGEMT
# =========================
with st.sidebar:
    st.header("Progress Tracker")
    st.markdown(f"**Student:** {st.session_state.student_name}")
    st.markdown(f"**Group Name:** {st.session_state.group_name}")
    st.markdown(f"**Team Number:** {st.session_state.team_number}")

    team_tasks_preview = df[df["Student Team"] == st.session_state.team_number]
    if "task_idx" in st.session_state and not team_tasks_preview.empty:
        current_task_preview = team_tasks_preview.iloc[st.session_state.task_idx]
        st.markdown(
            f"""
        **Subtask:** {current_task_preview['Subtask Name']}  
        **Bag:** {current_task_preview['Bag']}  
        **Collect Parts:** {'✅' if st.session_state.get('collected_parts_confirmed', False) else '❌'}
        """
        )
        if current_task_preview["Subassembly"]:
            st.markdown("**Subassembly:**")
            for page in current_task_preview["Subassembly"]:
                done = page in st.session_state.get("subassembly_confirmed_pages", set())
                st.markdown(f"- Page {page}: {'✅' if done else '❌'}")
        if current_task_preview["Final Assembly"]:
            st.markdown("**Final Assembly:**")
            for page in current_task_preview["Final Assembly"]:
                done = page in st.session_state.get("finalassembly_confirmed_pages", set())
                st.markdown(f"- Page {page}: {'✅' if done else '❌'}")
        if st.session_state.get("step", 0) == 4:
            st.markdown("**Handover:** ✅")

    # ---- AGEMT expander ----
    with st.expander("💬 AGEMT", expanded=False):
        st.markdown("Ask a question about your current step.")
        step_keys = ["q_step0", "q_step1", "q_step2", "q_step3", "q_step4"]
        current_step = st.session_state.get("step", 0)

        if current_step in range(len(step_keys)):
            key = step_keys[current_step]
            user_question = st.text_input("Your question to AGEMT:", key=key)
            if user_question and user_question.lower() != "n":
                task_idx_q = st.session_state.get("task_idx", 0)
                team_tasks_q = df[df["Student Team"] == st.session_state.team_number]
                if team_tasks_q.empty:
                    st.warning("No subtasks for current team.")
                else:
                    current_task_q = team_tasks_q.iloc[task_idx_q]
                    idx_q = df.index.get_loc(current_task_q.name)
                    prev_row_q = df.iloc[idx_q - 1] if idx_q > 0 else None

                    context_q = {
                        "subtask_name": current_task_q["Subtask Name"],
                        "subassembly": current_task_q["Subassembly"],
                        "final_assembly": current_task_q["Final Assembly"],
                        "bag": current_task_q["Bag"],
                        "previous_step": prev_row_q["Subtask Name"] if prev_row_q is not None else None,
                        "team_number": st.session_state.team_number,
                        "task_sequence_text": format_task_sequence(df),
                    }
                    q_hash = get_question_hash(user_question, context_q)
                    if q_hash not in st.session_state:
                        answer = call_chatgpt(user_question, context_q)
                        st.session_state[q_hash] = answer

                    # 记录 AI 对话
                    st.session_state.chat_history.append(
                        {
                            "step": current_step,
                            "subtask": current_task_q["Subtask Name"],
                            "question": user_question,
                            "answer": st.session_state[q_hash],
                        }
                    )

                    show_gpt_response(st.session_state[q_hash])
        else:
            st.info("No active step to ask about.")


# =========================
# Main layout (center)
# =========================
left, center, _ = st.columns([1, 2, 1])

with center:
    team_tasks = df[df["Student Team"] == st.session_state.team_number]
    if team_tasks.empty:
        st.error(f"No subtasks found for Team {st.session_state.team_number}.")
        st.stop()

    # Initialize step-related state
    if "task_idx" not in st.session_state:
        st.session_state.task_idx = 0
        st.session_state.step = 0
        st.session_state.subassembly_confirmed_pages = set()
        st.session_state.finalassembly_confirmed_pages = set()
        st.session_state.previous_step_confirmed = False
        st.session_state.collected_parts_confirmed = False
        st.session_state.survey_submitted = False

    task_idx = st.session_state.task_idx
    step = st.session_state.step
    current_task = team_tasks.iloc[task_idx]
    context = {
        "subtask_name": current_task["Subtask Name"],
        "subassembly": current_task["Subassembly"],
        "final_assembly": current_task["Final Assembly"],
        "bag": current_task["Bag"],
        "previous_step": None,
    }

    # Progress bar
    total_steps = 5
    current_progress = min(step / (total_steps - 1), 1.0)
    subtask_id = current_task.get("Subtask ID", current_task["Subtask Name"])
    st.markdown(f"### 🧱 Subtask: {subtask_id}")
    st.progress(current_progress, text=f"Step {step + 1} of {total_steps}")

    # ---------- Step 1: Collect parts ----------
    if step == 0:
        st.subheader("Step 1: Collect required parts")
        part_img = f"combined_subtasks/{context['subtask_name']}.png"
        show_image(part_img, "Parts Required")
        if not st.session_state.collected_parts_confirmed:
            if st.button("I have collected all parts"):
                st.session_state.collected_parts_confirmed = True
                st.session_state.step = 1
                st.rerun()

    # ---------- Step 2: Subassembly ----------
    elif step == 1:
        if context["subassembly"]:
            st.subheader("Step 2: Perform subassembly")
            for page in context["subassembly"]:
                show_image(f"manuals/page_{page}.png", f"Subassembly - Page {page}")
                if page not in st.session_state.subassembly_confirmed_pages:
                    if st.button(f"✅ Confirm completed Subassembly - Page {page}"):
                        st.session_state.subassembly_confirmed_pages.add(page)
                        st.rerun()
            if len(st.session_state.subassembly_confirmed_pages) == len(context["subassembly"]):
                st.success("All subassembly pages completed!")
                st.session_state.step = 2
                st.rerun()
        else:
            st.session_state.step = 2
            st.rerun()

    # ---------- Step 3: Receive from previous team ----------
    elif step == 2:
        idx = df.index.get_loc(current_task.name)
        if idx > 0:
            prev_row = df.iloc[idx - 1]
            context["previous_step"] = prev_row["Subtask Name"]
            giver_team = prev_row["Student Team"]
            receiver_team = st.session_state.team_number
            st.subheader(f"Step 3: Receive from Team {giver_team}")
            show_image(f"handling-image/receive-t{giver_team}-t{receiver_team}.png")
            if not st.session_state.previous_step_confirmed:
                if st.button("I have received the product from the previous team"):
                    st.session_state.previous_step_confirmed = True
                    st.session_state.step = 3
                    st.rerun()
        else:
            st.session_state.previous_step_confirmed = True
            st.session_state.step = 3
            st.rerun()

    # ---------- Step 4: Final assembly ----------
    elif step == 3:
        st.subheader("Step 4: Perform the final assembly")
        subassembly_pages = set(context["subassembly"]) if context["subassembly"] else set()
        final_assembly_pages = context["final_assembly"]

        for page in final_assembly_pages:
            manual_path = f"manuals/page_{page}.png"
            show_image(manual_path, f"Final Assembly - Page {page}")

            if page not in st.session_state.finalassembly_confirmed_pages:
                if page in subassembly_pages:
                    if st.button(f"✅ The subassembled part for Page {page} is ready"):
                        st.session_state.finalassembly_confirmed_pages.add(page)
                        st.rerun()
                else:
                    if st.button(f"✅ Confirm completed Final Assembly - Page {page}"):
                        st.session_state.finalassembly_confirmed_pages.add(page)
                        st.rerun()

        if len(st.session_state.finalassembly_confirmed_pages) == len(final_assembly_pages):
            st.success("All final assembly pages completed!")
            st.session_state.step = 4
            st.rerun()

    # ---------- Step 5: Handover + Final Survey ----------
    elif step == 4:
        idx = df.index.get_loc(current_task.name)
        if idx + 1 < len(df):
            next_row = df.iloc[idx + 1]
            st.subheader(f"Step 5: Handover to Team {next_row['Student Team']}")
            show_image(
                f"handling-image/give-t{st.session_state.team_number}-t{next_row['Student Team']}.png"
            )
        else:
            st.subheader("🎉 You are the final team — no further handover needed.")

        st.success("✅ Subtask complete. Great work!")

        has_next_subtask = st.session_state.task_idx + 1 < len(team_tasks)

        if has_next_subtask:
            if st.button("Next Subtask"):
                st.session_state.task_idx += 1
                st.session_state.step = 0
                st.session_state.subassembly_confirmed_pages = set()
                st.session_state.finalassembly_confirmed_pages = set()
                st.session_state.previous_step_confirmed = False
                st.session_state.collected_parts_confirmed = False
                # 不清空 chat_history，让 AI 问答可以贯穿整个活动
                st.rerun()
        else:
            st.info("You have completed all your subtasks.")

            if "survey_submitted" not in st.session_state:
                st.session_state.survey_submitted = False

            st.markdown("---")
            st.markdown("### 📝 Final Survey")

            if not st.session_state.survey_submitted:
                with st.form("final_survey"):
                    st.markdown("Please complete this short survey.")

                    group_color = st.selectbox(
                        "Which group are you in?",
                        ["Red", "Yellow", "Blue", "Green"],
                        index=["Red", "Yellow", "Blue", "Green"].index(
                            st.session_state.group_name
                        )
                        if st.session_state.get("group_name") in ["Red", "Yellow", "Blue", "Green"]
                        else 0,
                    )

                    team_num = st.selectbox(
                        "Which team number are you in?",
                        [1, 2, 3, 4, 5],
                        index=[1, 2, 3, 4, 5].index(st.session_state.team_number)
                        if st.session_state.get("team_number") in [1, 2, 3, 4, 5]
                        else 0,
                    )

                    student_name = st.text_input(
                        "Enter your name:",
                        value=st.session_state.get("student_name", ""),
                    )

                    # —— 你图片里那套更细的问卷 —— 
                    task_completion_driver = st.text_area(
                        "What mainly led to the completion of your task? (可以用中文回答)",
                        placeholder="例如：团队合作、AI 建议、仔细阅读说明书 等……",
                    )

                    ai_feedback_accuracy = st.slider(
                        "How accurate was the AI feedback? (1-5)", 1, 5, 4
                    )
                    ai_feedback_helpfulness = st.slider(
                        "How helpful was the AI feedback? (1-5)", 1, 5, 4
                    )

                    score_improvement_ideas = st.text_area(
                        "What do you think would help you get a better score? (可以用中文回答)",
                        placeholder="例如：更多练习时间、更清晰的分工、更详细的说明……",
                    )

                    genai_improvement_ideas = st.text_area(
                        "How could GenAI further improve your efficiency? (信息 / 建议 / 指导)",
                        placeholder="例如：自动总结、实时检查错误、更具体的步骤指引……",
                    )

                    difficulty = st.slider("Task difficulty (1 easy - 5 hard)", 1, 5, 3)
                    enjoyment = st.slider(
                        "How enjoyable was the activity? (1-5)", 1, 5, 4
                    )
                    clarity = st.slider(
                        "How clear were the instructions? (1-5)", 1, 5, 4
                    )

                    free_feedback = st.text_area("Additional feedback:")

                    submitted = st.form_submit_button("Submit Survey")

                if submitted:
                    if not student_name.strip():
                        st.warning("Please enter your name before submitting.")
                    else:
                        # 把 chat_history 整理成字符串
                        history = st.session_state.get("chat_history", [])
                        if history:
                            lines = []
                            for i, h in enumerate(history, 1):
                                lines.append(f"[{i}] Step {h['step']} - {h['subtask']}")
                                lines.append(f"Q: {h['question']}")
                                lines.append(f"A: {h['answer']}")
                                lines.append("")
                            ai_conversation_log = "\n".join(lines)
                        else:
                            ai_conversation_log = ""

                        survey_row = {
                            "student_name": student_name.strip(),
                            "group_color": group_color,
                            "team_number": team_num,
                            "task_completion_driver": task_completion_driver.strip(),
                            "ai_feedback_accuracy_1_5": ai_feedback_accuracy,
                            "ai_feedback_helpfulness_1_5": ai_feedback_helpfulness,
                            "score_improvement_ideas": score_improvement_ideas.strip(),
                            "genai_improvement_ideas": genai_improvement_ideas.strip(),
                            "difficulty_1_5": difficulty,
                            "enjoyment_1_5": enjoyment,
                            "clarity_1_5": clarity,
                            "free_feedback": free_feedback.strip(),
                            "ai_conversation_log": ai_conversation_log,
                        }

                        if not os.path.exists(SURVEY_FILE):
                            pd.DataFrame([survey_row]).to_csv(SURVEY_FILE, index=False)
                        else:
                            pd.DataFrame([survey_row]).to_csv(
                                SURVEY_FILE, index=False, mode="a", header=False
                            )

                        st.session_state.survey_submitted = True
                        st.success("✅ Thank you! Your survey is saved.")
            else:
                st.success("✅ Thank you! Your survey is saved.")


# =========================
# Instructor Panel (password protected)
# =========================
st.markdown("---")
with st.expander("🔐 Instructor Panel (password required)", expanded=False):
    pwd = st.text_input("Instructor password:", type="password")
    if pwd:
        if pwd == ADMIN_PASSWORD:
            st.success("Instructor authenticated.")

            if os.path.exists(SURVEY_FILE):
                try:
                    df_survey = pd.read_csv(SURVEY_FILE)
                except Exception as e:
                    st.error(f"Error reading survey file: {e}")
                    df_survey = None

                if df_survey is not None:
                    st.success(f"Found {len(df_survey)} survey submissions.")
                    st.dataframe(df_survey, use_container_width=True)

                    # Excel download
                    output = io.BytesIO()
                    df_survey.to_excel(output, index=False, sheet_name="Survey Responses")
                    excel_data = output.getvalue()

                    st.download_button(
                        label="⬇️ Download survey_responses.xlsx",
                        data=excel_data,
                        file_name="survey_responses.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    # Delete all
                    if st.button("🗑 Delete ALL survey responses"):
                        try:
                            if os.path.exists(SURVEY_FILE):
                                os.remove(SURVEY_FILE)
                            st.success("All survey responses have been deleted.")
                        except Exception as e:
                            st.error(f"Unexpected error while deleting the survey file: {e}")
                        st.rerun()
                else:
                    st.info("No valid survey data to display.")
            else:
                st.info("No survey responses submitted yet.")
        else:
            st.error("Incorrect instructor password.")




