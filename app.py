import os
import io
import re
import streamlit as st
from typing import TypedDict, Annotated, List
import operator

# Core Agent Framework
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_core.tools import tool

# PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# 1. API & AGENT TOOL BINDING
os.environ["GROQ_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

# Initialize the Model
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@tool
def research_tool(topic: str):
    """RESEARCHER: Gathers verified facts and URLs for source attribution."""
    search = tavily.search(query=topic, search_depth="advanced")
    return {
        "notes": "\n".join([r['content'] for r in search['results']]), 
        "urls": [r['url'] for r in search['results']]
    }

@tool
def writer_tool(topic: str, content_type: str, notes: str, tone: str, feedback: str = ""):
    """WRITER: Generates professional content. Addresses feedback if provided."""
    prompt = f"Topic: {topic}\nType: {content_type}\nTone: {tone}\nNotes: {notes}\nFeedback: {feedback}"
    return llm.invoke(f"Draft content. No intro/outro/meta-talk. Just the content: {prompt}").content

@tool
def reviewer_tool(draft: str):
    """REVIEWER: Audits the draft for quality and provides a score."""
    return llm.invoke(f"Critique this draft. Score 1-10 and list 3 strengths: {draft}").content

@tool
def cleaner_tool(content: str):
    """CLEANER: Strips AI meta-commentary like 'I have rewritten the content'."""
    patterns = [r"(?i)I made the following.*", r"(?i)Refined for clarity.*", r"(?i)Note:.*", r"(?i)Simplified sentence.*"]
    clean = content
    for p in patterns:
        clean = re.split(p, clean, flags=re.DOTALL)[0]
    return clean.strip()

# This links the tools to the LLM's brain
bound_tools = [research_tool, writer_tool, reviewer_tool, cleaner_tool]
llm_with_tools = llm.bind_tools(bound_tools)

# 2. THE AGENTIC STATE MACHINE (LangGraph Logic)
class AgentState(TypedDict):
    topic: str
    content_type: str
    tone: str
    notes: str
    urls: List[str]
    draft: str
    review: str
    feedback: str

# Define nodes that use the bound tools
def research_node(state: AgentState):
    res = research_tool.invoke({"topic": state["topic"]})
    return {"notes": res["notes"], "urls": res["urls"]}

def writer_node(state: AgentState):
    draft = writer_tool.invoke({
        "topic": state["topic"], "content_type": state["content_type"],
        "notes": state["notes"], "tone": state["tone"], "feedback": state["feedback"]
    })
    clean_draft = cleaner_tool.invoke({"content": draft})
    return {"draft": clean_draft}

def reviewer_node(state: AgentState):
    audit = reviewer_tool.invoke({"draft": state["draft"]})
    return {"review": audit}

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)
app_graph = workflow.compile()

# 3. PDF EXPORT UTILITY

def generate_pdf(text, title, urls):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    story = [Paragraph(f"<b>{title.upper()}</b>", styles['Title']), Spacer(1, 12)]
    
    for para in text.split('\n'):
        if para.strip():
            story.append(Paragraph(para, styles['BodyText']))
            story.append(Spacer(1, 6))
            
    if urls:
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>VERIFIED SOURCES:</b>", styles['Heading3']))
        for url in urls:
            story.append(Paragraph(f'<font color="blue">{url}</font>', styles['BodyText']))
            
    doc.build(story)
    return buffer.getvalue()

# 4. STREAMLIT UI: ContentAlchemist
st.set_page_config(page_title="ContentAlchemist", layout="wide", page_icon="🧪")
st.title("🧪 ContentAlchemist")
st.caption("Bound-Tool Agentic Intelligence Suite")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
    st.session_state.ui_step = "idle"

with st.sidebar:
    st.header("⚗️ Alchemy Lab")
    topic_in = st.text_input("Topic", placeholder="e.g., AI for Rural Development")
    ctype_in = st.selectbox("Format", ["Blog Post", "Newsletter", "Product Write-up"])
    tone_in = st.selectbox("Tone", ["Formal Corporate", "Casual/Engaging", "Technical/Academic"])
    
    if st.button("🚀 Begin Transmutation"):
        initial_state = {
            "topic": topic_in, "content_type": ctype_in, "tone": tone_in,
            "notes": "", "urls": [], "draft": "", "review": "", "feedback": ""
        }
        
        # Progress Tracking
        bar = st.progress(0)
        status_msg = st.empty()
        
        with st.status("Agents activating tools...", expanded=True) as s:
            status_msg.text("📡 Calling Research Tool...")
            bar.progress(25)
            r_state = research_node(initial_state)
            
            status_msg.text("🖋️ Calling Writer Tool...")
            bar.progress(50)
            w_state = writer_node({**initial_state, **r_state})
            
            status_msg.text("🧐 Calling Reviewer Tool...")
            bar.progress(75)
            rev_state = reviewer_node({**initial_state, **r_state, **w_state})
            
            bar.progress(100)
            status_msg.text("✅ Content Synthesized.")
            
            st.session_state.agent_state = {**initial_state, **r_state, **w_state, **rev_state}
            st.session_state.ui_step = "gate"
            s.update(label="Process Complete", state="complete")

# HUMAN IN THE LOOP SECTION
if st.session_state.ui_step == "gate":
    st.divider()
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("✨ Purified Output")
        st.markdown(st.session_state.agent_state["draft"])
        
    with c2:
        st.subheader("🧐 Review Audit")
        st.info(st.session_state.agent_state["review"])
        
        st.divider()
        st.subheader("🚦 Decision Gate")
        satisfied = st.radio("Are changes required?", ["No, Publish", "Yes, Rewrite"], index=0)
        
        if satisfied == "Yes, Rewrite":
            fb = st.text_area("What should the agent change?")
            if st.button("🔄 Trigger Rewrite Tool"):
                with st.status("Agent re-calculating..."):
                    st.session_state.agent_state["feedback"] = fb
                    # Invoke tools again via nodes
                    upd_w = writer_node(st.session_state.agent_state)
                    upd_r = reviewer_node({**st.session_state.agent_state, **upd_w})
                    st.session_state.agent_state.update({**upd_w, **upd_r})
                    st.rerun()
        else:
            if st.button("✨ Finalize & Celebrate"):
                st.balloons()
                st.session_state.ui_step = "finished"
                st.rerun()

# DOWNLOAD SECTION
if st.session_state.ui_step == "finished":
    st.success("✅ Content Ready for Export!")
    pdf = generate_pdf(st.session_state.agent_state["draft"], topic_in, st.session_state.agent_state["urls"])
    
    if st.download_button("📥 Download PDF", data=pdf, file_name="ContentAlchemist_Project.pdf"):
        st.balloons()
        
    if st.button("🆕 New Project"):
        st.session_state.ui_step = "idle"
        st.rerun()