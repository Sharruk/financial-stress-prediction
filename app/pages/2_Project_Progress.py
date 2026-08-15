import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api.dependencies import get_project_status

st.title("Project Progress")

status = get_project_status()

if "error" in status:
    st.error(status["error"])
else:
    st.subheader(status.get("project_name", "Zindi Project"))
    
    tasks = status.get("status", {})
    
    col1, col2, col3 = st.columns(3)
    completed = [k for k, v in tasks.items() if v == "COMPLETED"]
    in_progress = [k for k, v in tasks.items() if v == "IN PROGRESS"]
    planned = [k for k, v in tasks.items() if v == "PLANNED"]
    blocked = [k for k, v in tasks.items() if v == "BLOCKED"]
    
    with col1:
        st.success(f"**COMPLETED ({len(completed)})**")
        for t in completed: st.write(f"- {t}")
        
    with col2:
        st.info(f"**IN PROGRESS ({len(in_progress)})**")
        for t in in_progress: st.write(f"- {t}")
        
    with col3:
        st.warning(f"**PLANNED / BLOCKED ({len(planned) + len(blocked)})**")
        for t in planned: st.write(f"- {t} (Planned)")
        for t in blocked: st.write(f"- {t} (Blocked)")
