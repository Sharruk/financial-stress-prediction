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
    verified = [k for k, v in tasks.items() if v == "VERIFIED"]
    implemented = [k for k, v in tasks.items() if v == "IMPLEMENTED"]
    pending = [k for k, v in tasks.items() if v == "PENDING TRAINING MACHINE"]
    planned = [k for k, v in tasks.items() if v in ["PLANNED", "BLOCKED"]]
    
    with col1:
        st.success(f"**VERIFIED / IMPLEMENTED ({len(verified) + len(implemented)})**")
        for t in verified: st.write(f"- {t} *(Verified)*")
        for t in implemented: st.write(f"- {t} *(Implemented)*")
        
    with col2:
        st.info(f"**PENDING TRAINING ({len(pending)})**")
        for t in pending: st.write(f"- {t}")
        
    with col3:
        st.warning(f"**PLANNED / BLOCKED ({len(planned)})**")
        for t in planned: st.write(f"- {t}")
