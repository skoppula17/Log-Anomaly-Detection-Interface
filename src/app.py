import streamlit as st
import pandas as pd
import json
import torch
from pathlib import Path
import os
import io

from src.models.deeplog_lstm import DeepLogLSTM
from src.eval import session_is_anomalous, make_next_event_windows, _predict_next_topk
from src.data.hdfs import encode_sequences

st.set_page_config(
    page_title="SOC Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SOC Log Session Anomaly Detector")
st.markdown("""
Welcome to the Anomaly Detection Dashboard. 
Upload a CSV of preprocessed log sessions (`Event_traces.csv`) to identify suspicious event sequences.
""")

# Default paths for Checkpoint 1
DEFAULT_CKPT = "runs/cp1/model.pt"
DEFAULT_META = "runs/cp1/meta.json"

@st.cache_resource
def load_model(ckpt_path: str, meta_path: str):
    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads(Path(meta_path).read_text())
    vocab = meta["vocab"]
    window = int(meta["window"])
    emb_dim = int(meta["emb_dim"])
    hidden_dim = int(meta["hidden_dim"])

    model = DeepLogLSTM(vocab_size=len(vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    return model, meta, device

model, meta, device = load_model(DEFAULT_CKPT, DEFAULT_META)

if model is None:
    st.warning(f"No trained model found at `{DEFAULT_CKPT}`. Please run training first.")
else:
    st.sidebar.success("✅ Model loaded successfully")
    window_val = int(meta["window"])
    
    st.sidebar.header("Inference Settings")
    topk_val = st.sidebar.slider("Top-K Threshold", min_value=1, max_value=20, value=9, help="A higher top-k makes the model more permissive (fewer anomalies).")
    
    uploaded_file = st.file_uploader("Upload Event_traces.csv", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Parsing uploaded file..."):
            df = pd.read_csv(uploaded_file)
            
            # Identify columns
            bid_col = "BlockId"
            seq_col = "EventSequence"
            for c in df.columns:
                if c.lower() == "blockid": bid_col = c
                if c.lower() in ("eventsequence", "event_sequence", "events"): seq_col = c
                
            if bid_col not in df.columns or seq_col not in df.columns:
                st.error("Uploaded CSV must contain BlockId and EventSequence columns.")
                st.stop()
                
            block_ids = df[bid_col].astype(str).tolist()
            seqs = []
            for s in df[seq_col].astype(str).tolist():
                seqs.append([t.strip() for t in s.split() if t.strip()])
                
            enc = encode_sequences(seqs, meta["vocab"])
            
            # Evaluate anomalies comprehensively 
            # We will calculate "count of unusual transitions" to give the UI more flavor
            results = []
            progress_bar = st.progress(0)
            
            for i, seq in enumerate(enc):
                windows = make_next_event_windows(seq, window_val)
                unusual_count = 0
                if windows:
                    for xw, y_true in windows:
                        preds = _predict_next_topk(model, device, xw, topk=topk_val)
                        if y_true not in preds:
                            unusual_count += 1
                
                is_anom = unusual_count > 0
                results.append({
                    "Block ID": block_ids[i],
                    "Status": "🚨 Anomalous" if is_anom else "✅ Normal",
                    "Unusual Transitions": unusual_count,
                    "Total Transitions": len(windows)
                })
                
                if i % 100 == 0:
                    progress_bar.progress(min(i / len(enc), 1.0))
            
            progress_bar.empty()
            
            res_df = pd.DataFrame(results)
            anom_count = len(res_df[res_df["Status"] == "🚨 Anomalous"])
            total_count = len(res_df)
            
            hCol1, hCol2, hCol3 = st.columns(3)
            hCol1.metric("Total Sessions", total_count)
            hCol2.metric("Anomalous Sessions", anom_count)
            hCol3.metric("Normal Sessions", total_count - anom_count)
            
            st.subheader("Session Analysis Details")
            # Show anomalous sessions first
            st.dataframe(res_df.sort_values(by="Unusual Transitions", ascending=False), use_container_width=True)
