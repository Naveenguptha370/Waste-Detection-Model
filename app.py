import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Waste Detection",
    page_icon="♻️",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("""
<div style="
background: linear-gradient(90deg,#00C853,#00E676);
padding:25px;
border-radius:15px;
text-align:center;
color:white;
">
<h1>♻️ Smart Waste Detection System</h1>
<h4>AI Powered Waste Classification Dashboard</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- TOP CARDS ----------------
c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("🟦 Dry Waste","♻️")
c2.metric("🟩 Wet Waste","🌱")
c3.metric("🟥 Sanitary","🧴")
c4.metric("⚡ E-Waste","💻")
c5.metric("🏥 Medical","🩺")

st.divider()

# ---------------- CLASSES ----------------
classes = [
    "dry",
    "ewaste",
    "medical",
    "sanitary",
    "wet"
]

# ---------------- LOAD MODEL ----------------
model = efficientnet_v2_s(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "waste_model.pth",
        map_location="cpu"
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- SIDEBAR ----------------
st.sidebar.title("♻️ Waste Categories")

st.sidebar.success("Supported Classes")

st.sidebar.write("🟦 Dry Waste")
st.sidebar.write("🟩 Wet Waste")
st.sidebar.write("🟥 Sanitary Waste")
st.sidebar.write("⚡ E-Waste")
st.sidebar.write("🏥 Medical Waste")

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload or capture a waste image to classify it."
)

# ---------------- INPUT ----------------
uploaded = st.file_uploader(
    "📤 Upload Waste Image",
    type=["jpg","jpeg","png"]
)

camera = st.camera_input(
    "📸 Capture Waste Image"
)

if uploaded or camera:

    file = uploaded if uploaded else camera

    image = Image.open(file).convert("RGB")

    col1,col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(
            outputs,
            dim=1
        )[0]

    confidence, pred = torch.max(
        probs,
        0
    )

    prediction = classes[pred.item()]

    emoji = {
        "dry":"🟦",
        "wet":"🟩",
        "sanitary":"🟥",
        "ewaste":"⚡",
        "medical":"🏥"
    }

    recommendations = {
        "dry":"Recycle in Dry Waste Bin",
        "wet":"Dispose in Compost/Wet Waste Bin",
        "sanitary":"Dispose in Sanitary Waste Bin",
        "ewaste":"Send to E-Waste Collection Center",
        "medical":"Dispose through Biomedical Waste System"
    }

    waste_info = {
        "dry":"Recyclable materials like paper, plastic and cardboard.",
        "wet":"Organic waste that can be composted.",
        "sanitary":"Personal hygiene products requiring special handling.",
        "ewaste":"Electronic devices and components.",
        "medical":"Potentially infectious healthcare waste."
    }

    impact = {
        "dry":"♻️ High Recycling Potential",
        "wet":"🌱 Compost Friendly",
        "sanitary":"⚠️ Requires Safe Disposal",
        "ewaste":"☣️ Contains Hazardous Components",
        "medical":"🏥 Biomedical Handling Required"
    }

    with col2:

        st.subheader("🔍 Prediction Result")

        st.success(
            f"{emoji[prediction]} {prediction.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence.item()*100:.2f}%"
        )

        # Gauge Meter
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence.item()*100,
            title={'text':"Confidence Score"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':"green"}
            }
        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    st.divider()

    # ---------------- BAR CHART ----------------
    st.subheader("📊 Probability Distribution")

    df = pd.DataFrame({
        "Waste Type": classes,
        "Probability": probs.numpy()*100
    })

    fig_bar = px.bar(
        df,
        x="Waste Type",
        y="Probability",
        color="Waste Type",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    # ---------------- DONUT CHART ----------------
    st.subheader("🍩 Prediction Breakdown")

    fig_pie = px.pie(
        df,
        names="Waste Type",
        values="Probability",
        hole=0.5
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # ---------------- DETAILS ----------------
    c1,c2 = st.columns(2)

    with c1:
        st.subheader("♻️ Waste Information")
        st.info(
            waste_info[prediction]
        )

    with c2:
        st.subheader("🌍 Environmental Impact")
        st.success(
            impact[prediction]
        )

    st.subheader("💡 Disposal Recommendation")

    st.markdown(f"""
    <div style="
    background:#E8F5E9;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;">
    {recommendations[prediction]}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📋 Detailed Probability Table")

    df["Probability"] = (
        df["Probability"]
    ).round(2)

    st.dataframe(
        df,
        use_container_width=True
    )

# ---------------- FOOTER ----------------
st.divider()

f1,f2,f3 = st.columns(3)

f1.metric(
    "Waste Categories",
    "5"
)

f2.metric(
    "Model",
    "EfficientNetV2-S"
)

f3.metric(
    "Framework",
    "PyTorch"
)

st.caption(
    "♻️ Smart Waste Detection System | AI for Sustainable Waste Management"
)