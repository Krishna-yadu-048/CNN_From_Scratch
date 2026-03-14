"""
Streamlit app for the Cats vs Dogs classifier.

Run with: streamlit run app.py
"""

import streamlit as st
import torch
import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.model import CatDogCNN

# ========== Page Config ==========

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom Styling ==========

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }

    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.7;
    }

    .prediction-box {
        text-align: center;
        padding: 1.8rem;
        border-radius: 16px;
        margin: 1rem 0;
    }

    .prediction-box.cat {
        background: linear-gradient(135deg, #4ECDC420, #4ECDC410);
        border: 2px solid #4ECDC4;
    }

    .prediction-box.dog {
        background: linear-gradient(135deg, #FF6B6B20, #FF6B6B10);
        border: 2px solid #FF6B6B;
    }

    .prediction-box h2 {
        font-size: 2rem;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #ffffff15;
    }

    .metric-card h3 {
        font-size: 0.85rem;
        opacity: 0.6;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    div[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ========== Helper Functions ==========

@st.cache_resource
def load_model():
    """Load the trained model weights."""
    model = CatDogCNN(dropout_rate=0.5)
    model_path = os.path.join("models", "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )
        model.eval()
        return model, True
    return model, False


@st.cache_data
def load_training_history():
    """Load saved training metrics."""
    history_path = os.path.join("models", "training_history.json")

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return None


def preprocess_image(image):
    """Apply the same transforms we used for validation."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)


def predict(model, image):
    """Run inference on a single image."""
    with torch.no_grad():
        img_tensor = preprocess_image(image)
        output = model(img_tensor).squeeze()
        prob = torch.sigmoid(output).item()

    predicted_class = "Dog" if prob > 0.5 else "Cat"
    confidence = prob if prob > 0.5 else 1 - prob
    return predicted_class, confidence, prob


# ========== Plotly Chart Builders ==========

def make_training_curves(history):
    """Build interactive training curves."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss", "Accuracy"),
        horizontal_spacing=0.1
    )

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=history['train_loss'],
        mode='lines+markers', name='Train',
        line=dict(color='#FF6B6B', width=2.5), marker=dict(size=5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'],
        mode='lines+markers', name='Validation',
        line=dict(color='#4ECDC4', width=2.5), marker=dict(size=5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=history['train_acc'],
        mode='lines+markers', name='Train',
        line=dict(color='#FF6B6B', width=2.5), marker=dict(size=5),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_acc'],
        mode='lines+markers', name='Validation',
        line=dict(color='#4ECDC4', width=2.5), marker=dict(size=5),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif", size=12),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(title_text="Epoch", gridcolor='rgba(255,255,255,0.07)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)')

    return fig


def make_confusion_matrix_fig(history):
    """Build confusion matrix as a matplotlib figure."""
    y_true = np.array(history['test_labels'])
    y_pred = np.array(history['test_preds'])
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
        annot_kws={"size": 20, "weight": "bold"},
        linewidths=2, linecolor='#333',
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel("Actual", fontsize=12, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()

    return fig


def make_roc_curve(history):
    """Build ROC curve with AUC."""
    from sklearn.metrics import roc_curve, auc

    y_true = np.array(history['test_labels'])
    y_probs = np.array(history['test_probs'])
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'AUC = {roc_auc:.3f}',
        line=dict(color='#FF6B6B', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random', line=dict(color='gray', dash='dash', width=1)
    ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="ROC Curve", x=0.5),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif"),
        legend=dict(x=0.6, y=0.1, bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)')

    return fig


def make_confidence_distribution(history):
    """Histogram of prediction confidences split by true class."""
    y_true = np.array(history['test_labels'])
    y_probs = np.array(history['test_probs'])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_probs[y_true == 0], name="Actual: Cat",
        marker_color='#4ECDC4', opacity=0.7, nbinsx=25
    ))
    fig.add_trace(go.Histogram(
        x=y_probs[y_true == 1], name="Actual: Dog",
        marker_color='#FF6B6B', opacity=0.7, nbinsx=25
    ))

    fig.update_layout(
        template="plotly_dark",
        barmode='overlay',
        title=dict(text="Prediction Confidence", x=0.5),
        xaxis_title="P(Dog)",
        yaxis_title="Count",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif"),
        legend=dict(x=0.75, y=0.95, bgcolor='rgba(0,0,0,0)')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.07)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.07)')

    return fig


# ========== Main App ==========

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🐱 Cats vs Dogs Classifier 🐶</h1>
        <p>A CNN built from scratch with PyTorch — no pretrained models</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model and history
    model, model_loaded = load_model()
    history = load_training_history()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 About This Project")
        st.markdown(
            "This is a binary image classifier built using a custom CNN "
            "architecture. The model was trained on the Kaggle Cats vs Dogs "
            "dataset (~25k images)."
        )

        st.markdown("---")
        st.markdown("### Architecture")
        st.markdown(
            "**4 Conv Blocks** (Conv → BN → ReLU → Pool)  \n"
            "**Classifier:** FC(512) → Dropout → FC(1)  \n"
            "**Input:** 128×128 RGB images"
        )

        st.markdown("---")
        st.markdown("### Tech Stack")
        st.markdown("PyTorch · Streamlit · Plotly · Seaborn")

        if history:
            st.markdown("---")
            st.markdown("### Quick Stats")
            st.metric("Test Accuracy", f"{history['test_acc']:.1%}")
            st.metric("Test Loss", f"{history['test_loss']:.4f}")
            st.metric("Epochs Trained", len(history['train_loss']))

    # Main content — two tabs
    tab1, tab2 = st.tabs(["🔍 Predict", "📊 Model Performance"])

    # ===== TAB 1: Prediction =====
    with tab1:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown("### Upload an Image")
            st.markdown("Drop in a photo of a cat or dog and see what the model thinks.")

            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Your image", use_container_width=True)

        with col_result:
            if uploaded_file is not None and model_loaded:
                predicted_class, confidence, raw_prob = predict(model, image)
                css_class = "dog" if predicted_class == "Dog" else "cat"
                emoji = "🐶" if predicted_class == "Dog" else "🐱"

                st.markdown(f"""
                <div class="prediction-box {css_class}">
                    <h2>{emoji} {predicted_class}</h2>
                    <p style="font-size: 1.2rem; opacity: 0.8; margin-top: 0.5rem;">
                        Confidence: <strong>{confidence:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.markdown("#### Confidence Breakdown")

                cat_prob = 1 - raw_prob
                dog_prob = raw_prob

                prob_fig = go.Figure()
                prob_fig.add_trace(go.Bar(
                    y=[""], x=[cat_prob], name="Cat 🐱",
                    orientation='h', marker_color='#4ECDC4',
                    text=f"{cat_prob:.1%}", textposition='inside',
                    textfont=dict(size=14, color='white')
                ))
                prob_fig.add_trace(go.Bar(
                    y=[""], x=[dog_prob], name="Dog 🐶",
                    orientation='h', marker_color='#FF6B6B',
                    text=f"{dog_prob:.1%}", textposition='inside',
                    textfont=dict(size=14, color='white')
                ))
                prob_fig.update_layout(
                    barmode='stack',
                    template="plotly_dark",
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(orientation='h', x=0.3, y=-0.5),
                    xaxis=dict(visible=False, range=[0, 1]),
                    yaxis=dict(visible=False)
                )
                st.plotly_chart(prob_fig, use_container_width=True)

            elif uploaded_file is not None and not model_loaded:
                st.warning(
                    "⚠️ Model weights not found. Train the model first by running "
                    "the notebook, then restart this app."
                )
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem 1rem; opacity: 0.5;">
                    <p style="font-size: 3rem;">🐾</p>
                    <p>Upload an image to get a prediction</p>
                </div>
                """, unsafe_allow_html=True)

    # ===== TAB 2: Model Performance =====
    with tab2:
        if history is None:
            st.info(
                "📝 No training history found. Run the training notebook first "
                "to generate performance plots."
            )
            st.stop()

        # Metrics cards
        st.markdown("### Overview")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Test Accuracy</h3>
                <div class="value" style="color: #4ECDC4;">{history['test_acc']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Test Loss</h3>
                <div class="value" style="color: #FF6B6B;">{history['test_loss']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            best_epoch = np.argmin(history['val_loss']) + 1
            st.markdown(f"""
            <div class="metric-card">
                <h3>Best Epoch</h3>
                <div class="value" style="color: #FFE66D;">{best_epoch}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            final_lr = "0.001"  # initial LR
            st.markdown(f"""
            <div class="metric-card">
                <h3>Learning Rate</h3>
                <div class="value" style="color: #C792EA;">{final_lr}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Training curves
        st.markdown("### Training Curves")
        st.plotly_chart(make_training_curves(history), use_container_width=True)

        # Two columns for confusion matrix and ROC
        st.markdown("---")
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("### Confusion Matrix")
            cm_fig = make_confusion_matrix_fig(history)
            st.pyplot(cm_fig)

        with col_roc:
            st.markdown("### ROC Curve")
            st.plotly_chart(make_roc_curve(history), use_container_width=True)

        # Confidence distribution
        st.markdown("---")
        st.markdown("### Prediction Confidence Distribution")
        st.markdown(
            "This shows how confident the model is on test set predictions. "
            "Ideally, cats cluster near 0 and dogs cluster near 1."
        )
        st.plotly_chart(make_confidence_distribution(history), use_container_width=True)


if __name__ == "__main__":
    main()
