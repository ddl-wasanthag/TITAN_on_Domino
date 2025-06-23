# ──────────────────────────────────────────────────────────────────────────────
# DEBUG TRIPWIRE BLOCK (must come before any Streamlit commands)
# ──────────────────────────────────────────────────────────────────────────────

import traceback

# 1) Patch requests.get and requests.post to print a stack trace if called
import requests
import urllib3

_original_requests_get = requests.get
_original_requests_post = requests.post
_original_urllib3_pool = urllib3.PoolManager

def _debug_requests_get(*args, **kwargs):
    print("\n>>> DEBUG: requests.get() called! Stack trace:")
    traceback.print_stack()
    return _original_requests_get(*args, **kwargs)

def _debug_requests_post(*args, **kwargs):
    print("\n>>> DEBUG: requests.post() called! Stack trace:")
    traceback.print_stack()
    return _original_requests_post(*args, **kwargs)

def _debug_urllib3_poolmgr(*args, **kwargs):
    print("\n>>> DEBUG: urllib3.PoolManager() instantiated! Stack trace:")
    traceback.print_stack()
    return _original_urllib3_pool(*args, **kwargs)

requests.get = _debug_requests_get
requests.post = _debug_requests_post
urllib3.PoolManager = _debug_urllib3_poolmgr

# 2) Patch Hugging Face from_pretrained methods
import transformers

_original_acfp = transformers.AutoConfig.from_pretrained
_original_amfp = transformers.AutoModel.from_pretrained
_original_atfp = transformers.AutoTokenizer.from_pretrained
_original_ptfp = transformers.PreTrainedTokenizerFast.from_pretrained

def _debug_auto_config(cls, *args, **kwargs):
    print("\n>>> DEBUG: AutoConfig.from_pretrained() called! Stack trace:")
    traceback.print_stack()
    return _original_acfp(*args, **kwargs)

def _debug_auto_model(cls, *args, **kwargs):
    print("\n>>> DEBUG: AutoModel.from_pretrained() called! Stack trace:")
    traceback.print_stack()
    return _original_amfp(*args, **kwargs)

def _debug_auto_tokenizer(cls, *args, **kwargs):
    print("\n>>> DEBUG: AutoTokenizer.from_pretrained() called! Stack trace:")
    traceback.print_stack()
    return _original_atfp(*args, **kwargs)

def _debug_pretrained_tokenizer_fast(cls, *args, **kwargs):
    print("\n>>> DEBUG: PreTrainedTokenizerFast.from_pretrained() called! Stack trace:")
    traceback.print_stack()
    return _original_ptfp(*args, **kwargs)

transformers.AutoConfig.from_pretrained    = classmethod(_debug_auto_config)
transformers.AutoModel.from_pretrained     = classmethod(_debug_auto_model)
transformers.AutoTokenizer.from_pretrained = classmethod(_debug_auto_tokenizer)
transformers.PreTrainedTokenizerFast.from_pretrained = classmethod(_debug_pretrained_tokenizer_fast)

# 3) Patch huggingface_hub.HfApi.__init__ and hf_hub_download if available
try:
    from huggingface_hub import HfApi, hf_hub_download
    _original_hfapi_init = HfApi.__init__
    _original_hf_hub_download = hf_hub_download

    def _debug_hfapi_init(self, *args, **kwargs):
        print("\n>>> DEBUG: HfApi.__init__() called! Stack trace:")
        traceback.print_stack()
        return _original_hfapi_init(self, *args, **kwargs)

    def _patched_hf_hub_download(repo_id, filename, *args, **kwargs):
        print(f"\n>>> DEBUG: hf_hub_download called with repo_id={repo_id}, filename={filename}. Stack trace:")
        traceback.print_stack()
        # Attempt to load from local directory
        local_path = Path("/mnt/titan_model") / filename
        if local_path.exists():
            return str(local_path)
        # Fallback to original (will trigger error if gated)
        return _original_hf_hub_download(repo_id, filename, *args, **kwargs)

    HfApi.__init__ = _debug_hfapi_init
    import huggingface_hub
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    hf_hub_download = _patched_hf_hub_download
except ImportError:
    HfApi = None

# ──────────────────────────────────────────────────────────────────────────────
# End of debug tripwire block
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st

st.set_page_config(
    page_title="TITAN WSI Embedding Generator with Enhanced Visualization",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys  # required for sys.path manipulation
import importlib
import json
import torch
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from io import BytesIO
import logging
import time
from transformers import logging as hf_logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import scipy.stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "/mnt/titan_model"

def setup_local_model_env():
    """Setup environment for local model loading without HF tokens"""
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DISABLE_TELEMETRY'] = '1'

    model_path = Path(LOCAL_MODEL_DIR)
    if not model_path.exists():
        st.sidebar.error(f"❌ Local model directory not found: {LOCAL_MODEL_DIR}")
        return False

    config_file = model_path / "config.json"
    if not config_file.exists():
        st.sidebar.error("❌ config.json not found in model directory")
        return False

    python_files = list(model_path.glob("*.py"))
    if not python_files:
        st.sidebar.error("❌ No Python files found – TITAN needs modeling_titan.py, configuration_titan.py, etc.")
        return False
    else:
        st.sidebar.success(f"✅ Found {len(python_files)} Python files for custom model code")
        for py_file in python_files:
            st.sidebar.info(f"  📄 {py_file.name}")

    files = list(model_path.iterdir())
    st.sidebar.info(f"📁 Total files in model directory: {len(files)}")

    weight_files = []
    for fname in ["model.safetensors", "pytorch_model.bin", "conch_v1_5_pytorch_model.bin"]:
        if (model_path / fname).exists():
            weight_files.append(fname)
    if not weight_files:
        st.sidebar.warning("⚠️ No model weight files found (model.safetensors, pytorch_model.bin, or conch_v1_5_pytorch_model.bin)")
    else:
        st.sidebar.success(f"✅ Found model weights: {weight_files}")

    st.sidebar.success("🔧 Configured for local-only model loading")
    return True

@st.cache_resource
def load_titan_model():
    """Load and cache the TITAN model from local directory"""
    setup_local_model_env()
    processor = TITANProcessor(LOCAL_MODEL_DIR)
    return processor if processor.load_model() else None

class TITANProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.conch = None
        self.eval_transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_error = None

    def load_model(self):
        """
        Load the TITAN model from the local directory by directly instantiating
        the class and manually loading weights (no HF online calls).
        """
        hf_logging.set_verbosity_error()
        try:
            with st.spinner("Loading TITAN model from local directory..."):
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
                os.environ["DISABLE_TELEMETRY"] = "1"

                model_dir = Path(self.model_path)
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

                # Ensure the parent directory is on sys.path so we can import titan_model
                parent_dir = model_dir.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))

                st.info(f"📁 Model directory: {model_dir}")
                st.info(f"Files in {model_dir}: {[f.name for f in model_dir.iterdir()]}")

                # 1) Patch Conch_Tokenizer to load from local files instead of HF
                try:
                    ctmod = importlib.import_module("titan_model.conch_tokenizer")
                    from transformers import PreTrainedTokenizerFast

                    _original_ct_init = ctmod.Conch_Tokenizer.__init__
                    def _patched_conch_init(self, context_length):
                        # Load tokenizer.json from local model dir
                        tok_path = Path(self._tokenizer_base_dir) if hasattr(self, '_tokenizer_base_dir') else Path(self.model_path)
                        json_file = tok_path / "tokenizer.json"
                        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(json_file))
                        self.context_length = context_length
                    ctmod.Conch_Tokenizer.__init__ = _patched_conch_init
                    st.info("🔧 Patched Conch_Tokenizer to load from local tokenizer.json")
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error("❌ Failed to patch Conch_Tokenizer; stack trace:")
                    st.code(tb)
                    raise

                # 2) Import TitanConfig from local configuration_titan.py
                try:
                    conf_mod = importlib.import_module("titan_model.configuration_titan")
                    TitanConfig = getattr(conf_mod, "TitanConfig")
                    st.info("✅ Imported TitanConfig")
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error("❌ Could not import TitanConfig; stack trace:")
                    st.code(tb)
                    raise

                # 3) Load config dict from config.json
                with open(model_dir / "config.json", "r") as f:
                    cfg_dict = json.load(f)
                config = TitanConfig(**cfg_dict)
                st.info("✅ Loaded TitanConfig from local config.json")

                # 4) Import Titan class from local modeling_titan.py
                try:
                    mod_mod = importlib.import_module("titan_model.modeling_titan")
                    Titan = getattr(mod_mod, "Titan")
                    st.info("✅ Imported Titan class")
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error("❌ Could not import Titan; stack trace:")
                    st.code(tb)
                    raise

                # 5) Instantiate Titan directly from config
                try:
                    # Ensure Conch_Tokenizer sees the correct base directory
                    ctmod.Conch_Tokenizer._tokenizer_base_dir = self.model_path
                    self.model = Titan(config)
                    st.info("🔧 Instantiated Titan model from local config")
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error("❌ Error inside Titan.__init__; stack trace:")
                    st.code(tb)
                    raise

                # 6) Manually load state dict from safetensors or .bin
                weight_file_path = None
                if (model_dir / "model.safetensors").exists():
                    weight_file_path = model_dir / "model.safetensors"
                    try:
                        from safetensors.torch import load_file as load_safetensors
                        state_dict = load_safetensors(str(weight_file_path))
                    except ImportError:
                        state_dict = torch.load(str(weight_file_path), map_location="cpu")
                elif (model_dir / "pytorch_model.bin").exists():
                    weight_file_path = model_dir / "pytorch_model.bin"
                    state_dict = torch.load(str(weight_file_path), map_location="cpu")

                if weight_file_path:
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        st.warning(f"⚠️ Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        st.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                    st.success(f"✅ Loaded weights from {weight_file_path.name}")
                else:
                    st.warning("⚠️ No weight file found; model remains randomly initialized")

                # 7) Move model to the correct device and half precision if available
                self.model = self.model.to(self.device)
                if torch.cuda.is_available():
                    self.model = self.model.half()

                # 8) Extract CONCH encoder and eval_transform
                try:
                    self.conch, self.eval_transform = self.model.return_conch()
                    st.success("✅ Loaded CONCH encoder from TITAN model")
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error("❌ Error in return_conch(); stack trace:")
                    st.code(tb)
                    raise

                return True

        except Exception as e:
            self.load_error = str(e)
            st.error(f"❌ Failed to load model: {e}")
            return False

    def encode_slide_from_features(self, features, coords, patch_size_lv0):
        """Encode slide from pre-extracted patch features."""
        try:
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", torch.float16), torch.inference_mode():
                slide_embedding = self.model.encode_slide_from_patch_features(
                    features.to(self.device),
                    coords.to(self.device),
                    patch_size_lv0
                )
            return slide_embedding.cpu().numpy()
        except Exception as e:
            raise Exception(f"Failed to encode slide: {e}")

def visualize_wsi_h5(features, coords, patch_size_lv0, filename):
    """
    WSI H5 visualization showing:
     - patch distribution (Scatter)
     - basic slide statistics
     - feature‐value histogram
    """

    st.subheader(f"🔬 WSI Visualization – {filename}")

    # Convert tensors to NumPy if needed
    coords_np = coords.numpy() if isinstance(coords, torch.Tensor) else coords
    features_np = features.numpy() if isinstance(features, torch.Tensor) else features

    col1, col2 = st.columns([2, 1])

    # ────────────────────────────────
    # 1) Patch Distribution Scatter
    # ────────────────────────────────
    with col1:
        st.write("**Patch Distribution Map**")

        fig = go.Figure()

        # Ensure coords_np is shape (N, 2)
        pts_count = coords_np.shape[0]
        color_vals = list(range(pts_count))  # must be a list, not a bare range

        fig.add_trace(
            go.Scatter(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Patch Index", x=1.02),
                ),
                text=[f"Patch {i}<br>X: {x}<br>Y: {y}"
                      for i, (x, y) in enumerate(coords_np)],
                hovertemplate="<b>%{text}</b><extra></extra>",
                name="Patches",
            )
        )

        fig.update_layout(
            title=f"WSI Patch Distribution (Total: {pts_count} patches)",
            xaxis_title="X Coordinate (pixels)",
            yaxis_title="Y Coordinate (pixels)",
            showlegend=False,
            height=500,
            margin=dict(r=80),
            yaxis=dict(autorange="reversed")   # ← use update_layout instead of update_yaxis
        )

        st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────
    # 2) Basic Slide Statistics
    # ────────────────────────────────
    with col2:
        st.write("**WSI Statistics**")

        slide_width = coords_np[:, 0].max() + patch_size_lv0
        slide_height = coords_np[:, 1].max() + patch_size_lv0

        stats_data = [
            ("Total Patches", pts_count),
            ("Patch Size", f"{patch_size_lv0}px"),
            ("X Range", f"{coords_np[:, 0].min()}–{coords_np[:, 0].max()}"),
            ("Y Range", f"{coords_np[:, 1].min()}–{coords_np[:, 1].max()}"),
            ("Est. Slide Size", f"{slide_width}×{slide_height}px"),
            ("Feature Dimensions", features_np.shape[1]),
            ("Coverage", f"{pts_count * patch_size_lv0**2 / (slide_width * slide_height) * 100:.1f}%")
        ]

        for label, value in stats_data:
            st.metric(label, value)

        st.write("**Feature Statistics**")
        feature_stats = [
            ("Mean", f"{features_np.mean():.4f}"),
            ("Std Dev", f"{features_np.std():.4f}"),
            ("Min Value", f"{features_np.min():.4f}"),
            ("Max Value", f"{features_np.max():.4f}"),
            ("Sparsity", f"{(features_np == 0).sum() / features_np.size * 100:.1f}%")
        ]
        for label, value in feature_stats:
            st.text(f"{label}: {value}")

    # ────────────────────────────────
    # 3) Feature‐Value Histogram
    # ────────────────────────────────
    st.write("**Feature Value Distribution**")

    flat_size = features_np.size
    sample_size = min(10000, flat_size)
    if flat_size > sample_size:
        feature_sample = np.random.choice(features_np.flatten(), sample_size, replace=False)
        st.info(f"Showing distribution of {sample_size:,} randomly sampled feature values")
    else:
        feature_sample = features_np.flatten()

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=feature_sample,
            nbinsx=50,
            opacity=0.7,
            name="Feature Values"
        )
    )
    fig_hist.update_layout(
        title="Distribution of Feature Values",
        xaxis_title="Feature Value",
        yaxis_title="Frequency",
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)


def process_h5_file(uploaded_file):
    """Process an uploaded H5 file and extract features + coords."""
    try:
        file_bytes = uploaded_file.read()
        h5_buffer = BytesIO(file_bytes)
        with h5py.File(h5_buffer, "r") as f:
            st.write("**H5 File Structure:**")
            structure_info = []

            def collect_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    structure_info.append(f"📊 Dataset: `{name}` – shape {obj.shape}, type {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    structure_info.append(f"📁 Group: `{name}`")

            f.visititems(collect_structure)
            for info in structure_info[:10]:
                st.write(info)
            if len(structure_info) > 10:
                st.write(f"... and {len(structure_info) - 10} more items")

            if "features" in f and "coords" in f:
                features = torch.from_numpy(f["features"][:])
                coords = torch.from_numpy(f["coords"][:])
                patch_size_lv0 = f["coords"].attrs.get("patch_size_level0", 256)
                if "patch_size_level0" not in f["coords"].attrs:
                    st.warning("No `patch_size_level0` found in coords attrs; defaulting to 256")

                st.success(f"✅ Extracted features: {features.shape}, coords: {coords.shape}")
                st.info(f"Patch size level 0: {patch_size_lv0}")

                visualize_wsi_h5(features, coords, patch_size_lv0, uploaded_file.name)
                return features, coords, patch_size_lv0
            else:
                st.error("❌ H5 file must contain `features` and `coords` datasets")
                return None, None, None

    except Exception as e:
        st.error(f"❌ Error processing H5 file: {e}")
        return None, None, None

def enhanced_embedding_visualizations(embeddings_dict):
    """
    Enhanced visualization suite for TITAN WSI embeddings with multiple analysis views
    """
    if len(embeddings_dict) < 1:
        st.info("No embeddings to visualize")
        return
    
    st.header("🎯 Enhanced Embedding Analysis & Visualizations")
    
    # Prepare data
    emb_list = []
    labels = []
    colors = px.colors.qualitative.Set1[:len(embeddings_dict)]
    
    for fname, emb in embeddings_dict.items():
        emb_list.append(emb.flatten())
        labels.append(fname)
    
    emb_array = np.vstack(emb_list)
    n_samples, n_features = emb_array.shape
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dimensionality Reduction", 
        "🔍 Statistical Analysis", 
        "🌐 Distance & Similarity", 
        "🎯 Clustering Analysis",
        "📈 Individual Embedding Profiles"
    ])
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: DIMENSIONALITY REDUCTION
    # ═══════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("📊 Dimensionality Reduction Visualizations")
        
        col1, col2 = st.columns(2)
        
        # PCA Analysis
        with col1:
            st.write("**🔸 Principal Component Analysis (PCA)**")
            
            if len(embeddings_dict) == 1:
                st.info("📊 Single sample detected - showing embedding distribution")
                
                # For single sample, show the embedding values as a line plot
                single_emb = emb_array[0]
                fig_single = go.Figure()
                fig_single.add_trace(go.Scatter(
                    y=single_emb,
                    mode='lines+markers',
                    name=labels[0],
                    line=dict(width=2),
                    marker=dict(size=3)
                ))
                fig_single.update_layout(
                    title=f"Embedding Profile - {labels[0]}",
                    xaxis_title="Feature Index",
                    yaxis_title="Embedding Value",
                    height=400
                )
                st.plotly_chart(fig_single, use_container_width=True)
                
                # Show statistics
                stats_data = {
                    'Total Features': len(single_emb),
                    'Mean': f"{single_emb.mean():.4f}",
                    'Std Dev': f"{single_emb.std():.4f}",
                    'Range': f"{single_emb.max() - single_emb.min():.4f}",
                    'L2 Norm': f"{np.linalg.norm(single_emb):.4f}"
                }
                
                for stat, value in stats_data.items():
                    st.metric(stat, value)
                    
            else:
                # Multiple samples - proceed with PCA
                n_components = min(len(embeddings_dict), 5, n_features)
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(emb_array)
                
                # Ensure we have at least 2 components for 2D plot
                if pca_result.shape[1] >= 2:
                    # PCA 2D plot
                    fig_pca = px.scatter(
                        x=pca_result[:, 0], 
                        y=pca_result[:, 1],
                        text=labels,
                        color=labels,
                        title=f"PCA 2D Projection (Explains {pca.explained_variance_ratio_[:2].sum():.1%} of variance)",
                        labels={
                            'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                            'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
                        },
                        width=500, height=400
                    )
                    fig_pca.update_traces(textposition="top center", marker_size=15)
                    fig_pca.update_layout(showlegend=True)
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    # Only 1 component available
                    fig_pca_1d = px.scatter(
                        x=pca_result[:, 0], 
                        y=[0] * len(labels),  # All points on same y-level
                        text=labels,
                        color=labels,
                        title=f"PCA 1D Projection (Explains {pca.explained_variance_ratio_[0]:.1%} of variance)",
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 'y': ''},
                        width=500, height=200
                    )
                    fig_pca_1d.update_traces(textposition="top center", marker_size=15)
                    fig_pca_1d.update_layout(showlegend=True, yaxis=dict(showticklabels=False))
                    st.plotly_chart(fig_pca_1d, use_container_width=True)
                    st.info("Only 1 principal component available - showing 1D projection")
                
                # PCA explained variance
                fig_var = px.bar(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=pca.explained_variance_ratio_,
                    title="Explained Variance by Principal Component",
                    labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
                )
                st.plotly_chart(fig_var, use_container_width=True)
        
        # t-SNE Analysis (if multiple samples)
        with col2:
            if len(embeddings_dict) > 1:
                st.write("**🔸 t-SNE Visualization**")
                
                # Check if we have enough samples for t-SNE
                n_samples = len(embeddings_dict)
                min_perplexity = 1
                max_perplexity = min(30, max(1, n_samples - 1))
                
                if n_samples < 3:
                    st.info("t-SNE requires at least 3 samples for meaningful visualization")
                elif max_perplexity < min_perplexity:
                    st.warning("Not enough samples for reliable t-SNE visualization")
                else:
                    try:
                        with st.spinner("Computing t-SNE... (this may take a moment)"):
                            # Use safe perplexity value
                            perplexity = min(5, max(1, n_samples - 1))
                            tsne = TSNE(
                                n_components=2, 
                                perplexity=perplexity, 
                                random_state=42, 
                                n_iter=1000,
                                learning_rate='auto',
                                init='random'
                            )
                            tsne_result = tsne.fit_transform(emb_array)
                        
                        fig_tsne = px.scatter(
                            x=tsne_result[:, 0], 
                            y=tsne_result[:, 1],
                            text=labels,
                            color=labels,
                            title=f"t-SNE 2D Projection (perplexity={perplexity})",
                            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                            width=500, height=400
                        )
                        fig_tsne.update_traces(textposition="top center", marker_size=15)
                        st.plotly_chart(fig_tsne, use_container_width=True)
                        
                        st.info(f"📊 Used perplexity: {perplexity} (adjusted for {n_samples} samples)")
                        
                    except Exception as e:
                        st.error(f"❌ t-SNE computation failed: {str(e)}")
                        st.info("This can happen with very similar embeddings or insufficient samples")
            else:
                st.info("t-SNE requires multiple samples for comparison")
        
        # 3D PCA if enough components
        if len(embeddings_dict) > 2 and 'pca_result' in locals() and pca_result.shape[1] >= 3:
            st.write("**🔸 3D PCA Visualization**")
            fig_3d = px.scatter_3d(
                x=pca_result[:, 0], 
                y=pca_result[:, 1], 
                z=pca_result[:, 2],
                text=labels,
                color=labels,
                title=f"3D PCA Projection (Explains {pca.explained_variance_ratio_[:3].sum():.1%} of variance)",
                labels={
                    'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                    'z': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
                }
            )
            fig_3d.update_traces(marker_size=8)
            st.plotly_chart(fig_3d, use_container_width=True)
        elif len(embeddings_dict) <= 2:
            st.info("3D PCA requires at least 3 samples")
        else:
            st.info("Insufficient components for 3D PCA visualization")
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🔍 Statistical Analysis of Embeddings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Embedding Statistics Summary**")
            
            # Create statistics dataframe
            stats_data = []
            for fname, emb in embeddings_dict.items():
                flat_emb = emb.flatten()
                stats_data.append({
                    'Filename': fname,
                    'Dimensions': emb.size,
                    'Mean': f"{flat_emb.mean():.4f}",
                    'Std Dev': f"{flat_emb.std():.4f}",
                    'Min': f"{flat_emb.min():.4f}",
                    'Max': f"{flat_emb.max():.4f}",
                    'L2 Norm': f"{np.linalg.norm(flat_emb):.4f}",
                    'Sparsity (%)': f"{(flat_emb == 0).sum() / len(flat_emb) * 100:.2f}%"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Distribution comparison
            st.write("**📈 Embedding Value Distributions**")
            
            fig_dist = go.Figure()
            for i, (fname, emb) in enumerate(embeddings_dict.items()):
                flat_emb = emb.flatten()
                # Sample for performance if too large
                if len(flat_emb) > 10000:
                    sample_emb = np.random.choice(flat_emb, 10000, replace=False)
                else:
                    sample_emb = flat_emb
                
                fig_dist.add_trace(go.Histogram(
                    x=sample_emb,
                    name=fname,
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig_dist.update_layout(
                title="Distribution of Embedding Values",
                xaxis_title="Embedding Value",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.write("**🎯 Feature Importance Analysis**")
            
            if len(embeddings_dict) == 1:
                st.info("📊 Single sample analysis")
                
                # For single sample, show feature statistics
                single_emb = emb_array[0]
                
                # Feature value distribution
                fig_hist_single = px.histogram(
                    x=single_emb,
                    nbins=50,
                    title=f"Feature Value Distribution - {labels[0]}",
                    labels={'x': 'Feature Value', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_hist_single, use_container_width=True)
                
                # Top and bottom features by absolute value
                abs_values = np.abs(single_emb)
                top_k = min(10, len(single_emb))
                top_indices = np.argsort(abs_values)[-top_k:][::-1]
                
                feature_data = []
                for i, idx in enumerate(top_indices):
                    feature_data.append({
                        'Rank': i + 1,
                        'Feature Index': idx,
                        'Value': f"{single_emb[idx]:.6f}",
                        'Abs Value': f"{abs_values[idx]:.6f}"
                    })
                
                feature_df = pd.DataFrame(feature_data)
                st.write(f"**Top {top_k} Features by Magnitude**")
                st.dataframe(feature_df, use_container_width=True)
                
            else:
                # Multiple samples - compute feature statistics across all embeddings
                feature_means = np.mean(emb_array, axis=0)
                feature_stds = np.std(emb_array, axis=0)
                feature_importance = feature_stds  # Use std as importance measure
                
                # Top important features
                top_k = min(20, len(feature_importance))
                top_indices = np.argsort(feature_importance)[-top_k:]
                
                fig_importance = px.bar(
                    x=feature_importance[top_indices],
                    y=[f'Feature {i}' for i in top_indices],
                    orientation='h',
                    title=f"Top {top_k} Most Variable Features",
                    labels={'x': 'Standard Deviation', 'y': 'Feature Index'}
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature correlation heatmap (if multiple samples)
            if len(embeddings_dict) > 1:
                st.write("**🔗 Feature Correlation Matrix**")
                
                try:
                    # Sample features for visualization
                    n_features_viz = min(50, n_features)
                    if n_features > n_features_viz:
                        selected_features = np.random.choice(n_features, n_features_viz, replace=False)
                    else:
                        selected_features = np.arange(n_features)
                    
                    # Compute correlation matrix
                    selected_data = emb_array[:, selected_features]
                    
                    # Check if we have enough variance for correlation
                    if selected_data.std(axis=0).min() > 1e-10:  # Check for non-zero variance
                        corr_matrix = np.corrcoef(selected_data.T)
                        
                        # Handle NaN values
                        if np.isnan(corr_matrix).any():
                            st.warning("⚠️ Some features have constant values, correlation matrix may be incomplete")
                            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                        
                        fig_corr = px.imshow(
                            corr_matrix,
                            title=f"Feature Correlation Matrix ({len(selected_features)} features)",
                            color_continuous_scale="RdBu_r",
                            aspect="auto",
                            zmin=-1,
                            zmax=1
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.warning("⚠️ Features have insufficient variance for correlation analysis")
                        
                except Exception as e:
                    st.error(f"❌ Could not compute feature correlation matrix: {str(e)}")
                    st.info("This can happen with identical embeddings or insufficient data")
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: DISTANCE & SIMILARITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🌐 Distance & Similarity Analysis")
        
        if len(embeddings_dict) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📏 Distance Matrices**")
                
                # Cosine similarity
                cosine_sim = cosine_similarity(emb_array)
                fig_cosine = px.imshow(
                    cosine_sim,
                    x=labels, y=labels,
                    title="Cosine Similarity Matrix",
                    color_continuous_scale="RdYlBu",
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig_cosine, use_container_width=True)
                
                # Euclidean distances
                euclidean_dist = euclidean_distances(emb_array)
                fig_euclidean = px.imshow(
                    euclidean_dist,
                    x=labels, y=labels,
                    title="Euclidean Distance Matrix",
                    color_continuous_scale="Viridis",
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig_euclidean, use_container_width=True)
            
            with col2:
                st.write("**🌳 Hierarchical Clustering**")
                
                try:
                    # Hierarchical clustering dendrogram
                    linkage_matrix = linkage(emb_array, method='ward')
                    
                    # Create dendrogram using matplotlib then display
                    fig, ax = plt.subplots(figsize=(10, 6))
                    dendro = dendrogram(linkage_matrix, labels=labels, ax=ax)
                    plt.title('Hierarchical Clustering Dendrogram')
                    plt.xlabel('Samples')
                    plt.ylabel('Distance')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"❌ Could not generate dendrogram: {str(e)}")
                    st.info("This can happen with identical embeddings or insufficient samples")
                
                st.write("**📊 Pairwise Similarities**")
                
                try:
                    # Create pairwise similarity table
                    similarity_data = []
                    for i in range(len(labels)):
                        for j in range(i+1, len(labels)):
                            similarity_data.append({
                                'Sample 1': labels[i],
                                'Sample 2': labels[j],
                                'Cosine Similarity': f"{cosine_sim[i, j]:.4f}",
                                'Euclidean Distance': f"{euclidean_dist[i, j]:.4f}"
                            })
                    
                    if similarity_data:
                        sim_df = pd.DataFrame(similarity_data)
                        st.dataframe(sim_df, use_container_width=True)
                    else:
                        st.info("No pairwise comparisons available")
                        
                except Exception as e:
                    st.error(f"❌ Could not compute pairwise similarities: {str(e)}")
        else:
            st.info("📊 Multiple samples required for distance/similarity analysis")
            st.write("**🔍 Single Sample Analysis**")
            
            # For single sample, show embedding characteristics
            single_emb = emb_array[0]
            single_name = labels[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Embedding Characteristics**")
                char_metrics = {
                    'L1 Norm': np.linalg.norm(single_emb, ord=1),
                    'L2 Norm': np.linalg.norm(single_emb, ord=2),
                    'L∞ Norm': np.linalg.norm(single_emb, ord=np.inf),
                    'Mean Absolute Value': np.mean(np.abs(single_emb)),
                    'RMS Value': np.sqrt(np.mean(single_emb**2))
                }
                
                for metric, value in char_metrics.items():
                    st.metric(metric, f"{value:.4f}")
            
            with col2:
                st.write("**📊 Value Statistics**")
                
                # Percentile information
                percentiles = [1, 5, 25, 50, 75, 95, 99]
                perc_values = np.percentile(single_emb, percentiles)
                
                perc_data = []
                for p, v in zip(percentiles, perc_values):
                    perc_data.append({
                        'Percentile': f"{p}th",
                        'Value': f"{v:.6f}"
                    })
                
                perc_df = pd.DataFrame(perc_data)
                st.dataframe(perc_df, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: CLUSTERING ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🎯 Clustering Analysis")
        
        if len(embeddings_dict) > 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 K-Means Clustering**")
                
                try:
                    # Determine optimal number of clusters
                    max_clusters = min(5, len(embeddings_dict) - 1)  # Ensure max_clusters < n_samples
                    k_range = range(2, max_clusters + 1)
                    
                    if len(k_range) == 0:
                        st.warning("Need at least 3 samples for clustering analysis")
                    else:
                        inertias = []
                        
                        for k in k_range:
                            try:
                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                kmeans.fit(emb_array)
                                inertias.append(kmeans.inertia_)
                            except Exception as e:
                                st.warning(f"Clustering with k={k} failed: {str(e)}")
                                continue
                        
                        if inertias:
                            # Elbow plot
                            fig_elbow = px.line(
                                x=list(k_range)[:len(inertias)], y=inertias,
                                title="K-Means Elbow Plot",
                                labels={'x': 'Number of Clusters', 'y': 'Inertia'},
                                markers=True
                            )
                            st.plotly_chart(fig_elbow, use_container_width=True)
                            
                            # Perform clustering with optimal k
                            valid_k_range = list(k_range)[:len(inertias)]
                            if valid_k_range:
                                optimal_k = st.selectbox("Select number of clusters:", valid_k_range, index=0)
                                
                                try:
                                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                                    cluster_labels = kmeans.fit_predict(emb_array)
                                    
                                    # Visualize clusters in PCA space
                                    pca_for_cluster = PCA(n_components=2)
                                    pca_result_cluster = pca_for_cluster.fit_transform(emb_array)
                                    
                                    fig_cluster = px.scatter(
                                        x=pca_result_cluster[:, 0], 
                                        y=pca_result_cluster[:, 1],
                                        color=[f"Cluster {i}" for i in cluster_labels],
                                        text=labels,
                                        title=f"K-Means Clustering (k={optimal_k}) in PCA Space",
                                        labels={'x': 'PC1', 'y': 'PC2'}
                                    )
                                    fig_cluster.update_traces(textposition="top center", marker_size=12)
                                    st.plotly_chart(fig_cluster, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"❌ Clustering visualization failed: {str(e)}")
                        else:
                            st.error("❌ All clustering attempts failed")
                            
                except Exception as e:
                    st.error(f"❌ Clustering analysis failed: {str(e)}")
            
            with col2:
                st.write("**📋 Cluster Assignments**")
                
                try:
                    if 'cluster_labels' in locals() and 'kmeans' in locals():
                        cluster_df = pd.DataFrame({
                            'Filename': labels,
                            'Cluster': [f"Cluster {i}" for i in cluster_labels],
                            'Distance to Centroid': [
                                f"{np.linalg.norm(emb_array[i] - kmeans.cluster_centers_[cluster_labels[i]]):.4f}"
                                for i in range(len(labels))
                            ]
                        })
                        st.dataframe(cluster_df, use_container_width=True)
                        
                        # Cluster statistics
                        st.write("**📊 Cluster Statistics**")
                        for cluster_id in range(optimal_k):
                            cluster_mask = cluster_labels == cluster_id
                            cluster_samples = np.array(labels)[cluster_mask]
                            if len(cluster_samples) > 0:
                                st.write(f"**Cluster {cluster_id}**: {', '.join(cluster_samples)}")
                    else:
                        st.info("Complete clustering analysis above to see assignments")
                        
                except Exception as e:
                    st.warning(f"Could not display cluster assignments: {str(e)}")
        else:
            st.info("At least 3 samples required for meaningful clustering analysis")
    
    # ═══════════════════════════════════════════════════════════════════
    # TAB 5: INDIVIDUAL EMBEDDING PROFILES
    # ═══════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("📈 Individual Embedding Profiles")
        
        selected_sample = st.selectbox("Select sample to analyze:", labels)
        selected_idx = labels.index(selected_sample)
        selected_emb = emb_array[selected_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**📊 Profile for: {selected_sample}**")
            
            # Embedding value distribution
            fig_profile = go.Figure()
            fig_profile.add_trace(go.Scatter(
                y=selected_emb,
                mode='lines+markers',
                name=selected_sample,
                line=dict(width=1),
                marker=dict(size=2)
            ))
            fig_profile.update_layout(
                title=f"Embedding Values Profile - {selected_sample}",
                xaxis_title="Feature Index",
                yaxis_title="Embedding Value",
                height=400
            )
            st.plotly_chart(fig_profile, use_container_width=True)
            
            # Statistical summary
            st.write("**📋 Statistical Summary**")
            profile_stats = {
                'Mean': selected_emb.mean(),
                'Std Dev': selected_emb.std(),
                'Min': selected_emb.min(),
                'Max': selected_emb.max(),
                'Range': selected_emb.max() - selected_emb.min(),
                'L1 Norm': np.linalg.norm(selected_emb, ord=1),
                'L2 Norm': np.linalg.norm(selected_emb, ord=2),
                'Non-zero Elements': np.count_nonzero(selected_emb),
                'Sparsity (%)': (selected_emb == 0).sum() / len(selected_emb) * 100
            }
            
            for stat, value in profile_stats.items():
                if isinstance(value, float):
                    st.metric(stat, f"{value:.4f}")
                else:
                    st.metric(stat, value)
        
        with col2:
            st.write("**🔍 Detailed Analysis**")
            
            # Histogram of embedding values
            fig_hist_profile = px.histogram(
                x=selected_emb,
                nbins=50,
                title=f"Distribution of Values - {selected_sample}",
                labels={'x': 'Embedding Value', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_hist_profile, use_container_width=True)
            
            # Top positive and negative values
            st.write("**🔺 Extreme Values**")
            
            # Top 10 highest and lowest values
            top_indices = np.argsort(selected_emb)
            highest_10 = top_indices[-10:][::-1]  # Reverse for descending order
            lowest_10 = top_indices[:10]
            
            extreme_data = []
            for i, idx in enumerate(highest_10):
                extreme_data.append({
                    'Rank': f"High #{i+1}",
                    'Feature Index': idx,
                    'Value': f"{selected_emb[idx]:.6f}"
                })
            for i, idx in enumerate(lowest_10):
                extreme_data.append({
                    'Rank': f"Low #{i+1}",
                    'Feature Index': idx,
                    'Value': f"{selected_emb[idx]:.6f}"
                })
            
            extreme_df = pd.DataFrame(extreme_data)
            st.dataframe(extreme_df, use_container_width=True)
        
        # Comparison with other samples (if available)
        if len(embeddings_dict) > 1:
            st.write("**⚖️ Comparison with Other Samples**")
            
            comparison_sample = st.selectbox(
                "Compare with:", 
                [label for label in labels if label != selected_sample]
            )
            comparison_idx = labels.index(comparison_sample)
            comparison_emb = emb_array[comparison_idx]
            
            # Side-by-side comparison
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                y=selected_emb,
                mode='lines',
                name=selected_sample,
                opacity=0.7
            ))
            fig_comparison.add_trace(go.Scatter(
                y=comparison_emb,
                mode='lines',
                name=comparison_sample,
                opacity=0.7
            ))
            fig_comparison.update_layout(
                title=f"Embedding Comparison: {selected_sample} vs {comparison_sample}",
                xaxis_title="Feature Index",
                yaxis_title="Embedding Value",
                height=400
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Difference plot
            diff = selected_emb - comparison_emb
            fig_diff = px.bar(
                y=diff,
                title=f"Difference: {selected_sample} - {comparison_sample}",
                labels={'x': 'Feature Index', 'y': 'Difference'}
            )
            st.plotly_chart(fig_diff, use_container_width=True)


def add_embedding_export_options(embeddings_dict):
    """
    Add advanced export options for embeddings with multiple formats
    """
    st.subheader("💾 Advanced Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📊 Export Formats**")
        
        # JSON export (existing)
        download_data = {}
        for fname, emb in embeddings_dict.items():
            download_data[fname] = {
                "embedding": emb.tolist(),
                "shape": emb.shape,
                "metadata": {
                    "generated_by": "TITAN",
                    "timestamp": time.time(),
                    "statistics": {
                        "mean": float(emb.mean()),
                        "std": float(emb.std()),
                        "min": float(emb.min()),
                        "max": float(emb.max()),
                        "l2_norm": float(np.linalg.norm(emb.flatten()))
                    }
                }
            }
        
        json_str = json.dumps(download_data, indent=2)
        st.download_button(
            label="📥 Download JSON (with metadata)",
            data=json_str,
            file_name="titan_embeddings_detailed.json",
            mime="application/json"
        )
    
    with col2:
        st.write("**📈 CSV Export**")
        
        # CSV export
        csv_data = []
        for fname, emb in embeddings_dict.items():
            flat_emb = emb.flatten()
            row = {"filename": fname}
            for i, val in enumerate(flat_emb):
                row[f"feature_{i:04d}"] = val
            csv_data.append(row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_string = csv_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_string,
            file_name="titan_embeddings.csv",
            mime="text/csv"
        )
    
    with col3:
        st.write("**🔢 NumPy Export**")
        
        if len(embeddings_dict) == 1:
            # Single embedding
            only_fname, only_emb = next(iter(embeddings_dict.items()))
            np_bytes = BytesIO()
            np.save(np_bytes, only_emb)
            st.download_button(
                label="📥 Download .npy (single)",
                data=np_bytes.getvalue(),
                file_name=f"{only_fname}_embedding.npy",
                mime="application/octet-stream"
            )
        else:
            # Multiple embeddings as array
            emb_array = np.vstack([emb.flatten() for emb in embeddings_dict.values()])
            np_bytes = BytesIO()
            np.savez_compressed(np_bytes, 
                               embeddings=emb_array, 
                               labels=list(embeddings_dict.keys()))
            st.download_button(
                label="📥 Download .npz (all)",
                data=np_bytes.getvalue(),
                file_name="titan_embeddings_all.npz",
                mime="application/octet-stream"
            )


def compute_embedding_quality_metrics(embeddings_dict):
    """
    Compute quality metrics for embeddings
    """
    if len(embeddings_dict) < 2:
        return None
    
    emb_array = np.vstack([emb.flatten() for emb in embeddings_dict.values()])
    
    # Compute various quality metrics
    metrics = {
        'inter_sample_variance': np.var(emb_array, axis=0).mean(),
        'intra_sample_consistency': np.mean([np.var(emb.flatten()) for emb in embeddings_dict.values()]),
        'feature_utilization': (emb_array != 0).mean(),
        'dynamic_range': emb_array.max() - emb_array.min(),
        'signal_to_noise_ratio': np.mean(emb_array) / np.std(emb_array) if np.std(emb_array) > 0 else 0
    }
    
    return metrics


def generate_embedding_fingerprint(embedding):
    """
    Generate a compact fingerprint/hash for an embedding
    """
    flat_emb = embedding.flatten()
    
    # Create statistical fingerprint
    fingerprint = {
        'hash': hash(flat_emb.tobytes()),
        'checksum': np.sum(flat_emb),
        'moments': {
            'mean': float(flat_emb.mean()),
            'std': float(flat_emb.std()),
            'skewness': float(scipy.stats.skew(flat_emb)),
            'kurtosis': float(scipy.stats.kurtosis(flat_emb))
        },
        'percentiles': {
            '1st': float(np.percentile(flat_emb, 1)),
            '25th': float(np.percentile(flat_emb, 25)),
            '50th': float(np.percentile(flat_emb, 50)),
            '75th': float(np.percentile(flat_emb, 75)),
            '99th': float(np.percentile(flat_emb, 99))
        }
    }
    
    return fingerprint


def main():
    st.title("🔬 TITAN WSI Embedding Generator")
    st.markdown("Upload WSI H5 files to visualize patches and generate embeddings using the TITAN model")

    # Sidebar: Model info
    with st.sidebar:
        st.header("⚙️ Model Information")

        model_path = Path(LOCAL_MODEL_DIR)
        if model_path.exists():
            st.success("📁 Local Model Dir: ✅ Found")
            files = list(model_path.iterdir())
            st.info(f"📊 Files: {len(files)} found")
        else:
            st.error("📁 Local Model Dir: ❌ Not found")

        cache_path = Path("/mnt/hf_cache")
        if cache_path.exists():
            st.success("💾 Cache Dir: ✅ Found")
        else:
            st.warning("💾 Cache Dir: ⚠️ Not found")

        if "titan_processor" not in st.session_state:
            st.session_state.titan_processor = load_titan_model()

        if st.session_state.titan_processor is not None:
            st.success("✅ TITAN Model Loaded")
            st.info(f"🔧 Device: {st.session_state.titan_processor.device}")
        else:
            st.error("❌ Model Not Loaded")
            if st.button("🔄 Retry Loading Model"):
                st.cache_resource.clear()
                st.session_state.titan_processor = load_titan_model()
                st.rerun()

        st.header("📁 Local Model Files")
        if model_path.exists():
            with st.expander("View Model Files"):
                for f in sorted(files)[:15]:
                    if f.is_file():
                        size_mb = f.stat().st_size / (1024 * 1024)
                        st.text(f"📄 {f.name} ({size_mb:.1f} MB)")
                    else:
                        st.text(f"📁 {f.name}/")
                if len(files) > 15:
                    st.text(f"... and {len(files) - 15} more")
        else:
            st.error("❌ Model directory not found")
            st.info("Expected location: /mnt/titan_model")

    # If model not loaded, stop
    if st.session_state.titan_processor is None:
        st.error("⚠️ Please load the TITAN model first (check sidebar)")
        return

    # Initialize embeddings store
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = {}

    st.header("📤 Upload WSI Files")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Choose H5 files containing WSI features",
            type=["h5", "hdf5"],
            accept_multiple_files=True,
            help="Upload H5 files with `features` and `coords` datasets"
        )
    with col2:
        if st.button("🗑️ Clear All Results"):
            st.session_state.embeddings = {}
            st.rerun()

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            fname = uploaded_file.name
            if fname not in st.session_state.embeddings:
                st.subheader(f"Processing: {fname}")
                features, coords, patch_size_lv0 = process_h5_file(uploaded_file)
                if features is not None:
                    try:
                        with st.spinner(f"Generating embedding for {fname}..."):
                            start = time.time()
                            emb = st.session_state.titan_processor.encode_slide_from_features(
                                features, coords, patch_size_lv0
                            )
                            duration = time.time() - start
                        st.session_state.embeddings[fname] = emb
                        st.success(f"✅ Generated embedding for {fname}")
                        st.info(f"Embedding shape: {emb.shape}")
                        st.info(f"Processing time: {duration:.2f} seconds")
                    except Exception as e:
                        st.error(f"❌ Failed to generate embedding: {e}")

    # ENHANCED RESULTS SECTION
    if st.session_state.embeddings:
        st.header("📊 Results & Comprehensive Analysis")
        
        # Quick overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(st.session_state.embeddings))
        with col2:
            total_features = sum(emb.size for emb in st.session_state.embeddings.values())
            st.metric("Total Features", f"{total_features:,}")
        with col3:
            avg_emb_size = np.mean([emb.size for emb in st.session_state.embeddings.values()])
            st.metric("Avg Embedding Size", f"{avg_emb_size:.0f}")
        with col4:
            processing_complete = len(st.session_state.embeddings)
            st.metric("Processing Status", f"{processing_complete} Complete")
        
        # Basic Summary Table
        with st.expander("📋 Detailed Summary Table", expanded=True):
            summary_data = []
            for fname, emb in st.session_state.embeddings.items():
                flat_emb = emb.flatten()
                summary_data.append({
                    "Filename": fname,
                    "Shape": str(emb.shape),
                    "Total Elements": f"{emb.size:,}",
                    "Mean": f"{flat_emb.mean():.4f}",
                    "Std Dev": f"{flat_emb.std():.4f}",
                    "Min": f"{flat_emb.min():.4f}",
                    "Max": f"{flat_emb.max():.4f}",
                    "L2 Norm": f"{np.linalg.norm(flat_emb):.4f}",
                    "Sparsity (%)": f"{(flat_emb == 0).sum() / len(flat_emb) * 100:.2f}%"
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # Enhanced Visualizations
        enhanced_embedding_visualizations(st.session_state.embeddings)
        
        # Advanced Export Options
        add_embedding_export_options(st.session_state.embeddings)
        
        # Additional Analysis Options
        st.subheader("🔬 Additional Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧮 Generate Analysis Report"):
                st.info("📝 Generating comprehensive analysis report...")
                
                report_data = {
                    "analysis_timestamp": time.time(),
                    "total_samples": len(st.session_state.embeddings),
                    "samples": {}
                }
                
                for fname, emb in st.session_state.embeddings.items():
                    flat_emb = emb.flatten()
                    report_data["samples"][fname] = {
                        "shape": emb.shape,
                        "statistics": {
                            "mean": float(flat_emb.mean()),
                            "std": float(flat_emb.std()),
                            "min": float(flat_emb.min()),
                            "max": float(flat_emb.max()),
                            "l1_norm": float(np.linalg.norm(flat_emb, ord=1)),
                            "l2_norm": float(np.linalg.norm(flat_emb, ord=2)),
                            "sparsity_percent": float((flat_emb == 0).sum() / len(flat_emb) * 100)
                        },
                        "fingerprint": generate_embedding_fingerprint(emb)
                    }
                
                # Add comparative analysis if multiple samples
                if len(st.session_state.embeddings) > 1:
                    emb_array = np.vstack([emb.flatten() for emb in st.session_state.embeddings.values()])
                    cosine_sim = cosine_similarity(emb_array)
                    euclidean_dist = euclidean_distances(emb_array)
                    
                    report_data["comparative_analysis"] = {
                        "cosine_similarity_matrix": cosine_sim.tolist(),
                        "euclidean_distance_matrix": euclidean_dist.tolist(),
                        "sample_labels": list(st.session_state.embeddings.keys()),
                        "quality_metrics": compute_embedding_quality_metrics(st.session_state.embeddings)
                    }
                
                report_json = json.dumps(report_data, indent=2)
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report_json,
                    file_name=f"titan_analysis_report_{int(time.time())}.json",
                    mime="application/json"
                )
                st.success("✅ Analysis report generated!")
        
        with col2:
            if len(st.session_state.embeddings) > 1:
                if st.button("🔍 Run Similarity Analysis"):
                    with st.spinner("Computing detailed similarity metrics..."):
                        emb_list = [emb.flatten() for emb in st.session_state.embeddings.values()]
                        labels = list(st.session_state.embeddings.keys())
                        
                        # Compute multiple similarity metrics
                        emb_array = np.vstack(emb_list)
                        cosine_sim = cosine_similarity(emb_array)
                        euclidean_dist = euclidean_distances(emb_array)
                        
                        # Correlation coefficient
                        correlation_matrix = np.corrcoef(emb_array)
                        
                        st.success("✅ Similarity analysis complete!")
                        
                        # Display most and least similar pairs
                        st.write("**🎯 Most Similar Pairs:**")
                        max_sim_idx = np.unravel_index(
                            np.argmax(cosine_sim - np.eye(len(labels))), 
                            cosine_sim.shape
                        )
                        st.write(f"• {labels[max_sim_idx[0]]} ↔ {labels[max_sim_idx[1]]}: {cosine_sim[max_sim_idx]:.4f}")
                        
                        st.write("**🎯 Most Different Pairs:**")
                        min_sim_idx = np.unravel_index(np.argmin(cosine_sim), cosine_sim.shape)
                        st.write(f"• {labels[min_sim_idx[0]]} ↔ {labels[min_sim_idx[1]]}: {cosine_sim[min_sim_idx]:.4f}")
            else:
                st.info("Upload multiple samples for similarity analysis")
        
        # Performance insights
        with st.expander("⚡ Performance Insights"):
            st.write("**🚀 Processing Performance**")
            
            # Estimate processing metrics
            total_features = sum(emb.size for emb in st.session_state.embeddings.values())
            avg_processing_time = 2.5  # Placeholder - you can track actual times
            
            perf_cols = st.columns(3)
            with perf_cols[0]:
                st.metric("Features/sec", f"{total_features/avg_processing_time:.0f}")
            with perf_cols[1]:
                st.metric("Memory Usage", f"~{total_features * 4 / 1024/1024:.1f} MB")
            with perf_cols[2]:
                device = st.session_state.titan_processor.device
                st.metric("Processing Device", str(device))
            
            st.write("**💡 Optimization Tips**")
            st.info("• Batch multiple slides for more efficient processing")
            st.info("• Use GPU acceleration when available")
            st.info("• Consider dimensionality reduction for large embedding sets")


if __name__ == "__main__":
    main()