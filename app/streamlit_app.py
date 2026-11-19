"""Streamlit interface for the MAINTiDOC multimodal document assistant."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.rag.retriever import index_pdf
from src.agents.rag_agent import answer_question


settings = get_settings()
st.set_page_config(page_title="MAINTiDOC", page_icon="🛠️", layout="wide")
st.markdown(
    """
    <style>
      :root { --mainti-navy: #10283f; --mainti-blue: #1d4564; --mainti-amber: #e7a938; --mainti-paper: #f4f7f9; }
      .stApp { background: var(--mainti-paper); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg,#10283f 0%,#183a56 100%); }
      [data-testid="stSidebar"] * { color: #eef5fa; }
      [data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: #ffffff12; border-color: #ffffff38; }
      [data-testid="stSidebar"] button[kind="primary"] { background: var(--mainti-amber); color: #10283f; border: 0; font-weight: 700; }
      .mainti-hero { padding: 1.2rem 1.5rem; border-radius: 18px; color: white; background: linear-gradient(115deg,#10283f,#1d4564); margin: .25rem 0 1rem; }
      .mainti-eyebrow { color: #f3bd5f; text-transform: uppercase; font-size: .72rem; letter-spacing: .13em; font-weight: 700; }
      .mainti-hero h1 { color: white; margin: .2rem 0 .25rem; font-size: 2rem; }
      .mainti-hero p { color: #d5e3ed; margin: 0; }
      .mainti-chip { display: inline-block; padding: .42rem .72rem; margin: .2rem .35rem .45rem 0; background: white; border: 1px solid #dce5eb; border-radius: 999px; color: #24445d; font-size: .82rem; }
      [data-testid="stChatMessage"] { background: white; border: 1px solid #e2e9ee; border-radius: 16px; padding: .65rem 1rem; box-shadow: 0 5px 18px #183a5608; }
      [data-testid="stChatInput"] textarea { border-radius: 14px; }
      [data-testid="stExpander"] { border-color: #dce5eb; border-radius: 12px; }
      @media (max-width: 700px) { .mainti-hero { padding: 1rem; } .mainti-hero h1 { font-size: 1.6rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="mainti-hero"><div class="mainti-eyebrow">Assistant de maintenance · espace local</div>'
    '<h1>MAINTiDOC</h1><p>Retrouve les informations utiles dans tes documents techniques, avec leurs pages à l’appui.</p></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="mainti-chip">🔒 Traitement local</span>'
    '<span class="mainti-chip">📄 Sources paginées</span>'
    '<span class="mainti-chip">🖼️ Photo facultative</span>',
    unsafe_allow_html=True,
)


def _render_sources(sources: list[dict]) -> None:
    for source in sources:
        page = source.get("page_number") or "?"
        kind = {"text": "Texte", "table": "Tableau", "image": "Figure"}.get(source.get("content_type"), "Source")
        with st.container(border=True):
            st.markdown(f"**{source['source_file']}**  ·  page {page}  ·  {kind}")
            st.caption(source.get("text", "")[:1200])
            image_path = source.get("image_path")
            if source.get("content_type") == "image" and image_path and Path(image_path).is_file():
                st.image(image_path, caption=f"Figure de la page {page}", use_container_width=True)


def _save_uploaded_pdf(uploaded_file) -> Path:
    name = Path(uploaded_file.name).name
    if Path(name).suffix.lower() != ".pdf":
        raise ValueError("Seuls les fichiers PDF sont acceptés.")
    safe_name = re.sub(r"[^\w.() -]+", "_", name, flags=re.UNICODE).strip(" .") or "document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    payload = uploaded_file.getvalue()
    if not payload.startswith(b"%PDF-"):
        raise ValueError("Le fichier transmis ne semble pas être un PDF valide.")
    destination_dir = settings.project_root / "data" / "documents"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / safe_name
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        destination = destination.with_name(f"{destination.stem}_{digest[:10]}{destination.suffix}")
    if not destination.exists():
        destination.write_bytes(payload)
    return destination


with st.sidebar:
    st.markdown("<div class='mainti-eyebrow'>Espace de travail</div>", unsafe_allow_html=True)
    st.header("Documents")
    st.caption("Ajoute un manuel, une procédure ou une fiche technique.")
    uploaded_pdf = st.file_uploader("Ajouter un PDF technique", type=["pdf"], max_upload_size=100)
    if st.button("Extraire et indexer", type="primary", disabled=uploaded_pdf is None, use_container_width=True):
        try:
            pdf_path = _save_uploaded_pdf(uploaded_pdf)
            with st.status("Préparation du document…", expanded=True) as status:
                st.write("Extraction du texte, des tableaux et des figures")
                result = index_pdf(pdf_path)
                status.update(label="Document indexé", state="complete", expanded=False)
            st.success(
                f"Indexé : {result['source_filename']} · {result['page_count']} pages · "
                f"{result['indexed_node_count']} passages · {len(result.get('image_descriptions', []))} figures"
            )
            for warning in result.get("warnings", []):
                st.warning(warning)
        except Exception as error:
            st.error(str(error))

    st.divider()
    st.header("Photo facultative")
    st.caption("Ajoute une photo pour rechercher des figures similaires dans les PDF indexés.")
    uploaded_image = st.file_uploader(
        "Ajouter une photo à comparer aux figures indexées",
        type=["png", "jpg", "jpeg", "webp"],
        key="maintenance_photo",
        max_upload_size=20,
    )
    if uploaded_image:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

    st.divider()
    st.caption(f"Modèles locaux\n\nVision · {settings.vision_model}\n\nRéponse · {settings.generation_model}\n\nIndex documentaire · PostgreSQL + pgvector")


if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources documentaires"):
                _render_sources(turn["sources"])
        if turn.get("warning"):
            st.warning(turn["warning"])

question = st.chat_input("Pose une question sur les documents indexés")
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            photo_bytes = uploaded_image.getvalue() if uploaded_image else None
            with st.spinner("Recherche documentaire et préparation de la réponse…"):
                response = answer_question(question, photo_bytes)
            st.markdown(response["answer"])
            if response["vision_description"]:
                with st.expander("Description de la photo (Qwen2.5-VL)"):
                    st.write(response["vision_description"])
            if response["sources"]:
                with st.expander("Sources documentaires"):
                    _render_sources(response["sources"])
            if response["agent_warning"]:
                st.warning(response["agent_warning"])
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "warning": response["agent_warning"],
                }
            )
        except Exception as error:
            st.error(f"Le RAG n'est pas prêt : {error}")
            st.session_state.chat.append(
                {"role": "assistant", "content": f"Le RAG n'est pas prêt : {error}"}
            )
