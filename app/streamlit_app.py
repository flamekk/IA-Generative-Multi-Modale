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
st.title("MAINTiDOC")
st.caption("Assistant documentaire local pour les PDF techniques — réponses avec références de page.")


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
    st.header("Documents")
    uploaded_pdf = st.file_uploader("Ajouter un PDF technique", type=["pdf"], max_upload_size=100)
    if st.button("Extraire et indexer", type="primary", disabled=uploaded_pdf is None, use_container_width=True):
        try:
            pdf_path = _save_uploaded_pdf(uploaded_pdf)
            with st.spinner("Extraction, analyse des figures et indexation PGVector…"):
                result = index_pdf(pdf_path)
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
    uploaded_image = st.file_uploader(
        "Ajouter une photo à comparer aux figures indexées",
        type=["png", "jpg", "jpeg", "webp"],
        key="maintenance_photo",
        max_upload_size=20,
    )
    if uploaded_image:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

    st.divider()
    st.caption(f"Vision : {settings.vision_model}\n\nGénération : {settings.generation_model}\n\nIndex : PostgreSQL + pgvector")


if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources documentaires"):
                for source in turn["sources"]:
                    page = source.get("page_number") or "?"
                    st.markdown(f"**{source['source_file']} — page {page}** · {source['content_type']}")
                    st.caption(source["text"][:1200])
                    image_path = source.get("image_path")
                    if source["content_type"] == "image" and image_path and Path(image_path).is_file():
                        st.image(image_path, width=360)
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
                    for source in response["sources"]:
                        page = source.get("page_number") or "?"
                        st.markdown(f"**{source['source_file']} — page {page}** · {source['content_type']}")
                        st.caption(source["text"][:1200])
                        image_path = source.get("image_path")
                        if source["content_type"] == "image" and image_path and Path(image_path).is_file():
                            st.image(image_path, width=360)
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
