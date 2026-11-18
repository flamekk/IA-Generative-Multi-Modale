# MAINTiDOC

Assistant documentaire local pour PDF techniques, conçu pour les techniciens de maintenance. Le premier incrément intégré comprend l’ingestion, l’indexation vectorielle PostgreSQL/pgvector, la recherche textuelle et visuelle, l’analyse de figures par Qwen2.5-VL, un agent LangChain avec Qwen3/Ollama et une interface Streamlit.

## État et architecture

Le pipeline d’ingestion existant extrait les pages, le texte, les images et les tableaux avec PyMuPDF. `src/rag/retriever.py` transforme ces sorties en passages LlamaIndex, conserve les numéros de page et métadonnées, décrit les figures, puis stocke les passages BGE-M3 et les vecteurs CLIP dans deux collections PGVector distinctes. BGE-M3 fournit la recherche sémantique multilingue sur les textes; CLIP relie les descriptions et les figures pour la recherche visuelle.

Le flux de question-réponse est : Streamlit → agent LangChain → outils de recherche LlamaIndex/PGVector (BGE-M3 pour texte, CLIP pour images) → Qwen3 via Ollama. Une photo facultative est décrite par Qwen2.5-VL et comparée aux figures indexées avec CLIP. Les réponses doivent citer le fichier et la page; les résultats affichent aussi les extraits et les figures retrouvées.

Le modèle CLIP configuré est `openai/clip-vit-base-patch32`, déjà présent dans le cache local vérifié; il sert aux figures. BGE-M3 est servi localement par Ollama pour les textes. L’OCR des pages scannées, reranking/fine-tuning, outils MCP, historique d’équipement, tickets avec approbation, bot et déploiement multi-service restent des jalons ultérieurs de la feuille de route.

## Préparation Windows / PowerShell

Depuis la racine `C:\Users\PC\Desktop\MAINTiDOC` :

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose up -d pgvector
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

L’API Ollama doit être active sur `http://127.0.0.1:11434`. Les modèles `qwen2.5vl:3b` et `qwen3:4b` ont déjà été validés et sont installés dans l’environnement observé. Si nécessaire :

```powershell
ollama pull qwen2.5vl:3b
ollama pull qwen3:4b
```

Le premier chargement du modèle CLIP exige que son dépôt soit déjà dans le cache Hugging Face ; l’application ne télécharge pas de poids au démarrage. Le poste analysé possède le cache `openai/clip-vit-base-patch32`.

## Utilisation

1. Dans la barre latérale, téléverser un PDF et choisir **Extraire et indexer**.
2. Poser une question dans le champ de chat. Les passages, tableaux et descriptions de figures sont recherchés dans PGVector.
3. Pour une question liée à une photo, joindre une image dans la barre latérale. L’application la décrit avec Qwen2.5-VL et recherche les figures proches avec CLIP.
4. Ouvrir **Sources documentaires** sous la réponse pour vérifier les pages et les extraits utilisés.

Les PDF téléversés sont conservés sous `data/documents/`. Les résultats d’extraction restent dans `data/extracted/`, et les manifestes dans `data/processed/`. L’index pgvector utilise le volume Docker `maintidoc_pgdata` et survit aux redémarrages. Ne pas supprimer ce volume pour arrêter le service.

## Ingestion en ligne de commande

Depuis PowerShell, se placer dans `C:\Users\PC\Desktop\MAINTiDOC`. La commande d’extraction seule écrit les sorties documentaires, sans appeler les modèles ni PostgreSQL :

```powershell
.\.venv\Scripts\python.exe -m src.ingestion.pdf_loader ".\attention is all you need.pdf"
```

Pour extraire et indexer un document dans les deux collections PGVector, démarrer d’abord Ollama et PostgreSQL, puis exécuter :

```powershell
docker compose up -d pgvector
.\.venv\Scripts\python.exe -c "from src.rag.retriever import index_pdf; m=index_pdf(r'.\attention is all you need.pdf'); print({k:m.get(k) for k in ['document_id','page_count','text_node_count','image_node_count','indexed_node_count']}); print('warnings=',m.get('warnings',[]))"
```

Pour poser une question de contrôle à l’agent local et afficher les sources :

```powershell
.\.venv\Scripts\python.exe -c "from src.agents.rag_agent import answer_question; r=answer_question('Comment le produit scalaire mis à l’échelle est-il calculé ?'); print(r['answer']); print([(s['source_file'],s['page_number'],s['content_type']) for s in r['sources'][:6]]); print('agent_warning=',r['agent_warning'])"
```

L’interface Streamlit réalise le même flux avec **Extraire et indexer**. Le manifeste d’ingestion et les index gardent les numéros de page et les métadonnées de source.

## Dépendances

Les exigences ciblent Python 3.12. PyTorch et Transformers sont déjà installés dans la venv observée; la liste de dépendances les conserve pour une installation reproductible. Le téléchargement d’un modèle Hugging Face peut être nécessaire sur une nouvelle machine.

## Contrôles observés

- Extraction PyMuPDF de `attention is all you need.pdf` : 11 pages, 36 247 caractères, 3 images et 3 tableaux (pages 6, 8 et 9), sans avertissement.
- Indexation intégrée : 216 passages texte/tableau avec BGE-M3 (vecteurs de 1 024 dimensions) et 3 figures avec CLIP (vecteurs de 512 dimensions), répartis entre `maintidoc_text` et `maintidoc_images`; 219 nœuds au total, sans avertissement.
- Recherche et génération : la question de contrôle sur le produit scalaire mis à l’échelle retrouve les pages 3 et 4. Qwen3 répond avec la formule `Q · K / √dk`, cite la page 4 et l’agent LangChain ne signale aucun repli.
- PostgreSQL/pgvector : conteneur `maintidoc-pgvector` actif et sain, avec le volume persistant `maintidoc_maintidoc_pgdata`.
- Ollama local : `bge-m3`, `qwen2.5vl:3b` et `qwen3:4b` présents. La description des 3 figures est conservée dans `captions.json`.

## Limites et sécurité

Les réponses ne constituent ni un diagnostic garanti ni une consigne de pilotage de machine. Les documents et images téléversés sont des données non fiables; l’agent n’a pas d’outil d’écriture ni d’accès à un système de maintenance. La création de tickets, les actions MCP et les intégrations Telegram/Discord ne font pas partie de cet incrément.
