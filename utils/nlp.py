import os
import json
import spacy
import asyncio
import hashlib
import logging
from google import generativeai as genai
from google.generativeai.types import GenerationConfig
from gliner import GLiNER
from state import RBIState
from utils.schema import DecompositionPackSchema
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# 1. Logger Setup
# =====================================================================
# Initialize a module-level logger
logger = logging.getLogger(__name__)

# Optional: Set a default level if the root logger isn't configured yet in app.py
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# =====================================================================
# 2. Local Environment & Memory Initialization
# =====================================================================
logger.info("Loading NLP models and Long-Term Memory...")

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
    logger.debug("SpaCy 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.info("SpaCy model not found. Downloading 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load Production-Grade GLiNER
try:
    gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1") 
    logger.debug("GLiNER model loaded successfully.")
except Exception as e:
    logger.warning(f"GLiNER failed to load. Falling back to SpaCy exclusively. Error: {e}")
    gliner_model = None

# Load Long-Term Domain Glossary
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOSSARY_PATH = os.path.join(BASE_DIR, "long_term_memory", "domain_glossary.json")

domain_terms = []
try:
    with open(GLOSSARY_PATH, 'r') as f:
        glossary_data = json.load(f)
        
        # Extract just the "term" values from the list of dictionaries
        domain_terms = [item["term"] for item in glossary_data if "term" in item]
        
    # Inject deterministic rules into SpaCy
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "RBI_TERM", "pattern": [{"LOWER": term.lower()}]} for term in domain_terms]
    ruler.add_patterns(patterns)
    logger.info(f"Successfully loaded {len(domain_terms)} glossary terms into NLP pipeline.")
except Exception as e:
    logger.error(f"Could not load domain_glossary.json. Expected at {GLOSSARY_PATH}. Error: {e}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_FALLBACK_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# =====================================================================
# 3. The LangGraph Node Function
# =====================================================================
async def decompose_and_paraphrase_node(state: RBIState) -> dict:
    original_query = state.get("query", "").strip()
    session_id = state.get("session_id", "unknown_session")
    
    logger.info(f"[Session: {session_id}] Starting decomposition node for query.")
    
    if not original_query:
        logger.warning(f"[Session: {session_id}] Empty query received. Short-circuiting.")
        return {
            "decomposition": {
                "strategy": "SINGLE",
                "paraphrased_global_query": "",
                "sub_queries": []
            }
        }

    def execute_nlp_pipeline() -> dict:
        # ---------------------------------------------------------
        # Step A: Local Extractive Layer (GLiNER + SpaCy Anchoring)
        # ---------------------------------------------------------
        extracted_entities = []
        doc = nlp(original_query)
        
        if gliner_model:
            labels = ["PERSON", "ORGANIZATION", "TIME", "MONEY", "EVENT", "PRODUCT", "LAW", "FINANCIAL_METRIC", "REGULATION", "LOCATION", "DATE"]
            extracted = gliner_model.predict_entities(original_query, labels=labels)
            extracted_entities = [ent["text"] for ent in extracted]
        
        # Combine GLiNER entities with SpaCy's deterministic glossary matches
        extracted_entities.extend([ent.text for ent in doc.ents if ent.text not in extracted_entities])
        
        logger.debug(f"[Session: {session_id}] Extracted anchoring entities: {extracted_entities}")
        
        # Structural check for splits
        conjunctions = [token.text for token in doc if token.pos_ == "CCONJ"]
        has_complex_structure = len(conjunctions) > 0 or len(doc) > 15

        # ---------------------------------------------------------
        # Step B: Build Domain-Aware Prompts
        # ---------------------------------------------------------
        glossary_context = f"CRITICAL: The following terms are strict RBI domains. DO NOT translate or expand them: {', '.join(domain_terms[:50])}" if domain_terms else ""
        
        system_instruction = (
            "You are the Core Decomposition Engine of DRISHTI, an RBI (Reserve Bank of India) banking supervision AI.\n"
            "Your task is to analyze user queries and break them down into distinct, structured logical sub-queries.\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER translate, expand, or alter Indian banking acronyms (e.g., PCA, NPA, CRAR, SLR, MOR, FEMA, NBFC).\n"
            "2. Keep those acronyms exactly as they are written by the user.\n"
            "3. Ensure that sub-queries requiring sequential processing specify their dependencies explicitly.\n"
            f"{glossary_context}"
        )

        user_prompt = f"""
        Analyze, paraphrase, and decompose the following regulatory query.
        
        Original User Query: "{original_query}"
        Locally Identified Entities (Anchor Words): {extracted_entities}
        Structural Complexity Hint: {"High" if has_complex_structure else "Low"}
        
        Instructions:
        - If the query requires checking multiple storage systems (e.g., checking circular compliance text AND looking up a bank's financial penalty metric), create a multi-part split.
        - Match sub-queries with realistic 'target_source_hint' values:
            * 'SQL' for tabular items like bank ratios, penalties, financial views.
            * 'NoSQL' for event logs, audit traces, raw JSON packets.
            * 'Vector' for search queries inside policy circulars, rulebooks, master directions.
        """

        # ---------------------------------------------------------
        # Step C: Controlled LLM Execution
        # ---------------------------------------------------------
        try:
            logger.debug(f"[Session: {session_id}] Sending decomposition prompt to Gemini API.")
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=DecompositionPackSchema,
                    temperature=0.1 
                )
            )
            
            validated_data = DecompositionPackSchema.model_validate_json(response.text)
            logger.info(f"[Session: {session_id}] Query successfully decomposed and validated.")
            return {"decomposition": validated_data.model_dump()}

        except Exception as e:
            logger.error(f"[Session: {session_id}] LLM Generation or Validation failed. Using safe fallback. Error: {e}", exc_info=True)
            return {
                "decomposition": {
                    "strategy": "SINGLE",
                    "paraphrased_global_query": original_query,
                    "sub_queries": [
                        {
                            "id": "sub_q_1",
                            "text": original_query,
                            "target_source_hint": "Unknown",
                            "dependencies": []
                        }
                    ]
                }
            }

    def check_query_cache() -> str:
        doc = nlp(original_query.lower())
        tokens = sorted([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        query_hash = hashlib.md5("_".join(tokens).encode()).hexdigest()
        logger.debug(f"[Session: {session_id}] Generated query hash: {query_hash}")
        return query_hash

    # =====================================================================
    # 4. Async Dispatch
    # =====================================================================
    logger.debug(f"[Session: {session_id}] Dispatching NLP pipeline tasks in parallel...")
    results = await asyncio.gather(
        asyncio.to_thread(execute_nlp_pipeline),
        asyncio.to_thread(check_query_cache)
    )
    
    decomp_result, query_hash = results
    
    logger.info(f"[Session: {session_id}] NLP Node execution complete.")
    return decomp_result