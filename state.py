import operator
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated

# =====================================================================
# Sub-Schemas for Type Hinting Complex State Keys
# =====================================================================

class SubQueryDict(TypedDict):
    """Represents an atomic query extracted during the decomposition phase."""
    id: str
    text: str
    target_source_hint: Literal["SQL", "NoSQL", "Vector", "Unknown"]
    dependencies: List[str]

class DecompositionPackDict(TypedDict):
    """The structured payload containing entities, filters, and sub-queries."""
    strategy: Literal["SINGLE", "PARALLEL_SPLIT", "SEQUENTIAL_CHAIN"]
    paraphrased_global_query: str
    sub_queries: List[SubQueryDict]
    entities: Optional[List[str]]
    join_hints: Optional[Dict[str, Any]]

# =====================================================================
# Main State Schema definition for LangGraph
# =====================================================================

class RBIState(TypedDict):
    """
    Core State Graph for DRISHTI. 
    Passed and mutated through all L1/L2, execution, and synthesis nodes.
    """
    
    # --- P1: Gate & Session Context ---
    query: str  # Raw user input
    dept: str  # User department (e.g., DEPT_RISK, DEPT_FRAUD)
    session_id: str  # Session UUID for audit tracking
    is_valid: bool  # L1 gate result
    invalid_reason: Optional[str]  # Rejection reason from regex/structural checks
    error: Optional[str]  # General error message if a node fails

    # --- P2: Understanding & Classification ---
    decomposition: DecompositionPackDict  # Entities, filters, sub-queries
    paraphrases: List[str]  # Formal + keyword versions for vector search
    sources: List[str]  # Target sources: ["sql", "nosql", "vector"]
    query_type: str  # E.g., report, lookup, viz, multi_hop
    
    # --- P3: Planning ---
    execution_plan: List[str]  # Ordered DAG of tool calls

    # --- P4: Execution & Data Results ---
    sql_result: Optional[Dict[str, Any]]  # Rows + metadata from Postgres
    nosql_result: Optional[Dict[str, Any]]  # Docs + metadata from MongoDB
    vector_result: Optional[Dict[str, Any]]  # Chunks + scores from Chroma/VectorDB
    joined_result: Optional[Dict[str, Any]]  # Merged multi-hop data with provenance

    # --- P5: Quality Gate ---
    validation_status: Literal["PASS", "FAIL", "LOW_CONF", "PENDING"]  # Gate result
    confidence: float  # Value between 0.0 and 1.0

    # --- P6: Output & Audit ---
    viz_spec: Optional[Dict[str, Any]]  # Chart.js JSON specification
    final_response: str  # Synthesized answer streamed to UI
    
    # Annotated with operator.add so LangGraph appends to the list instead of overwriting.
    audit_log: Annotated[List[Dict[str, Any]], operator.add]  # Append-only trace