from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

# =====================================================================
# 1. Pydantic Schemas for Strict Output Contracts
# =====================================================================

class SubQuerySchema(BaseModel):
    id: str = Field(description="Unique identifier for the sub-query, e.g., 'sub_q_1'")
    text: str = Field(description="The standalone, isolated sub-query text.")
    target_source_hint: Literal["SQL", "NoSQL", "Vector", "Unknown"] = Field(
        description="Pre-classification hint indicating which storage layer likely holds this data."
    )
    dependencies: List[str] = Field(
        description="List of sub_q_ids that must complete execution *before* this sub-query can run. Empty list if independent."
    )

class DecompositionPackSchema(BaseModel):
    strategy: Literal["SINGLE", "PARALLEL_SPLIT", "SEQUENTIAL_CHAIN"] = Field(
        description="Execution routing strategy for the downstream planner."
    )
    paraphrased_global_query: str = Field(
        description="A cleaned, grammar-corrected, and unified version of the original user query that preserves specialized banking terms."
    )
    sub_queries: List[SubQuerySchema] = Field(
        description="The collection of broken-down atomic sub-queries."
    )