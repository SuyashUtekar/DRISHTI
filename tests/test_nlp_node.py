import asyncio
import time
import os

# Set fallback key for testing if not exported in your terminal
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "YOUR_ACTUAL_API_KEY_HERE"

from utils.nlp import decompose_and_paraphrase_node
from state import RBIState

# A diverse set of test queries designed to stress-test the new glossary anchoring
TEST_QUERIES = [
    # 1. Tests Domain Glossary Injection (Acronyms shouldn't be expanded/lost)
    "What is the impact of CRAR guidelines under the SARFAESI act for Urban Co-operative Banks?",
    
    # 2. Tests Structural Split (AND conjunction)
    "Which banks have an NPA greater than 10% and what are their CAMELS ratings?",
    
    # 3. Tests Multi-Hop Entity Extraction (SQL + NoSQL hints)
    "Show me the fraud reports and reporting delay penalties for Punjab National Bank.",
    
    # 4. Simple Lookup / Vector Search
    "Summarize the master circular on digital payment metrics."
]

async def run_tests():
    print("==================================================")
    print("🧪 DRISHTI: Testing NLP Node (with Domain Glossary)")
    print("==================================================\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"▶ Test Case {i}")
        print(f"  Query: '{query}'")
        
        # Construct the mock LangGraph state
        state: RBIState = {
            "query": query,
            "dept": "DEPT_RISK",
            "session_id": "test-session-001"
        }
        
        # Measure true async latency
        start_time = time.perf_counter()
        
        try:
            result = await decompose_and_paraphrase_node(state)
            elapsed_time = (time.perf_counter() - start_time) * 1000
            
            # Extract data safely for console printing
            decomp = result.get("decomposition", {})
            strategy = decomp.get("strategy", "UNKNOWN")
            entities = decomp.get("entities", [])
            sub_queries = decomp.get("sub_queries", [])
            paraphrases = result.get("paraphrases", [])

            print(f"  ⏱️  Latency: {elapsed_time:.2f}ms")
            print(f"  ⚙️  Strategy: {strategy}")
            print(f"  🧲  Anchored Entities: {entities}")
            
            print("  🧩  Sub-Queries:")
            for sq in sub_queries:
                print(f"      - [{sq.get('target_source_hint')}] {sq.get('text')}")
                
            print("  📝  Paraphrases:")
            for p in paraphrases:
                print(f"      - {p}")
                
        except Exception as e:
            print(f"  ❌ Error executing node: {e}")
            
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_tests())