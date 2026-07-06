import re

def rewrite_query(original_query: str) -> list:
    """
    Expands the query deterministically using production-grade industrial 
    automation synonyms instead of relying on fragile local LLM generations.
    
    Args:
        original_query (str): The raw technical user question.
        
    Returns:
        list: Up to 3 high-quality unique expanded search trajectories.
    """
    expanded_queries = [original_query]
    query_lower = original_query.lower()
    
    # Core engineering synonym dictionary for predictive maintenance mapping
    # grease -> lubricant, lubrication specifications, grease type
    # spindle -> axis, shaft, joint mechanisms
    # scara -> robotic arm, manipulator arm
    
    # Track 1: Focus explicitly on lubricants and lubrication criteria
    if "grease" in query_lower:
        alt_1 = original_query.replace("grease", "lubricant specifications")
        expanded_queries.append(alt_1)
    elif "maintenance" in query_lower:
        alt_1 = original_query.replace("maintenance", "periodic inspection parameters")
        expanded_queries.append(alt_1)
        
    # Track 2: Focus on technical mechanical components and structural synonyms
    alt_2 = original_query
    if "spindle" in query_lower:
        alt_2 = alt_2.replace("spindle", "axis structure")
    if "scara" in query_lower:
        alt_2 = alt_2.replace("scara", "industrial robotic manipulator")
        
    if alt_2 != original_query:
        expanded_queries.append(alt_2)
        
    # Safeguard: Keep data stream unique and capped at maximum 3 clean trajectories
    return list(set(expanded_queries))[:3]