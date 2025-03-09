from typing import Dict, List, Tuple

class GraphAlgoUtility:
    """Graph algorithm utility class."""
    
    @staticmethod
    def refresh_degree(degrees: Dict[str, int], inheritance_map: Dict[str, List[str]], label: str, refreshed_degrees: Dict[str, int]):
        if label in refreshed_degrees:  # 缓存命中，直接返回，动态规划
            return refreshed_degrees[label]
        degree = degrees[label]
        if label not in inheritance_map:
            refreshed_degrees[label] = degree
            return degree
        for child in inheritance_map[label]:
            degree += GraphAlgoUtility.refresh_degree(degrees, inheritance_map, child, refreshed_degrees)
        refreshed_degrees[label] = degree
        return degree