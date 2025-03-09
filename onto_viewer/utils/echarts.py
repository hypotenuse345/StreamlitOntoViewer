from typing import Literal

class EchartsUtility:
    """Utility class for creating Echarts graphs"""
    @staticmethod
    def create_normal_edge(source, target, label: str, line_type: Literal["solid", "dashed", "dotted"] = "solid"):
        return {
            "source": source, "target": target, 
            "label": {"show": False, "formatter": label},
            "lineStyle": {
                "curveness": 0.1,
                "width": 2,
                "type": 'solid'
            },
            "symbol": ['none', 'arrow'], # 添加箭头
            "symbolSize": 10 # 箭头大小
        }
    
    @staticmethod
    def create_normal_echart_options(echarts_graph_info, title: str, label_visible: bool = True):
        options = {
            "title": {
                "text": title,
                "subtext": "Default layout",
                "top": "bottom",
                "left": "right",
            },
            "tooltip": {
                "trigger": 'item',
                "formatter": '<b>{b}</b>',
                "position": "top",
            },
            "legend": [{"data": [a["name"] for a in echarts_graph_info["categories"]]}],
            "series": [
                {
                    "name": title,
                    "type": "graph",
                    "layout": "force",
                    "data": echarts_graph_info["nodes"],
                    "links": echarts_graph_info["links"],
                    "categories": echarts_graph_info["categories"],
                    "roam": True,
                    "label": {
                        "show": label_visible,
                        "position": "bottom",
                        "formatter": "{b}"
                    },
                    "emphasis": {"focus": "adjacency", "lineStyle": {"width": 10}},
                    "force": {"repulsion": 100}
                }
            ]
        }
        return options
