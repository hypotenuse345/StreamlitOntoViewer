import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_extras.grid import grid as st_grid
import streamlit_pydantic as sp
from streamlit_echarts import st_echarts

from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, List, Dict, Any, Literal, Union

import pandas as pd
import networkx as nx
import pyvis.network as net
import rdflib
from rdflib import RDF, RDFS, OWL
import ifcopenshell
import os
import logging
import re

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import cse_core.conceptual_defn
from cse_core.frontend.agents.langgraph_agents import AgentEnum, BaseAgent
import cse_core.conceptual_defn.information_entity
from cse_core.conceptual_defn.namespaces import bind_prefixes, DOLCE_ULTRA_LITE
from cse_core.conceptual_defn.ontology import define_ontology, ontology as DRSLEO_info

from cse_core.server.databases.graph_stores import OntotextGraphDBGraph, RDFLibGraphStore

from cse_core.utils import EchartsUtility, GraphAlgoUtility

from .base import StreamlitBaseApp, setup_logging

class BONTApp(StreamlitBaseApp):
    """Building Ontology Navigator"""
    app_name: str = "Building Ontology Navigator"
    app_description: str = "Building Ontology Navigator"
    app_icon: str = "🏗️"
    app_logo: str = "Not available yet"
    app_color: str = "#008080"
    app_version: str = "0.0.1"
    app_author: str = "Zeyu Pan"
    app_author_email: str = "panzeyu@sjtu.edu.cn"
    app_license: str = "MIT"
    
    #region Graph Status Page
    @st.fragment
    def graph_status_subpage_render_namespaces(self):
        namespaces = self.ontology_graph.namespaces()
        namespaces = {k: v for k, v in namespaces}
        search_value = st.text_input("请输入查询关键词")
        if search_value:
            namespaces = {k: v for k, v in namespaces.items() if search_value.lower() in k.lower() or search_value.lower() in v.lower()}
            
        # 渲染，使用st.columns
        st.dataframe(
            pd.DataFrame({"Prefix": namespaces.keys(), "Namespace": namespaces.values()}),
            use_container_width=True,
            hide_index=True,
            column_order=["Prefix", "Namespace"],
        )
    
    @st.fragment
    def graph_status_subpage_render_classes(self):
        def render_selected_class_echarts(ontology_graph, class_iri):
            class_iri = rdflib.URIRef(class_iri)
            class_label = class_iri.n3(ontology_graph.namespace_manager)
            echarts_graph_info = {}
            echarts_graph_info["nodes"] = []
            echarts_graph_info["links"] = []
            echarts_graph_info["categories"] = []
            echarts_graph_info["categories"].append({"name": "Class"})
            
            echarts_graph_info["nodes"].append({
                "id": class_label, "name": class_label, "category": 0})

            # 子类
            subclasses = ontology_graph.subjects(RDFS.subClassOf, class_iri, unique=True)
            
            # 添加节点和边
            if subclasses:
                for subclass in subclasses:
                    subclass_label = subclass.n3(ontology_graph.namespace_manager)
                    echarts_graph_info["nodes"].append({
                        "id": subclass_label, "name": subclass_label, "category": 0})
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(subclass_label, class_label, "rdfs:subClassOf"))
                    
            # 父类
            superclasses = ontology_graph.objects(class_iri, RDFS.subClassOf, unique=True)
            if superclasses:
                for superclass in superclasses:
                    superclass_label = superclass.n3(ontology_graph.namespace_manager)
                    echarts_graph_info["nodes"].append({
                        "id": superclass_label, "name": superclass_label, "category": 0
                    })
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(class_label, superclass_label, "rdfs:subClassOf"))

            st_echarts(EchartsUtility.create_normal_echart_options(echarts_graph_info, class_label), height="400px")
    
        grid = st_grid([2, 1])
        main_col, info_col = grid.container(), grid.container()
        with main_col:
            classes = self.classes

            classes = {rec.n3(self.ontology_graph.namespace_manager): rec for rec in classes if not rec.n3(self.ontology_graph.namespace_manager).startswith("_:")}
            search_value = st.text_input("请输入查询关键词", key="search_classes")
            if search_value:
                classes = {k: v for k, v in classes.items() if search_value.lower() in k.lower() or search_value.lower() in v.lower()}
                
            keys = list(classes.keys())
            values = list(classes.values())
            event = st.dataframe(
                {"Namespace": [kk.split(":")[0] for kk in keys], "LocalName": keys, "URIRef": values},
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
        if event.selection["rows"]:
            with info_col:
                render_selected_class_echarts(self.ontology_graph, values[event.selection["rows"][0]])
    
    @st.fragment
    def graph_status_subpage_render_properties(self):
        def render_selected_prop_echarts(ontology_graph, prop_iri):
            prop_iri = rdflib.URIRef(prop_iri)
            echarts_graph_info = {}
            echarts_graph_info["nodes"] = []
            echarts_graph_info["links"] = []
            echarts_graph_info["categories"] = []
            echarts_graph_info["categories"].append({"name": "Property"})
            echarts_graph_info["categories"].append({"name": "Class"})
            
            prop_label = prop_iri.n3(ontology_graph.namespace_manager)
            echarts_graph_info["nodes"].append({
                "id": prop_label, "name": prop_label, "category": 0})

            # 子属性
            subprops = ontology_graph.subjects(RDFS.subPropertyOf, prop_iri, unique=True)
            if subprops:
                for subprop in subprops:
                    subprop_label = subprop.n3(ontology_graph.namespace_manager)
                    echarts_graph_info["nodes"].append({
                        "id": subprop_label, "name": subprop_label, "category": 0})
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(subprop_label, prop_label, "rdfs:subPropertyOf"))
                    
            # 父属性
            superprops = ontology_graph.objects(prop_iri, RDFS.subPropertyOf, unique=True)
            if superprops:
                for superprop in superprops:
                    superprop_label = superprop.n3(ontology_graph.namespace_manager)
                    echarts_graph_info["nodes"].append({
                        "id": superprop_label, "name": superprop_label, "category": 0})
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(prop_label, superprop_label, "rdfs:subPropertyOf"))
            # echarts_graph_info["label"] = 
            
            st_echarts(EchartsUtility.create_normal_echart_options(echarts_graph_info, prop_label), height="400px")
        
        grid = st_grid([2, 1])
        main_col, info_col = grid.container(), grid.container()
        with main_col:
            properties = self.properties
            search_value = st.text_input("请输入查询关键词", key="search_props")
            props_to_df = {"Namespace":[], "PropType":[], "LocalName":[], "URIRef":[]}
            for prop_type in ["ObjectProperty", "DatatypeProperty", "AnnotationProperty"]:
                for prop in properties[prop_type]:
                    prop_label = prop.n3(self.ontology_graph.namespace_manager)
                    if search_value and (search_value.lower() not in prop.lower() and search_value.lower() not in prop_label.lower()):
                        continue
                    props_to_df["Namespace"].append(prop_label.split(":")[0])
                    props_to_df["PropType"].append(prop_type)
                    props_to_df["LocalName"].append(prop_label)
                    props_to_df["URIRef"].append(prop)
            event = st.dataframe(
                props_to_df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
        if event.selection["rows"]:
            with info_col:
                render_selected_prop_echarts(self.ontology_graph, props_to_df["URIRef"][event.selection["rows"][0]])
    
    def _graph_status_subpage_get_inheritance_map(self, echarts_graph_info, predicate: rdflib.URIRef, obj_range: List[str]):
        import numpy as np
        inheritance_map = {}
        degrees = {}
        pred_label = predicate.n3(self.ontology_graph.namespace_manager)
        for s, o in self.ontology_graph.subject_objects(predicate=predicate):
            if o not in obj_range:
                continue
            # 将RDF对象转换为缩写
            s_label = s.n3(self.ontology_graph.namespace_manager)
            o_label = o.n3(self.ontology_graph.namespace_manager)
            if o_label not in inheritance_map:
                inheritance_map[o_label] = []
            inheritance_map[o_label].append(s_label)
            if s_label not in degrees:
                degrees[s_label] = 1
            else:
                degrees[s_label] += 1
            if o_label not in degrees:
                degrees[o_label] = 1
            else:
                degrees[o_label] += 1
            # 在有向图中添加边，边的标签为谓词
            echarts_graph_info["links"].append(
                EchartsUtility.create_normal_edge(
                    s_label, o_label, 
                    label=pred_label
                )
            )
        
        refreshed_degrees = {}
        
        for label in degrees:
            GraphAlgoUtility.refresh_degree(degrees, inheritance_map, label, refreshed_degrees)    
        
        for i, clss in enumerate(obj_range):
            s_label = clss.n3(self.ontology_graph.namespace_manager)
            echarts_graph_info["nodes"].append({
                "id": s_label,
                "name": s_label,
                "category": 0,
                "symbol": 'circle',
                "symbolSize":10 + np.log(refreshed_degrees[s_label]) * 7 if s_label in refreshed_degrees else 10,
                # "symbolSize":[200, 20],
                "draggable": False,
                "value": clss
            })
    
    
    def graph_status_subpage_render_class_hierarchy(self):
        echarts_graph_info = {}
        echarts_graph_info["nodes"] = []
        echarts_graph_info["links"] = []
        echarts_graph_info["categories"] = []
        echarts_graph_info["categories"].append({"name": "Class"})
        
        # id_map = {}
        type_list = self.classes
        
        # id_map[RDFS.subClassOf.n3(self.ontology_graph.namespace_manager)] = RDFS.subClassOf
        self._graph_status_subpage_get_inheritance_map(echarts_graph_info, RDFS.subClassOf, type_list)
        
        # echarts_graph_info["label"] = 
        s = st_echarts(
            EchartsUtility.create_normal_echart_options(echarts_graph_info, f"Class Hierarchy\n\nTotal:{len(type_list)}", label_visible=False), 
            height="500px",
            events={
                "click": "function(params) { return params.value }",
            }
        )
        return s
    
    def graph_status_subpage_render_property_hierarchy(self):
        echarts_graph_info = {}
        echarts_graph_info["nodes"] = []
        echarts_graph_info["links"] = []
        echarts_graph_info["categories"] = []
        echarts_graph_info["categories"].append({"name": "Property"})
        
        properties = self.properties
        
        props_to_df = {"Namespace":[], "PropType":[], "LocalName":[], "URIRef":[]}
        for prop_type in ["ObjectProperty", "DatatypeProperty", "AnnotationProperty"]:
            for prop in properties[prop_type]:
                prop_label = prop.n3(self.ontology_graph.namespace_manager)
                props_to_df["Namespace"].append(prop_label.split(":")[0])
                props_to_df["PropType"].append(prop_type)
                props_to_df["LocalName"].append(prop_label)
                props_to_df["URIRef"].append(prop)
        self._graph_status_subpage_get_inheritance_map(echarts_graph_info, RDFS.subPropertyOf, props_to_df["URIRef"])
        
        s = st_echarts(
            EchartsUtility.create_normal_echart_options(echarts_graph_info, f"Property Hierarchy\n\nTotal:{len(props_to_df['URIRef'])}", label_visible=False), 
            height="500px",
            events={
                "click": "function(params) { return params.value }",
            }
        )
        return s
    
    def graph_status_subpage_display_graph_basic_info_widget(self, container):
        def get_properties(g: rdflib.Graph):
            property_dict = {}
            property_dict["ObjectProperty"] = [rec["property"] for rec in g.query("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT DISTINCT ?property WHERE {
                ?property rdf:type owl:ObjectProperty .
            }""")]
            property_dict["DatatypeProperty"] = [rec["property"] for rec in g.query("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT DISTINCT ?property WHERE {
                ?property rdf:type owl:DatatypeProperty .
            }""")]
            property_dict["AnnotationProperty"] = [rec["property"] for rec in g.query("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT DISTINCT ?property WHERE {
                ?property rdf:type owl:AnnotationProperty .
            }""")]
            return property_dict
        with container.container(border=True):
            grid = st_grid([1, 1],[1, 1],[1, 1])
            
            delta = len(self.ontology_graph) - st.session_state.triple_count
            
            st.session_state.triple_count = len(self.ontology_graph)
            grid.metric(label="三元组数量", value=st.session_state.triple_count, delta = delta)
        
            classes = list(self.ontology_graph.subjects(predicate=RDF.type, object=OWL.Class, unique=True))
            classes = [clss for clss in classes if not clss.n3(self.ontology_graph.namespace_manager).startswith("_:")]
            self._classes = classes
            grid.metric(label="类型数量", value=len(self.classes))
            properties = get_properties(self.ontology_graph)
            self._properties = properties
            grid.metric(label="对象属性数量", value=len(self.properties["ObjectProperty"]))
            grid.metric(label="数据属性数量", value=len(self.properties["DatatypeProperty"]))
            grid.metric(label="注释属性数量", value=len(self.properties["AnnotationProperty"]))
            grid.container()
    
    def graph_status_subpage_display_metadata(self, node_iri, container):
        import asyncio
        from langchain_core.documents import Document
        from langchain_core.messages import AIMessage
        async def render_results(agent: BaseAgent, documents: List[Document]):
            """Render results from the agent."""
            results = []
            async for wrapped_data in agent.astream(documents, None):
                content = ""
                if type(wrapped_data[1]).__name__ == "list":
                    for elem in wrapped_data[1]:
                        content += str(elem)+"\n\n"
                        if type(elem).__name__ == "tuple":
                            for sub_elem in elem:
                                st.write(sub_elem, unsafe_allow_html=True)
                        else:
                            st.write(elem, unsafe_allow_html=True)
                else:
                    content += str(wrapped_data[1])
                    st.write(wrapped_data[1])
                results.append(AIMessage(content=content))
            return results
        
        node_iri = rdflib.URIRef(node_iri)
        metadata = ""
        metadata += f"**Local Name:** {node_iri.n3(self.ontology_graph.namespace_manager)}\n\n"
        metadata += f"**IRI:** {node_iri}\n\n"
        metadata += f"**Namespace:** {node_iri.n3(self.ontology_graph.namespace_manager).split(':')[0]}\n\n"
        labels = sorted(list(self.ontology_graph.objects(node_iri, RDFS.label)), key=lambda x: x.language)
        for label in labels:
            # st.markdown(f"**Label ({label.language if label.language else 'en'}):** {label}")
            metadata += f"**Label ({label.language if label.language else 'en'}):** {label}\n\n"
        comments = list(self.ontology_graph.objects(node_iri, RDFS.comment))
        for comment in comments:
            # st.markdown(f"**Comment:** {comment}")
            metadata += f"**Comment:** {comment}\n\n"
        # response_placeholder = st.empty()
        with container.popover("元数据", use_container_width=True):
            st.markdown(metadata)
        if container.button("生成描述", use_container_width=True):
            with container.container(height=400):
                prompt_text = f"请根据以下元数据生成一个中文描述，尽量简约且结构化，富有设计感。不必解释渲染过程，只输出渲染文本。\n\nMetadata:{metadata}"
            
                with st.spinner("生成描述中...", show_time=True):
                    response = asyncio.run(render_results(self.chat_agent, [Document(page_content=prompt_text)]))
                # for response in response:
                #     st.markdown(response.content)
                
                # self.process_user_input(response_placeholder, metadata, self.chat_agent)
    
    def parse_dul_owl_widget(self, container):
        with container:
            DUL_File = st.file_uploader("Upload DUL File", type=["ttl", "rdf", "owl"])
            if DUL_File is not None:
                ext = os.path.splitext(DUL_File.name)[1]
                if ext == ".ttl":
                    format = "turtle"
                elif ext == ".rdf":
                    format = "xml"
                elif ext == ".owl":
                    format = "xml"
                # 将上传的文件读取为字符串
                file_content = DUL_File.read().decode("utf-8")
                # 使用rdflib库解析字符串为RDF图对象
                g = rdflib.Graph()
                define_ontology(g)
                g.parse(data=file_content, format=format)
                bind_prefixes(g)
                # 将RDF图对象存储在session_state中
                st.session_state["ontology_graph"] = g
                st.rerun()
                
    def import_drsleo_classes_widget(self, container):
        import inspect
        import pkgutil
        from cse_core.conceptual_defn.base import NamedIndividual
        import cse_core
        with container:
            if st.button("导入DRSLEO定义的类型", use_container_width=True):
                with st.spinner('正在导入DRSLEO定义的类型...', show_time=True):
                    # 遍历当前库中的类声明，将其添加到Graph中
                    inheritance_map = {}
                    def find_subclasses(package):
                        walked_modules = set()
                        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
                            if name in walked_modules:
                                continue
                            module = loader.find_module(name).load_module(name)
                            walked_modules.add(name)
                            
                            for name, obj in inspect.getmembers(module, inspect.isclass):
                                if NamedIndividual.__name__ in [ii.__name__ for ii in inspect.getmro(obj)] and obj is not NamedIndividual:
                                    super_class = [ii._type_iri_cls.default for ii in obj.__bases__][0]
                                    if super_class not in inheritance_map:
                                        inheritance_map[super_class] = set()
                                    inheritance_map[super_class].add(obj._type_iri_cls.default)
                    
                    find_subclasses(cse_core)
                    for acls, subclasses in inheritance_map.items():
                        for subclass in subclasses:
                            if subclass.startswith(DOLCE_ULTRA_LITE):
                                continue
                            self.ontology_graph.add((acls, RDF.type, OWL.Class))
                            self.ontology_graph.add((subclass, RDF.type, OWL.Class))
                            self.ontology_graph.add((subclass, RDFS.subClassOf, acls))
    
    def export_ontology_widget(self, container):
        with container:
            out_dir = st.text_input("输出目录", value="./dbs/rdflib_graph")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            if st.button("导出DRSLEO本体", use_container_width=True):
                with st.spinner('正在导出DRSLEO本体...', show_time=True):
                    self.ontology_graph.serialize(
                        destination=os.path.join(out_dir, "drsleo.ttl"), 
                        format="turtle",
                        encoding="utf-8"
                    )
    
    @st.fragment
    def graph_status_subpage_visualization(self):
        option_to_visualize = st.selectbox("选择要可视化的内容", ["类继承关系", "属性继承关系"])
        
        grid = st_grid([2, 1])
        
        main_col, info_col = grid.container(), grid.container()
        with main_col:
            
            # if st.button("可视化", use_container_width=True):
            with st.spinner("正在生成图...", show_time=True):
                if option_to_visualize == "类继承关系":
                    selected_iri = self.graph_status_subpage_render_class_hierarchy()   # Echarts方式
                elif option_to_visualize == "属性继承关系":
                    selected_iri = self.graph_status_subpage_render_property_hierarchy()
                st.success("已生成图！")
        if selected_iri:
            self.graph_status_subpage_display_metadata(selected_iri, info_col)
        
    def graph_status_subpage(self):
        # 占位：边栏
        with st.sidebar:
            sidetab1, sidetab2 = st.tabs(["基本信息", "开发者信息"])
        self.graph_status_subpage_display_graph_basic_info_widget(sidetab1)
        self.import_drsleo_classes_widget(sidetab1)
        
        self.display_creator_widget(sidetab2)
        # 占位： 主页面
        main_col = st.container()
        with main_col:
            maintab1, maintab2, maintab3, maintab4, maintab5, maintab6 = st.tabs(["本体可视化", "命名空间", "类", "属性", "日志", "测试"])
            
        st.session_state["streamlit_logger"] = setup_logging(maintab5)
        
        with maintab1.container():
            self.graph_status_subpage_visualization()
        
        with maintab2.container():
            self.graph_status_subpage_render_namespaces()
        
        with maintab3.container():
            self.graph_status_subpage_render_classes()
            
        with maintab4.container():
            self.graph_status_subpage_render_properties()
            
        with maintab6.container():
            self.test_widget()
    #endregion
    
    def test_widget(self):
        pass
    
    def model_post_init(self, __context):
        super().model_post_init(__context)
    
    _ontology_graph: Union[rdflib.Graph, rdflib.Dataset] = PrivateAttr(default=None)
    
    @property
    def ontology_graph(self) -> Union[rdflib.Graph, rdflib.Dataset]:
        return self._ontology_graph
    
    _classes: List[str] = PrivateAttr(default_factory=list)

    @property
    def classes(self) -> List[str]:
        return self._classes

    _properties: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    @property
    def properties(self) -> Dict[str, List[str]]:
        return self._properties
    
    @property
    def chat_agent(self) -> BaseAgent:
        return st.session_state["agents"][AgentEnum.CHAT.value]
    
    def run(self):
        if st.session_state.get("ontology_graph", None) is None:
            self.parse_dul_owl_widget(st.sidebar)
                    
        if st.session_state.get("ontology_graph", None) is None:
            st.warning("Please upload the DUL file.")
            st.stop()
            
        self._ontology_graph = st.session_state["ontology_graph"] 
        if st.session_state.get("triple_count") is None:
            st.session_state.triple_count = len(self.ontology_graph)
        
        with st.sidebar:
            subpage_option = st.selectbox("子页面导航", ["图谱状态"])
            
        if subpage_option == "图谱状态":
            self.graph_status_subpage()
        # elif subpage_option == "变更类":
        #     self.change_classes_widget()
        # elif subpage_option == "变更属性":
        #     self.change_properties_widget()
        # elif subpage_option == "变更日志":
        #     self.change_log_widget()
        
        self.export_ontology_widget(st.sidebar)