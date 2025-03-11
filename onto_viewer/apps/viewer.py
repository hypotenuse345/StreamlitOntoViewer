import streamlit as st
from streamlit_extras.grid import grid as st_grid
from streamlit_echarts import st_echarts

from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, List, Dict, Any, Literal, Union

import pandas as pd
import rdflib
from rdflib import RDF, RDFS, OWL, SKOS
import os

from ..utils import EchartsUtility, GraphAlgoUtility

from .base import StreamlitBaseApp

class OntoViewerApp(StreamlitBaseApp):
    """Ontology Viewer App"""

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
                    if superclass_label.startswith("_:"):
                        continue
                    echarts_graph_info["nodes"].append({
                        "id": superclass_label, "name": superclass_label, "category": 0
                    })
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(class_label, superclass_label, "rdfs:subClassOf"))

            st_echarts(EchartsUtility.create_normal_echart_options(echarts_graph_info, class_label), height="400px")
    
        grid = st_grid([2, 1, 1])
        main_col, graph_col, info_col = grid.container(), grid.container(), grid.container()
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
            with graph_col:
                selected_iri = values[event.selection["rows"][0]]
                render_selected_class_echarts(self.ontology_graph, selected_iri)
            self.graph_status_subpage_display_metadata(selected_iri, info_col) 
    
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
            nodes_instantiated = [prop_label]
            echarts_graph_info["nodes"].append({
                "id": prop_label, "name": prop_label, "category": 0})

            # 子属性
            subprops = ontology_graph.subjects(RDFS.subPropertyOf, prop_iri, unique=True)
            if subprops:
                for subprop in subprops:
                    subprop_label = subprop.n3(ontology_graph.namespace_manager)
                    if subprop_label not in nodes_instantiated:
                        echarts_graph_info["nodes"].append({
                            "id": subprop_label, "name": subprop_label, "category": 0})
                        nodes_instantiated.append(subprop_label)
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(subprop_label, prop_label, "rdfs:subPropertyOf"))
                    
            # 父属性
            superprops = ontology_graph.objects(prop_iri, RDFS.subPropertyOf, unique=True)
            if superprops:
                for superprop in superprops:
                    superprop_label = superprop.n3(ontology_graph.namespace_manager)
                    if superprop_label.startswith("_:"):
                        continue
                    if superprop_label not in nodes_instantiated:
                        echarts_graph_info["nodes"].append({
                            "id": superprop_label, "name": superprop_label, "category": 0})
                        nodes_instantiated.append(superprop_label)
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(prop_label, superprop_label, "rdfs:subPropertyOf"))
            # echarts_graph_info["label"] = 
            
            # owl:inverseOf
            inverse_of = ontology_graph.objects(prop_iri, OWL.inverseOf, unique=True)
            # inverse_of_re = ontology_graph.subjects(OWL.inverseOf, prop_iri, unique=True)
            if inverse_of:
                for inverse_prop in inverse_of:
                    inverse_prop_label = inverse_prop.n3(ontology_graph.namespace_manager)
                    if inverse_prop_label not in nodes_instantiated:
                        echarts_graph_info["nodes"].append({
                            "id": inverse_prop_label, "name": inverse_prop_label, "category": 0})
                        nodes_instantiated.append(inverse_prop_label)
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(prop_label, inverse_prop_label, "owl:inverseOf", line_type="dashed", show_label=True, curveness=0.2))
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(inverse_prop_label, prop_label, "owl:inverseOf", line_type="dashed", show_label=True, curveness=0.2))
            
            # rdfs:domain
            domains = ontology_graph.objects(prop_iri, RDFS.domain, unique=True)
            if domains:
                for domain in domains:
                    domain_label = domain.n3(ontology_graph.namespace_manager)
                    if domain_label not in nodes_instantiated:
                        echarts_graph_info["nodes"].append({
                            "id": domain_label, "name": domain_label, "category": 1})
                        nodes_instantiated.append(domain_label)
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(prop_label, domain_label, "rdfs:domain", line_type="dashed", show_label=True))
                    
            # rdfs:range
            ranges = ontology_graph.objects(prop_iri, RDFS.range, unique=True)
            if ranges:
                for range in ranges:
                    range_label = range.n3(ontology_graph.namespace_manager)
                    if range_label not in nodes_instantiated:
                        echarts_graph_info["nodes"].append({
                            "id": range_label, "name": range_label, "category": 1})
                        nodes_instantiated.append(range_label)
                    echarts_graph_info["links"].append(EchartsUtility.create_normal_edge(prop_label, range_label, "rdfs:range", line_type="dashed", show_label=True))
            options = EchartsUtility.create_normal_echart_options(echarts_graph_info, prop_label)
            st_echarts(options, height="400px")
            # st.write(options)
        
        grid = st_grid([2, 1, 1])
        main_col, graph_col, info_col = grid.container(), grid.container(), grid.container()
        with main_col:
            properties = self.properties
            search_value = st.text_input("请输入查询关键词", key="search_props")
            props_to_df = {"Namespace":[], "LocalName":[], "PropType":[], "URIRef":[]}
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
            with graph_col:
                selected_iri = props_to_df["URIRef"][event.selection["rows"][0]]
                render_selected_prop_echarts(self.ontology_graph, selected_iri)
            self.graph_status_subpage_display_metadata(selected_iri, info_col)
    
    @st.fragment
    def graph_status_subpage_render_instances(self):
        grid = st_grid([2, 1])
        main_col, info_col = grid.container(), grid.container()
        type_list = self.classes_with_individuals
        type_local_names = [t.n3(self.ontology_graph.namespace_manager) for t in type_list]
        type_map = {tloc:tiri for tiri, tloc in zip(type_list, type_local_names)}
        
        with main_col:
            search_class = st.selectbox("请选择需要查询的类", type_local_names, key="search_class")
            instances = self.ontology_graph.subjects(RDF.type, type_map[search_class])
            instances_df = {"Namespace":[], "LocalName":[], "URIRef":[]}
            for instance in instances:
                instance_label = instance.n3(self.ontology_graph.namespace_manager)
                instances_df["Namespace"].append(instance_label.split(":")[0])
                instances_df["LocalName"].append(instance_label)
                instances_df["URIRef"].append(instance)
        
            event = st.dataframe(
                instances_df,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
        if event.selection["rows"]:
            with info_col:
                selected_iri = instances_df["URIRef"][event.selection["rows"][0]]
                self.graph_status_subpage_display_metadata(selected_iri, info_col)
        
    def _graph_status_subpage_get_inheritance_map(self, echarts_graph_info, predicate: rdflib.URIRef, obj_range: List[str]):
        import numpy as np
        inheritance_map = {}
        degrees = {}
        pred_label = predicate.n3(self.ontology_graph.namespace_manager)
        obj_range_copy = set(obj_range.copy())
        for s, o in self.ontology_graph.subject_objects(predicate=predicate, unique=True):
            # 将RDF对象转换为缩写
            s_label = s.n3(self.ontology_graph.namespace_manager)
            o_label = o.n3(self.ontology_graph.namespace_manager)
            if o not in obj_range:
                if o_label.startswith("_:"):
                    continue
                else:
                    obj_range_copy.add(o)
            
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
        
        nodes_initiated = set()
        for i, clss in enumerate(obj_range_copy):
            s_label = clss.n3(self.ontology_graph.namespace_manager)
            if s_label in nodes_initiated:
                continue
            nodes_initiated.add(s_label)
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
    
    def graph_status_subpage_render_class_hierarchy(self, option_to_label_visualization: bool=False):
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
            EchartsUtility.create_normal_echart_options(echarts_graph_info, f"Class Hierarchy\n\nTotal:{len(type_list)}", label_visible=option_to_label_visualization), 
            height="500px",
            events={
                "click": "function(params) { return params.value }",
            }
        )
        return s
    
    def graph_status_subpage_render_property_hierarchy(self, option_to_label_visualization: bool=False):
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
        
        options = EchartsUtility.create_normal_echart_options(echarts_graph_info, f"Property Hierarchy\n\nTotal:{len(props_to_df['URIRef'])}", label_visible=option_to_label_visualization)
        # st.write(options)
        s = st_echarts(
            options=options,
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
        
        def get_classes_having_instances(g: rdflib.Graph):
            return [rec["class"] for rec in g.query("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            SELECT DISTINCT ?class WHERE {
                ?instance rdf:type ?class .
                FILTER (?class != owl:ObjectProperty && ?class != owl:DatatypeProperty && ?class != owl:Class && ?class != owl:AnnotationProperty && ?class != rdfs:Class && ?class != rdf:Property && ?class != owl:Restriction && ?class != owl:Ontology)
            }
            """)]
        
        def show_more(obj, g: rdflib.Graph):
            if isinstance(obj, rdflib.Literal):
                return obj
            elif isinstance(obj, rdflib.URIRef):
                o_label = obj.n3(g.namespace_manager)
                return f"[{o_label}]({obj})"
            # metadata = "[\n\n"
            metadata = ""
            for p, o in g.predicate_objects(subject=obj):
                p_label = p.n3(g.namespace_manager)
                if p_label == "rdf:first" or p_label == "rdf:rest":
                    if o == RDF.nil:
                        continue
                    metadata += f"\n\n{show_more(o, g)}\n\n\n\n"
                else:
                    metadata += f"&emsp;&emsp;**{p.n3(g.namespace_manager)}**: {show_more(o, g)};"
            # metadata += "]\n\n"
            return metadata
        
        @st.cache_resource
        def get_metadata_of_ontology(triple_count: int, _g: rdflib.Graph):
            ontos = list(_g.subjects(predicate=RDF.type, object=OWL.Ontology, unique=True))
            if len(ontos) == 0:
                return None
            ont = ontos[0]
            metadata = ""
            metadata += f"**Ontology IRI**: {ont.n3(_g.namespace_manager)}\n\n"
            for p, o in _g.predicate_objects(subject=ont):
                metadata += f"&emsp;**{p.n3(_g.namespace_manager)}**: {show_more(o, _g)}\n\n"
            return metadata
        
        delta = len(self.ontology_graph) - st.session_state.triple_count
            
        st.session_state.triple_count = len(self.ontology_graph)
        with container.container():
            st.header(f"🕸️ {st.session_state.ontology_filename} 📊")
            with st.popover("元数据", use_container_width=True):
                    metadata = get_metadata_of_ontology(st.session_state.triple_count, self.ontology_graph)
                    if metadata is not None:
                        st.markdown(metadata)
                    else:
                        st.markdown("未找到元数据")
                    
        with container.container(border=True):
            grid = st_grid([1, 1],[1, 1],[1, 1])
            
            
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
            
            self._classes_with_individuals = get_classes_having_instances(self.ontology_graph)
            grid.metric(label="有实例的类型数量", value=len(self.classes_with_individuals))

            self.export_ontology_widget(container)
    
    def graph_status_subpage_display_metadata(self, node_iri, container):
        node_iri = rdflib.URIRef(node_iri)
        metadata = ""
        metadata += f"**IRI:** {node_iri}\n\n"
        metadata += f"**Namespace:** {node_iri.n3(self.ontology_graph.namespace_manager).split(':')[0]}\n\n"
        labels = self.ontology_graph.objects(node_iri, RDFS.label)
        try:
            labels = sorted(list(labels), key=lambda x: x.language)
        except:
            pass
        
        for label in labels:
            # st.markdown(f"**Label ({label.language if label.language else 'en'}):** {label}")
            metadata += f"**Label ({label.language if label.language else 'en'}):** {label}\n\n"
        comments = list(self.ontology_graph.objects(node_iri, RDFS.comment, unique=True))
        for comment in comments:
            # st.markdown(f"**Comment:** {comment}")
            metadata += f"**Comment:** {comment}\n\n"
        definitions = list(self.ontology_graph.objects(node_iri, SKOS.definition, unique=True))
        for definition in definitions:
            # st.markdown(f"**Definition:** {definition}")
            metadata += f"**Definition:** {definition}\n\n"
        general_concepts = list(self.ontology_graph.objects(node_iri, SKOS.broader, unique=True))    
        for general_concept in general_concepts:
            # st.markdown(f"**General Concept:** {general_concept}")
            metadata += f"**General Concept:** {general_concept.n3(self.ontology_graph.namespace_manager)}\n\n"
        specific_concepts = list(self.ontology_graph.subjects(SKOS.broader, node_iri, unique=True))
        for specific_concept in specific_concepts:
            # st.markdown(f"**Specific Concept:** {specific_concept}")
            metadata += f"**Specific Concept:** {specific_concept.n3(self.ontology_graph.namespace_manager)}\n\n"
        
        # 若当前节点是为属性，则进一步考虑owl约束
        if node_iri in self.properties["ObjectProperty"]:
            is_asymmetric = self.ontology_graph.query(
                f"ASK {{<{node_iri}> a owl:AsymmetricProperty.}}"
            )
            if is_asymmetric.askAnswer:
                # st.markdown(f"**Asymmetric:** True")
                metadata += f"**Asymmetric:** True\n\n"
            is_reflexive = self.ontology_graph.query(
                f"ASK {{<{node_iri}> a owl:ReflexiveProperty.}}"
            )
            if is_reflexive.askAnswer:
                # st.markdown(f"**Reflexive:** True")
                metadata += f"**Reflexive:** True\n\n"
            is_irreflexive = self.ontology_graph.query(
                f"ASK {{<{node_iri}> a owl:IrreflexiveProperty.}}"
            )
            if is_irreflexive.askAnswer:
                # st.markdown(f"**Irreflexive:** True")
                metadata += f"**Irreflexive:** True\n\n"
            is_symmetric = self.ontology_graph.query(
                f"ASK {{<{node_iri}> a owl:SymmetricProperty.}}"
            )
            if is_symmetric.askAnswer:
                # st.markdown(f"**Symmetric:** True")
                metadata += f"**Symmetric:** True\n\n"
            is_transitive = self.ontology_graph.query(
                f"ASK {{<{node_iri}> a owl:TransitiveProperty.}}"
            )
            if is_transitive.askAnswer:
                # st.markdown(f"**Transitive:** True")
                metadata += f"**Transitive:** True\n\n"
        # response_placeholder = st.empty()
        
        with container.container():
            st.write(f"**{node_iri.n3(self.ontology_graph.namespace_manager)}** \n\n")
            with st.container(height=500, border=False):
                st.markdown(metadata)
        
    def parse_dul_owl_widget(self, container):
        with container:
            DUL_File = st.file_uploader("Upload Ontology File", type=["ttl", "rdf", "owl"])
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
                g.parse(data=file_content, format=format)
                g.bind("dul", rdflib.Namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"))
                # 将RDF图对象存储在session_state中
                st.session_state["ontology_graph"] = g
                st.session_state["ontology_filename"] = os.path.splitext(DUL_File.name)[0]
                st.rerun()
            resource_list = os.listdir("./resources/ontologies")
            selected = st.selectbox("Default Ontologies", resource_list, label_visibility="collapsed")
            if st.button("Test with default ontology", use_container_width=True):
                g = rdflib.Graph()
                ext = os.path.splitext(selected)[1]
                if ext == ".ttl":
                    format = "turtle"
                elif ext == ".rdf":
                    format = "xml"
                elif ext == ".owl":
                    format = "xml"
                g.parse(f"./resources/ontologies/{selected}", format=format)
                g.bind("dul", rdflib.Namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"))
                st.session_state["ontology_graph"] = g
                st.session_state["ontology_filename"] = os.path.splitext(selected)[0]
                st.rerun()
        
    def export_ontology_widget(self, container):
        with container:
            out_dir = st.text_input("输出目录", value="./dbs/rdflib_graph")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            if st.button("另存为", use_container_width=True):
                with st.spinner('正在导出本体...', show_time=True):
                    self.ontology_graph.serialize(
                        destination=os.path.join(out_dir, "ontology.ttl"), 
                        format="turtle",
                        encoding="utf-8"
                    )
    
    @st.fragment
    def graph_status_subpage_visualization(self):
        grid = st_grid([5, 1])
        option_to_visualize = grid.selectbox("选择要可视化的内容", ["类继承关系", "属性继承关系"], label_visibility="collapsed")
        option_to_label_visualization = grid.checkbox("是否显示标签")
        
        grid = st_grid([2, 1])
        
        main_col, info_col = grid.container(), grid.container()
        with main_col:
            
            # if st.button("可视化", use_container_width=True):
            with st.spinner("正在生成图...", show_time=True):
                if option_to_visualize == "类继承关系":
                    selected_iri = self.graph_status_subpage_render_class_hierarchy(option_to_label_visualization)   # Echarts方式
                elif option_to_visualize == "属性继承关系":
                    selected_iri = self.graph_status_subpage_render_property_hierarchy(option_to_label_visualization)
                st.success("已生成图！")
        if selected_iri:
            self.graph_status_subpage_display_metadata(selected_iri, info_col)

    def graph_status_subpage_render_original_file(self):
        hide_file = st.checkbox("隐藏原文件", value=True)
        if not hide_file:
            self.display_rdf_data_widget(st.container(), self.ontology_graph)
    
    def graph_status_subpage(self):
        # 占位：边栏
        with st.sidebar:
            sidetab1, sidetab2 = st.tabs(["基本信息 📝", "开发者信息 👨‍💻"])
            
        self.graph_status_subpage_display_graph_basic_info_widget(sidetab1)
        self.display_creator_widget(sidetab2)
        with st.sidebar:
            if st.button("重置查看器 🔄", type="primary", use_container_width=True):
                st.session_state["ontology_graph"] = None
                st.rerun()
        # 占位： 主页面
        main_col = st.container()
        with main_col:
            maintab1, maintab2, maintab3, maintab4, maintab5, maintab6 = st.tabs([
                "本体可视化 🌐", 
                "命名空间 📚", 
                "类 🏷️", 
                "属性 🔗", 
                "实例 📦", 
                "原文件内容 📄"
            ])
            
        with maintab1.container():
            self.graph_status_subpage_visualization()
        
        with maintab2.container():
            self.graph_status_subpage_render_namespaces()
        
        with maintab3.container():
            self.graph_status_subpage_render_classes()
            
        with maintab4.container():
            self.graph_status_subpage_render_properties()
        
        with maintab5.container():
            self.graph_status_subpage_render_instances()
        
        with maintab6.container():
            self.graph_status_subpage_render_original_file()

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

    _classes_with_individuals: List[str] = PrivateAttr(default_factory=list)

    @property
    def classes_with_individuals(self) -> List[str]:
        return self._classes_with_individuals

    _properties: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    @property
    def properties(self) -> Dict[str, List[str]]:
        return self._properties
    
    def run(self):
        if st.session_state.get("ontology_graph", None) is None:
            self.parse_dul_owl_widget(st.sidebar)
                    
        if st.session_state.get("ontology_graph", None) is None:
            st.warning("Please upload the ontology file or test with the default.")
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
