import streamlit as st
import logging

from typing import List, Dict, Annotated
from pydantic import BaseModel, Field, PrivateAttr, computed_field

import rdflib

class PersonInfo(BaseModel):
    name: str = Field(alias="name")
    email: str = Field(alias="emailAddress")
    last_name: str = Field(alias="familyName")
    first_name: str = Field(alias="givenName")

class StreamlitBaseApp(BaseModel):
    """Base class for all apps."""
    config_path: Annotated[str, Field(default="config.toml", description="Path to the config file.")]
    secrets_config_path: Annotated[str, Field(default=".streamlit/secrets.toml", description="Path to the secrets config file.")]
    output_dir: Annotated[str, Field(default="./outputs", description="Path to the output directory.")]
    
    def display_creator_widget(self, container):
        def display_person_info_widget(person_info: PersonInfo):
            return f"""
                <div class="card">
                    <h3>{person_info.name}</h3>
                    <p><i class="fas fa-envelope"></i> <strong>邮箱:</strong> {person_info.email}</p>
                    <p><i class="fas fa-user"></i> <strong>姓:</strong> {person_info.last_name}</p>
                    <p><i class="fas fa-user"></i> <strong>名:</strong> {person_info.first_name}</p>
                </div>
                """
        with container:
            # 自定义CSS
            html_content = """
            <style>
            .card {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 20px;
                margin: 10px 0;
            }
            .card h3, .card p {
                margin: 0;
                padding: 0;
            }
            .card h3 {
                font-size: 1.5em;
                color: #343a40;
            }
            .card p {
                color: #6c757d;
            }
            </style>
            """
            html_content += '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">'
            
            person = PersonInfo(
                name="Zeyu Pan",
                emailAddress="panzeyu@sjtu.edu.cn",
                familyName="Pan",
                givenName="Zeyu",
            )
            html_content += display_person_info_widget(person)
            st.markdown(
                html_content,
                unsafe_allow_html=True,
            )
    
    def display_rdf_data_widget(self, container, graph: rdflib.Graph):
        def wrap_turtle_string(text: str):
            # 定义一个函数 wrap_turtle_string，接受一个字符串参数 text
            # 返回一个字符串，该字符串以 "```turtle" 开始，以 "```" 结束，中间包含传入的 text
            return f"""```turtle\n
        {text}\n"""
        with container:
            # 使用Streamlit的write函数输出提示信息，告知用户接下来将展示RDF数据
            # st.write("Here is the RDF data:")
            # 使用Streamlit的expander组件创建一个可折叠展开的区域，标题为"View RDF data"
            with st.expander("**RDF数据**"):
                # 将RDF图对象g序列化为Turtle格式的字符串，并通过wrap_turtle_string函数对其进行格式化
                # 然后使用Streamlit的write函数在展开区域中显示格式化后的Turtle字符串
                st.write(wrap_turtle_string(graph.serialize(format="turtle")))
  
    def model_post_init(self, __context):
        logging.info("[APP] " +f" New app of {self.__class__.__name__} started ")
        
    def run(self):
        raise NotImplementedError("Please implement the run method in your app class.")