import streamlit as st
from streamlit_extras.grid import grid as st_grid
import os
import time
import pandas as pd
# import asyncio
import json

from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from typing import List, Dict, Annotated, Any, Tuple, Optional
from pydantic import BaseModel, Field, PrivateAttr, computed_field
from .base import StreamlitBaseApp

class RDFQueryApp(StreamlitBaseApp):
    """RDF Query App"""
    _query_history: StreamlitChatMessageHistory = PrivateAttr(default=None)
    
    @property
    def query_history(self) -> StreamlitChatMessageHistory:
        if self._query_history is None:
            self._query_history = self._initialize_history("query_history")
        return self._query_history
    
    def add_query_to_history(self, container, natural_language_query: str, sparql_query: str = None, sparql_query_results: pd.DataFrame = None):
        with container.container():
            # 检查会话状态中是否存在SPARQL查询和查询结果
            if sparql_query is not None and sparql_query_results is not None:
                # 将用户的消息添加到查询历史中
                # 用户的消息包括自然语言查询和SPARQL查询，使用Markdown格式展示SPARQL查询
                self.query_history.add_user_message(
                    HumanMessage('{}\n\n```sparql\n{}\n```'.format(natural_language_query, sparql_query))
                )
                # 将AI的消息添加到查询历史中
                # AI的消息是查询结果的JSON格式字符串，使用JSON格式化工具进行缩进
                self.query_history.add_ai_message(
                    AIMessage("{}".format(sparql_query_results.to_json(indent=4)))
                )
                st.info("Query added to history! 📝")
            else:
                st.warning("No query or result to add to history. ⚠️")
    
    def save_query_history(self):
        import datetime  # 导入datetime模块用于获取当前日期和时间
        import os  # 导入os模块用于文件和目录操作
        import json  # 导入json模块用于处理JSON数据

        # Get the current date and time
        now = datetime.datetime.now()

        # Format the date and time as a string
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        out_dir = "./query_history"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, f"query_history_{timestamp}.json")

        # Save the history to a JSON file
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([{"type": message.type, "data": message.model_dump()} for message in self.query_history.messages], f, ensure_ascii=False, indent=4)
            
        st.session_state["logger"].info(f"History saved to {out_file}")
    
    def run_sparql_query_widget(self, g, query_str):
        # 尝试执行SPARQL查询
        try:
            # 打印查询字符串
            print(query_str)
            # 使用图对象g执行SPARQL查询
            results = g.query(query_str)
            # 显示查询成功的信息
            st.success("Query executed successfully! 🎉")
            # 如果查询有结果
            if results:
                # 显示查询结果的提示信息
                st.write("Here is the result of the query:")
                # 将查询结果转换为DataFrame，列名为结果变量的名称
                df = pd.DataFrame(results, columns=[str(kk) for kk in results.vars])
                # 显示查询结果
                st.write(df)
                # 将查询结果存储在会话状态中
                st.session_state["sparql_query_results"] = df
            else:
                # 如果查询没有结果，显示警告信息
                st.warning("No results found for the query. ⚠️")
                # 将会话状态中的查询结果设置为None
                st.session_state["sparql_query_results"] = None
        # 捕获任何异常
        except Exception as e:
            # 显示查询执行错误的警告信息
            st.error(f"Error executing query: {e} ❌")
            # 显示无效SPARQL查询的警告信息
            st.warning("Invalid SPARQL query. Please try again. ⚠️")
            # 将会话状态中的查询结果设置为None
            st.session_state["sparql_query_results"] = None

    def sparql_query_history_editor_widget(self, container, natural_language_query: str):
        import datetime
        # Get the current date and time
        now = datetime.datetime.now()
        # Format the date and time as a string
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        with container:
            grid = st_grid([1,1,1])
            grid.button("写入查询历史", 
                    on_click=self.add_query_to_history, 
                    kwargs={"container": container,
                            "natural_language_query": natural_language_query,
                            "sparql_query": st.session_state.get("sparql_query", None),
                            "sparql_query_results": st.session_state.get("sparql_query_results", None)}, 
                    use_container_width=True)
            # st.button("保存查询历史", on_click=self.save_query_history, use_container_width=True)
            grid.download_button(
                label="保存查询历史", 
                data=json.dumps([{"type": message.type, "data": message.model_dump()} 
                                 for message in self.query_history.messages], ensure_ascii=False, indent=4), 
                file_name=f"query_history_{timestamp}.json", mime="application/json", use_container_width=True)
            
            grid.button("清空查询历史", on_click=self.query_history.clear, use_container_width=True, type="primary")

    def sparql_query_history_container_widget(self, container):
        import json
        with container:
            # with st.popover("查询历史", icon="🗃️", use_container_width=True):
            with st.container(height=600, border=True):
                st.write("🗃️ 查询历史")
                for msg in self.query_history.messages:
                    with st.chat_message(msg.type):
                        if msg.type == "ai" or msg.type == "assistant":
                            try:
                                content = json.loads(msg.content)
                                st.dataframe(pd.DataFrame(content), use_container_width=True, hide_index=False)
                                # st.write("```json\n{}\n```".format(msg.content))
                            except json.JSONDecodeError:
                                st.error("Invalid JSON content")
                        else:
                            st.markdown(msg.content)
