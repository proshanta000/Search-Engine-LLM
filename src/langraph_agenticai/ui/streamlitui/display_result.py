from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st

class DisplayResultStreamlit:
    def __init__(self, graph, user_message):
        self.graph = graph
        self.user_message = user_message

    def _extract_text(self, content):
        """Safely extracts string content even if it's a list (multimodal)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            # Common structure for multimodal: [{'type': 'text', 'text': '...'}]
            if isinstance(content[0], dict):
                return content[0].get("text", "")
        return ""

    def display_result_on_ui(self):
        # 1. Render History first
        for msg in st.session_state.chat_history:
            txt = self._extract_text(msg.content)
            if txt and not isinstance(msg, ToolMessage):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(txt)

        # 2. Handle the NEW message
        if self.user_message:
            # Save and show user message
            with st.chat_message("user"):
                st.markdown(self.user_message)
            
            user_msg = HumanMessage(content=self.user_message)
            st.session_state.chat_history.append(user_msg)

            # 3. Call the Graph
            with st.spinner("Searching and generating..."):
                config = {"configurable": {"thread_id": "current_session"}}
                inputs = {"messages": st.session_state.chat_history}
                
                result = self.graph.invoke(inputs, config)
                
                # Get only the new messages added by the graph
                new_msgs = result["messages"][len(st.session_state.chat_history):]
                
                for m in new_msgs:
                    st.session_state.chat_history.append(m)
                    txt = self._extract_text(m.content)
                    
                    # Display assistant response
                    if isinstance(m, AIMessage) and txt.strip():
                        # Filter out internal "loop" messages
                        if "previous response was cut off" not in txt.lower():
                            with st.chat_message("assistant"):
                                st.markdown(txt)