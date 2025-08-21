# app/agents/agent.py

import logging
from typing import Dict
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.services.llm_service import llm_service
from app.agents.tools import tools as langchain_tools  # ← This imports the LIST
from langchain.agents import create_structured_chat_agent

logger = logging.getLogger(__name__)

class LangChainAgent:
    """LangChain agent with proper session handling."""
    
    def __init__(self):
        self.agent_executor = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools."""
        try:
            # System prompt with session placeholder
            system_prompt = """You are an expert market research analyst. 
            Use the appropriate tool based on the user's request.

            Available Tools: {tool_names}

            Tool Descriptions: {tools}

            Current Session: {session_id}"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            agent = create_structured_chat_agent(
                llm=llm_service.model,
                tools=langchain_tools,  # ← FIXED: langchain_tools IS the list
                prompt=prompt
            )

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,  # ← FIXED: langchain_tools IS the list
                verbose=True,
                handle_parsing_errors=True
            )

            logger.info("✅ LangChain agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def run_agent(self, query: str, session_id: str) -> str:
        """Run the agent with proper session context."""
        try:
            logger.info(f"Running agent for: '{query}' with session: {session_id}")
            
            result = self.agent_executor.invoke({
                "input": query,
                "session_id": session_id
            })
            
            logger.info("✅ Agent execution completed")
            return result["output"]
            
        except Exception as e:
            error_msg = f"Agent execution failed: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

# Global instance
langchain_agent = LangChainAgent()