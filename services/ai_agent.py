# Standard library imports
import json
import logging
import os
from typing import Dict, Optional

# Third-party imports
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LogisticsAIAgent:
    """AI Agent for Logistics chatbot using LangChain with FAQ, handbook, schedule, and get_help tools"""
    
    def __init__(self, chromadb_dir: str = "chroma_db", faq_collection: str = "faqs", handbook_collection: str = "handbook", schedule_collection: str = "schedules"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=self.openai_api_key
        )
        
        # Initialize ChromaDB for FAQ, handbook, and schedule search
        self.chromadb_dir = chromadb_dir
        self.faq_collection = faq_collection
        self.handbook_collection = handbook_collection
        self.schedule_collection = schedule_collection
        self.chroma_db_faqs = None
        self.chroma_db_handbook = None
        self.chroma_db_schedules = None
        self._initialize_chromadb()
        
        # Initialize ReAct agent with FAQ, handbook, schedule, and get_help tools
        self.react_agent = self._create_react_agent()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB connections for FAQ, handbook, and schedule search"""
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        
        # Initialize FAQ ChromaDB
        try:
            if os.path.exists(self.chromadb_dir):
                self.chroma_db_faqs = Chroma(
                    persist_directory=self.chromadb_dir,
                    embedding_function=embeddings,
                    collection_name=self.faq_collection
                )
                logger.info(f"✅ ChromaDB FAQ collection initialized: {self.faq_collection}")
            else:
                logger.warning(f"⚠️ ChromaDB directory not found: {self.chromadb_dir}. FAQ search will not be available.")
        except Exception as e:
            logger.warning(f"⚠️ Error initializing FAQ ChromaDB: {e}. FAQ search will not be available.")
        
        # Initialize Handbook ChromaDB
        try:
            if os.path.exists(self.chromadb_dir):
                self.chroma_db_handbook = Chroma(
                    persist_directory=self.chromadb_dir,
                    embedding_function=embeddings,
                    collection_name=self.handbook_collection
                )
                logger.info(f"✅ ChromaDB Handbook collection initialized: {self.handbook_collection}")
            else:
                logger.warning(f"⚠️ ChromaDB directory not found: {self.chromadb_dir}. Handbook search will not be available.")
        except Exception as e:
            logger.warning(f"⚠️ Error initializing Handbook ChromaDB: {e}. Handbook search will not be available.")
        
        # Initialize Schedule ChromaDB
        try:
            if os.path.exists(self.chromadb_dir):
                self.chroma_db_schedules = Chroma(
                    persist_directory=self.chromadb_dir,
                    embedding_function=embeddings,
                    collection_name=self.schedule_collection
                )
                logger.info(f"✅ ChromaDB Schedule collection initialized: {self.schedule_collection}")
            else:
                logger.warning(f"⚠️ ChromaDB directory not found: {self.chromadb_dir}. Schedule search will not be available.")
        except Exception as e:
            logger.warning(f"⚠️ Error initializing Schedule ChromaDB: {e}. Schedule search will not be available.")
    
    def _create_react_agent(self):
        """Create ReAct agent with FAQ search, handbook search, schedule search, and get_help tools"""
        
        def search_faqs(question: str) -> str:
            """Search FAQs for relevant information about logistics operations, procedures, policies, and best practices.
            
            Use this tool to find answers to questions about:
            - Standard operating procedures (receiving, shipping, inventory management)
            - Safety procedures and requirements
            - Employee policies and procedures
            - Equipment operation and handling
            - Compliance and regulatory requirements
            - Quality assurance procedures
            - Emergency procedures
            - And other logistics-related topics
            
            Args:
                question: The user's question or query about logistics operations
                
            Returns:
                JSON string with relevant FAQs or error message
            """
            try:
                if not self.chroma_db_faqs:
                    return json.dumps({
                        "text_msg": "FAQ database is not available. Please contact support for assistance.",
                        "human_assistance_required": True
                    })
                
                # Perform similarity search
                results = self.chroma_db_faqs.similarity_search_with_score(question, k=3)
                logger.info(f"📊 Found {len(results)} relevant FAQs")
                
                if results:
                    # Format FAQs for response
                    faq_list = []
                    for doc, score in results:
                        metadata = doc.metadata
                        question_text = metadata.get("question", "")
                        answer_text = metadata.get("answer", "")
                        
                        if question_text and answer_text:
                            faq_list.append({
                                "question": question_text,
                                "answer": answer_text,
                                "similarity_score": float(score)
                            })
                    
                    if faq_list:
                        # Format FAQs as readable text
                        faq_text = "\n\n".join([
                            f"Q: {faq['question']}\nA: {faq['answer']}"
                            for faq in faq_list
                        ])
                        
                        return json.dumps({
                            "text_msg": f"Found relevant information:\n\n{faq_text}",
                            "human_assistance_required": False
                        })
                    else:
                        return json.dumps({
                            "text_msg": "No relevant FAQs found. Please contact support for assistance.",
                            "human_assistance_required": True
                        })
                else:
                    logger.warning("❌ No FAQs found")
                    return json.dumps({
                        "text_msg": "No relevant FAQs found. Please contact support for assistance.",
                        "human_assistance_required": True
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error in search_faqs: {str(e)}")
                return json.dumps({
                    "text_msg": f"Error searching FAQs: {str(e)}. Please contact support for assistance.",
                    "human_assistance_required": True
                })
        
        def search_handbook(question: str) -> str:
            """Search the logistics handbook for detailed information, procedures, guidelines, and best practices.
            
            Use this tool to find answers to questions about:
            - Detailed operational procedures and guidelines
            - Comprehensive safety protocols and requirements
            - Driving and transportation procedures
            - Manpower management guidelines
            - Detailed compliance and regulatory information
            - Best practices and standards
            - Training requirements and procedures
            - Equipment specifications and usage
            - And other detailed logistics information from the handbook
            
            Args:
                question: The user's question or query about logistics operations
                
            Returns:
                JSON string with relevant handbook content or error message
            """
            try:
                if not self.chroma_db_handbook:
                    return json.dumps({
                        "text_msg": "Handbook database is not available. Please contact support for assistance.",
                        "human_assistance_required": True
                    })
                
                # Perform similarity search
                results = self.chroma_db_handbook.similarity_search_with_score(question, k=3)
                logger.info(f"📚 Found {len(results)} relevant handbook sections")
                
                if results:
                    # Format handbook content for response
                    handbook_sections = []
                    for doc, score in results:
                        content = doc.page_content
                        metadata = doc.metadata
                        source = metadata.get("source", "handbook")
                        
                        if content:
                            handbook_sections.append({
                                "content": content,
                                "source": source,
                                "similarity_score": float(score)
                            })
                    
                    if handbook_sections:
                        # Format handbook content as readable text
                        handbook_text = "\n\n".join([
                            f"Section from {section['source']}:\n{section['content']}"
                            for section in handbook_sections
                        ])
                        
                        return json.dumps({
                            "text_msg": f"Found relevant information from the handbook:\n\n{handbook_text}",
                            "human_assistance_required": False
                        })
                    else:
                        return json.dumps({
                            "text_msg": "No relevant handbook content found. Please contact support for assistance.",
                            "human_assistance_required": True
                        })
                else:
                    logger.warning("❌ No handbook content found")
                    return json.dumps({
                        "text_msg": "No relevant handbook content found. Please contact support for assistance.",
                        "human_assistance_required": True
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error in search_handbook: {str(e)}")
                return json.dumps({
                    "text_msg": f"Error searching handbook: {str(e)}. Please contact support for assistance.",
                    "human_assistance_required": True
                })
        
        def search_schedules(question: str) -> str:
            """Search driver schedules and delivery routes for information about daily routes, deliveries, packages, destinations, and delivery times.
            
            Use this tool to find answers to questions about:
            - Driver routes and schedules for today
            - Delivery destinations and addresses
            - Package details (quantity, type, contents, temperature requirements)
            - Delivery times and deadlines
            - Special instructions for deliveries
            - Pickup locations and warehouse information
            - Route distances and estimated drive times
            - Package priorities and handling requirements
            
            Note: Default driver is DRV001 (John Martinez). When users ask about "my route", "my deliveries", etc., search for this driver.
            
            Args:
                question: The user's question or query about driver schedules, routes, or deliveries
                
            Returns:
                JSON string with relevant schedule information or error message
            """
            try:
                if not self.chroma_db_schedules:
                    return json.dumps({
                        "text_msg": "Schedule database is not available. Please contact support for assistance.",
                        "human_assistance_required": True
                    })
                
                # Enhance query with driver context if it's a personal question
                # Default driver: DRV001 (John Martinez)
                enhanced_question = question
                personal_indicators = ["my route", "my delivery", "my schedule", "my stop", "my package", "my deliveries", "my stops"]
                if any(indicator in question.lower() for indicator in personal_indicators):
                    # Add driver context to improve search relevance
                    enhanced_question = f"{question} DRV001 John Martinez"
                    logger.info(f"Enhanced schedule query with driver context: {enhanced_question}")
                
                # Perform similarity search
                results = self.chroma_db_schedules.similarity_search_with_score(enhanced_question, k=3)
                logger.info(f"📅 Found {len(results)} relevant schedule entries")
                
                if results:
                    # Format schedule information for response
                    schedule_list = []
                    for doc, score in results:
                        metadata = doc.metadata
                        content = doc.page_content
                        doc_type = metadata.get("type", "unknown")
                        
                        schedule_entry = {
                            "type": doc_type,
                            "driver_name": metadata.get("driver_name", "Unknown"),
                            "driver_id": metadata.get("driver_id", "N/A"),
                            "content": content,
                            "similarity_score": float(score)
                        }
                        
                        # Add stop-specific information if available
                        if doc_type == "stop":
                            schedule_entry["stop_number"] = metadata.get("stop_number")
                            schedule_entry["destination"] = metadata.get("destination_name")
                            schedule_entry["delivery_time"] = metadata.get("delivery_time")
                            schedule_entry["deadline"] = metadata.get("deadline")
                            schedule_entry["priority"] = metadata.get("priority")
                        
                        schedule_list.append(schedule_entry)
                    
                    if schedule_list:
                        # Format schedule information - content is already in natural language format
                        # from LLM conversion, so we can use it directly with minimal formatting
                        schedule_text_parts = []
                        for entry in schedule_list:
                            # Content is already in readable natural language format
                            # Just add a separator between multiple results
                            schedule_text_parts.append(entry['content'])
                        
                        schedule_text = "\n\n---\n\n".join(schedule_text_parts)
                        
                        return json.dumps({
                            "text_msg": schedule_text,
                            "human_assistance_required": False
                        })
                    else:
                        return json.dumps({
                            "text_msg": "No relevant schedule information found. Please contact support for assistance.",
                            "human_assistance_required": True
                        })
                else:
                    logger.warning("❌ No schedule entries found")
                    return json.dumps({
                        "text_msg": "No relevant schedule information found. Please contact support for assistance.",
                        "human_assistance_required": True
                    })
                    
            except Exception as e:
                logger.error(f"❌ Error in search_schedules: {str(e)}")
                return json.dumps({
                    "text_msg": f"Error searching schedules: {str(e)}. Please contact support for assistance.",
                    "human_assistance_required": True
                })
        
        def get_help() -> str:
            """Get help information - this requires human assistance."""
            return json.dumps({
                "text_msg": "I'm connecting you with our support team. Please wait while I arrange for someone to assist you.",
                "human_assistance_required": True
            })
        
        # Create system prompt for ReAct agent
        react_system_prompt = """You are a helpful logistics assistant for a medical distributor logistics department.

DRIVER INFORMATION:
- Default Driver ID: DRV001
- Default Driver Name: John Martinez
- When users ask about "my route", "my deliveries", "my schedule", etc., they are referring to driver DRV001 (John Martinez). You should automatically search for this driver's schedule information without asking for driver details.

CRITICAL RULES:
- For simple greetings like "Hi", "Hello", "Hey", "Bye", "Thank you", "Thanks" - do not use any tools and provide the final answer immediately. Keep responses concise and natural (e.g., "You're welcome!" for "Thank you").
- NEVER mention "tools", "FAQ tool", "handbook tool", "schedule tool", "get_help tool", "human assistance", "database", "search", or any internal system processes to the user. Always respond as if you naturally know or don't know the information directly.
- NEVER explain your process. Do NOT say things like "I searched the FAQs", "I found this in the handbook", "I checked the schedule", "I found this in the database", "The system shows", etc. Just provide the information naturally or use get_help naturally without explaining your process.
- Keep responses direct and to the point - avoid generic closing phrases like "If you have any other questions", "feel free to ask", etc. End your response naturally after providing the information.
- When using search_faqs, search_handbook, or search_schedules tools, synthesize the information from multiple sources if needed and provide a clear, comprehensive answer. Do NOT just list the information - provide a natural response based on what you found.
- SCOPE LIMITATION: ONLY answer questions related to logistics operations, medical distribution logistics, warehouse operations, shipping/receiving, inventory management, safety procedures, compliance, employee policies, equipment handling, transportation, and other logistics-related topics. For questions completely unrelated to logistics (e.g., general knowledge, entertainment, sports, politics, personal advice, cooking recipes, technology tutorials unrelated to logistics, etc.), politely decline by saying something like "I'm a logistics assistant and can only help with logistics-related questions. Please ask me about logistics operations, procedures, or policies." Do NOT use any tools for irrelevant questions - just provide a polite decline message directly.

EMERGENCY RULE: For any immediate/emergency/incident/accident situations (fire, medical emergency, safety concerns, accidents, urgent issues, maintenance issues), ALWAYS use get_help tool immediately. Do NOT try to answer directly or search FAQs/handbook first.

You have access to the following tools:
1. search_faqs(question) - Search the FAQ database for quick answers about standard operating procedures, policies, and common questions. Use this for: standard procedures, quick policy questions, common operational questions, employee policies, basic safety procedures. FAQs provide concise, direct answers.
2. search_handbook(question) - Search the comprehensive logistics handbook for detailed information, in-depth procedures, guidelines, and best practices. Use this for: detailed operational procedures, comprehensive safety protocols, driving and transportation guidelines, manpower management, detailed compliance information, training requirements, equipment specifications, and when you need more comprehensive information than FAQs provide.
3. search_schedules(question) - Search driver schedules and delivery routes for information about daily routes, deliveries, packages, destinations, and delivery times. Use this for: questions about driver routes ("What's my route for today?", "What deliveries do I have?"), delivery destinations and addresses, package details and contents, delivery times and deadlines, special instructions, pickup locations, route distances, and package priorities. This tool provides real-time schedule and delivery information. IMPORTANT: The default driver is DRV001 (John Martinez). When users ask about "my route", "my deliveries", "my schedule", etc., automatically search for DRV001's schedule without asking for driver details.
4. get_help() - Request human assistance. ALWAYS use this immediately for: (1) emergency/incident/accident situations, (2) requests requiring actions (cancellations, refunds, order changes, billing disputes, complaints, etc.), (3) explicit requests to connect with support ("connect me with support", "I need to speak with someone", "get me a human", etc.), (4) when search_faqs, search_handbook, and search_schedules don't provide sufficient information. IMPORTANT: When you use get_help tool, use the EXACT response message from the tool - do NOT modify it or add placeholder text. Simply use the tool's response as your final answer.

Instructions:
- Scope Check: FIRST, determine if the question is related to logistics operations. If it's completely unrelated (e.g., general knowledge, entertainment, personal advice, etc.), politely decline without using any tools.
- Tool Usage Hierarchy (ONLY for logistics-related questions): 
  * For questions about driver routes, schedules, deliveries, packages, destinations, or delivery times, FIRST try search_schedules tool.
  * For quick, standard questions about procedures or policies, FIRST try search_faqs tool.
  * For detailed, comprehensive questions or when FAQs don't provide enough detail, try search_handbook tool.
  * You can use MULTIPLE tools if needed - for example, if a question involves both schedule information and procedures, use both search_schedules and search_faqs.
  * If none of the search tools provide sufficient information or the question requires human intervention, use get_help tool.
- For general logistics questions that you can answer naturally without tools, provide helpful responses.
- When tools return information, synthesize it into a clear, natural response. Combine information from multiple sources if needed. Don't just repeat the format - provide the information as if you know it directly.
- If you cannot provide information after trying the search tools, use get_help tool.

Examples:
- "What's my route for today?" / "What deliveries do I have?" / "Where do I need to deliver packages?": Use search_schedules tool to find driver route and delivery information for DRV001 (John Martinez). Do NOT ask which driver - assume it's the default driver.
- "What packages am I delivering to Medical City Plano?": Use search_schedules tool to find specific delivery and package details for DRV001 (John Martinez).
- "What's my last stop today?": Use search_schedules tool to find the last stop for DRV001 (John Martinez).
- "How do I handle temperature-sensitive medications?": First try search_faqs for a quick answer. If more detail is needed, also try search_handbook.
- "What are the detailed safety procedures for driving?": Use search_handbook tool for comprehensive information.
- "What is the procedure for receiving medical supplies?": First try search_faqs, then search_handbook if more detail is needed.
- "Connect me with support team" / "I need to speak with someone": Use get_help tool immediately and use the tool's response message exactly as provided.
- Fire/emergency: Use get_help tool immediately. Do NOT try search tools first.
- "What's the weather today?" / "How do I cook pasta?" / "Tell me a joke": Politely decline without using tools - "I'm a logistics assistant and can only help with logistics-related questions. Please ask me about logistics operations, procedures, or policies."

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search_faqs, search_handbook, search_schedules, get_help]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

        # Create the ReAct agent with FAQ search, handbook search, schedule search, and get_help tools
        tools = [search_faqs, search_handbook, search_schedules, get_help]
        return create_agent(
            self.llm,
            tools=tools,
            system_prompt=react_system_prompt
        )
    
    def process_message(self, question: str) -> Dict[str, any]:
        """Process a user message using ReAct agent with tool usage
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: Dictionary with 'msg' (str) and 'is_assistance_required' (bool)
        """
        try:
            logger.info(f"\n🤖 ReAct Agent Processing")
            logger.info("=" * 50)
            logger.info(f"📝 Question: {question}")
            logger.info("")
            
            # Use ReAct agent to process the message with streaming
            logger.info("🔄 Starting ReAct reasoning process...")
            chunks = self.react_agent.stream({
                "messages": [{"role": "user", "content": question}],
                "stream_mode": "updates"
            })
            
            # Process and log the streaming response
            result = self._process_react_stream(chunks)
            
            logger.info(f"\n✅ ReAct processing completed!")
            logger.info("=" * 50)
            
            # Return dict with message and assistance requirement
            if isinstance(result, dict):
                return result
            else:
                # Fallback: If result is not a dict (shouldn't happen), wrap it
                return {
                    "msg": str(result),
                    "is_assistance_required": False
                }
                
        except Exception as e:
            logger.error(f"❌ Error in ReAct processing: {str(e)}")
            return {
                "msg": f"Error processing message with ReAct agent: {str(e)}",
                "is_assistance_required": True
            }
    
    def _process_react_stream(self, chunks):
        """Process ReAct agent streaming chunks and return final response with assistance requirement.
        
        Returns:
            dict: Dictionary with 'msg' (str) and 'is_assistance_required' (bool)
        """
        step_count = 0
        final_response = ""
        is_assistance_required = False
        get_help_message = None  # Store get_help tool's message if used
        
        for chunk in chunks:
            step_count += 1
            
            # Handle CrewAI model chunks
            if 'model' in chunk:
                model_msg = chunk['model']['messages'][-1]
                finish_reason = model_msg.response_metadata.get('finish_reason', 'unknown')
                
                if finish_reason == 'tool_calls' and hasattr(model_msg, 'tool_calls') and model_msg.tool_calls:
                    # REASON: AI is thinking about what tools to use
                    logger.info(f"🤔 REASON (Step {step_count}):")
                    logger.info(f"   The AI is deciding to use multiple tools:")
                    for i, tool_call in enumerate(model_msg.tool_calls, 1):
                        logger.info(f"   {i}. Tool: '{tool_call['name']}' with args: {tool_call['args']}")
                    logger.info("")
                    
                elif finish_reason == 'stop' and model_msg.content:
                    # Parse ReAct format from the content
                    self._parse_react_content(model_msg.content, step_count)
                    final_response = model_msg.content
            
            # Handle traditional agent chunks (for backward compatibility)
            elif 'agent' in chunk:
                agent_msg = chunk['agent']['messages'][-1]
                if hasattr(agent_msg, 'tool_calls') and agent_msg.tool_calls:
                    # REASON: AI is thinking about what tool to use
                    logger.info(f"🤔 REASON (Step {step_count}):")
                    logger.info(f"   The AI is deciding to use the '{agent_msg.tool_calls[0]['name']}' tool")
                    logger.info(f"   Arguments: {agent_msg.tool_calls[0]['args']}")
                    logger.info("")
                    
            elif 'tools' in chunk:
                tool_msg = chunk['tools']['messages'][-1]
                # ACT: Tool is being executed
                logger.info(f"⚡ ACT (Step {step_count}):")
                logger.info(f"   Tool '{tool_msg.name}' executed")
                
                # Check if get_help was called
                if tool_msg.name == 'get_help':
                    is_assistance_required = True
                    # Capture get_help tool's message to use as final response
                    try:
                        result_data = json.loads(tool_msg.content)
                        if isinstance(result_data, dict) and 'text_msg' in result_data:
                            get_help_message = result_data['text_msg']
                            logger.info(f"   📝 Text Message: {get_help_message}")
                            logger.info(f"   🤝 Human Assistance Required: {result_data['human_assistance_required']}")
                        else:
                            logger.info(f"   Result: {tool_msg.content}")
                    except (json.JSONDecodeError, TypeError):
                        logger.info(f"   Result: {tool_msg.content}")
                    logger.info("")
                    continue  # Skip further processing, we'll use get_help message
                
                # Parse structured response if it's JSON
                try:
                    result_data = json.loads(tool_msg.content)
                    if isinstance(result_data, dict) and 'text_msg' in result_data:
                        logger.info(f"   📝 Text Message: {result_data['text_msg']}")
                        logger.info(f"   🤝 Human Assistance Required: {result_data['human_assistance_required']}")
                        # Update assistance requirement if tool indicates it's needed
                        if result_data.get('human_assistance_required', False):
                            is_assistance_required = True
                    else:
                        logger.info(f"   Result: {tool_msg.content}")
                except (json.JSONDecodeError, TypeError):
                    logger.info(f"   Result: {tool_msg.content}")
                logger.info("")
                
            elif 'agent' in chunk and chunk['agent']['messages'][-1].content:
                # Parse ReAct format from the content
                final_msg = chunk['agent']['messages'][-1]
                self._parse_react_content(final_msg.content, step_count)
                final_response = final_msg.content

            else:
                logger.info(f"   Step {step_count}: No action taken")
                logger.info("")
        
        # If get_help was used, use its message directly instead of agent's response
        if get_help_message:
            msg = get_help_message
        else:
            # Extract the actual answer from ReAct format if present
            msg = self._extract_final_answer(final_response)
        
        # Return dict with message and assistance requirement
        return {
            "msg": msg,
            "is_assistance_required": is_assistance_required
        }
    
    def _extract_final_answer(self, content: str) -> str:
        """Extract the final answer from ReAct format content.
        
        Args:
            content: The full ReAct format content (may include Thought, Action, Final Answer, etc.)
        
        Returns:
            str: The extracted final answer, or the original content if no "Final Answer:" section found
        """
        if not content:
            return ""
        
        # Look for "Final Answer:" in the content
        lines = content.split('\n')
        final_answer_started = False
        final_answer_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('Final Answer:'):
                final_answer_started = True
                # Extract text after "Final Answer:"
                answer_text = line_stripped.replace('Final Answer:', '', 1).strip()
                if answer_text:
                    final_answer_lines.append(answer_text)
            elif final_answer_started:
                # Check if we hit a new section (e.g., another "Question:", "Thought:", etc.)
                if line_stripped and ':' in line_stripped and any(
                    line_stripped.startswith(prefix) 
                    for prefix in ['Question:', 'Thought:', 'Action:', 'Action Input:', 'Observation:']
                ):
                    break
                else:
                    final_answer_lines.append(line_stripped)
        
        # If we found a final answer section, return it; otherwise return original content
        if final_answer_lines:
            return '\n'.join(final_answer_lines).strip()
        else:
            # Return original content if no "Final Answer:" section found
            return content.strip()
    
    def _parse_react_content(self, content: str, step_count: int):
        """Parse ReAct format content and display it properly."""
        lines = content.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Question:'):
                logger.info(f"❓ QUESTION (Step {step_count}):")
                logger.info(f"   {line}")
                logger.info("")
                
            elif line.startswith('Thought:'):
                if current_section == 'thought':
                    logger.info(f"   {line}")
                else:
                    logger.info(f"🤔 THOUGHT (Step {step_count}):")
                    logger.info(f"   {line}")
                    current_section = 'thought'
                logger.info("")
                
            elif line.startswith('Action:'):
                logger.info(f"⚡ ACTION (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = 'action'
                logger.info("")
                
            elif line.startswith('Action Input:'):
                logger.info(f"📝 ACTION INPUT (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = 'action_input'
                logger.info("")
                
            elif line.startswith('Observation:'):
                logger.info(f"👁️ OBSERVATION (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = 'observation'
                logger.info("")
                
            elif line.startswith('Final Answer:'):
                logger.info(f"🎯 FINAL ANSWER (Step {step_count}):")
                logger.info(f"   {line}")
                current_section = 'final_answer'
                logger.info("")
                
            else:
                # Continue the current section
                if current_section and line:
                    logger.info(f"   {line}")

