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
    """AI Agent for Logistics using LangChain with FAQ, handbook, schedule, conversation history, and get_help tools"""
    
    def __init__(self, chromadb_dir: str = "chroma_db", faq_collection: str = "faqs", handbook_collection: str = "handbook", schedule_collection: str = "schedules", conversation_db=None, session_id=None):
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
        
        # Store conversation database and session ID for conversation history
        self.conversation_db = conversation_db
        self.session_id = session_id
        
        # Initialize ReAct agent with FAQ, handbook, schedule, conversation history, and get_help tools
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
            
            IMPORTANT: When interpreting "first stop" vs "last stop":
            - "First stop" = the stop with the smallest stop number (e.g., stop 1) OR the earliest delivery time
            - "Last stop" = the stop with the largest stop number (e.g., stop 6) OR the latest delivery time
            - Always prioritize stop number over time when determining first/last stops
            
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
                
                # Perform similarity search - retrieve all available records
                # Using a high k value to ensure we get all schedule entries
                results = self.chroma_db_schedules.similarity_search_with_score(enhanced_question, k=50)
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
        
        def get_conversation_history(limit: int = 10) -> str:
            """Get recent conversation history to understand context and provide more tailored responses.
            
            CRITICAL: ALWAYS use this tool FIRST when:
            - The user's question is short, ambiguous, or incomplete (e.g., "and time", "and distance", "what about it", "tell me more")
            - The user's question refers to something mentioned earlier (e.g., "What about that delivery?", "Tell me more about that")
            - You need context about previous questions or topics discussed
            - The user asks follow-up questions that require understanding previous conversation
            - The question seems unrelated to logistics but might be a follow-up to a logistics question
            - You want to provide more personalized responses based on conversation history
            
            IMPORTANT: For short phrases like "and time", "and distance", "what about...", "tell me more", etc., you MUST use this tool FIRST before determining if the question is logistics-related. These are likely follow-up questions to previous logistics queries.
            
            Args:
                limit: Number of recent messages to retrieve (default: 10, max: 50)
                
            Returns:
                JSON string with conversation history or error message
            """
            try:
                if not self.conversation_db or not self.session_id:
                    return json.dumps({
                        "text_msg": "Conversation history is not available.",
                        "human_assistance_required": False
                    })
                
                # Limit the number of messages retrieved
                limit = min(max(1, limit), 50)
                
                # Get conversation history
                history = self.conversation_db.get_conversation_history(
                    session_id=self.session_id,
                    limit=limit
                )
                
                if not history:
                    return json.dumps({
                        "text_msg": "No previous conversation history found.",
                        "human_assistance_required": False
                    })
                
                # Format conversation history for the agent
                history_parts = []
                for msg in history:
                    role_label = "User" if msg['role'] == 'user' else "Assistant"
                    history_parts.append(f"{role_label}: {msg['content']}")
                
                history_text = "\n".join(history_parts)
                
                return json.dumps({
                    "text_msg": f"Recent conversation history ({len(history)} messages):\n\n{history_text}",
                    "human_assistance_required": False
                })
                
            except Exception as e:
                logger.error(f"❌ Error retrieving conversation history: {str(e)}")
                return json.dumps({
                    "text_msg": f"Error retrieving conversation history: {str(e)}",
                    "human_assistance_required": False
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
- NEVER mention "tools", "FAQ tool", "handbook tool", "schedule tool", "conversation history tool", "get_help tool", "human assistance", "database", "search", "previous conversation", or any internal system processes to the user. Always respond as if you naturally know or don't know the information directly.
- NEVER explain your process. Do NOT say things like "I searched the FAQs", "I found this in the handbook", "I checked the schedule", "I looked at our previous conversation", "I found this in the database", "The system shows", etc. Just provide the information naturally or use get_help naturally without explaining your process.
- Keep responses direct and to the point - NEVER add generic closing phrases or invitations to ask more questions. Do NOT end with phrases like "If you have any other questions", "feel free to ask", "If you need more details", "Let me know if you need anything else", "Is there anything else I can help with?", etc. Simply end your response immediately after providing the requested information. Do NOT add any closing statements or invitations.
- ONLY provide the information that was specifically requested. Do NOT add extra details, additional information, or supplementary content that was not asked for. If the user asks for "contacts", provide ONLY contact information. If they ask for "addresses", provide ONLY addresses. If they ask for "packages", provide ONLY package information. Do NOT add special instructions, additional notes, or other details unless explicitly requested.
- When using search_faqs, search_handbook, or search_schedules tools, extract and provide ONLY the information that directly answers the user's question. Do NOT add extra context, instructions, or details that were not requested. Provide the information as if you know it directly, but ONLY what was asked for.

RESPONSE FORMATTING:
- Structure your responses in a well-formatted, readable format using Markdown:
  * Use **bold** for important information (e.g., locations, times, priorities)
  * Use bullet points (- or *) for lists of items, instructions, or multiple pieces of information
  * Use numbered lists (1., 2., 3.) for sequential steps or ordered information
  * Use headers (## or ###) to organize different sections when providing comprehensive information
  * Use line breaks to separate different topics or sections
  * Use tables when presenting structured data (e.g., multiple stops, packages with details, route information, delivery schedules, contact information, package lists, procedure steps with requirements, comparison data, etc.)
  * Use code blocks or inline code for specific IDs, codes, or technical terms
- For schedule information, ALWAYS use tables when displaying multiple stops, routes, or delivery information. Organize schedule data in tabular format with columns such as:
  * Stop Number | Destination | Address | Delivery Time | Deadline | Priority | Packages | Contact
  * For single stop details, you can use a structured format, but for multiple stops or route overviews, ALWAYS use a table
  * Include ONLY the information that was requested in the table. If the user asks for "contacts", include only contact-related columns. If they ask for "addresses", include only address-related columns. Do NOT add extra columns or information that was not requested.
  * Do NOT add additional details like special instructions, package contents, temperature requirements, etc. unless the user specifically asks for them. Only provide what was requested.
- For other structured information, use tables when appropriate:
  * Multiple packages with details (Package ID, Description, Quantity, Type, Priority, Requirements)
  * Contact information for multiple locations (Location, Contact Name, Phone, Role)
  * Procedure steps with requirements (Step, Action, Requirements, Notes)
  * Comparison data or lists with multiple attributes
  * Any data that has multiple items with the same set of attributes
- For procedures, organize by:
  * Main procedure title/heading
  * Step-by-step instructions (numbered list)
  * Important notes or warnings (bold or separate section)
  * Related information or requirements
- Keep formatting consistent and professional - use proper spacing, clear hierarchy, and logical organization
- SCOPE LIMITATION: ONLY answer questions related to logistics operations, medical distribution logistics, warehouse operations, shipping/receiving, inventory management, safety procedures, compliance, employee policies, equipment handling, transportation, and other logistics-related topics. IMPORTANT: Questions about delivery locations, addresses, contacts at delivery locations, route information, schedules, packages, and anything related to driver routes or deliveries ARE logistics questions. For questions completely unrelated to logistics (e.g., general knowledge, entertainment, sports, politics, personal advice, cooking recipes, technology tutorials unrelated to logistics, etc.), politely decline by saying something like "I'm a logistics assistant and can only help with logistics-related questions. Please ask me about logistics operations, procedures, or policies." CRITICAL: Before rejecting ANY question, you MUST first check if it might be related to logistics by using appropriate tools (search_schedules, get_conversation_history, search_faqs, search_handbook). Only reject questions that are clearly and obviously unrelated to logistics AFTER checking tools and confirming they are not logistics-related.

EMERGENCY RULE: For any immediate/emergency/incident/accident situations (fire, medical emergency, safety concerns, accidents, urgent issues, maintenance issues), ALWAYS use get_help tool immediately. Do NOT try to answer directly or search FAQs/handbook first.

You have access to the following tools:
1. search_faqs(question) - Search the FAQ database for quick answers about standard operating procedures, policies, and common questions. Use this for: standard procedures, quick policy questions, common operational questions, employee policies, basic safety procedures. FAQs provide concise, direct answers.
2. search_handbook(question) - Search the comprehensive logistics handbook for detailed information, in-depth procedures, guidelines, and best practices. Use this for: detailed operational procedures, comprehensive safety protocols, driving and transportation guidelines, manpower management, detailed compliance information, training requirements, equipment specifications, and when you need more comprehensive information than FAQs provide.
3. search_schedules(question) - Search driver schedules and delivery routes for information about daily routes, deliveries, packages, destinations, and delivery times. Use this for: questions about driver routes ("What's my route for today?", "What deliveries do I have?"), delivery destinations and addresses, package details and contents, delivery times and deadlines, special instructions, pickup locations, route distances, package priorities, contact information at delivery locations, and ANY questions mentioning location names, addresses, or delivery destinations (e.g., "Who should I contact at [location]?", "What's the address for [location]?", "Tell me about [location]"). This tool provides real-time schedule and delivery information. IMPORTANT: The default driver is DRV001 (John Martinez). When users ask about "my route", "my deliveries", "my schedule", etc., automatically search for DRV001's schedule without asking for driver details. CRITICAL: When users ask about "first stop" or "last stop", interpret as follows: "first stop" = the stop with the smallest stop number OR the earliest delivery time; "last stop" = the stop with the largest stop number OR the latest delivery time. Always identify the correct stop based on stop number (primary) or delivery time (if stop numbers are not available). CRITICAL: If a question mentions ANY location name, address, or delivery destination (even if it seems like a general question), ALWAYS use this tool FIRST to check if it's a logistics-related location before making any scope decisions.
4. get_conversation_history(limit) - Retrieve recent conversation history to understand context and provide more tailored, personalized responses. CRITICAL: ALWAYS use this FIRST when: (1) the user's question is short, ambiguous, or incomplete (e.g., "and time", "and distance", "what about it", "tell me more"), (2) the user's question refers to something mentioned earlier (e.g., "What about that delivery?", "Tell me more", "What was that again?"), (3) you need context about previous questions or topics, (4) the user asks follow-up questions, (5) the question seems unrelated but might be a follow-up to a logistics question. The limit parameter (default: 10, max: 50) controls how many recent messages to retrieve. Use this tool BEFORE scope checks and BEFORE other tools when the question is ambiguous or short.
5. get_help() - Request human assistance. ALWAYS use this immediately for: (1) emergency/incident/accident situations, (2) requests requiring actions (cancellations, refunds, order changes, billing disputes, complaints, etc.), (3) explicit requests to connect with support ("connect me with support", "I need to speak with someone", "get me a human", etc.), (4) when search_faqs, search_handbook, and search_schedules don't provide sufficient information. IMPORTANT: When you use get_help tool, use the EXACT response message from the tool - do NOT modify it or add placeholder text. Simply use the tool's response as your final answer.

Instructions:
- CRITICAL: ALWAYS CHECK TOOLS FIRST before rejecting any question. Do NOT make scope decisions without first checking if the question might be logistics-related using appropriate tools.
- Context Check FIRST: For short, ambiguous, or incomplete questions (e.g., "and time", "and distance", "what about it", "tell me more", "that one", etc.), ALWAYS use get_conversation_history FIRST before making any decisions. These are likely follow-up questions to previous logistics queries. DO NOT reject them as non-logistics without checking context.
- Location/Address Check: If a question mentions ANY location name, address, delivery destination, or contact information (e.g., "Who should I contact at [location]?", "Tell me about [location]", "What's at [address]?"), ALWAYS use search_schedules FIRST to check if it's a logistics-related location. Do NOT reject these questions without checking - they are likely logistics questions about delivery locations.
- Scope Check: ONLY after checking conversation history and/or tools, determine if the question is related to logistics operations. If it's completely unrelated (e.g., general knowledge, entertainment, personal advice, etc.) AND not a follow-up to a logistics question AND tools confirm it's not logistics-related, politely decline. However, if tools return relevant logistics information, the question IS logistics-related and you must answer it.
- Context Awareness: If the user's question seems to refer to something mentioned earlier, is a follow-up question, or uses vague references (e.g., "that delivery", "what about it", "tell me more", "and time", "and distance"), FIRST use get_conversation_history to understand the context before using other tools or making scope decisions.
- Tool Usage Hierarchy (ONLY for logistics-related questions): 
  * If context is needed, FIRST use get_conversation_history to understand previous conversation.
  * For questions about driver routes, schedules, deliveries, packages, destinations, or delivery times, use search_schedules tool.
  * When search_schedules returns multiple stops and the user asks about "first stop" or "last stop", identify the correct stop: "first" = smallest stop number (e.g., stop 1) or earliest delivery time; "last" = largest stop number (e.g., stop 6) or latest delivery time. Always check the stop_number metadata to determine first/last.
  * For quick, standard questions about procedures or policies, use search_faqs tool.
  * For detailed, comprehensive questions or when FAQs don't provide enough detail, use search_handbook tool.
  * You can use MULTIPLE tools if needed - for example, if a question involves both schedule information and procedures, use both search_schedules and search_faqs.
  * If none of the search tools provide sufficient information or the question requires human intervention, use get_help tool.
- For general logistics questions that you can answer naturally without tools, provide helpful responses.
- When tools return information, extract and provide ONLY the information that directly answers the user's question. Do NOT add extra details, special instructions, or additional information that was not requested. Use conversation history to tailor your response style and provide more personalized answers. Combine information from multiple sources only if needed to answer the specific question. Don't just repeat the format - provide the information as if you know it directly, but ONLY what was asked for.
- Response Formatting: Always format your responses using Markdown for better readability. Use bold text for key information, bullet points for lists, headers for sections, and proper spacing. CRITICAL: For schedule information with multiple stops or routes, ALWAYS use tables to display the data in a structured, easy-to-read format. Use tables for any structured data with multiple items sharing the same attributes (stops, packages, contacts, procedures with requirements, etc.). Structure schedule information clearly with location, time, packages, and instructions in tabular format when showing multiple items. Format procedures with clear headings and numbered steps. Make responses visually organized and easy to scan.
- Personalization: Use conversation history to remember user preferences, previous topics discussed, and provide continuity in the conversation. Reference previous exchanges naturally when relevant.
- If you cannot provide information after trying the search tools, use get_help tool.

Examples:
- "What's my route for today?" / "What deliveries do I have?" / "Where do I need to deliver packages?": Use search_schedules tool to find driver route and delivery information for DRV001 (John Martinez). ALWAYS format the response using a table with columns: Stop Number | Destination | Address | Delivery Time | Deadline | Priority | Packages | Contact. Include all stops in the table. After the table, provide additional details like special instructions or package contents if needed. Do NOT ask which driver - assume it's the default driver.
- "What packages am I delivering to Medical City Plano?": Use search_schedules tool to find specific delivery and package details for DRV001 (John Martinez). If there are multiple packages, format them in a table with columns: Package ID | Description | Quantity | Type | Contents | Priority | Requirements. For single package, use structured format with bold destination and clear sections.
- "What's my last stop today?" / "What's my first stop?": Use search_schedules tool to find the appropriate stop for DRV001 (John Martinez). Remember: "first stop" = smallest stop number or earliest time; "last stop" = largest stop number or latest time. When the tool returns multiple stops, identify the correct one based on stop number (first = lowest number, last = highest number). For single stop details, format with bold location, time information in a structured way, package details in a table if multiple packages, and special instructions as a bulleted list. If showing multiple stops for context, use a table.
- "What about that delivery?" / "Tell me more about that" / "What was that again?": FIRST use get_conversation_history to understand what the user is referring to, then use appropriate tools based on context.
- "and time" / "and distance" / "and packages" / "what about it": These are short follow-up questions. ALWAYS use get_conversation_history FIRST to see the previous question (e.g., if previous question was about travel distance, "and time" means travel time). Then use appropriate tools (e.g., search_schedules) based on the context.
- User asks "what's my total travel distance?" → You answer with distance. User then asks "and time" → FIRST use get_conversation_history to see they asked about distance, then use search_schedules to find travel time information.
- "How do I handle temperature-sensitive medications?": First try search_faqs for a quick answer. If more detail is needed, also try search_handbook. Format procedures with clear headings, numbered steps, and important notes in bold.
- "What are the detailed safety procedures for driving?": Use search_handbook tool for comprehensive information. Format with section headers, numbered steps, and warnings/important notes in bold.
- "What is the procedure for receiving medical supplies?": First try search_faqs, then search_handbook if more detail is needed. Format with clear structure: procedure title, step-by-step instructions (numbered), and any special requirements or notes.
- "Connect me with support team" / "I need to speak with someone": Use get_help tool immediately and use the tool's response message exactly as provided.
- Fire/emergency: Use get_help tool immediately. Do NOT try search tools first.
- "Who should I contact at THR Dallas?" / "Who do I contact at [delivery location]?": This is a logistics question about delivery location contacts. Use search_schedules tool to find contact information for the delivery location. Do NOT reject this - it's clearly logistics-related.
- "contacts for my deliveries?" / "What are the contacts?": Use search_schedules tool to find contact information. Provide ONLY a table with contact information (Stop, Destination, Contact Person, Phone). Do NOT add special instructions, addresses, or any other details - ONLY contacts as requested.
- "What's the address for Medical City Plano?" / "Tell me about [location name]": Use search_schedules tool to find information about the delivery location. These are logistics questions.
- Table Format Example for Routes: When displaying route information, use a table like this:
  | Stop | Destination | Address | Delivery Time | Deadline | Priority | Packages | Contact |
  |------|-------------|---------|---------------|----------|----------|----------|---------|
  | 1 | Medical City Plano | 3901 W 15th St, Plano, TX 75075 | 12:00 PM | 12:00 PM | High | 5 totes | Receiving Dept (+1-555-2001) |
  | 2 | THR Dallas | 8200 Walnut Hill Ln, Dallas, TX 75231 | 2:00 PM | 3:00 PM | Critical | 3 crates | Mike Chen (+1-555-2002) |
  This makes the information easy to scan and compare across stops.
- "What's the weather today?" / "How do I cook pasta?" / "Tell me a joke": These are clearly unrelated to logistics. However, if a question mentions a location name that might be a delivery destination, ALWAYS check search_schedules first before rejecting. Only reject questions that are obviously and clearly unrelated to logistics after confirming with tools.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search_faqs, search_handbook, search_schedules, get_conversation_history, get_help]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

        # Create the ReAct agent with FAQ search, handbook search, schedule search, conversation history, and get_help tools
        tools = [search_faqs, search_handbook, search_schedules, get_conversation_history, get_help]
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

