import streamlit as st
import os
import logging
import uuid
from dotenv import load_dotenv

# Set up logging to display in terminal (must be before importing agent)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration if already configured
)
logger = logging.getLogger(__name__)

load_dotenv()

# Import agent and database after logging is configured
from services.ai_agent import LogisticsAIAgent
from services.conversation_db import ConversationDB

# Show title and description.
st.title("🚚 Logistics Agent")
st.write(
    "This is a logistics assistant agent powered by ReAct and LangChain. "
    "It can help you with logistics operations, procedures, schedules, and more. "
)

# Check for API key in environment variables or secrets, otherwise ask user
openai_api_key = (
    os.getenv("OPENAI_API_KEY") 
    or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
    or st.text_input("OpenAI API Key", type="password")
)

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Set the API key as environment variable if it's not already set
    # (required by LogisticsAIAgent which reads from os.getenv)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Initialize the AI agent in session state (only once to avoid re-initialization)
    if "ai_agent" not in st.session_state:
        with st.spinner("Initializing logistics AI agent..."):
            try:
                # Initialize conversation database first (needed for agent)
                conversation_db = ConversationDB()
                session_id = str(uuid.uuid4())
                
                # Initialize agent with conversation database and session ID
                st.session_state.ai_agent = LogisticsAIAgent(
                    conversation_db=conversation_db,
                    session_id=session_id
                )
                logger.info("✅ AI agent initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize AI agent: {str(e)}")
                st.session_state.ai_agent = None

    # Only proceed if agent is initialized
    if st.session_state.get("ai_agent") is not None:
        # Get conversation database and session ID from agent
        if "conversation_db" not in st.session_state:
            st.session_state.conversation_db = st.session_state.ai_agent.conversation_db
        
        if "session_id" not in st.session_state:
            st.session_state.session_id = st.session_state.ai_agent.session_id
            logger.info(f"Session ID: {st.session_state.session_id}")
        
        # Ensure agent has the latest references
        st.session_state.ai_agent.conversation_db = st.session_state.conversation_db
        st.session_state.ai_agent.session_id = st.session_state.session_id
        
        # Create a session state variable to store the chat messages. This ensures that the
        # messages persist across reruns.
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Try to load conversation history from database
            try:
                history = st.session_state.conversation_db.get_conversation_history(
                    st.session_state.session_id
                )
                if history:
                    # Convert database format to streamlit format
                    for msg in history:
                        st.session_state.messages.append({
                            "role": msg["role"],
                            "content": msg["content"],
                            "assistance_required": msg["assistance_required"]
                        })
                    logger.info(f"Loaded {len(history)} messages from database")
            except Exception as e:
                logger.warning(f"Could not load conversation history: {e}")

        # Display the existing chat messages via `st.chat_message`.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show assistance indicator if this was a message requiring human assistance
                if message.get("assistance_required", False):
                    st.info("🤝 Human assistance may be required for this query.")

        # Create a chat input field to allow the user to enter a message. This will display
        # automatically at the bottom of the page.
        if prompt := st.chat_input("Ask about logistics operations, schedules, or procedures..."):

            # Store and display the current prompt.
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Save user message to database
            try:
                st.session_state.conversation_db.save_message(
                    session_id=st.session_state.session_id,
                    role="user",
                    content=prompt,
                    assistance_required=False
                )
            except Exception as e:
                logger.error(f"Error saving user message to database: {e}")

            # Generate a response using the LangChain agent.
            with st.chat_message("assistant"):
                with st.spinner("Processing your query..."):
                    try:
                        result = st.session_state.ai_agent.process_message(question=prompt)
                        
                        # Extract response message and assistance requirement
                        response_message = result.get('msg', 'No response generated.')
                        is_assistance_required = result.get('is_assistance_required', False)
                        
                        # Display the response
                        st.markdown(response_message)
                        
                        # Show assistance indicator if needed
                        if is_assistance_required:
                            st.info("🤝 Human assistance may be required for this query.")
                        
                        # Store the response in session state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response_message,
                            "assistance_required": is_assistance_required
                        })
                        
                        # Save assistant message to database
                        try:
                            st.session_state.conversation_db.save_message(
                                session_id=st.session_state.session_id,
                                role="assistant",
                                content=response_message,
                                assistance_required=is_assistance_required,
                                metadata={"query": prompt}
                            )
                        except Exception as e:
                            logger.error(f"Error saving assistant message to database: {e}")
                            
                    except Exception as e:
                        error_message = f"❌ Error processing query: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_message,
                            "assistance_required": True
                        })
                        
                        # Save error message to database
                        try:
                            st.session_state.conversation_db.save_message(
                                session_id=st.session_state.session_id,
                                role="assistant",
                                content=error_message,
                                assistance_required=True,
                                metadata={"error": str(e), "query": prompt}
                            )
                        except Exception as db_error:
                            logger.error(f"Error saving error message to database: {db_error}")
