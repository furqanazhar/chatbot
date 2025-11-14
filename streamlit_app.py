import streamlit as st
import os
import logging
import uuid
import base64
from dotenv import load_dotenv

# Configure page settings and theme
st.set_page_config(
    page_title="RX Logistics Agent",
    page_icon="static/logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

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

# Load custom CSS from external file
def load_css():
    try:
        with open("static/style.css", "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logger.warning("CSS file not found at static/style.css")
    except Exception as e:
        logger.error(f"Error loading CSS file: {e}")

# Load CSS
load_css()

# Display title with logo at the top of the page using fixed header
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_base64 = get_base64_image("static/logo.png")
if logo_base64:
    st.markdown(f"""
        <div class="header-container">
            <img src="data:image/png;base64,{logo_base64}" style="width: 40px; height: 40px; object-fit: contain;" />
            <h1 style="margin: 0; padding: 0; line-height: 1; color: #ffffff; font-weight: 700; font-size: 1.8rem;">RX Logistics Agent</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    # Fallback: use columns if logo can't be loaded
    col1, col2 = st.columns([0.08, 0.92], gap="small")
    with col1:
        try:
            st.image("data/logo.png", use_container_width=False, width=40)
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
    with col2:
        st.markdown('<div style="display: flex; align-items: center; height: 40px;"><h1 style="margin: 0; padding: 0; line-height: 1; display: inline-block;">RX Logistics Agent</h1></div>', unsafe_allow_html=True)

st.markdown(
    '<p style="color: #d1d5db; margin-bottom: 1.5rem;">This is a logistics assistant agent powered by ReAct and LangChain. '
    'It can help you with logistics operations, procedures, schedules, and more.</p>',
    unsafe_allow_html=True
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

        # Store the current prompt in session state (will be displayed by the loop above)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
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

        # Force immediate rerun to display user message right away
        st.rerun()

    # Process the last user message if it hasn't been responded to yet
    # This check runs on every rerun, not just when a new message is sent
    if (st.session_state.messages and 
        st.session_state.messages[-1]["role"] == "user" and
        not hasattr(st.session_state, "_processing_message")):
        
        # Set flag to prevent duplicate processing
        st.session_state._processing_message = True
        
        # Get the last user message
        last_user_message = st.session_state.messages[-1]["content"]
        
        # Generate a response using the LangChain agent.
        with st.spinner("Processing your query..."):
            try:
                result = st.session_state.ai_agent.process_message(question=last_user_message)
                
                # Extract response message and assistance requirement
                response_message = result.get('msg', 'No response generated.')
                is_assistance_required = result.get('is_assistance_required', False)
                
                # Store the response in session state (will be displayed by the loop above)
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
                        metadata={"query": last_user_message}
                    )
                except Exception as e:
                    logger.error(f"Error saving assistant message to database: {e}")
                
                # Clear processing flag and force a rerun to display the new messages
                if "_processing_message" in st.session_state:
                    del st.session_state._processing_message
                st.rerun()
                    
            except Exception as e:
                error_message = f"❌ Error processing query: {str(e)}"
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
                        metadata={"error": str(e), "query": last_user_message}
                    )
                except Exception as db_error:
                    logger.error(f"Error saving error message to database: {db_error}")
                
                # Clear processing flag and force a rerun to display the error message
                if "_processing_message" in st.session_state:
                    del st.session_state._processing_message
                st.rerun()
