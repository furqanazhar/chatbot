import streamlit as st
import os
import logging
import uuid
import base64
from dotenv import load_dotenv

# Configure page settings and theme
st.set_page_config(
    page_title="RX Logistics Agent",
    page_icon="data/logo.png",
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

# Custom CSS for theme customization
st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #4c63d2 0%, #5a3d7a 100%);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-top: 1rem;
        max-width: 100% !important;
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Title styling */
    h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* General text color for dark theme */
    .main .block-container p,
    .main .block-container div,
    .main .block-container span {
        color: #e5e7eb;
    }
    
    /* Chat message containers - increase width */
    .stChatMessage {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.3rem !important;
        margin-top: 0 !important;
        background: linear-gradient(135deg, #5568d3 0%, #6b4a8a 100%) !important;
        max-width: 100% !important;
        width: 100% !important;
        box-sizing: border-box;
    }
    
    /* Ensure markdown content within chat messages doesn't overflow */
    .stChatMessage .stMarkdown {
        max-width: 100% !important;
        width: 100% !important;
        overflow-x: hidden;
        box-sizing: border-box;
    }
    
    /* Increase width of element containers holding chat messages */
    [data-testid="element-container"] {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Ensure vertical blocks containing messages use full width */
    [data-testid="stVerticalBlock"] {
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Ensure tables within chat messages fit properly */
    .stChatMessage table {
        max-width: 100% !important;
        width: 100% !important;
        table-layout: auto;
        box-sizing: border-box;
    }
    
    /* Ensure table cells don't overflow and size based on content */
    .stChatMessage table th,
    .stChatMessage table td {
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }
    
    /* Make first column (Stop) narrower since it only has numbers */
    .stChatMessage table th:first-child,
    .stChatMessage table td:first-child {
        width: auto;
        min-width: 50px;
        max-width: 80px;
    }
    
    /* Spacing between chat messages */
    [data-testid="stChatMessage"] {
        margin-bottom: 0.3rem !important;
        margin-top: 0 !important;
    }
    
    /* Remove gap in chat message container */
    .stChatMessage > div {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Spacing between consecutive messages */
    [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
        margin-top: 0.3rem !important;
    }
    
    /* Remove excessive gaps in chat container */
    [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
        margin-bottom: 0.3rem !important;
    }
    
    /* Small gap in Streamlit's internal containers */
    [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }
    
    /* Spacing from element containers */
    [data-testid="element-container"] {
        margin-bottom: 0.3rem !important;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #5c6fd8 0%, #6d4d8f 100%) !important;
        border-left: 4px solid #667eea;
        color: #ffffff;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant"] {
        background: linear-gradient(135deg, #5568d3 0%, #6b4a8a 100%) !important;
        border-left: 4px solid #764ba2;
        color: #ffffff;
    }
    
    /* Markdown content in messages */
    .stChatMessage .stMarkdown,
    .stChatMessage .stMarkdown p,
    .stChatMessage .stMarkdown div {
        color: #e5e7eb !important;
    }
    
    /* User message text */
    [data-testid="stChatMessage"]:has(div[data-testid="user"]) .stMarkdown,
    [data-testid="stChatMessage"]:has(div[data-testid="user"]) .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Chat input styling - simple and clean */
    .stChatInput {
        padding: 0 !important;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
    }
    
    .stChatInput > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stChatInput > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 25px;
        box-shadow: none !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        position: relative;
        padding: 0.5rem 1rem !important;
    }
    
    /* Remove any pseudo-elements that might create double round shapes */
    .stChatInput > div > div::before,
    .stChatInput > div > div::after {
        display: none !important;
        content: none !important;
    }
    
    /* Remove any inner shadows or overlays */
    .stChatInput > div > div > div::before,
    .stChatInput > div > div > div::after {
        display: none !important;
        content: none !important;
    }
    
    /* Remove box-shadow from nested elements */
    .stChatInput > div > div > div {
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Remove any background overlays or nested rounded shapes */
    .stChatInput > div > div > div:first-child {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        border-radius: 0 !important;
    }
    
    /* Remove borders from nested containers that might create double shapes */
    .stChatInput > div > div > div {
        border: none !important;
        border-radius: 0 !important;
    }
    
    /* Keep border-radius only on the outer container */
    .stChatInput > div > div {
        border-radius: 25px !important;
    }
    
    /* Simple input field styling */
    .stChatInput input {
        color: #ffffff !important;
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    
    /* Remove focus outline that might create visual artifacts */
    .stChatInput input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Reduce vertical size of bottom container (white/gradient area) - minimal */
    [data-testid="stAppViewContainer"] > [data-testid="stVerticalBlock"]:last-child {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stVerticalBlock"]:has(.stChatInput) {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="element-container"]:has(.stChatInput) {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Further reduce spacing around chat input container */
    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:has(.stChatInput) {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all spacing from containers holding chat input */
    [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"]:has(.stChatInput) {
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
    }
    
    [data-testid="stAppViewContainer"] [data-testid="element-container"]:has(.stChatInput) {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #5568d3 0%, #6b4a8a 100%);
        border-left: 4px solid #667eea;
        border-radius: 5px;
        color: #ffffff;
    }
    
    /* Error boxes */
    .stError {
        background: linear-gradient(135deg, #6b4a8a 0%, #7a3d5a 100%);
        border-left: 4px solid #764ba2;
        border-radius: 5px;
        color: #ffffff;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #6366f1;
    }
    
    /* Markdown tables */
    .stChatMessage table,
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #5568d3 0%, #6b4a8a 100%);
        table-layout: auto;
    }
    
    /* Ensure table cells wrap text properly and size based on content */
    table th,
    table td {
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
    }
    
    /* Make first column (Stop) narrower since it only has numbers */
    table th:first-child,
    table td:first-child {
        width: auto;
        min-width: 50px;
        max-width: 80px;
    }
    
    table th {
        background: #667eea !important;
        color: #ffffff;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        border: 1px solid #7a5fa8;
        font-size: 0.85rem !important;
    }
    
    table td {
        padding: 10px 12px;
        border-bottom: 1px solid #6b4a8a;
        color: #ffffff;
        font-size: 0.8rem !important;
    }
    
    table tr:hover {
        background: linear-gradient(135deg, #5c6fd8 0%, #6d4d8f 100%);
    }
    
    /* Code blocks */
    code {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    
    pre {
        background: linear-gradient(135deg, #5568d3 0%, #6b4a8a 100%);
        color: #ffffff;
        border: 1px solid #667eea;
    }
    
    /* Remove Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide the menu button (hamburger menu) */
    .stDeployButton {display: none;}
    
    /* Hide the header toolbar */
    header[data-testid="stHeader"] {display: none;}
    
    /* Hide any link buttons in the header */
    .stApp > header > div:first-child {display: none;}
    
    /* Hide the menu button specifically */
    button[title="View app source"] {display: none;}
    button[title="Settings"] {display: none;}
    
    /* Hide header completely */
    #stAppViewContainer > header {display: none !important;}
    
    /* Title with logo container */
    .title-with-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .title-with-logo img {
        height: 40px;
        width: auto;
        object-fit: contain;
    }
    
    .title-with-logo h1 {
        margin: 0;
        flex: 1;
    }
    
    /* Align columns vertically - ensure logo and title are on same line */
    [data-testid="column"] {
        display: flex;
        align-items: flex-start;
    }
    
    /* Ensure logo column aligns to top */
    [data-testid="column"]:first-child {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    
    /* Ensure title column aligns to same baseline */
    [data-testid="column"]:last-child {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding-left: 0.5rem !important;
    }
    
    /* Reduce gap between columns */
    [data-testid="column"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    /* Header area at top of page */
    .header-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        z-index: 999;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* Add padding to body to account for fixed header */
    .main .block-container {
        padding-top: 6rem;
    }
    
    /* Hide Streamlit header to use custom header */
    header[data-testid="stHeader"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Display title with logo at the top of the page using fixed header
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_base64 = get_base64_image("data/logo.png")
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
