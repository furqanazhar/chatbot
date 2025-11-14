# 🚚 RX Logistics Agent

An intelligent logistics assistant agent powered by LangChain, ReAct agent, and OpenAI. This agent helps with logistics operations, procedures, schedules, and provides context-aware responses using conversation history.

## Features

- **ReAct Agent**: Uses LangChain's ReAct agent framework for intelligent tool usage and reasoning
- **Multiple Knowledge Sources**:
  - FAQ database for quick answers about standard procedures and policies
  - Comprehensive handbook search for detailed operational guidelines
  - Schedule search for driver routes, deliveries, and package information
- **Conversation History**: Stores all conversations in SQLite database for context-aware responses
- **Vector Search**: Uses ChromaDB with OpenAI embeddings for semantic search
- **LLM-Powered Data Processing**: Schedule data is converted to natural language using LLM before embedding
- **Context Awareness**: Automatically retrieves conversation history for follow-up questions
- **Default Driver Support**: Hardcoded driver information (DRV001 - John Martinez) for seamless queries

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## Setup

1. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**

   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Alternatively, you can provide the API key through Streamlit secrets or input it directly in the app.

3. **Set up ChromaDB collections** (Optional - if you have data files)

   - **FAQs**: Place FAQ data and run `ops/faqs/create_faq_embeddings.py`
   - **Handbook**: Place handbook PDFs and run `ops/handbook/create_handbook_embeddings.py`
   - **Schedules**: Place schedule JSON and run `ops/schedule/create_schedule_embeddings.py`

   Example for schedules:
   ```bash
   python ops/schedule/create_schedule_embeddings.py
   ```

4. **Run the app**

   ```bash
   streamlit run streamlit_app.py
   ```

   The app will be available at `http://localhost:8501`

## Project Structure

```
agent/
├── streamlit_app.py          # Main Streamlit application
├── services/
│   ├── ai_agent.py          # LangChain ReAct agent with tools
│   └── conversation_db.py   # SQLite conversation storage
├── ops/
│   ├── schedule/            # Schedule data and embeddings
│   ├── faqs/                # FAQ data and embeddings
│   └── handbook/            # Handbook data and embeddings
├── chroma_db/               # ChromaDB vector database
├── conversations.db         # SQLite conversation database (auto-created)
└── requirements.txt         # Python dependencies
```

## Usage

### Example Queries

- **Schedule Questions**:
  - "What's my route for today?"
  - "What's my last stop today?"
  - "What packages am I delivering to Medical City Plano?"
  - "What's my total travel distance?"

- **Procedure Questions**:
  - "How do I handle temperature-sensitive medications?"
  - "What are the safety procedures for driving?"
  - "What is the procedure for receiving medical supplies?"

- **Follow-up Questions**:
  - "and time?" (after asking about distance)
  - "tell me more about that"
  - "what about that delivery?"

### Features in Action

- **Context Awareness**: The agent automatically retrieves conversation history for ambiguous or follow-up questions
- **Smart Stop Identification**: Understands "first stop" (smallest stop number) vs "last stop" (largest stop number)
- **Default Driver**: Automatically uses DRV001 (John Martinez) for personal schedule queries
- **Conversation Persistence**: All conversations are stored in SQLite for future reference

## Configuration

- **Default Driver**: DRV001 (John Martinez) - hardcoded in agent prompt
- **ChromaDB Directory**: `chroma_db/` (default)
- **Conversation Database**: `conversations.db` (auto-created)
- **LLM Model**: GPT-4o (configurable in `services/ai_agent.py`)

## Notes

- The conversation database (`conversations.db`) is automatically created on first run
- ChromaDB collections must be created before using FAQ/handbook/schedule search
- All conversations are stored locally in SQLite
- The agent uses ReAct framework for tool selection and reasoning
