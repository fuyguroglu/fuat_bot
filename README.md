# Fuat_bot 🤖

A personal AI assistant built step-by-step as a learning project, inspired by OpenClaw. Features include file operations, 3-tier memory system, and semantic PDF document search (RAG).

## 🎯 Current Status

**Core Features Complete**: CLI Agent + Memory System + RAG (PDF Search)

This bot can now:
- 📁 Manage files in a sandboxed workspace
- 🧠 Remember context across sessions (3-tier memory)
- 📄 Search PDF documents semantically (RAG system)
- 🔍 Answer questions with automatic citations
- 🤖 Support multiple LLM providers (Anthropic, Gemini, Ollama)

## Project Roadmap

### Phase 1: CLI Agent with Tools ✅ COMPLETE
- [x] Basic project structure
- [x] Config management with pydantic
- [x] Core tools: `read_file`, `write_file`, `list_directory`, `bash`
- [x] Agent loop with tool calling
- [x] Session persistence (JSONL)
- [x] Interactive CLI with Rich
- [x] Multi-provider support (Anthropic, Gemini, Ollama)

### Phase 2: Memory System ✅ COMPLETE
- [x] 3-tier memory architecture (JSON, SQLite, ChromaDB)
- [x] Working memory for recent context
- [x] Long-term facts for persistent knowledge
- [x] Semantic memory with embeddings
- [x] Automatic memory injection into prompts
- [x] 6 memory tools (remember/recall for each tier)

### RAG System: PDF Document Search ✅ COMPLETE
- [x] PDF text extraction with page tracking
- [x] Smart chunking with sentence boundaries
- [x] Document indexing (single file or directory)
- [x] Semantic search with vector embeddings
- [x] Cross-encoder re-ranking for relevance
- [x] Automatic citations (document + page numbers)
- [x] 6 RAG tools (extract, index, search, list, delete)

### Phase 3: Skills System (Future)
- [ ] Skills system (markdown files injected into system prompt)
- [ ] Domain-specific knowledge injection
- [ ] Better error handling and recovery

### Phase 4: Telegram Integration (Future)
- [ ] Basic Telegram bot setup
- [ ] DM handling with allowlist
- [ ] Message routing to agent
- [ ] Media support (images, files)

### Phase 5: Gateway Architecture (Future)
- [ ] Separate Gateway and Agent processes
- [ ] WebSocket API for control
- [ ] Multiple channel support
- [ ] Web UI for management

---

## Quick Start

### Prerequisites
- Python 3.11+
- Conda (recommended) or venv
- API key for your chosen provider (or use local Ollama):
  - **Anthropic (Claude)** - get from https://console.anthropic.com/ (recommended for RAG)
  - **Google Gemini** - get from https://makersuite.google.com/app/apikey
  - **Ollama (Local)** - install from https://ollama.ai/ (free, private)

### Setup

```bash
# Clone/navigate to the project
cd Fuat_bot

# Create conda environment
conda create -n fuat_bot python=3.11 -y
conda activate fuat_bot

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your API key:
# - Set LLM_PROVIDER to "anthropic", "gemini", or "openai"
# - Add the corresponding API key (ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.)
# - Set MODEL_NAME to your desired model
```

### Usage

```bash
# Start interactive chat
python -m fuat_bot chat

# Resume a specific session
python -m fuat_bot chat --session 20260203_143022

# Send a single message (non-interactive)
python -m fuat_bot chat -m "What files are in my workspace?"

# List saved sessions
python -m fuat_bot sessions
```

---

## LLM Provider Support

Fuat_bot supports multiple LLM providers:

| Provider | Status | Models Available | Best For |
|----------|--------|------------------|----------|
| **Anthropic** | ✅ Fully supported | Claude Sonnet 4, Claude Opus 4 | RAG, Complex reasoning |
| **Google Gemini** | ✅ Fully supported | Gemini 2.5 Flash/Pro, 2.0 Flash | Fast responses, Cost-effective |
| **Ollama (Local)** | ✅ Fully supported | Llama 3.2/3.3, Mistral, Phi3, Qwen | Privacy, Offline, Free |
| **OpenAI** | ⏳ Planned | GPT-4, GPT-3.5-turbo | Coming soon |

**To use a different provider:**
1. Edit `.env` and set `LLM_PROVIDER=anthropic` (or `gemini`, `ollama`)
2. Add the corresponding API key (cloud providers only)
3. Set `MODEL_NAME` to a model supported by that provider

**For local Ollama:**
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.2:3b  # Recommended for tool calling (2GB)
# Then set LLM_PROVIDER=ollama in .env
```

---

## Project Structure

```
Fuat_bot/
├── fuat_bot/
│   ├── __init__.py      # Package init
│   ├── __main__.py      # Entry point for `python -m fuat_bot`
│   ├── cli.py           # Command-line interface (Typer + Rich)
│   ├── config.py        # Configuration management (Pydantic)
│   ├── agent.py         # Core agent loop + session management
│   ├── tools.py         # Tool definitions and implementations
│   ├── memory.py        # 3-tier memory system (JSON, SQLite, ChromaDB)
│   └── rag.py           # RAG system (PDF extraction, chunking, search)
├── workspace/           # Agent's working directory (sandboxed)
├── sessions/            # Session transcripts (JSONL files)
├── memory/              # Memory storage
│   ├── working/         # Working memory (JSON files)
│   ├── longterm.db      # Long-term facts (SQLite)
│   └── chromadb/        # Semantic memory + document chunks
├── .env                 # Your configuration (git-ignored)
├── .env.example         # Template for .env
├── requirements.txt     # Python dependencies
├── CLAUDE.md            # Detailed project documentation for AI
└── README.md            # This file
```

---

## How It Works

### The Agent Loop

```
User Message
     │
     ▼
┌─────────────┐
│   LLM API   │◄────────┐
│  (Claude)   │         │
└──────┬──────┘         │
       │                │
       ▼                │
  Tool calls?           │
       │                │
   Yes │ No             │
       │  └──► Return text response
       ▼                │
  Execute tools         │
       │                │
       └────────────────┘
```

### Tools Available

**File Operations:**
| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a file in the workspace |
| `write_file` | Write content to a file |
| `list_directory` | List files and folders |

**Memory System (3-tier):**
| Tool | Description |
|------|-------------|
| `remember_context` | Store temporary context (working memory - JSON) |
| `recall_context` | Retrieve recent working memories |
| `remember_fact` | Store permanent facts (long-term - SQLite) |
| `recall_facts` | Search long-term facts by text/category |
| `remember_semantically` | Store with embeddings (semantic - ChromaDB) |
| `search_memories` | Find similar memories by meaning |

**RAG System (PDF Search):**
| Tool | Description |
|------|-------------|
| `extract_pdf_text` | Extract text from PDF pages |
| `index_document` | Index single PDF for semantic search |
| `index_directory` | Batch index all PDFs in directory |
| `search_documents` | Search indexed PDFs semantically |
| `list_indexed_documents` | List all indexed documents |
| `delete_indexed_document` | Remove indexed document |

**System:**
| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands (use with caution) |

---

## 📚 RAG System: Searching PDFs

The RAG (Retrieval-Augmented Generation) system lets you search PDF documents semantically and get answers with citations.

### Quick Start with RAG

```bash
# 1. Start the bot
python -m fuat_bot chat

# 2. Index your PDFs
> Index the file "regulations.pdf" in category "regulations"

# 3. Ask questions
> What is the late homework submission policy?
# Returns: Relevant passages with page numbers

# 4. Index entire directories
> Index all PDFs in the "docs/" directory

# 5. List what's indexed
> List all indexed documents
```

### RAG Features

- **Semantic Search**: Finds relevant content by meaning, not just keywords
- **Smart Chunking**: Preserves sentence boundaries for better context
- **Re-ranking**: Cross-encoder improves result relevance
- **Citations**: Automatic page numbers for every answer
- **Configurable**: Adjust chunk size, overlap, retrieval limits in `.env`

### Use Case Example: School Assistant

```
1. Index all regulation PDFs once:
   "Index all PDFs in workspace/regulations/"

2. Students ask questions:
   "Can I submit homework 2 days late?"

3. Bot searches indexed PDFs and answers:
   "According to the Late Work Policy (regulations.pdf, page 12),
    students may submit homework up to 3 days late with a 10%
    penalty per day..."
```

---

## Security Notes

- ✅ All file operations are sandboxed to the `workspace/` directory
- ✅ PDF operations limited to workspace PDFs only
- ✅ Memory operations limited to `memory/` directory
- ⚠️ Bash commands have unrestricted system access (use with caution)
- 🔒 API keys are stored in `.env` (never commit this!)
- 📝 Sessions contain full conversation history (be mindful of sensitive data)

---

## 📖 Documentation

- **CLAUDE.md** - Comprehensive project documentation for AI assistants (architecture, implementation details, configuration)
- **README.md** (this file) - Quick start guide and overview

## Learning Resources

- [Anthropic API Docs](https://docs.anthropic.com/) - Claude API reference
- [OpenClaw](https://github.com/openclaw/openclaw) - The inspiration for this project
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Anthropic's guide
- [Sentence Transformers](https://www.sbert.net/) - Embeddings library used for RAG

---

## License

MIT - Do whatever you want with it! 🎉
