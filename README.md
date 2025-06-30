# I-CAN-IISc-Conversational-Academic-Navigator

## Getting Started

### Prerequisites

- Python 3.7+ installed
- Git installed

### Clone the repo

```bash
git clone https://github.com/PrasannaBV/I-CAN-IISc-Conversational-Academic-Navigator.git
cd I-CAN-IISc-Conversational-Academic-Navigator




Project Structure


├── src/ # Source code root
│ ├── config/ # Configuration files and parameters
│ ├── data/ # vectorstore/FAISS
│ ├
│ ├── ingestion/ # Modules for website, PDF, and manual text ingestion
│ ├── prompting/ # Prompt templates and prompt handling logic
│ ├── retrieval/ # RAG (Retrieval-Augmented Generation) logic
│ ├── reward/ # Reward model training, scoring, evaluation
│
│ ├── ui/ # Streamlit User interface components 
│ ├── utility/ # Helper functions, common utilities
│ └── validation/ # Evaluation scripts, testing, and validation logic
│
├── requirements.txt # Python dependencies
├── README.md # Project overview and documentation
└── setup.py # Package installation and distribution config





The code for the respective features is in different branches as mentioned below.

main/
│
├── master/                 # MVP ready code base, core pipeline
│
├── ingestion/              # Website & PDF ingestion, manual text support
│
├── prompting/              # Prompting logic and template changes
│
├── langgraph/              # LangGraph-based orchestration implementation
│
├── agent-delegation/       # Router Agent (Orchestrator), multi-agent delegation
│
├── MCP/                    # Simulated server for MCP, Realtime API integration
│
├── rewardmodel/            # Trained distilbert-base-uncased model, training scripts and data
│
└── personalization/        # Structured user context injection for personalization




