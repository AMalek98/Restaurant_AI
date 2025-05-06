# 🍽️ Restaurant AI Assistant

![Restaurant AI Assistant](https://img.shields.io/badge/AI-Powered-brightgreen)
![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-orange)
![Anthropic Claude](https://img.shields.io/badge/Claude-3.5%20Haiku-purple)

> An intelligent conversation system that manages restaurant interactions using state-of-the-art AI technologies

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Technical Implementation](#-technical-implementation)
- [Example Interactions](#-example-interactions)
- [Data Flow](#-data-flow)
- [Installation & Setup](#-installation--setup)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Project Overview

AlFarooj Restaurant AI Assistant is a sophisticated conversational AI system designed to manage customer interactions in a restaurant setting. The system uses advanced state-machine architecture through LangGraph to understand customer intentions, provide menu information, process food orders with modifications, and handle the entire ordering workflow through natural language conversations.

This project demonstrates expertise in:
- 🧠 Conversation flow management using state machines
- 🔄 Multi-step workflow orchestration
- 💬 Natural language processing and intention detection
- 📊 Vector database integration for semantic search
- 🗃️ Database interactions for order persistence
- 🤖 Advanced prompt engineering techniques

## 🔑 Key Features

- **Intent Detection**: Automatically identifies whether customers are greeting, inquiring about the menu, placing an order, or asking other questions
- **Semantic Menu Search**: Uses vector embeddings to find menu items even when customers use approximate descriptions
- **Order Management**: Tracks items, quantities, special requests, and modifications across conversations
- **Dynamic Order Modification**: Allows adding items or modifying existing ones with special instructions
- **Price Calculation**: Automatically looks up prices and calculates order totals
- **Order Persistence**: Saves orders to both JSON files and PostgreSQL database for record-keeping
- **Interactive Conversation**: Maintains natural dialogue flow throughout the entire interaction

## 🏗️ Architecture

The system is built on a state machine architecture using LangGraph, ensuring robust conversation flow:

```
START
  └── Detect Intention
      ├── Handle Greeting
      ├── Handle Menu Inquiry
      ├── Extract Order
      │   └── Fetch Prices
      │       └── Confirm Order
      │           ├── Calculate Total Price
      │           │   └── Generate Response
      │           │       └── Save Order
      │           │           └── END
      │           ├── Prompt for Extra Item
      │           │   └── Extract Order (loop)
      │           ├── Prompt for Modification
      │           │   └── Calculate Total Price (recalculate)
      │           └── END (cancel)
      └── Handle Other Inquiries
          └── END
```

## 💻 Technical Implementation

### State Management

The system uses a sophisticated TypedDict state model to track:
- Food items and quantities
- Special notes and modifications
- Price information
- Order status
- Conversation history
- User intentions

### Vector Database Integration

Menu items are stored in a Chroma vector database with embeddings, enabling:
- Semantic search capabilities
- Price lookups
- Menu inquiries using natural language

### Intention Detection

The system uses Claude 3.5 Haiku to classify user input into four main categories:
- Greeting
- Menu inquiry
- Order placement
- Other inquiries

### Dynamic Workflow

The state machine dynamically adapts based on user inputs:
- Handling order additions
- Processing item modifications
- Providing confirmation steps
- Calculating and recalculating totals

### Database Integration

Orders are persisted through:
- JSON file storage with unique timestamps
- PostgreSQL integration with normalized tables
- Transaction management with proper rollback handling

### Installation

```bash
git clone https://github.com/AMalek98/Restaurant_AI.git
pip install -r requirements.txt

