from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import  OllamaEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
import operator
from typing import TypedDict, Annotated, List, Dict, Literal, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.constants import Send
from langchain_anthropic import ChatAnthropic
import psycopg
import base64
from IPython.display import display, HTML
from datetime import datetime
import os
os.environ["ANTHROPIC_API_KEY"] = ""
# Define the state with additional intention field
class OverallState(TypedDict):
    names: Annotated[List[str], operator.add]
    user_input: str
    prices: Annotated[List[Dict[str, str]], operator.add]  
    quantities: Annotated[List[int], operator.add]
    total_price: float
    response: str
    item_details: Annotated[List[Dict[str, str]], operator.add]
    intention: str  # New field for intention detection
    confirmation: str
    order_in_progress: bool
    notes: List[str]  # Changed from Annotated to regular List since we'll replace it completely
    modified_item_index: int 
    conversation_history: Annotated[List[Dict[str, str]], operator.add]  # Store conversation history

# Create a response model for structured output
class FoodItems(BaseModel):
    names: List[str]
    quantities: List[int]
    notes: Optional[List[str]] = None 

# Create a model for intention detection
class IntentionType(BaseModel):
    intention: Literal["greeting", "menu_inquiry", "order", "other"]

def initialize_components():
    """Initialize all required components"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    loader = CSVLoader("menu_data.csv", encoding="windows-1252", metadata_columns=['price'])
    documents = loader.load()
    db = Chroma.from_documents(documents, embeddings)
    llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022")
    return embeddings, documents, db, llm

# Initialize components once
embeddings, documents, db, llm = initialize_components()

# Prompts
intention_prompt = """Determine the customer's intention from this message: {user_input}

Choose exactly ONE of the following intentions:
- "greeting": If the customer is saying hello or greeting
- "menu_inquiry": If the customer is asking about the menu or what's available
- "order": If the customer is trying to place an order for food items
- "other": For any other type of inquiry

Return ONLY the intention type, nothing else.
"""

order_prompt = """Extract ONLY the food names from this order and quantities and notes (example: extra tomatoes...) of each food name: {user_input}
Return ONLY a list of food names, quantities and notes nothing else.
IF a Quantity is mentioned in letters return it in numbers.
IF there are no notes return an empty list.
"""

menu_prompt = """You are a friendly restaurant assistant.
The customer is asking about the menu. Create a helpful response about the menu offerings.
Use this information about our menu items:
{menu_items}

Customer's question: {user_input}
"""

greeting_prompt = """You are a friendly restaurant assistant. 
The customer has greeted you. Respond with a warm welcome and briefly mention that you can help with menu inquiries and taking orders.

Customer's greeting: {user_input}
"""

response_prompt = """
You are a friendly restaurant order assistant. 
Create a detailed order confirmation with the following item details:
{item_details}

Total Price: ${total_price:.2f}

Craft a warm, personalized order confirmation message that highlights the specific items, their individual prices, and quantities.
Welcome to AlFarooj!
"""

def detect_intention(state: OverallState):
    """Detect the user's intention from their input"""
    prompt = intention_prompt.format(user_input=state["user_input"])
    response = llm.with_structured_output(IntentionType).invoke(prompt)
    
    # Add to conversation history
    return {
        "intention": response.intention,
        "conversation_history": [{"role": "user", "content": state["user_input"]}]
    }

def route_by_intention(state: OverallState):
    """Route to different nodes based on the detected intention"""
    intention = state["intention"]
    if intention == "greeting":
        return "handle_greeting"
    elif intention == "menu_inquiry":
        return "handle_menu_inquiry" 
    elif intention == "order":
        return "extract_order"
    else:
        return "handle_other"

def handle_greeting(state: OverallState):
    """Handle greeting interactions"""
    prompt = greeting_prompt.format(user_input=state["user_input"])
    response = llm.invoke(prompt).content
    
    # Add to conversation history
    return {
        "response": response,
        "conversation_history": [{"role": "assistant", "content": response}]
    }

def handle_menu_inquiry(state: OverallState):
    """Handle menu inquiries with images"""
    # Get menu items from the vector store
    menu_items = []
    for doc in documents:
        item_name = doc.page_content
        price = doc.metadata.get("price", "Price not available")
        menu_items.append(f"{item_name}: ${price}")
    
    menu_items_str = "\n".join(menu_items)
    prompt = menu_prompt.format(
        user_input=state["user_input"],
        menu_items=menu_items_str
    )
    response = llm.invoke(prompt).content
    
    # Display menu images
    display_menu_images()
    
    # Add to conversation history
    return {
        "response": response,
        "conversation_history": [{"role": "assistant", "content": response}]
    }

def display_menu_images():
    """Display menu images in the notebook"""
    try:
        # Check if menu images exist and display them
        menu_images = []
        for i in range(0, 5):
            image_path = f"menu{i}.jpg"
            if os.path.exists(image_path):
                menu_images.append(image_path)
        
        if menu_images:
            print("Menu Images:")
            for img_path in menu_images:
                try:
                    with open(img_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        display(HTML(f'<img src="data:image/jpeg;base64,{img_data}" alt="{img_path}" width="400"/>'))
                except Exception as e:
                    print(f"Error displaying {img_path}: {str(e)}")
        else:
            print("No menu images found. Please make sure menu1.jpg through menu4.jpg exist in the working directory.")
    except Exception as e:
        print(f"Error displaying menu images: {str(e)}")

def handle_other(state: OverallState):
    """Handle other types of inquiries"""
    prompt = f"""You are a friendly restaurant assistant. The customer has asked: {state["user_input"]}
    
    Respond politely and helpfully. If appropriate, mention that you can help with menu inquiries and taking orders."""
    response = llm.invoke(prompt).content
    
    # Add to conversation history
    return {
        "response": response,
        "conversation_history": [{"role": "assistant", "content": response}]
    }

def extract_order(state: OverallState):
    prompt = order_prompt.format(user_input=state["user_input"])
    response = llm.with_structured_output(FoodItems).invoke(prompt)
    
    # Make sure notes list is provided and has the right length
    notes = response.notes if response.notes else [""] * len(response.names)
    if len(notes) < len(response.names):
        notes.extend([""] * (len(response.names) - len(notes)))
    
    # Check if this is a new order or adding to existing order
    if state.get("order_in_progress", False):
        # This is an addition to an existing order
        return {
            "names": response.names, 
            "quantities": response.quantities,
            "notes": notes,  # Full replacement of notes
            "conversation_history": [{"role": "system", "content": f"Added to order: {', '.join([f'{q} {n}' for q, n in zip(response.quantities, response.names)])}"}]
        }
    else:
        # This is a new order, set the flag
        return {
            "names": response.names, 
            "quantities": response.quantities,
            "notes": notes,  # Full replacement of notes
            "order_in_progress": True,
            "conversation_history": [{"role": "system", "content": f"New order started: {', '.join([f'{q} {n}' for q, n in zip(response.quantities, response.names)])}"}]
        }

def continue_to_prices(state: OverallState):
    return [Send("fetch_prices", {"name": n}) for n in state["names"]]

class PriceState(TypedDict):
    name: str

def fetch_prices(state: PriceState):
    docs = db.similarity_search(state["name"], k=1)
    price = docs[0].metadata.get("price", "Price not found") if docs else "Price not found"
    return {"prices": [{"name": state["name"], "price": price}]}

def calculate_total_price(state: OverallState):
    import json
    import os
    from datetime import datetime
    
    total = 0.0
    item_details = []
    
    # Debug - print all state components
    print("\nDEBUG - State before calculation:")
    print(f"Names: {state['names']}")
    print(f"Quantities: {state['quantities']}")
    print(f"Notes: {state['notes']}")
    
    # Build a dictionary to consolidate items with the same name AND same notes
    item_consolidation = {}
    
    # Make sure we have notes for all items
    notes = state.get("notes", [""] * len(state["names"]))
    if len(notes) < len(state["names"]):
        notes.extend([""] * (len(state["names"]) - len(notes)))
    
    # First, consolidate items with the same name and same modification by summing their quantities
    for price_info, name, quantity, note in zip(state["prices"], state["names"], state["quantities"], notes):
        # Create a unique key using both the item name and the note
        item_key = f"{name}_{note}"
        
        if item_key not in item_consolidation:
            item_consolidation[item_key] = {
                "name": name,
                "price": float(price_info["price"]),
                "quantity": quantity,
                "note": note
            }
        else:
            # Add quantity to existing item
            item_consolidation[item_key]["quantity"] += quantity
    
    # Now process the consolidated items
    for item_key, details in item_consolidation.items():
        price = details["price"]
        quantity = details["quantity"]
        item_total = price * quantity
        total += item_total
        
        item_details.append({
            "name": details["name"],
            "price": f"{price:.2f}",
            "quantity": quantity,
            "item_total": f"{item_total:.2f}",
            "note": details["note"]
        })
    
    print(f"\n*** FINAL ORDER SUMMARY ***")
    for item in item_details:
        # Only display the note if it's not empty
        note_display = f" ({item['note']})" if item['note'] else ""
        print(f"{item['quantity']} x {item['name']}{note_display} @ ${item['price']} = ${item['item_total']}")
    print(f"Total: ${total:.2f}\n")
    
    # Create order data to save in JSON
    current_time = datetime.now()
    order_data = {
        "order_id": current_time.strftime("%Y%m%d%H%M%S"),  # Create unique ID based on timestamp
        "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": item_details,
        "total_price": f"{total:.2f}"
    }
    
    # Save to JSON file
    orders_file = "orders.json"
    
    # Check if file exists and load existing orders
    if os.path.exists(orders_file):
        try:
            with open(orders_file, 'r') as file:
                orders = json.load(file)
        except json.JSONDecodeError:
            # If file is empty or invalid JSON
            orders = {"orders": []}
    else:
        orders = {"orders": []}
    
    # Add new order to orders list
    orders["orders"].append(order_data)
    
    # Write back to file
    with open(orders_file, 'w') as file:
        json.dump(orders, file, indent=4)
    
    print(f"Order saved to {orders_file} with ID: {order_data['order_id']}")
    
    return {
        "total_price": total, 
        "item_details": item_details
    }

def confirm_order(state: OverallState):
    # Make sure we have notes for all items
    notes = state.get("notes", [""] * len(state["names"]))
    if len(notes) < len(state["names"]):
        notes.extend([""] * (len(state["names"]) - len(notes)))
    
    # Display the current order with modifications for the user to confirm
    print("\n*** CURRENT ORDER ***")
    for price_info, name, quantity, note in zip(state["prices"], state["names"], state["quantities"], notes):
        note_display = f" ({note})" if note else ""
        print(f"{quantity} x {name}{note_display} @ ${price_info['price']}")
    
    # In real implementation, this would prompt user for input
    input_text = input(f"Please confirm your order. Do you want to proceed? (yes/no/add/modify): ")
    return {
        "confirmation": input_text,
        "conversation_history": [{"role": "system", "content": f"Order confirmation requested: {input_text}"}]
    }

def should_calculate_add_or_end(state: OverallState):
    # Check the confirmation and route accordingly
    confirmation = state.get("confirmation", "").lower()
    if confirmation == "yes":
        return "calculate_total_price"
    elif confirmation == "add":
        return "prompt_for_extra_item"
    elif confirmation == "modify":
        return "prompt_for_modification"
    else:
        return END

def prompt_for_extra_item(state: OverallState):
    # Ask user what they'd like to add
    extra_item = input("What would you like to add to your order? ")
    return {
        "user_input": extra_item,
        "conversation_history": [{"role": "user", "content": f"Adding to order: {extra_item}"}]
    }

def prompt_for_modification(state: OverallState):
    # Make sure notes exist in the state with proper length
    notes = state.get("notes", [""] * len(state["names"]))
    if len(notes) < len(state["names"]):
        notes.extend([""] * (len(state["names"]) - len(notes)))
    
    # Display current items for reference
    print("\nCurrent items in order:")
    for i, (name, note) in enumerate(zip(state["names"], notes)):
        note_display = f" ({note})" if note else ""
        print(f"{i+1}. {name}{note_display}")
    
    # Ask which item to modify and how
    item_index = int(input("\nWhich item number would you like to modify? ")) - 1
    
    if 0 <= item_index < len(state["names"]):
        modification = input(f"Enter modification for {state['names'][item_index]} (e.g., 'no onions'): ")
        
        # Create a completely new notes list
        new_notes = list(notes)  # Make a copy of the current notes
        new_notes[item_index] = modification  # Update the specific note
        
        print(f"DEBUG - New notes list: {new_notes}")
        
        # Return the entire updated notes list to replace the old one
        return {
            "notes": new_notes, 
            "modified_item_index": item_index,
            "conversation_history": [{"role": "system", "content": f"Modified {state['names'][item_index]} with note: {modification}"}]
        }
    else:
        print("Invalid item number.")
        return {
            "conversation_history": [{"role": "system", "content": "Invalid item number provided for modification"}]
        }

def generate_response(state: OverallState):
    # Format item details into a readable string
    item_details_str = "\n".join([
        f"{item['quantity']} x {item['name']} (${item['price']} each) - Total: ${item['item_total']}"
        for item in state["item_details"]
    ])
    
    # Format the response prompt with order details
    formatted_prompt = response_prompt.format(
        item_details=item_details_str,
        total_price=state["total_price"]
    )
    
    # Generate a friendly response using the LLM
    response = llm.invoke(formatted_prompt).content
    
    return {
        "response": response,
        "conversation_history": [{"role": "assistant", "content": response}]
    }

def save_order(state: OverallState):
    """
    Save order data to PostgreSQL database.
    This function will be the last node in the workflow.
    """
    import psycopg2
    from datetime import datetime
    
    # Database connection parameters
    DB_PARAMS = {
        'dbname': 'your_database_name_here',  # Replace with your actual database name
        'user': 'your_username_here',  # Replace with your actual username
        # Note: For security reasons, avoid hardcoding passwords in your code
        'password': 'your_password_here',  # Replace with your actual password
        'host': 'localhost',
        'port': '5432'
    }
    
    # Get current date and time
    now = datetime.now()
    current_date = now.date()
    current_time = now.time()
    
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        
        # Begin transaction
        conn.autocommit = False
        
        # Insert into order_history table
        cursor.execute(
            "INSERT INTO order_history (order_date, order_time, total_price) VALUES (%s, %s, %s) RETURNING order_id",
            (current_date, current_time, state["total_price"])
        )
        
        # Get the generated order_id
        order_id = cursor.fetchone()[0]
        
        # Insert each item into order_details table
        for item in state["item_details"]:
            # Set default note to 'no' if empty
            note = item["note"] if item["note"].strip() else "no"
            
            cursor.execute(
                "INSERT INTO order_details (order_id, item_name, quantity, item_price, notes) VALUES (%s, %s, %s, %s, %s)",
                (order_id, item["name"], item["quantity"], float(item["price"]), note)
            )
        
        # Commit the transaction
        conn.commit()
        print(f"\nOrder successfully saved to database with order ID: {order_id}")
        
    except Exception as e:
        # If an error occurs, rollback the transaction
        if conn:
            conn.rollback()
        print(f"Database error: {e}")
        
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    # If we have a response from previous node, use it; otherwise, generate one
    if "response" not in state or not state["response"]:
        return generate_response(state)
    
    # Pass through the response from previous node
    return {"response": state.get("response", "Order processed successfully!")}

def run_interactive_chatbot():
    """
    Run an interactive conversational chatbot allowing natural dialogue with the system
    """
    # Initialize a thread
    thread = {"configurable": {"thread_id": "interactive_session"}}
    
    print("\n💬 Welcome to AlFarooj Restaurant Chat Assistant! 💬")
    print("You can ask about our menu, place orders, or just say hello.")
    print("Type 'exit' to end the conversation.\n")
    
    conversation_active = True
    
    while conversation_active:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using AlFarooj Restaurant Chat Assistant. Goodbye!")
            conversation_active = False
            break
        
        # Initialize state with user input
        state = {"user_input": user_input}
        
        try:
            # Process workflow
            final_state = None
            for step in app.stream(state, thread):
                # Keep track of the most recent state
                final_state = step
                
                # Only display output for terminal nodes (those with responses)
                if "response" in list(step.values())[0]:
                    print(f"\nAssistant: {list(step.values())[0]['response']}")
            
            # If order was placed and completed, ask if user wants to start a new conversation
            if final_state and END in final_state:
                restart = input("\nWould you like to place another order or ask something else? (yes/no): ")
                if restart.lower() not in ["yes", "y", "sure", "okay", "ok"]:
                    print("Thank you for using AlFarooj Restaurant Chat Assistant. Goodbye!")
                    conversation_active = False
        
        except Exception as e:
            print(f"Sorry, I encountered an error: {str(e)}")
            print("Let's try again. How can I help you?")

# Initialize the graph
def build_graph():
    # Initialize the graph
    graph = StateGraph(OverallState)

    # Add all nodes
    graph.add_node("detect_intention", detect_intention)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_menu_inquiry", handle_menu_inquiry)
    graph.add_node("handle_other", handle_other)
    graph.add_node("extract_order", extract_order)
    graph.add_node("fetch_prices", fetch_prices)
    graph.add_node("calculate_total_price", calculate_total_price)
    graph.add_node("confirm_order", confirm_order)
    graph.add_node("prompt_for_extra_item", prompt_for_extra_item)
    graph.add_node("prompt_for_modification", prompt_for_modification)
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_order", save_order)

    # Add edges
    graph.add_edge(START, "detect_intention")
    graph.add_conditional_edges("detect_intention", route_by_intention, 
                             ["handle_greeting", "handle_menu_inquiry", "extract_order", "handle_other"])

    # Connect greeting, menu_inquiry, and other nodes directly to END
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_menu_inquiry", END)
    graph.add_edge("handle_other", END)

    # Original order workflow connections
    graph.add_conditional_edges("extract_order", continue_to_prices, ["fetch_prices"])
    graph.add_edge("fetch_prices", "confirm_order")
    
    # Add multi-way choice after confirmation
    graph.add_conditional_edges(
        "confirm_order",
        should_calculate_add_or_end,
        ["calculate_total_price", "prompt_for_extra_item", "prompt_for_modification", END]
    )

    # After prompting for extra items, go back to extract_order to process them
    graph.add_edge("prompt_for_extra_item", "extract_order")

    # After modifying an order, go back to confirm_order to verify changes
    graph.add_edge("prompt_for_modification", "calculate_total_price")

    graph.add_edge("calculate_total_price", "generate_response")
    graph.add_edge("generate_response", "save_order")
    graph.add_edge("save_order", END)

    # Compile the graph
    return graph.compile(checkpointer=MemorySaver())

# Build the graph
app = build_graph()

# Run the interactive chatbot
if __name__ == "__main__":
    run_interactive_chatbot()