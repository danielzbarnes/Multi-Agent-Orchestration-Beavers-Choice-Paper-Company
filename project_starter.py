import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Set up and load your env parameters and instantiate your model.

dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

model = OpenAIServerModel(model_id="gpt-4-0613", api_key=API_KEY, api_base=BASE_URL, max_tokens=200)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

@tool
def check_inventory(item_names: list[str], as_of_date: str) -> Dict:
    """
    Tool for checking inventory levels of specified items as of a given date.

    This function retrieves the current stock levels for each item in `item_names`
    using the `get_stock_level` function, and returns a dictionary mapping item names
    to their available stock quantities.

    Args:
        item_names (list[str]): A list of item names to check inventory for.
        as_of_date (str): The date (in ISO format) to check inventory levels as of.

    Returns:
        Dict: A dictionary where keys are item names and values are the current stock levels.
              If an item is not found, it will be included with a stock level of 0.
    """
    results = {}
    inventory = get_all_inventory(as_of_date)
    for item in item_names:
        results[item] = inventory.get(item, 0)
    return results

@tool
def reorder_inventory(item_name: str, quantity: int, order_date: str) -> str:
    """
    Tool for placing a stock order to replenish inventory for a specific item.

    This function calculates the total cost of the order based on the unit price from the inventory,
    creates a transaction record for the stock order, and estimates the delivery date using the
    `get_supplier_delivery_date` function.

    Args:
        item_name (str): The name of the item to reorder.
        quantity (int): The number of units to order.
        order_date (str): The date (in ISO format) when the order is placed.

    Returns:
        str: A message confirming the order placement and estimated delivery date, or an error message if the item is not found.
    """
    # Fetch unit price from inventory
    inventory_df = pd.read_sql("SELECT * FROM inventory WHERE item_name = :item_name", db_engine, params={"item_name": item_name})
    
    if inventory_df.empty:
        return f"Error: Item '{item_name}' not found in inventory."

    unit_price = inventory_df.iloc[0]["unit_price"]
    total_cost = unit_price * quantity

    # Create stock order transaction
    create_transaction(
        item_name=item_name,
        transaction_type="stock_orders",
        quantity=quantity,
        price=total_cost,
        date=order_date
    )

    # Estimate delivery date
    delivery_date = get_supplier_delivery_date(order_date, quantity)

    return f"Order placed for {quantity} units of '{item_name}' at a total cost of ${total_cost:.2f}. Estimated delivery date: {delivery_date}."

@tool
def get_inventory_snapshot(as_of_date: str) -> Dict:
    """
    Tool for retrieving a comprehensive snapshot of the current inventory as of a specific date.

    This function provides a detailed overview of all items in the inventory, including their
    current stock levels, unit prices, and total value based on the stock quantity. It also
    calculates the overall inventory valuation.

    Args:
        as_of_date (str): The date (in ISO format) to retrieve the inventory snapshot for.

    Returns:
        Dict: A dictionary containing:
            - 'inventory': A list of dictionaries for each item with keys 'item_name', 'stock', 'unit_price', and 'value'.
            - 'total_inventory_value': The total value of all inventory items combined.
    """
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    snapshot = []
    total_value = 0.0

    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        total_value += item_value

        snapshot.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    return {
        "inventory": snapshot,
        "total_inventory_value": total_value,
    }


# Tools for quoting agent

@tool
def generate_quote(order_size: str, as_of_date: str, items_requested: Dict[str, int] = {}, event_type: str ="") -> Dict:
    
    """
    Tool for generating a pricing quote for a customer paper order, applying bulk discounts.
    
    Discount tiers:
        - Small (1-100 units): No discount
        - Medium (101-500 units): 5% discount
        - Large (501-1000 units): 10% discount
        - Extra Large (1001+ units): 15% discount
        
    Args:
        order_size (str): The size category of the order (e.g., 'small', 'medium', 'large', 'extra_large').
        as_of_date (str): The date (in ISO format) to calculate inventory and pricing as of.
        items_requested (Dict[str, int], optional): A dictionary mapping item names to quantities requested. Default is an empty dict.
        event_type (str, optional): The type of event for which the quote is being generated (e.g., 'wedding', 'corporate event'). Default is an empty string.
        
    Returns:
        Dict: A dictionary containing:
            - 'total_amount': The final quoted price after applying discounts.
            - 'quote_explanation': A breakdown of the pricing calculation, including any discounts applied.
    """
    
    discount_mapping = {
        "small": 0.0,
        "medium": 0.05,
        "large": 0.10,
        "extra_large": 0.15
    }
    discount_rate = discount_mapping.get(order_size.lower(), 0.0)
    
    inv_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    price_map = dict(zip(inv_df["item_name"], inv_df["unit_price"]))
    all_inventory = get_all_inventory(as_of_date)
    
    missing_items = [item for item in items_requested if inventory.get(item,0) < qty]
    
    
    if missing_items:
        return {
            "fulfilled": False,
            "total_amount": 0.0,
            "items_sold": {},
            "reason": f"Insufficient stock for: {', '.join(missing_items)}",
            "customer_message": (
                "We’re unable to fully fulfill your request at this time due to "
                f"limited stock of: {', '.join(missing_items)}."
            )
        }

    subtotal = sum(price_map[item] * qty for item, qty in items_requested.items())

    total = round(subtotal * (1 - discount_rate), 2)

    discount_msg = (
        f"A {int(discount_rate * 100)}% bulk discount has been applied."
        if discount_rate > 0 else
        "No bulk discount applies for this order size."
    )

    return {
        "fulfilled": True,
        "total_amount": total,
        "items_sold": items_requested,
        "reason": "",
        "customer_message": (
            f"Your order can be fulfilled. {discount_msg} "
            f"The total price is ${total:.2f}."
        )
    }
    
       
@tool
def search_quote_history_tool(search_terms: List[str], limit: int = 5) -> List[Dict[str, any]]:
    """
    Tool for searching historical quotes based on provided search terms.

    This function allows users to query past quotes by matching keywords against the original customer requests and quote explanations. It returns a list of relevant quotes sorted by most recent order date.

    Args:
        search_terms (List[str]): A list of keywords to search for in the quote history.
        limit (int, optional): The maximum number of matching quotes to return. Default is 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a matching quote with fields such as 'original_request', 'total_amount', 'quote_explanation', 'job_type', 'order_size', 'event_type', and 'order_date'.
    """
    # Implementation is handled by the underlying function defined above.
    return search_quote_history(search_terms, limit)


# Tools for ordering agent

@tool
def get_cash_balance_tool(as_of_date: str) -> float:
    """
    Tool for retrieving the current cash balance as of a specific date.

    This function calculates the net cash balance by summing all sales revenue and subtracting all stock purchase costs recorded in the transactions table up to the given date.

    Args:
        as_of_date (str): The date (in ISO format) to calculate the cash balance as of.

    Returns:
        float: The net cash balance available as of the specified date.
    """
    return get_cash_balance(as_of_date)

@tool
def generate_financial_report_tool(as_of_date: str) -> Dict:
    """
    Tool for generating a comprehensive financial report as of a specific date.

    This function compiles key financial metrics including cash balance, inventory valuation, total assets, and identifies top-selling products. It provides a detailed snapshot of the company's financial health at the given point in time.

    Args:
        as_of_date (str): The date (in ISO format) to generate the financial report for.

    Returns:
        Dict: A dictionary containing:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    return generate_financial_report(as_of_date)

@tool
def place_order(items_sold: Dict, total_price: float, order_date: str) -> str:
    """
    Tool for recording a sales transaction when an order is placed.

    This function creates a new transaction record in the database for the sale of specified items at a given total price and date. It also returns a confirmation message summarizing the order details.

    Args:
        items_sold (Dict): A dictionary mapping item names to quantities sold.
        total_price (float): The total price of the order.
        order_date (str): The date (in ISO format) when the order is placed.

    Returns:
        str: A confirmation message indicating that the order has been recorded, including the total price and items sold.
    """
    for item_name, quantity in items_sold.items():
        create_transaction(
            item_name=item_name,
            transaction_type="sales",
            quantity=quantity,
            price=total_price,  # Assuming total_price is for the entire order; adjust if needed
            date=order_date
        )
    return f"Order recorded for items: {items_sold} with a total price of ${total_price:.2f} on {order_date}."

@tool
def get_sales_report(as_of_date: str) -> Dict:
    """
    Tool for generating a sales report as of a specific date.

    This function retrieves all sales transactions up to the given date and compiles a report that includes total revenue, number of transactions, and a breakdown of items sold.

    Args:
        as_of_date (str): The date (in ISO format) to generate the sales report for.
    Returns:
        Dict: A dictionary containing:
            - 'total_revenue': The total revenue from sales up to the given date.
            - 'transaction_count': The number of sales transactions recorded.
            - 'items_sold': A dictionary mapping item names to total quantities sold.
    """
    sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
    """
    sales_data = pd.read_sql(sales_query, db_engine, params={"date": as_of_date})
    
    total_revenue = sales_data["total_revenue"].sum()
    transaction_count = len(sales_data)
    items_sold = dict(zip(sales_data["item_name"], sales_data["total_units"]))
    
    return {
        "total_revenue": total_revenue,
        "transaction_count": transaction_count,
        "items_sold": items_sold,
    }


# Set up your agents and create an orchestration agent that will manage them.

inventory_agent = ToolCallingAgent(
    name="InventoryAgent",
    model=model,
    tools=[check_inventory, reorder_inventory, get_inventory_snapshot],
    description="Agent responsible for managing inventory levels, checking stock, and placing orders to suppliers."
)

quoting_agent = ToolCallingAgent(
    name="QuotingAgent",
    model=model,
    tools=[generate_quote, search_quote_history_tool],
    description="Agent responsible for generating pricing quotes for customer orders, applying discounts, and searching historical quotes for reference."
)

finance_agent = ToolCallingAgent(
    name="FinanceAgent",
    model=model,
    tools=[get_cash_balance_tool, generate_financial_report_tool],
    description="Agent responsible for monitoring the company's financial health, including cash balance and inventory valuation, and generating comprehensive financial reports."
)

ordering_agent = ToolCallingAgent(
    name="OrderingAgent",
    model=model,
    tools=[place_order, get_sales_report],
    description="Agent responsible for processing customer orders, recording sales transactions, and generating sales reports."
)

@tool
def ask_inventory_agent(request: str) -> str:
    """
    Tool for the Orchestrator to query the InventoryAgent with a specific request.

    This function allows the Orchestrator to send a request to the InventoryAgent and receive a response, facilitating communication between agents in the multi-agent system.

    Args:
        request (str): The request or inquiry to be sent to the InventoryAgent.

    Returns:
        str: The response generated by the InventoryAgent after processing the request.
    """
    return str(inventory_agent.run(request, max_tokens=500))

@tool
def ask_quoting_agent(request: str) -> str:
    """
    Tool for the Orchestrator to query the QuotingAgent with a specific request.

    This function allows the Orchestrator to send a request to the QuotingAgent and receive a response, facilitating communication between agents in the multi-agent system.

    Args:
        request (str): The request or inquiry to be sent to the QuotingAgent.

    Returns:
        str: The response generated by the QuotingAgent after processing the request.
    """
    return str(quoting_agent.run(request))

@tool
def ask_ordering_agent(request: str) -> str:
    """
    Tool for the Orchestrator to query the OrderingAgent with a specific request.

    This function allows the Orchestrator to send a request to the OrderingAgent and receive a response, facilitating communication between agents in the multi-agent system.

    Args:
        request (str): The request or inquiry to be sent to the OrderingAgent.

    Returns:
        str: The response generated by the OrderingAgent after processing the request.
    """
    return str(ordering_agent.run(request))

@tool
def ask_finance_agent(request: str) -> str:
    """
    Tool for the Orchestrator to query the FinanceAgent with a specific request.

    This function allows the Orchestrator to send a request to the FinanceAgent and receive a response, facilitating communication between agents in the multi-agent system.

    Args:
        request (str): The request or inquiry to be sent to the FinanceAgent.

    Returns:
        str: The response generated by the FinanceAgent after processing the request.
    """
    return str(finance_agent.run(request))


orchestrator = ToolCallingAgent(
    name="Orchestrator",
    model=model,
    tools=[ask_inventory_agent, ask_quoting_agent, ask_finance_agent, ask_ordering_agent],
    description="Central agent responsible for coordinating worker agents and delegating tasks to them."
)

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator agent for Beaver's Choice Paper Company. Your role is to manage and coordinate the InventoryAgent, QuotingAgent, and OrderingAgent to fulfill customer requests effectively.

When a customer request comes in, you will:
1. Analyze the request to understand the customer's needs, including the type of paper, quantity, and event context.
2. Check inventory levels using the InventoryAgent to determine if the requested items are  available.
3. If inventory is insufficient, instruct the InventoryAgent to place a stock order to replenish the inventory.
4. Use the QuotingAgent to generate a pricing quote for the customer based on the requested items and quantities, applying any relevant discounts.
5. If the customer accepts the quote, coordinate with the OrderingAgent to record the sale and update inventory levels accordingly.
6. Continuously monitor inventory and sales data to ensure that the company maintains optimal stock levels and meets customer demand efficiently. Always provide clear and concise responses to customer inquiries, and

You must ALWAYS return valid JSON with the following fields:
{
  "fulfilled": true/false,
  "total_amount": float,
  "items_sold": { "item_name": quantity, ... },
  "reason": "string if unfulfilled",
  "customer_message": "string"
}

"""

def call_orchestrator(request: str) -> str:
    """
    Function to send a customer request to the Orchestrator agent and receive a response.

    Args:
        request (str): The customer's request or inquiry that needs to be processed.

    Returns:
        str: The response generated by the Orchestrator agent after processing the request through the appropriate tools and agents.
    """
    
    prompt = ORCHESTRATOR_SYSTEM_PROMPT + "\n\nCustomer Request: " + request
    
    response = orchestrator.run(prompt)
    
    return str(response)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def summarize_results(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    df.to_csv("test_results.csv", index=False)

    fulfilled_count = int(df["fulfilled"].sum())
    cash_changes = int(df["cash_balance"].diff().fillna(0).ne(0).sum())
    unfulfilled_df = df[df["fulfilled"] == False]

    print("\n=== Evaluation Summary ===")
    print(f"Fulfilled requests: {fulfilled_count}")
    print(f"Cash balance changes: {cash_changes}")
    print(f"Unfulfilled requests: {len(unfulfilled_df)}")

    if len(unfulfilled_df) > 0:
        print("\nSample unfulfilled requests:")
        print(unfulfilled_df[["request_id", "unfulfilled_reason"]].head())

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    
    inv_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    price_map = dict(zip(inv_df["item_name"], inv_df["unit_price"]))

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")
        order_size = str(row.get("order_size", "small")).lower().strip()

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"
        response = call_orchestrator(request_with_date)

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        # response = call_your_multi_agent_system(request_with_date)
        
        try:
            response_data = json.loads(response)
            """
            total_amount = response_dict.get("total_amount", 0.0)
            items_sold = response_dict.get("items_sold", {})
            if items_sold and total_amount > 0:
                place_order(items_sold, total_amount, request_date)
            """
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Raw response: {response}")
            
        fulfilled = response_data.get("fulfilled", False)
        total_amount = response_data.get("total_amount", 0.0)
        items_sold = response_data.get("items_sold", {})
        reason = response_data.get("reason", "")
        
        if fulfilled and items_sold and total_amount > 0:
            order_response = place_order(items_sold, total_amount, request_date)
            print(f"Order Response: {order_response}")
            
            

        # Update state
        report = generate_financial_report_tool(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "fulfilled": fulfilled,
                "total_amount": total_amount,
                "items_sold": json.dumps(items_sold),
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "unfulfilled_reason": reason,
                "response": json.dumps(response_data)
            }
        )

        summarize_results(results)


        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
