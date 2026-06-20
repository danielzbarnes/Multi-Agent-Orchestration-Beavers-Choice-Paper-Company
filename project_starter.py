import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import re
import csv
from dotenv import load_dotenv
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

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

load_dotenv()

MODEL_ID = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")


# -----------------------
# FINANCE TOOLS
# -----------------------

@tool
def get_cash_balance_tool(as_of_date: str) -> float:
    """
    Return the cash balance as of a given date.

    Args:
        as_of_date (str): ISO date string (YYYY-MM-DD) representing the cutoff date.

    Returns:
        float: Cash balance as of the given date.
    """
    
    return get_cash_balance(as_of_date)

@tool
def generate_financial_report_tool(as_of_date: str) -> dict:
    """
    Generate a full financial report as of a given date.

    Args:
        as_of_date (str): ISO date string (YYYY-MM-DD).

    Returns:
        dict: Financial report including cash, inventory value, and assets.
    """
    
    return generate_financial_report(as_of_date)

# -----------------------
# INVENTORY TOOLS
# -----------------------

@tool
def get_inventory_snapshot_tool(as_of_date: str) -> dict:
    """
    Return inventory snapshot as of a given date.

    Args:
        as_of_date (str): ISO date string (YYYY-MM-DD).

    Returns:
        dict: Mapping of item_name → stock.
    """
    
    return get_all_inventory(as_of_date)

@tool
def get_stock_level_tool(item_name: str, as_of_date: str) -> dict:
    """
    Return stock level for a specific item.

    Args:
        item_name (str): Name of the item.
        as_of_date (str): ISO date string (YYYY-MM-DD).

    Returns:
        dict: { "item_name": str, "current_stock": int }
    """
    
    df = get_stock_level(item_name, as_of_date)
    return {"item_name": item_name, "current_stock": int(df["current_stock"].iloc[0])}

@tool
def reorder_inventory_tool(item_name: str, quantity: int, order_date: str) -> dict:
    """
    Compute supplier delivery date and return reorder info.

    Args:
        item_name (str): Name of the item to reorder.
        quantity (int): Number of units to order.
        order_date (str): ISO date string (YYYY-MM-DD).

    Returns:
        dict: { item_name, quantity, delivery_date }
    """
    
    delivery = get_supplier_delivery_date(order_date, quantity)
    return {
        "item_name": item_name,
        "quantity": quantity,
        "delivery_date": delivery,
    }

# -----------------------
# QUOTING TOOLS
# -----------------------

@tool
def search_quote_history_tool(search_terms: list, limit: int = 5) -> list:
    """
    Search historical quotes for matching terms.

    Args:
        search_terms (list): Keywords to search for.
        limit (int): Max number of results.

    Returns:
        list: Matching quote records.
    """
    
    return search_quote_history(search_terms, limit)

@tool
def generate_quote_tool(items: list, as_of_date: str, event_type: str) -> dict:
    """
    Generate a quote using inventory and pricing rules.

    Args:
        items (list): List of { item_name, quantity }.
        as_of_date (str): ISO date string.
        event_type (str): Event type (ceremony, parade, etc.).

    Returns:
        dict: Quote with totals and line items.
    """
    
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    snapshot = get_all_inventory(as_of_date)

    margin = 0.30
    total = 0
    line_items = []

    for item in items:
        name = item["item_name"]
        qty = item["quantity"]

        row = inventory_df[inventory_df["item_name"].str.lower() == name.lower()]
        unit_price = float(row["unit_price"].iloc[0]) if not row.empty else 0.10

        base = unit_price * qty
        line_total = base * (1 + margin)
        total += line_total

        line_items.append({
            "item_name": name,
            "requested_qty": qty,
            "available_stock": snapshot.get(name, 0),
            "unit_price": unit_price,
            "line_total": round(line_total, 2),
        })

    return {
        "total_amount": round(total, 2),
        "line_items": line_items,
        "quote_explanation": f"Margin {margin:.0%}, event={event_type}",
    }


# -----------------------
# ORDERING TOOLS
# -----------------------

@tool
def place_sales_order_tool(items_sold: list, total_price: float, order_date: str) -> dict:
    """
    Record a sales transaction.

    Args:
        items_sold (list): Line items sold.
        total_price (float): Total sale amount.
        order_date (str): ISO date string.

    Returns:
        dict: Status message.
    """
    
    create_transaction(
        item_name=None,
        transaction_type="sales",
        quantity=None,
        price=total_price,
        date=order_date,
    )
    return {"status": "sale_recorded"}

@tool
def place_stock_order_tool(item_name: str, quantity: int, unit_price: float, order_date: str) -> dict:
    """
    Record a stock order transaction.

    Args:
        item_name (str): Item to order.
        quantity (int): Units to order.
        unit_price (float): Price per unit.
        order_date (str): ISO date string.

    Returns:
        dict: Stock order details.
    """
    
    total_price = quantity * unit_price
    create_transaction(
        item_name=item_name,
        transaction_type="stock_orders",
        quantity=quantity,
        price=total_price,
        date=order_date,
    )
    return {
        "status": "stock_ordered",
        "item_name": item_name,
        "quantity": quantity,
        "total_price": total_price,
    }
    
# -----------------------
# AGENTS
# -----------------------

SHARED_MODEL = OpenAIServerModel(
    model_id=MODEL_ID,
    api_key=API_KEY,
    api_base=BASE_URL
)

class BaseWorkerAgent(ToolCallingAgent):
    def __init__(self, tools, name, description):
        super().__init__(
            model=SHARED_MODEL,
            tools=tools,
            name=name,
            description=description,        
        )
        
        try:
            self.final_answer_tool = None
        except Exception:
            pass
        try:
            self._final_answer_tool = None
        except Exception:
            pass
        
class FinanceAgent(BaseWorkerAgent):
    def __init__(self):
        super().__init__(
            tools=[get_cash_balance_tool, generate_financial_report_tool],
            name="finance_agent",
            description="Handles cash balance, spending checks, and financial reports."
        )

class InventoryAgent(BaseWorkerAgent):
    def __init__(self):
        super().__init__(
            tools=[
                get_inventory_snapshot_tool,
                get_stock_level_tool,
                reorder_inventory_tool,
            ],
            name="inventory_agent",
            description="Checks stock levels and handles reorders."
        )

class QuotingAgent(BaseWorkerAgent):
    def __init__(self):
        super().__init__(
            tools=[
                search_quote_history_tool,
                generate_quote_tool,
            ],
            name="quoting_agent",
            description="Generates quotes using inventory and historical data."
        )


class OrderingAgent(BaseWorkerAgent):
    def __init__(self):
        super().__init__(
            tools=[
                place_sales_order_tool,
                place_stock_order_tool,
            ],
            name="ordering_agent",
            description="Places sales orders and stock orders."
        )
        
# -----------------------
# ORCHESTRATOR
# -----------------------
class Orchestrator:
    def __init__(self):
        self.finance = FinanceAgent()
        self.inventory = InventoryAgent()
        self.quoting = QuotingAgent()
        self.ordering = OrderingAgent()

    def run(self, structured_request: dict):
        """
        Main orchestrator entrypoint.
        Calls worker agents in the correct order and returns a unified response.
        """

        as_of = structured_request["as_of_date"]
        items = structured_request["items"]
        event_type = structured_request.get("event_type")

        # 1. Generate quote (agent call)
        raw_quote = call_agent_tool_strict(
            self.quoting,
            f"Call generate_quote_tool with items={items}, as_of_date='{as_of}', event_type='{event_type}'."
        )
        quote = ensure_dict_from_agent_result(self.quoting, raw_quote)
        """ 
        raw_quote = self.quoting.run(
            f"Use ONLY generate_quote_tool. Do NOT produce a natural-language final answer. "
            f"Call generate_quote_tool with items={items}, as_of_date='{as_of}', event_type='{event_type}'."
        )
        quote = ensure_dict_from_agent_result(self.quoting, raw_quote)
        """

        # 2. Reorder missing items based on quote's available_stock
        reorders = []
        for line in quote.get("line_items", []):
            # ensure numeric types
            try:
                available = int(line.get("available_stock", 0))
            except Exception:
                available = int(float(line.get("available_stock", 0) or 0))
            try:
                requested = int(line.get("requested_qty", 0))
            except Exception:
                requested = int(float(line.get("requested_qty", 0) or 0))

            if available < requested:
                missing = requested - available

                raw_reorder = self.inventory.run(
                    f"Use ONLY reorder_inventory_tool. Do NOT produce a natural-language final answer. "
                    f"Call reorder_inventory_tool with item_name='{line['item_name']}', quantity={missing}, order_date='{as_of}'."
                )
                reorder_info = ensure_dict_from_agent_result(self.inventory, raw_reorder)

                # place stock order (we don't need its return value here, but keep agent call)
                self.ordering.run(
                    f"Use ONLY place_stock_order_tool. Do NOT produce a natural-language final answer. "
                    f"Call place_stock_order_tool with item_name='{line['item_name']}', quantity={missing}, unit_price={line['unit_price']}, order_date='{as_of}'."
                )

                reorders.append(reorder_info)

        return {
            "quote": quote,
            "reorders": reorders,
            "delivery_date_requested": structured_request.get("delivery_date"),
        }


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def parse_order_like_string(s: str) -> Optional[Dict]:
    """
    Try to extract structured fields from common tool strings.
    Returns dict or None.
    Examples handled:
      - "Order for 1000 units of matte A3 paper placed. Delivery date: 2025-04-11"
      - "Order placed: 1000 matte A3 paper; delivery 2025-04-11"
      - "Placed order: 50 x A4 glossy paper (delivery 2025-04-08)"
    """
    if not isinstance(s, str):
        return None

    # Normalize whitespace
    text = " ".join(s.split())

    # Pattern 1: "Order for 1000 units of NAME placed. Delivery date: YYYY-MM-DD"
    m = re.search(r"Order\s+for\s+(\d+)\s+(?:units\s+of\s+)?(.+?)\s+placed[.\s].*?Delivery\s+date[:\s]+(\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
    if m:
        return {"item_name": m.group(2).strip(), "quantity": int(m.group(1)), "delivery_date": m.group(3), "raw_text": s}

    # Pattern 2: "Order placed: 1000 matte A3 paper; delivery 2025-04-11"
    m = re.search(r"Order\s+placed[:\s]+(\d+)\s+(.+?)[;,\s]+delivery[:\s]+(\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
    if m:
        return {"item_name": m.group(2).strip(), "quantity": int(m.group(1)), "delivery_date": m.group(3), "raw_text": s}

    # Pattern 3: "Placed order: 50 x A4 glossy paper (delivery 2025-04-08)"
    m = re.search(r"(\d+)\s*[xX]\s*(.+?)\s*\(.*?(\d{4}-\d{2}-\d{2}).*?\)", text)
    if m:
        return {"item_name": m.group(2).strip(), "quantity": int(m.group(1)), "delivery_date": m.group(3), "raw_text": s}

    # Pattern 4: "Delivery date: YYYY-MM-DD" and "qty" somewhere
    m_qty = re.search(r"(\d+)\s+(?:units|qty|quantity)\b", text, flags=re.IGNORECASE)
    m_date = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m_qty and m_date:
        # best-effort item name: text between qty and 'delivery' or end
        qty = int(m_qty.group(1))
        # try to find a plausible item name (words around qty)
        # fallback to raw_text if we can't
        name_match = re.search(r"(?:for|of)\s+(.+?)(?:\s+placed|\s+delivery|\s+on|$)", text, flags=re.IGNORECASE)
        item_name = name_match.group(1).strip() if name_match else None
        return {"item_name": item_name or "unknown", "quantity": qty, "delivery_date": m_date.group(1), "raw_text": s}

    return None


def ensure_dict_from_agent_result(agent, result: Any) -> Dict:
    """
    Ensure we get a dict from an agent call.
    - If result is dict -> return it.
    - If result is str -> try ast.literal_eval, json.loads, then pattern parsing.
    - If that fails -> try agent.memory.steps[-1].observations and repeat parsing.
    - If still fails -> raise ValueError.
    """
    # 1) Already a dict
    if isinstance(result, dict):
        return result

    # 2) If it's a string, try structured parsing
    if isinstance(result, str):
        s = result.strip()

        # 2a) Try JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2b) Try Python literal eval (safe-ish)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 2c) Try to extract a dict-like substring { ... } and eval it
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass

        # 2d) Try order-like string parsing
        parsed = parse_order_like_string(s)
        if parsed:
            return parsed

        # 2e) FINAL FALLBACK: extract delivery date from plain sentence
        if isinstance(result, str):
            # Look for a date
            m_date = re.search(r"(\d{4}-\d{2}-\d{2})", result)
            if m_date:
                delivery = m_date.group(1)

                # Try to recover item_name from sentence
                m_item = re.search(r"for\s+(.+?)\s+is\s+\d{4}-\d{2}-\d{2}", result, flags=re.IGNORECASE)
                item_name = m_item.group(1).strip() if m_item else "unknown"

                # Quantity cannot be extracted from this sentence, so we fallback
                # to the quantity we *asked* the tool to reorder.
                # We retrieve it from the last tool call arguments:
                try:
                    last_step = agent.memory.steps[-1]
                    args = getattr(last_step.action, "arguments", {})
                    qty = args.get("quantity", None)
                except Exception:
                    qty = None

                return {
                    "item_name": item_name,
                    "quantity": qty,
                    "delivery_date": delivery,
                    "raw_text": result,
                }

    # 3) Fallback: try agent memory last step observations
    try:
        last_step = getattr(agent, "memory", None)
        if last_step is not None:
            steps = getattr(agent.memory, "steps", None)
            if steps and len(steps) > 0:
                obs = getattr(steps[-1], "observations", None)
                if isinstance(obs, dict):
                    return obs
                if isinstance(obs, str):
                    # try same parsing on obs string
                    parsed = ensure_dict_from_agent_result(agent, obs)
                    if isinstance(parsed, dict):
                        return parsed
    except Exception:
        pass

    # 4) Nothing worked
    raise ValueError("Unable to extract dict result from agent run() output or memory.")


def call_agent_tool_strict(agent, prompt: str, retry_prompt: Optional[str] = None) -> Any:
    """
    Call agent.run(prompt). If the result is a string (final answer) or parsing fails,
    retry once with a stricter instruction that forces a tool-only call.
    Returns the raw agent.run() result (not parsed).
    """
    raw = agent.run(prompt)
    # If it's already a dict, return immediately
    if isinstance(raw, dict):
        return raw

    # If it's a string, try to parse quickly; if parsing fails, retry once with strict prompt
    try:
        _ = ensure_dict_from_agent_result(agent, raw)
        return raw
    except Exception:
        # Build a strict retry prompt if not provided
        strict = retry_prompt or (
            "Use ONLY the tool. Do NOT produce a natural-language final answer. "
            + prompt
        )
        raw2 = agent.run(strict)
        return raw2
    
    
def parse_request_from_csv_row(row):
    """
    Convert a row from quote_requests_sample.csv into a structured request
    for the multi-agent orchestrator.
    """
    
    raw_text = row["request"]
    event_type = row["event"]
    order_size = row["need_size"]
    request_date = row["request_date"].strftime("%Y-%m-%d")

    delivery_match = re.search(r"by ([A-Za-z]+\s+\d{1,2},\s*\d{4})", raw_text)
    delivery_date = (
        datetime.strptime(delivery_match.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
        if delivery_match else None
    )

    items = []
    for line in raw_text.splitlines():
        line = line.strip()

        # match patterns like:
        # - 50 sheets of cardstock
        # - 50 cardstock
        # - 50 units cardstock
        m = re.match(r"-?\s*(\d+)\s+(?:\w+\s+)?(?:of\s+)?(.+)", line, flags=re.IGNORECASE)
        if m:
            qty = int(m.group(1))
            name = m.group(2).strip()
            items.append({"item_name": name, "quantity": qty})

    return {
        "type": "quote",
        "as_of_date": request_date,
        "delivery_date": delivery_date,
        "items": items,
        "event_type": event_type,
        "order_size": order_size,
        "raw_text": raw_text,
    }

import json
from typing import Dict, List

import json
from typing import Dict, List

def format_money(v: float) -> str:
    return f"${v:,.2f}"

def quote_total(quote: Dict) -> float:
    return float(quote.get("total_amount", 0.0))

def compute_fulfilled_items_compact(quote: Dict) -> str:
    """
    Returns a compact semicolon-separated string of fulfilled items:
    "A4 glossy paper: 200/200 (fully); heavy cardstock: 0/100 (none)"
    Only include items with any fulfilled quantity (available > 0).
    """
    lines = quote.get("line_items", [])
    entries = []
    for li in lines:
        name = li.get("item_name", "unknown")
        requested = int(li.get("requested_qty", 0))
        available = li.get("available_stock", 0) or 0
        try:
            available = int(float(available))
        except Exception:
            available = 0
        fulfilled = min(available, requested)
        if fulfilled > 0:
            status = "fully" if fulfilled >= requested else "partial"
            entries.append(f"{name}: {fulfilled}/{requested} ({status})")
    return "; ".join(entries)

def build_human_response(quote: Dict, reorders: List[Dict]) -> str:
    """
    Build a short human-friendly sentence describing availability and restocks.
    Prioritize readable phrasing similar to the example the user provided.
    """
    lines = quote.get("line_items", [])
    available_names = []
    insufficient_names = []
    for li in lines:
        name = li.get("item_name", "unknown")
        requested = int(li.get("requested_qty", 0))
        available = li.get("available_stock", 0) or 0
        try:
            available = int(float(available))
        except Exception:
            available = 0
        if available >= requested and requested > 0:
            available_names.append(name)
        elif requested > 0:
            insufficient_names.append(name)

    parts = []
    if available_names:
        # natural join with commas and 'and'
        if len(available_names) == 1:
            parts.append(f"{available_names[0]} is available in sufficient quantity.")
        else:
            last = available_names[-1]
            rest = ", ".join(available_names[:-1])
            parts.append(f"{rest} and {last} are available in sufficient quantities.")
    if insufficient_names:
        # mention restock if reorders exist for those items
        restock_msgs = []
        for r in reorders:
            rn = r.get("item_name", "")
            # normalize names: some reorder parsers include qty text; prefer matching by substring
            for ins in insufficient_names:
                if ins.lower() in rn.lower() or rn.lower() in ins.lower():
                    date = r.get("delivery_date")
                    restock_msgs.append(f"{ins} is insufficient, restock expected {date}.")
                    break
            else:
                # no matching reorder found; generic note
                pass
        if restock_msgs:
            parts.append(" ".join(restock_msgs))
        else:
            # generic insufficient message
            if len(insufficient_names) == 1:
                parts.append(f"{insufficient_names[0]} is insufficient and will require restocking.")
            else:
                last = insufficient_names[-1]
                rest = ", ".join(insufficient_names[:-1])
                parts.append(f"{rest} and {last} are insufficient and will require restocking.")

    if not parts:
        parts.append("No items requested or all items handled.")

    # Compose final short response
    return " ".join(parts)

def build_csv_row(request_id: int,
                  as_of: str,
                  cash_balance: float,
                  inventory_value: float,
                  orchestrator_result: Dict) -> Dict:
    quote = orchestrator_result.get("quote", {})
    reorders = orchestrator_result.get("reorders", [])

    row = {
        "request_id": request_id,
        "request_date": as_of,
        "cash_balance": round(cash_balance, 2),
        "inventory_value": round(inventory_value, 2),
        "quote_total": round(quote_total(quote), 2),
        "fulfilled_items": compute_fulfilled_items_compact(quote),
        "response": build_human_response(quote, reorders)
    }
    return row

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

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    orchestrator = Orchestrator()

    results = []
    
    # CSV writer with your new columns
    with open("test_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "request_date",
                "cash_balance",
                "inventory_value",
                "quote_total",
                "fulfilled_items",
                "response",
            ],
        )
        writer.writeheader()

        for idx, row in quote_requests_sample.iterrows():
            request_date = row["request_date"].strftime("%Y-%m-%d")

            print(f"\n=== Request {idx+1} ===")
            print(f"Context: {row['job']} organizing {row['event']}")
            print(f"Request Date: {request_date}")
            print(f"Cash Balance: ${current_cash:.2f}")
            print(f"Inventory Value: ${current_inventory:.2f}")

            structured_request = parse_request_from_csv_row(row)

            orchestrator_result = orchestrator.run(structured_request)

            # Update financials after the run
            report = generate_financial_report(request_date)
            current_cash = report["cash_balance"]
            current_inventory = report["inventory_value"]

            # Build the CSV row using your new helper
            csv_row = build_csv_row(
                request_id=idx + 1,
                as_of=request_date,
                cash_balance=current_cash,
                inventory_value=current_inventory,
                orchestrator_result=orchestrator_result,
            )

            print(f"Response: {csv_row['response']}")
            print(f"Updated Cash: ${current_cash:.2f}")
            print(f"Updated Inventory: ${current_inventory:.2f}")

            writer.writerow(csv_row)
            results.append(csv_row)

            time.sleep(1)

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    results = run_test_scenarios()