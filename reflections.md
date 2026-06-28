# Reflections

## Workflow Diagram Explanation

The core idea here is that each domain is handled by a specialized agents managed by the Orchestrator Agent.
### The Finance Agent
This agent provides financial visibility and ensures the business can afford actions.
I arrived at the conclusion there would need to be an agent to handle the cash flow unrelated to the sales orders.
This helps isolate the sensitive infomation in `get_cash_balance` and `generate_financial_report` from processes that don't require it.
These functions are not needed as part of the inventory logic, quoting heuristics or order creation.
### The Inventory Agent
This agent tracks stock levels and manages replenishment.
Logically, there needs to be an agent that's sole job is to handle the inventory.
This would utilize the functions `get_stock_level`, `reorder_inventory`, and `check_inventory` to deal with the changing inventory, supplier lead times, restockign logic and physical constraints. If the inventory functions are mixed with orders or quoting it could lead to the quoting agent hallucinating stock levels or the ordering agent reordering inventory incorrectly. Keeping the inentory operations separate help prevent potential circular logic problems.
### The Quoting Agent
This agent generates price quotes based on history, inventory, and request details.
The quoting agent needed to be it's own agent to handle quote history, inventory prices, event types, and order sizes.
These are heavy reasoning functions that should not be complicated with placing orders, manging inventory or accessing financial reports.
If the quoting agent could place orders it could place orders before the client approves the quote.
This agent needs to be advisory, not authoritative.
### The Ordering Agent
This agent handles sales reporting and creating new orders.
I thought about naming it Sales Agent, but perfered the semantics of Ordering Agent.
This agent uses the `create_transaction` function to write-operations with real consequences.
The process of placing orders(sales) where financial transactions happen needs to be isolated.
The agent changes the system of record, it must only happen after validation and must not be triggered by quoting or inventory logic.
The Quoting Agent calling `place_order` would lead to chaos, and the Inventory agent would cause phantom orders.
One agent needs to be allowed to commit transactions and be isolated from other functions.

The Orchestrator Agent receives the Customer Request and breaks it into subtasks.  
It identifies each subtask and sends it to the correct specialist agent.  
Then it collects all the responses and compiles them into a Customer-Facing Response.

An example workflow: 
1. Orchestrator reaches out to Inventory Agent to “Check Stock”
2. Orchestrator then reaches out to Quoting Agent to “Generate Quote”
3. Orchestrator asks the Finance Agent to “Check Finance” (if needed)
4. Orchestrator sends the order to the Ordering Agent to “Process Order” (if customer accepts the quote)
5. Orchestrator provides final customer-facing response

## Test Results

All 20 quotes were processed with the systemfinishing with a cash balanace of $43224.70 and inventory value of $4940.30.  
Some strengths noted in the system:  
Five request show a change to the cash balance(IDs 7, 8, 9, 15, 19).  
The system also showed the partially fulfilled items for request IDs 8, 17, and 19.  
The response indicates where stock is insufficient and when restock is expected.  

## Improvements
The response output doesn't indicate exact reasons why a request isn't fulfilled, whether from lack of inventory or items not solde by the company.
Another improvement would be to output all fulfilled items, at present it appears to only display the partially filled orders.
