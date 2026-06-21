# Reflections

## Workflow Diagram Explanation

The core idea here is that each domain is handled by a specialized agents managed by the Orchestrator Agent.
- The Finance Agent provides financial visibility and ensures the business can afford actions.
- The Ordering Agent handle sales reporting and creating new orders
- The Quoting Agent Generate price quotes based on history, inventory, and request details.
- The Inventory Agent tracks stock levels and manages replenishment.

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
