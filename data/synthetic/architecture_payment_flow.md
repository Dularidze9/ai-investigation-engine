# Payment flow architecture

A card payment passes through these services in order: checkout-api, then payment-gateway, then fraud-check, then card-network-connector.

The payment-gateway waits for fraud-check synchronously, so any latency in fraud-check directly slows every payment.

After authorization, the ledger-service records the payment asynchronously and the notification-service emails the customer. Neither is on the critical path of authorization.
