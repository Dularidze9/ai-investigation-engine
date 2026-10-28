# Acme Payments Change Log, March 2026

CHG-1041 (March 3, 2026): Upgraded the ledger-service database driver to version 5.2. Owner: Ledger team. No customer impact expected.

CHG-1047 (March 9, 2026): Reduced the payment-gateway database connection pool from 200 to 50 connections to lower infrastructure cost. Owner: Payments Core team.

CHG-1052 (March 10, 2026): Enabled fraud-scoring model v3 in the fraud-check service. The payment-gateway calls fraud-check synchronously with a 2 second timeout. Owner: Risk team.

CHG-1060 (March 18, 2026): Rotated the TLS certificates of the card-network-connector. Owner: Payments Core team.

CHG-1063 (March 24, 2026): Migrated the notification-service to the new message queue cluster. Owner: Messaging team.
