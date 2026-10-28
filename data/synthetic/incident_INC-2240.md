# Incident INC-2240: Delayed payment emails on March 24, 2026

After the notification-service was migrated to the new message queue cluster (change CHG-1063), payment confirmation emails were delayed by up to 40 minutes.

Payment processing itself was not affected. Severity: SEV-3.
