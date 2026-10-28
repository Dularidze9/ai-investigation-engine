# Runbook: payment-gateway

The payment-gateway database connection pool must never be set below 150 connections in production. Below that, traffic peaks cause request queueing.

If gateway latency rises, first check fraud-check latency, because the gateway calls it synchronously.
