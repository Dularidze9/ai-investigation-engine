"""Public-demo spending cap, stored in the database so it survives restarts and is shared by workers.

reserve() books the worst-case cost of a query BEFORE running it, in one atomic UPDATE: two requests
racing for the last cents cannot both pass, because the row lock serializes them.
settle() replaces the reservation with the real cost afterwards.
"""
MAX_QUERY_COST = 0.02  # worst-case reservation per question (a typical question costs far less)


def reserve(conn, amount: float = MAX_QUERY_COST) -> bool:
    row = conn.execute(
        "UPDATE spend SET usd = usd + %s WHERE id = 1 AND usd + %s <= cap_usd RETURNING usd",
        (amount, amount)).fetchone()
    conn.commit()
    return row is not None


def settle(conn, reserved: float, actual: float) -> None:
    conn.execute("UPDATE spend SET usd = usd - %s + %s WHERE id = 1", (reserved, actual))
    conn.commit()


def status(conn) -> dict:
    usd, cap = conn.execute("SELECT usd, cap_usd FROM spend WHERE id = 1").fetchone()
    return {"spent_usd": float(usd), "cap_usd": float(cap)}
