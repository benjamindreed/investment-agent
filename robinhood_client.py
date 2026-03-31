"""
Robinhood integration via robin_stocks.

Wraps authentication, portfolio lookup, and order placement behind
a simple stateful class so the Streamlit app can stay clean.
"""
import os
import robin_stocks.robinhood as rh


class RobinhoodClient:
    def __init__(self):
        self._logged_in = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, username: str, password: str, mfa_code: str = "") -> tuple[bool, str]:
        """
        Log in to Robinhood.  Returns (success, message).
        Stores session so subsequent calls work without re-authenticating.
        """
        try:
            kwargs = dict(username=username, password=password, store_session=True)
            if mfa_code:
                kwargs["mfa_code"] = mfa_code
            rh.login(**kwargs)
            self._logged_in = True
            return True, "Logged in successfully."
        except Exception as e:
            self._logged_in = False
            return False, str(e)

    def logout(self):
        try:
            rh.logout()
        except Exception:
            pass
        self._logged_in = False

    @property
    def logged_in(self) -> bool:
        return self._logged_in

    # ------------------------------------------------------------------
    # Portfolio / account info
    # ------------------------------------------------------------------

    def get_buying_power(self) -> float:
        """Return available buying power in USD."""
        try:
            profile = rh.profiles.load_account_profile()
            return float(profile.get("buying_power") or 0)
        except Exception:
            return 0.0

    def get_holdings(self) -> list[dict]:
        """Return current stock positions as a list of dicts."""
        try:
            positions = rh.account.build_holdings()
            results = []
            for ticker, data in positions.items():
                results.append(
                    {
                        "ticker": ticker,
                        "name": data.get("name", ticker),
                        "quantity": float(data.get("quantity", 0)),
                        "average_buy_price": float(data.get("average_buy_price", 0)),
                        "equity": float(data.get("equity", 0)),
                        "percent_change": float(data.get("percent_change", 0)),
                    }
                )
            return sorted(results, key=lambda x: x["equity"], reverse=True)
        except Exception:
            return []

    def get_quote(self, ticker: str) -> dict:
        """Fetch a live quote for a single ticker."""
        try:
            q = rh.stocks.get_quotes(ticker)[0]
            return {
                "ask_price": float(q.get("ask_price") or 0),
                "bid_price": float(q.get("bid_price") or 0),
                "last_trade_price": float(q.get("last_trade_price") or 0),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_buy(self, ticker: str, shares: float) -> dict:
        """
        Place a fractional market buy order.

        Args:
            ticker: Stock symbol.
            shares: Number of shares (can be fractional).

        Returns dict with order details or {"error": "..."}.
        """
        try:
            order = rh.orders.order_buy_fractional_by_quantity(
                symbol=ticker,
                quantity=shares,
                timeInForce="gfd",
            )
            if order and "id" in order:
                return {
                    "order_id": order["id"],
                    "state": order.get("state", "unknown"),
                    "quantity": shares,
                    "ticker": ticker,
                    "type": "market_buy",
                }
            return {"error": f"Unexpected response: {order}"}
        except Exception as e:
            return {"error": str(e)}

    def place_limit_buy(self, ticker: str, shares: float, limit_price: float) -> dict:
        """
        Place a limit buy order.

        Args:
            ticker: Stock symbol.
            shares: Number of shares (fractional supported).
            limit_price: Maximum price per share.

        Returns dict with order details or {"error": "..."}.
        """
        try:
            order = rh.orders.order_buy_fractional_by_quantity(
                symbol=ticker,
                quantity=shares,
                timeInForce="gtc",
                limitPrice=limit_price,
            )
            if order and "id" in order:
                return {
                    "order_id": order["id"],
                    "state": order.get("state", "unknown"),
                    "quantity": shares,
                    "limit_price": limit_price,
                    "ticker": ticker,
                    "type": "limit_buy",
                }
            return {"error": f"Unexpected response: {order}"}
        except Exception as e:
            return {"error": str(e)}

    def place_dollar_buy(self, ticker: str, amount_usd: float) -> dict:
        """
        Place a fractional buy by dollar amount.

        Args:
            ticker: Stock symbol.
            amount_usd: Dollar amount to invest.

        Returns dict with order details or {"error": "..."}.
        """
        try:
            order = rh.orders.order_buy_fractional_by_price(
                symbol=ticker,
                amountInDollars=amount_usd,
                timeInForce="gfd",
            )
            if order and "id" in order:
                return {
                    "order_id": order["id"],
                    "state": order.get("state", "unknown"),
                    "amount_usd": amount_usd,
                    "ticker": ticker,
                    "type": "dollar_buy",
                }
            return {"error": f"Unexpected response: {order}"}
        except Exception as e:
            return {"error": str(e)}

    def get_open_orders(self) -> list[dict]:
        """Return all open/pending orders."""
        try:
            orders = rh.orders.get_all_open_stock_orders()
            results = []
            for o in orders:
                results.append(
                    {
                        "order_id": o.get("id"),
                        "ticker": rh.stocks.get_symbol_by_url(o.get("instrument", "")),
                        "side": o.get("side"),
                        "quantity": o.get("quantity"),
                        "price": o.get("price"),
                        "state": o.get("state"),
                        "created_at": o.get("created_at"),
                    }
                )
            return results
        except Exception:
            return []

    def cancel_order(self, order_id: str) -> tuple[bool, str]:
        """Cancel an open order by ID."""
        try:
            result = rh.orders.cancel_stock_order(order_id)
            return True, "Order cancelled."
        except Exception as e:
            return False, str(e)
