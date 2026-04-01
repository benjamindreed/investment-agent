"""
Multi-brokerage client layer.

Provides a common interface (BaseBrokerageClient) with concrete
implementations for each supported broker. All implementations expose
the same methods so app.py never needs to know which broker is active.

Supported brokers
-----------------
  Robinhood  — unofficial API via robin-stocks (email + password + MFA)
  Alpaca     — official REST API via alpaca-py  (API key + secret)
                 supports both paper-trading and live accounts

Not supported (no public retail trading API)
--------------------------------------------
  Fidelity, Vanguard, Schwab retail accounts
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseBrokerageClient(ABC):
    """Common interface every broker must implement."""

    @property
    @abstractmethod
    def broker_name(self) -> str: ...

    @property
    @abstractmethod
    def logged_in(self) -> bool: ...

    @abstractmethod
    def login(self, **credentials) -> Tuple[bool, str]: ...

    @abstractmethod
    def logout(self) -> None: ...

    @abstractmethod
    def get_buying_power(self) -> float: ...

    @abstractmethod
    def get_holdings(self) -> List[Dict]: ...

    @abstractmethod
    def place_market_buy(self, ticker: str, shares: float) -> Dict: ...

    @abstractmethod
    def place_limit_buy(self, ticker: str, shares: float, limit_price: float) -> Dict: ...

    @abstractmethod
    def place_dollar_buy(self, ticker: str, amount_usd: float) -> Dict: ...

    def get_open_orders(self) -> List[Dict]:
        return []

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        return False, "Not supported by this broker."


# ---------------------------------------------------------------------------
# Robinhood
# ---------------------------------------------------------------------------

class RobinhoodClient(BaseBrokerageClient):
    """Robinhood via the unofficial robin-stocks library."""

    def __init__(self):
        self._logged_in = False

    @property
    def broker_name(self) -> str:
        return "Robinhood"

    @property
    def logged_in(self) -> bool:
        return self._logged_in

    def login(self, username: str = "", password: str = "", mfa_code: str = "", **_) -> Tuple[bool, str]:
        try:
            import robin_stocks.robinhood as rh
            kwargs = dict(username=username, password=password, store_session=True)
            if mfa_code:
                kwargs["mfa_code"] = mfa_code
            rh.login(**kwargs)
            self._logged_in = True
            return True, "Connected to Robinhood."
        except Exception as e:
            self._logged_in = False
            return False, str(e)

    def logout(self) -> None:
        try:
            import robin_stocks.robinhood as rh
            rh.logout()
        except Exception:
            pass
        self._logged_in = False

    def get_buying_power(self) -> float:
        try:
            import robin_stocks.robinhood as rh
            profile = rh.profiles.load_account_profile()
            return float(profile.get("buying_power") or 0)
        except Exception:
            return 0.0

    def get_holdings(self) -> List[Dict]:
        try:
            import robin_stocks.robinhood as rh
            positions = rh.account.build_holdings()
            results = []
            for ticker, data in positions.items():
                results.append({
                    "ticker": ticker,
                    "name": data.get("name", ticker),
                    "quantity": float(data.get("quantity", 0)),
                    "average_buy_price": float(data.get("average_buy_price", 0)),
                    "equity": float(data.get("equity", 0)),
                    "percent_change": float(data.get("percent_change", 0)),
                })
            return sorted(results, key=lambda x: x["equity"], reverse=True)
        except Exception:
            return []

    def place_market_buy(self, ticker: str, shares: float) -> Dict:
        try:
            import robin_stocks.robinhood as rh
            order = rh.orders.order_buy_fractional_by_quantity(
                symbol=ticker, quantity=shares, timeInForce="gfd",
            )
            if order and "id" in order:
                return {"order_id": order["id"], "state": order.get("state", "unknown"),
                        "quantity": shares, "ticker": ticker, "type": "market_buy"}
            return {"error": f"Unexpected response: {order}"}
        except Exception as e:
            return {"error": str(e)}

    def place_limit_buy(self, ticker: str, shares: float, limit_price: float) -> Dict:
        try:
            import robin_stocks.robinhood as rh
            order = rh.orders.order_buy_fractional_by_quantity(
                symbol=ticker, quantity=shares, timeInForce="gtc", limitPrice=limit_price,
            )
            if order and "id" in order:
                return {"order_id": order["id"], "state": order.get("state", "unknown"),
                        "quantity": shares, "limit_price": limit_price,
                        "ticker": ticker, "type": "limit_buy"}
            return {"error": f"Unexpected response: {order}"}
        except Exception as e:
            return {"error": str(e)}

    def place_dollar_buy(self, ticker: str, amount_usd: float) -> Dict:
        try:
            import robin_stocks.robinhood as rh
            order = rh.orders.order_buy_fractional_by_price(
                symbol=ticker, amountInDollars=amount_usd, timeInForce="gfd",
            )
            if order and "id" in order:
                return {"order_id": order["id"], "state": order.get("state", "unknown"),
                        "amount_usd": amount_usd, "ticker": ticker, "type": "dollar_buy"}
            return {"error": f"Unexpected response: {order}"}
        except Exception as e:
            return {"error": str(e)}

    def get_open_orders(self) -> List[Dict]:
        try:
            import robin_stocks.robinhood as rh
            orders = rh.orders.get_all_open_stock_orders()
            return [
                {
                    "order_id": o.get("id"),
                    "ticker": rh.stocks.get_symbol_by_url(o.get("instrument", "")),
                    "side": o.get("side"),
                    "quantity": o.get("quantity"),
                    "price": o.get("price"),
                    "state": o.get("state"),
                    "created_at": o.get("created_at"),
                }
                for o in orders
            ]
        except Exception:
            return []

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        try:
            import robin_stocks.robinhood as rh
            rh.orders.cancel_stock_order(order_id)
            return True, "Order cancelled."
        except Exception as e:
            return False, str(e)


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

class AlpacaClient(BaseBrokerageClient):
    """
    Alpaca via the official alpaca-py SDK.

    Supports both paper-trading (paper=True) and live accounts (paper=False).
    Get API keys at: https://alpaca.markets
    """

    def __init__(self):
        self._logged_in = False
        self._client = None
        self._paper = True

    @property
    def broker_name(self) -> str:
        mode = "Paper" if self._paper else "Live"
        return f"Alpaca ({mode})"

    @property
    def logged_in(self) -> bool:
        return self._logged_in

    def login(self, api_key: str = "", api_secret: str = "", paper: bool = True, **_) -> Tuple[bool, str]:
        try:
            from alpaca.trading.client import TradingClient
            self._paper = paper
            self._client = TradingClient(api_key, api_secret, paper=paper)
            # Validate credentials by fetching account
            acct = self._client.get_account()
            if acct:
                self._logged_in = True
                mode = "paper" if paper else "live"
                return True, f"Connected to Alpaca ({mode} account)."
            return False, "Could not verify account."
        except Exception as e:
            self._logged_in = False
            self._client = None
            return False, str(e)

    def logout(self) -> None:
        self._logged_in = False
        self._client = None

    def get_buying_power(self) -> float:
        try:
            acct = self._client.get_account()
            return float(acct.buying_power or 0)
        except Exception:
            return 0.0

    def get_holdings(self) -> List[Dict]:
        try:
            positions = self._client.get_all_positions()
            results = []
            for p in positions:
                results.append({
                    "ticker": p.symbol,
                    "name": p.symbol,
                    "quantity": float(p.qty or 0),
                    "average_buy_price": float(p.avg_entry_price or 0),
                    "equity": float(p.market_value or 0),
                    "percent_change": float(p.unrealized_plpc or 0) * 100,
                })
            return sorted(results, key=lambda x: x["equity"], reverse=True)
        except Exception:
            return []

    def place_market_buy(self, ticker: str, shares: float) -> Dict:
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = MarketOrderRequest(
                symbol=ticker, qty=shares,
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            return {"order_id": str(order.id), "state": str(order.status),
                    "quantity": shares, "ticker": ticker, "type": "market_buy"}
        except Exception as e:
            return {"error": str(e)}

    def place_limit_buy(self, ticker: str, shares: float, limit_price: float) -> Dict:
        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = LimitOrderRequest(
                symbol=ticker, qty=shares, limit_price=limit_price,
                side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
            )
            order = self._client.submit_order(req)
            return {"order_id": str(order.id), "state": str(order.status),
                    "quantity": shares, "limit_price": limit_price,
                    "ticker": ticker, "type": "limit_buy"}
        except Exception as e:
            return {"error": str(e)}

    def place_dollar_buy(self, ticker: str, amount_usd: float) -> Dict:
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            # Alpaca supports notional (dollar-based) market orders natively
            req = MarketOrderRequest(
                symbol=ticker, notional=round(amount_usd, 2),
                side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            return {"order_id": str(order.id), "state": str(order.status),
                    "amount_usd": amount_usd, "ticker": ticker, "type": "dollar_buy"}
        except Exception as e:
            return {"error": str(e)}

    def get_open_orders(self) -> List[Dict]:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(req)
            return [
                {
                    "order_id": str(o.id),
                    "ticker": o.symbol,
                    "side": str(o.side),
                    "quantity": str(o.qty),
                    "price": str(o.limit_price),
                    "state": str(o.status),
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception:
            return []

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        try:
            from uuid import UUID
            self._client.cancel_order_by_id(UUID(order_id))
            return True, "Order cancelled."
        except Exception as e:
            return False, str(e)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BROKERS = {
    "Robinhood": RobinhoodClient,
    "Alpaca":    AlpacaClient,
}

BROKER_LOGOS = {
    "Robinhood": "🟢",
    "Alpaca":    "🦙",
}

# Brokers with no public retail API — shown greyed out in the UI
UNSUPPORTED_BROKERS = {
    "Fidelity":  "No public trading API for retail accounts.",
    "Schwab":    "Developer API restricted to institutional use.",
    "Vanguard":  "No third-party trading API available.",
}


def make_client(broker_name: str) -> BaseBrokerageClient:
    """Instantiate a fresh client for the given broker name."""
    cls = BROKERS.get(broker_name)
    if cls is None:
        raise ValueError(f"Unsupported broker: {broker_name}")
    return cls()
