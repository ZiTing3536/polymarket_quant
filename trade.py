#!/usr/bin/env python3
"""
Polymarket Live Trading Engine — trade.py
==========================================
Same Bayesian model as papertrade_engine.py.
Places REAL limit orders on Polymarket CLOB via py-clob-client.

One-time API key setup (run on your local machine with private key, NOT on VPS):
    from py_clob_client.client import ClobClient
    client = ClobClient("https://clob.polymarket.com", chain_id=137, key="0xYOUR_PRIVATE_KEY")
    creds = client.create_or_derive_api_creds()
    print(creds.api_key, creds.api_secret, creds.api_passphrase)

Then on the VPS set environment variables (never hardcode):
    export POLY_API_KEY="..."
    export POLY_API_SECRET="..."
    export POLY_API_PASSPHRASE="..."

Usage:
    python3 trade.py --backfill                        # real money
    python3 trade.py --backfill --dry-run              # simulate without placing orders
    python3 trade.py --backfill --capital 100 --risk 0.03
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLOB_HOST     = "https://clob.polymarket.com"
CHAIN_ID      = 137
CSV_FILE      = "polymarket_odds.csv"
TRADE_LOG     = "live_trade_log.csv"
MODEL_LOG     = "live_model_updates.csv"
REFRESH_SEC   = 1
CONTRACT_DUR  = 300
DASHBOARD_SEC = 30
MIN_SIZE      = 1.0    # Polymarket minimum order size in USDC


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colors (same as papertrade_engine)
# ─────────────────────────────────────────────────────────────────────────────

class C:
    R      = "\033[0m";  B      = "\033[1m";  DIM    = "\033[2m"
    RED    = "\033[31m"; GRN    = "\033[32m"; YLW    = "\033[33m"
    BGRN   = "\033[92m"; BRED   = "\033[91m"; BYLW   = "\033[93m"
    BWHT   = "\033[97m"; SEP    = "\033[2;34m"; LABEL  = "\033[2;37m"
    VALUE  = "\033[97m"; PROFIT = "\033[92m"; LOSS   = "\033[91m"
    WARN   = "\033[33m"; SKIP   = "\033[90m"; FILL   = "\033[92m"
    EXPIRE = "\033[90m"; ORDER  = "\033[93m"; HEADER = "\033[1;97m"

def fmt_pnl(v):
    c = C.PROFIT if v >= 0 else C.LOSS
    return f"{c}{C.B}{'+' if v>=0 else ''}{v:.2f}{C.R}"

def fmt_pct(v):
    c = C.PROFIT if v >= 0 else C.LOSS
    return f"{c}{'+' if v>=0 else ''}{v:.2f}%{C.R}"

def fmt_wr(v):
    c = C.PROFIT if v >= 85 else (C.YLW if v >= 70 else C.LOSS)
    return f"{c}{C.B}{v:.1f}%{C.R}"

def sep(char="─", w=68):
    return f"{C.SEP}{char*w}{C.R}"


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian math (identical to papertrade_engine)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default=None):
    try:
        f = float(val)
        return f if (f == f) else default
    except (ValueError, TypeError):
        return default

def _lgamma(x):
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _lgamma(1 - x)
    x -= 1
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    a = c[0]
    for i in range(1, 9): a += c[i] / (x + i)
    t = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)

def _beta_inc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1 - _beta_inc(b, a, 1 - x)
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    cf = d
    for m in range(1, 201):
        num_e = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        d = 1 + num_e * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + num_e / (c if abs(c) > 1e-30 else 1e-30); cf *= c * d
        num_o = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        d = 1 + num_o * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + num_o / (c if abs(c) > 1e-30 else 1e-30); cf *= c * d
        if abs(c * d - 1) < 1e-10: break
    return min(1.0, max(0.0, front * cf))

def beta_quantile(a, b, p):
    lo, hi, mid = 0.0, 1.0, a / (a + b)
    for _ in range(80):
        v = _beta_inc(a, b, mid)
        if abs(v - p) < 1e-9: break
        if v < p: lo = mid
        else:     hi = mid
        mid = (lo + hi) / 2
    return mid


# ─────────────────────────────────────────────────────────────────────────────
# Signal definitions (identical to papertrade_engine)
# ─────────────────────────────────────────────────────────────────────────────

HOUR_BUCKETS = [
    ("夜間 00-08", 0,  8),
    ("日間 08-16", 8,  16),
    ("晚間 16-24", 16, 24),
]

def hour_bucket(h):
    for name, lo, hi in HOUR_BUCKETS:
        if lo <= h < hi: return name
    return "晚間 16-24"

PROGRESS_BUCKETS = [
    ("早段<0.33",      0.00, 0.33),
    ("中段0.33-0.66",  0.33, 0.66),
    ("晚段≥0.66",      0.66, 1.01),
]

def progress_bucket(p):
    for name, lo, hi in PROGRESS_BUCKETS:
        if lo <= p < hi: return name
    return "晚段≥0.66"

def drift_bucket(d):
    if d > 0.005:  return "上升>0.005"
    if d < -0.005: return "下降<-0.005"
    return "穩定±0.005"

DRIFT_BUCKET_NAMES = ["上升>0.005", "穩定±0.005", "下降<-0.005"]

YES_PRICE_TIERS = [
    ("YES@0.90", 0.90, 0.90, 0.93),
    ("YES@0.85", 0.85, 0.85, 0.90),
    ("YES@0.80", 0.80, 0.80, 0.85),
]

NO_PRICE_TIERS = [
    ("NO@0.90",  0.90, 0.07, 0.10),
    ("NO@0.85",  0.85, 0.10, 0.15),
    ("NO@0.80",  0.80, 0.15, 0.20),
]

def build_signal_key(side, tier, hb, pb, db):
    return f"{side}::{tier}::{hb}::{pb}::{db}"


# ─────────────────────────────────────────────────────────────────────────────
# Sample
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    ts: str; slug: str; yes_p: float; no_p: float
    drift: float = 0.0; drift_abs: float = 0.0
    cum_drift: float = 0.0; drift_accel: float = 0.0
    progress: float = 0.5; uncertainty: float = 0.5
    hour: int = 0; outcome: Optional[int] = None

def make_sample(ts, slug, yes_p, no_p, drift=0.0, cum_drift=0.0,
                drift_accel=0.0, progress=0.5, outcome=None):
    try: h = datetime.fromisoformat(ts.replace(" ", "T")).hour
    except: h = 0
    return Sample(ts=ts, slug=slug, yes_p=yes_p, no_p=no_p,
                  drift=drift, drift_abs=abs(drift), cum_drift=cum_drift,
                  drift_accel=drift_accel, progress=round(progress, 4),
                  uncertainty=round(1 - abs(yes_p - 0.5) * 2, 4),
                  hour=h, outcome=outcome)


# ─────────────────────────────────────────────────────────────────────────────
# TierModel (identical to papertrade_engine)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TierModel:
    key: str; side: str; tier_name: str; entry_price: float
    hbucket: str; pbucket: str; dbucket: str
    yes_lo: float; yes_hi: float
    alpha0: float = 2.0; beta0: float = 2.0
    wins: int = 0; losses: int = 0

    @property
    def a(self): return self.alpha0 + self.wins
    @property
    def b(self): return self.beta0  + self.losses
    @property
    def n(self): return self.wins + self.losses
    @property
    def post_mean(self): return self.a / (self.a + self.b)
    @property
    def ci_lo(self): return beta_quantile(self.a, self.b, 0.025)
    @property
    def ci_hi(self): return beta_quantile(self.a, self.b, 0.975)
    @property
    def reliability(self):
        w = self.ci_hi - self.ci_lo
        return "high" if w < 0.05 else "mid" if w < 0.15 else "low"

    def matches_sample(self, s):
        if hour_bucket(s.hour) != self.hbucket:        return False
        if not (self.yes_lo <= s.yes_p < self.yes_hi): return False
        if progress_bucket(s.progress) != self.pbucket: return False
        if drift_bucket(s.drift) != self.dbucket:       return False
        return True

    def update(self, outcome):
        if self.side == "YES":
            if outcome == 1: self.wins   += 1
            else:            self.losses += 1
        else:
            if outcome == 0: self.wins   += 1
            else:            self.losses += 1

    def ev(self):
        p = self.entry_price
        if not (0.001 < p < 0.999): return -999.0
        payout = (1.0 - p) / p
        return self.ci_lo * payout - (1.0 - self.ci_lo)


def build_all_models(alpha0, beta0):
    models = {}
    for hname, _, _ in HOUR_BUCKETS:
        for pname, plo, phi in PROGRESS_BUCKETS:
            for dname in DRIFT_BUCKET_NAMES:
                for tier_name, entry_price, yes_lo, yes_hi in YES_PRICE_TIERS:
                    key = build_signal_key("YES", tier_name, hname, pname, dname)
                    models[key] = TierModel(key=key, side="YES", tier_name=tier_name,
                                            entry_price=entry_price,
                                            hbucket=hname, pbucket=pname, dbucket=dname,
                                            yes_lo=yes_lo, yes_hi=yes_hi,
                                            alpha0=alpha0, beta0=beta0)
                for tier_name, entry_price, yes_lo, yes_hi in NO_PRICE_TIERS:
                    key = build_signal_key("NO", tier_name, hname, pname, dname)
                    models[key] = TierModel(key=key, side="NO", tier_name=tier_name,
                                            entry_price=entry_price,
                                            hbucket=hname, pbucket=pname, dbucket=dname,
                                            yes_lo=yes_lo, yes_hi=yes_hi,
                                            alpha0=alpha0, beta0=beta0)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# ContractState
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractState:
    slug: str; yes_id: str = ""; no_id: str = ""
    obs: list = field(default_factory=list)
    closed: bool = False; outcome: Optional[int] = None
    start_ts_sec: Optional[float] = None

    def _slug_start_ts(self):
        import re as _re
        m = _re.search(r'-(\d{9,})$', self.slug)
        return float(m.group(1)) if m else None

    def add(self, ts, yes_p, no_p):
        self.obs.append((ts, yes_p, no_p))
        if self.start_ts_sec is None:
            self.start_ts_sec = self._slug_start_ts()
            if self.start_ts_sec is None:
                try: self.start_ts_sec = datetime.fromisoformat(ts.replace(" ","T")).timestamp()
                except: pass

    def prev_yes_p(self):
        return self.obs[-2][1] if len(self.obs) >= 2 else None

    def all_samples(self):
        if self.outcome is None or len(self.obs) < 2: return []
        obs = self.obs[:-1]; n = len(obs); out = []
        for i, (ts, yp, np_) in enumerate(obs):
            ypp = obs[i-1][1] if i > 0 else yp
            y0  = obs[0][1]
            da  = (yp - ypp) - (ypp - obs[i-2][1]) if i >= 2 else 0.0
            out.append(make_sample(ts, self.slug, yp, np_,
                                   drift=yp-ypp, cum_drift=yp-y0,
                                   drift_accel=da, progress=i/n if n else 0.5,
                                   outcome=self.outcome))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# LiveTrade — extends PaperTrade with real order tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveTrade:
    id:              int
    slug:            str
    model_key:       str
    side:            str
    tier_name:       str
    entry_ts:        str
    limit_price:     float
    stake:           float
    post_mean_entry: float
    ev_entry:        float
    yes_token_id:    str = ""
    no_token_id:     str = ""
    clob_order_id:   Optional[str]  = None    # Polymarket order ID
    filled:          bool           = False
    fill_ts:         Optional[str]  = None
    exit_ts:         Optional[str]  = None
    outcome:         Optional[int]  = None
    pnl:             Optional[float] = None
    pnl_pct_capital: Optional[float] = None

    @property
    def settled(self): return self.outcome is not None
    @property
    def expired(self): return self.settled and not self.filled

    def settle(self, outcome, exit_ts, capital):
        self.outcome = outcome; self.exit_ts = exit_ts
        if not self.filled:
            self.pnl = 0.0; self.pnl_pct_capital = 0.0; return
        p   = self.limit_price
        won = (self.side == "YES" and outcome == 1) or \
              (self.side == "NO"  and outcome == 0)
        self.pnl = self.stake * (1.0 - p) / p if won else -self.stake
        self.pnl_pct_capital = self.pnl / capital * 100 if capital > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Live Trading Engine
# ─────────────────────────────────────────────────────────────────────────────

class LiveTradingEngine:

    SKIP_UTC_HOURS = {13, 14, 20}

    def __init__(self, csv_file=CSV_FILE, capital=1000.0, risk_pct=0.03,
                 alpha0=2.0, beta0=2.0, min_ev=0.005, min_n=10,
                 dry_run=False, debug=False):

        self.csv_file    = csv_file
        self.initial_cap = capital
        self.capital     = capital
        self.risk_pct    = risk_pct
        self.min_ev      = min_ev
        self.min_n       = min_n
        self.dry_run     = dry_run
        self.debug       = debug

        self.models       = build_all_models(alpha0, beta0)
        self.contracts:   dict[str, ContractState] = {}
        self.open_trades: dict[str, LiveTrade]     = {}
        self.closed_trades: list[LiveTrade]         = []
        self.trade_id     = 0
        self.last_row     = 0
        self.updated_slugs: set = set()

        # Polymarket CLOB client
        self._clob = None
        if not dry_run:
            self._init_clob()

        self._init_logs()

    # ── CLOB client init ──────────────────────────────────────────────────

    def _init_clob(self):
        api_key    = os.environ.get("POLY_API_KEY")
        api_secret = os.environ.get("POLY_API_SECRET")
        api_pass   = os.environ.get("POLY_API_PASSPHRASE")

        if not all([api_key, api_secret, api_pass]):
            print(f"{C.LOSS}ERROR: Missing API credentials.{C.R}")
            print(f"{C.LABEL}Set environment variables:{C.R}")
            print(f"  export POLY_API_KEY=...")
            print(f"  export POLY_API_SECRET=...")
            print(f"  export POLY_API_PASSPHRASE=...")
            sys.exit(1)

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            creds = ApiCreds(api_key=api_key, api_secret=api_secret,
                             api_passphrase=api_pass)
            self._clob = ClobClient(CLOB_HOST, chain_id=CHAIN_ID, creds=creds)
            ok = self._clob.get_ok()
            print(f"{C.FILL}  CLOB connected: {ok}{C.R}")
        except Exception as e:
            print(f"{C.LOSS}  CLOB init failed: {e}{C.R}")
            sys.exit(1)

    # ── Log init ──────────────────────────────────────────────────────────

    def _init_logs(self):
        if not os.path.exists(TRADE_LOG):
            with open(TRADE_LOG, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "id", "slug", "model_key", "side", "tier",
                    "entry_ts", "limit_price", "stake",
                    "post_mean_entry", "ev_entry",
                    "clob_order_id", "filled", "fill_ts",
                    "exit_ts", "outcome", "pnl", "pnl_pct_capital",
                ])
        if not os.path.exists(MODEL_LOG):
            with open(MODEL_LOG, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "ts", "model_key", "side", "tier", "hbucket",
                    "n", "wins", "losses",
                    "post_mean", "ci_lo", "ci_hi", "reliability", "slug",
                ])

    # ── Backfill ──────────────────────────────────────────────────────────

    def backfill(self, rows):
        print(f"\n  [BACKFILL] {len(rows)} rows...")
        by_slug = defaultdict(list)
        for r in rows: by_slug[r["slug"]].append(r)

        trained = skipped = 0
        for slug, entries in by_slug.items():
            entries.sort(key=lambda x: x["timestamp"])
            closed_row = next(
                (e for e in entries
                 if e.get("market_closed","").strip().upper() in ("TRUE","1")), None)
            if not closed_row: skipped += 1; continue
            try: outcome = 1 if float(closed_row["yes_price"]) > 0.5 else 0
            except: skipped += 1; continue
            closed_idx = entries.index(closed_row)
            valid = entries[:closed_idx]
            if len(valid) < 2: skipped += 1; continue

            n = len(valid)
            for i, row in enumerate(valid):
                yp  = _safe_float(row.get("yes_price"))
                np_ = _safe_float(row.get("no_price"))
                if yp is None or not (0 < yp < 1): continue
                if np_ is None: np_ = 1 - yp
                ypp = _safe_float(valid[i-1].get("yes_price")) if i > 0 else None
                ypp = ypp if ypp is not None else yp
                y0  = _safe_float(valid[0].get("yes_price")) or yp
                da  = 0.0
                if i >= 2:
                    yp2 = _safe_float(valid[i-2].get("yes_price"))
                    if yp2 is not None: da = (yp - ypp) - (ypp - yp2)
                s = make_sample(row["timestamp"], slug, yp, np_,
                                drift=yp-ypp, cum_drift=yp-y0,
                                drift_accel=da, progress=i/n, outcome=outcome)
                for model in self.models.values():
                    if model.matches_sample(s): model.update(outcome)
            self.updated_slugs.add(slug)
            trained += 1

        print(f"  [BACKFILL] done: {trained} contracts, {skipped} skipped")
        self._print_model_table()

    # ── CSV read ──────────────────────────────────────────────────────────

    def _read_new_rows(self):
        rows = []
        try:
            with open(self.csv_file, "r", encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if i >= self.last_row: rows.append(row)
            self.last_row += len(rows)
        except FileNotFoundError: pass
        except Exception as e: print(f"  [READ ERROR] {e}")
        return rows

    # ── Skip hour check ────────────────────────────────────────────────────

    def _is_skip_hour(self, ts):
        try:
            h = datetime.fromisoformat(ts.replace(" ","T")).hour
            return h in self.SKIP_UTC_HOURS
        except: return False

    # ── Process row ───────────────────────────────────────────────────────

    def process_row(self, row):
        slug   = row.get("slug","").strip()
        ts     = row.get("timestamp","").strip()
        yes_id = row.get("yes_token_id","")
        no_id  = row.get("no_token_id","")
        closed = row.get("market_closed","").strip().upper() in ("TRUE","1")

        yp = _safe_float(row.get("yes_price"))
        np_= _safe_float(row.get("no_price"))
        if not slug or not ts or yp is None or not (0 < yp < 1): return
        if np_ is None: np_ = 1 - yp

        is_new = slug not in self.contracts
        if is_new:
            self.contracts[slug] = ContractState(slug=slug,
                                                  yes_id=yes_id, no_id=no_id)
        contract = self.contracts[slug]
        if contract.closed: return

        if closed:
            outcome = 1 if yp > 0.5 else 0
            contract.closed = True; contract.outcome = outcome
            contract.add(ts, yp, np_)
            self._settle_contract(slug, outcome, ts)
        else:
            contract.add(ts, yp, np_)
            if is_new: self._place_orders(slug, ts, yp, yes_id, no_id)
            self._check_fills(slug, ts, yp, contract)

    # ── Place limit orders ─────────────────────────────────────────────────

    def _place_orders(self, slug, ts, yes_p, yes_id, no_id):
        if self._is_skip_hour(ts):
            try:
                h_utc = datetime.fromisoformat(ts.replace(" ","T")).hour
                h_est = (h_utc - 5) % 24
                print(f"{C.SKIP}  SKIP [{h_est:02d}:xx EST]  {slug[-25:]}{C.R}")
            except: pass
            return

        try: hb = hour_bucket(datetime.fromisoformat(ts.replace(" ","T")).hour)
        except: hb = hour_bucket(0)

        for side, tiers, token_id in [
            ("YES", YES_PRICE_TIERS, yes_id),
            ("NO",  NO_PRICE_TIERS,  no_id),
        ]:
            # Find best EV model for this contract's opening conditions
            best_ev = self.min_ev; best_model = None
            pb = progress_bucket(0.0); db = drift_bucket(0.0)
            for pname, _, _ in PROGRESS_BUCKETS:
                for dname in DRIFT_BUCKET_NAMES:
                    for tier_name, _, _, _ in tiers:
                        key   = build_signal_key(side, tier_name, hb, pname, dname)
                        model = self.models.get(key)
                        if model and model.n >= self.min_n:
                            ev = model.ev()
                            if ev > best_ev:
                                best_ev = ev; best_model = model

            if best_model is None:
                if self.debug:
                    print(f"{C.SKIP}  NO SIGNAL [{side}] {hb}{C.R}")
                continue

            # Prevent duplicate order for same slug+side
            if any(k.startswith(f"{slug}::{side}::") for k in self.open_trades):
                continue

            stake = round(self.capital * self.risk_pct, 2)
            if stake < MIN_SIZE: continue

            limit = best_model.entry_price
            # Size = stake / limit_price (number of tokens)
            size  = round(stake / limit, 4)

            # Place real order on CLOB
            clob_order_id = None
            if not self.dry_run and self._clob and token_id:
                clob_order_id = self._place_clob_order(
                    token_id=token_id,
                    price=limit,
                    size=size,
                    side=side,
                )

            self.trade_id += 1
            trade = LiveTrade(
                id=self.trade_id, slug=slug,
                model_key=best_model.key,
                side=side, tier_name=best_model.tier_name,
                entry_ts=ts, limit_price=limit, stake=stake,
                post_mean_entry=best_model.post_mean, ev_entry=best_ev,
                yes_token_id=yes_id, no_token_id=no_id,
                clob_order_id=clob_order_id,
            )
            full_key = f"{slug}::{side}::{best_model.tier_name}"
            self.open_trades[full_key] = trade

            mode = "[DRY]" if self.dry_run else f"[{clob_order_id[:8] if clob_order_id else 'ERR'}]"
            s_col = C.PROFIT if side == "YES" else C.BRED
            print(f"\n{C.ORDER}  ORDER {mode}  {s_col}{C.B}{side:>3}{C.R}{C.ORDER}  "
                  f"#{trade.id:<4}  {best_model.tier_name}  [{hb}]{C.R}")
            print(f"  {C.LABEL}limit={C.VALUE}{limit:.4f}  "
                  f"{C.LABEL}size={C.VALUE}{size:.2f}  "
                  f"{C.LABEL}stake=${C.VALUE}{stake:.2f}  "
                  f"{C.LABEL}EV={C.FILL}{best_ev:+.4f}  "
                  f"{C.LABEL}post={C.VALUE}{best_model.post_mean*100:.1f}%{C.R}")

    def _place_clob_order(self, token_id, price, size, side):
        """Place a GTC limit order on Polymarket CLOB. Returns order ID or None."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side="BUY",     # We always BUY the token (YES token or NO token)
            )
            resp = self._clob.create_and_post_order(order_args)
            if resp and isinstance(resp, dict):
                oid = resp.get("orderID") or resp.get("id") or str(resp)
                print(f"{C.FILL}    CLOB order placed: {oid[:16]}{C.R}")
                return oid
        except Exception as e:
            print(f"{C.LOSS}    CLOB order failed: {e}{C.R}")
        return None

    def _cancel_clob_order(self, order_id):
        """Cancel a specific order on CLOB."""
        if not order_id or not self._clob: return
        try:
            self._clob.cancel_orders([order_id])
            print(f"{C.EXPIRE}    CLOB order cancelled: {order_id[:16]}{C.R}")
        except Exception as e:
            print(f"{C.WARN}    CLOB cancel failed: {e}{C.R}")

    # ── Fill check (same logic as papertrade_engine) ───────────────────────

    def _check_fills(self, slug, ts, yes_p, contract):
        prev = contract.prev_yes_p()
        if prev is None: prev = yes_p

        for key, trade in list(self.open_trades.items()):
            if not key.startswith(f"{slug}::") or trade.filled:
                continue

            filled_now = False

            if trade.side == "YES":
                if max(prev, yes_p) >= trade.limit_price:
                    trade.filled  = True
                    trade.fill_ts = ts
                    filled_now    = True
                    print(f"{C.FILL}  FILL  YES  #{trade.id:<4}  "
                          f"limit={trade.limit_price:.4f}  "
                          f"{prev:.4f}{C.BWHT}->{C.R}{C.FILL}{yes_p:.4f}{C.R}")

            elif trade.side == "NO":
                yes_trigger = 1.0 - trade.limit_price
                if min(prev, yes_p) <= yes_trigger:
                    trade.filled  = True
                    trade.fill_ts = ts
                    filled_now    = True
                    print(f"{C.FILL}  FILL   NO  #{trade.id:<4}  "
                          f"limit_no={trade.limit_price:.4f}  "
                          f"no:{1-prev:.4f}{C.BWHT}->{C.R}{C.FILL}{1-yes_p:.4f}{C.R}")

            # Cancel opposite side once one fills
            if filled_now:
                opposite = "NO" if trade.side == "YES" else "YES"
                for k in list(self.open_trades.keys()):
                    if k.startswith(f"{slug}::{opposite}::") and \
                       not self.open_trades[k].filled:
                        cancelled = self.open_trades.pop(k)
                        # Cancel on CLOB too
                        if not self.dry_run:
                            self._cancel_clob_order(cancelled.clob_order_id)
                        print(f"{C.EXPIRE}  CANCEL [{opposite}] #{cancelled.id:<4}  "
                              f"opposite filled{C.R}")

    # ── Settle ────────────────────────────────────────────────────────────

    def _settle_contract(self, slug, outcome, ts):
        icon_res  = "YES WIN" if outcome == 1 else "NO  WIN"
        settled_n = 0

        for key in list(self.open_trades.keys()):
            if not key.startswith(f"{slug}::"): continue
            trade = self.open_trades.pop(key)

            # Cancel unfilled orders on CLOB at settlement
            if not trade.filled and not self.dry_run:
                self._cancel_clob_order(trade.clob_order_id)

            trade.settle(outcome, ts, capital=self.capital)
            self.closed_trades.append(trade)
            self.capital += trade.pnl
            settled_n += 1

            if trade.expired:
                print(f"{C.EXPIRE}  EXPIR [{trade.side:>3}] #{trade.id:<4}  "
                      f"{trade.tier_name}{C.R}")
            else:
                won     = (trade.side == "YES" and outcome == 1) or \
                          (trade.side == "NO"  and outcome == 0)
                res_col = C.PROFIT if won else C.LOSS
                print(f"\n{res_col}{C.B}  {'WIN' if won else 'LOSS'}  [{trade.side:>3}]{C.R}  "
                      f"#{trade.id:<4}  {trade.tier_name}  {icon_res}")
                print(f"  {C.LABEL}PnL={fmt_pnl(trade.pnl)}  "
                      f"({fmt_pct(trade.pnl_pct_capital)} capital)  "
                      f"{C.LABEL}equity={C.VALUE}${self.capital:.2f}{C.R}")
            self._write_trade(trade)

        # Rolling model update (each contract only once)
        if slug not in self.updated_slugs:
            contract = self.contracts.get(slug)
            if contract:
                for s in contract.all_samples():
                    for model in self.models.values():
                        if model.matches_sample(s): model.update(outcome)
                for model in self.models.values():
                    self._write_model_update(model, ts, slug)
            self.updated_slugs.add(slug)

        if settled_n > 0: self._print_pnl_summary()

    # ── Write logs ────────────────────────────────────────────────────────

    def _write_trade(self, t):
        with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                t.id, t.slug, t.model_key, t.side, t.tier_name,
                t.entry_ts, f"{t.limit_price:.4f}", f"{t.stake:.4f}",
                f"{t.post_mean_entry:.4f}", f"{t.ev_entry:.6f}",
                t.clob_order_id or "",
                t.filled, t.fill_ts or "", t.exit_ts or "",
                t.outcome if t.outcome is not None else "",
                f"{t.pnl:.6f}" if t.pnl is not None else "",
                f"{t.pnl_pct_capital:.4f}" if t.pnl_pct_capital is not None else "",
            ])

    def _write_model_update(self, m, ts, slug):
        with open(MODEL_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ts, m.key, m.side, m.tier_name, m.hbucket,
                m.n, m.wins, m.losses,
                f"{m.post_mean:.4f}", f"{m.ci_lo:.4f}",
                f"{m.ci_hi:.4f}", m.reliability, slug,
            ])

    # ── Display ───────────────────────────────────────────────────────────

    def _print_model_table(self):
        rows = [m for m in self.models.values() if m.n >= self.min_n]
        if not rows: print(f"{C.WARN}  [MODEL] no sufficient samples{C.R}"); return
        rows.sort(key=lambda x: -x.ev())
        print(f"\n{sep()}")
        print(f"  {C.HEADER}MODEL RANKING{C.R}  {C.LABEL}{len(rows)} active  top 15{C.R}")
        print(sep())
        for m in rows[:15]:
            ev    = m.ev()
            ev_c  = C.PROFIT if ev > 0.02 else (C.YLW if ev > 0 else C.LOSS)
            s_col = C.PROFIT if m.side == "YES" else C.BRED
            print(f"  {s_col}{C.B}{m.side:>3}{C.R}  "
                  f"{C.VALUE}{m.tier_name:<8}{C.R}  "
                  f"{C.LABEL}{m.hbucket:<10}{C.R}  "
                  f"{C.DIM}{m.pbucket:<13}{C.R}  "
                  f"{C.DIM}{m.dbucket:<10}{C.R}  "
                  f"n={C.VALUE}{m.n:<5}{C.R}  "
                  f"post={m.post_mean*100:.1f}%  "
                  f"{ev_c}EV={ev*100:>+6.2f}%{C.R}  "
                  f"{C.LABEL}{m.reliability}{C.R}")
        print(sep())

    def _print_pnl_summary(self):
        filled = [t for t in self.closed_trades if t.filled]
        if not filled: return
        wins  = sum(1 for t in filled if (t.pnl or 0) > 0)
        total = sum(t.pnl or 0 for t in self.closed_trades)
        ret   = (self.capital - self.initial_cap) / self.initial_cap * 100
        print(f"\n{sep()}")
        print(f"  {C.LABEL}trades {C.VALUE}{len(filled)}{C.R}  "
              f"wr {fmt_wr(wins/len(filled)*100)}  "
              f"pnl {fmt_pnl(total)}  "
              f"return {fmt_pct(ret)}")
        print(sep())

    def print_dashboard(self):
        n         = len(self.closed_trades)
        filled    = [t for t in self.closed_trades if t.filled]
        expired   = [t for t in self.closed_trades if t.expired]
        pending   = sum(1 for t in self.open_trades.values() if not t.filled)
        ret_pct   = (self.capital - self.initial_cap) / self.initial_cap * 100
        wins      = sum(1 for t in filled if (t.pnl or 0) > 0)
        wr        = wins / len(filled) * 100 if filled else 0
        total_pnl = sum(t.pnl or 0 for t in self.closed_trades)
        fill_rate = len(filled) / n * 100 if n else 0
        now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        yes_f     = [t for t in filled if t.side == "YES"]
        no_f      = [t for t in filled if t.side == "NO"]
        yes_wr    = sum(1 for t in yes_f if (t.pnl or 0) > 0) / len(yes_f) * 100 if yes_f else 0
        no_wr     = sum(1 for t in no_f  if (t.pnl or 0) > 0) / len(no_f)  * 100 if no_f  else 0
        eq_col    = C.PROFIT if ret_pct >= 0 else C.LOSS
        mode_str  = f"{C.WARN}[DRY RUN]{C.R}" if self.dry_run else f"{C.PROFIT}[LIVE]{C.R}"

        print(f"\n{sep('=')}")
        print(f"  {C.HEADER}POLYMARKET LIVE ENGINE{C.R}  {mode_str}  {C.LABEL}{now_str}{C.R}")
        print(sep("="))
        print(f"  {C.LABEL}equity   start=${C.VALUE}{self.initial_cap:.2f}  "
              f"{C.LABEL}now={eq_col}{C.B}${self.capital:.2f}{C.R}  "
              f"{C.LABEL}return={fmt_pct(ret_pct)}{C.R}")
        print(f"  {C.LABEL}orders   total={C.VALUE}{n}  "
              f"{C.LABEL}filled={C.VALUE}{len(filled)}{C.LABEL}({fill_rate:.0f}%)  "
              f"{C.LABEL}expired={C.VALUE}{len(expired)}  "
              f"{C.LABEL}pending={C.VALUE}{pending}{C.R}")
        print(f"  {C.LABEL}win rate "
              f"{C.PROFIT}YES {fmt_wr(yes_wr)}{C.LABEL}({len(yes_f)})  "
              f"{C.BRED}NO {fmt_wr(no_wr)}{C.LABEL}({len(no_f)})  "
              f"{C.LABEL}overall {fmt_wr(wr)}{C.R}")
        print(f"  {C.LABEL}p&l      total={fmt_pnl(total_pnl)}  "
              f"{C.LABEL}avg/trade={fmt_pnl(total_pnl/len(filled)) if filled else C.VALUE+'N/A'+C.R}{C.R}")
        recent = [t for t in self.closed_trades if t.filled][-5:]
        if recent:
            print(sep())
            for t in reversed(recent):
                won   = (t.side=="YES" and t.outcome==1) or (t.side=="NO" and t.outcome==0)
                s_col = C.PROFIT if t.side == "YES" else C.BRED
                print(f"  {'>' if won else '<'}  "
                      f"{s_col}{C.B}{t.side:>3}{C.R}  "
                      f"{C.LABEL}{(t.fill_ts or t.entry_ts)[:16]}  "
                      f"{C.VALUE}{t.tier_name:<10}{C.R}  "
                      f"{fmt_pnl(t.pnl or 0)}  {fmt_pct(t.pnl_pct_capital or 0)}")
        print(sep("="))

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self, backfill_rows=None):
        if backfill_rows:
            self.backfill(backfill_rows)
            try:
                with open(self.csv_file, "r", encoding="utf-8") as f:
                    self.last_row = max(0, sum(1 for _ in f) - 1)
            except: pass

        print(f"\n{sep('=')}")
        print(f"  {C.HEADER}POLYMARKET LIVE ENGINE{C.R}  "
              f"{'[DRY RUN]' if self.dry_run else C.PROFIT+'[LIVE TRADING]'+C.R}")
        print(f"  {C.LABEL}capital  {C.VALUE}${self.capital:.2f}  "
              f"{C.LABEL}risk/bet {C.VALUE}{self.risk_pct*100:.1f}%{C.R}")
        skip_est = sorted((h-5) % 24 for h in self.SKIP_UTC_HOURS)
        print(f"  {C.WARN}skip UTC {sorted(self.SKIP_UTC_HOURS)} = EST {skip_est}{C.R}")
        print(sep("="))

        last_dashboard = 0; last_model = 0; total_new = 0

        while True:
            rows = self._read_new_rows()
            total_new += len(rows)
            for row in rows: self.process_row(row)

            now = time.time()
            if now - last_dashboard > DASHBOARD_SEC:
                ts_now = datetime.now().strftime("%H:%M:%S")
                active = sum(1 for c in self.contracts.values() if not c.closed)
                print(f"{C.SEP}  @ {ts_now}  "
                      f"+{len(rows)} rows  active={active}  "
                      f"open={len(self.open_trades)}  total={total_new}{C.R}",
                      flush=True)
                self.print_dashboard()
                last_dashboard = now
            if now - last_model > 300:
                self._print_model_table(); last_model = now

            time.sleep(REFRESH_SEC)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Polymarket Live Trading Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--csv",          default=CSV_FILE)
    ap.add_argument("--capital",      type=float, default=1000.0,
                    help="Starting capital in USDC (default 1000)")
    ap.add_argument("--risk",         type=float, default=0.03,
                    help="Risk per bet as fraction (default 0.03 = 3%%)")
    ap.add_argument("--alpha0",       type=float, default=2.0)
    ap.add_argument("--beta0",        type=float, default=2.0)
    ap.add_argument("--min-ev",       type=float, default=0.005)
    ap.add_argument("--min-n",        type=int,   default=10)
    ap.add_argument("--skip-hours",   default="13,14,20",
                    help="UTC hours to skip (default: 13,14,20 = US open/close)")
    ap.add_argument("--backfill",     action="store_true",
                    help="Train on existing CSV before going live")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Simulate without placing real orders")
    ap.add_argument("--debug",        action="store_true")
    args = ap.parse_args()

    engine = LiveTradingEngine(
        csv_file = args.csv,
        capital  = args.capital,
        risk_pct = args.risk,
        alpha0   = args.alpha0,
        beta0    = args.beta0,
        min_ev   = args.min_ev,
        min_n    = args.min_n,
        dry_run  = args.dry_run,
        debug    = args.debug,
    )

    try:
        skip = {int(x.strip()) for x in args.skip_hours.split(",") if x.strip()}
        engine.SKIP_UTC_HOURS = skip
    except: pass

    backfill_rows = None
    if args.backfill:
        try:
            with open(args.csv, "r", encoding="utf-8") as f:
                backfill_rows = list(csv.DictReader(f))
            print(f"  loaded {len(backfill_rows)} rows for backfill")
        except FileNotFoundError:
            print(f"  {C.WARN}CSV not found, starting fresh{C.R}")

    try:
        engine.run(backfill_rows=backfill_rows)
    except KeyboardInterrupt:
        print(f"\n\n  stopped")
        engine._print_model_table()
        engine.print_dashboard()
        sys.exit(0)


if __name__ == "__main__":
    main()
