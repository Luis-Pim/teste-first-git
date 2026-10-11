export type Dec = string | null;

export interface DividendStatsLite {
  last: { amount_per_share: string; com_date: string; ex_date: string | null; payment_date: string; kind: string; source: string } | null;
  sum_12m: string;
  mean_monthly_12m: Dec;
  dy_12m: Dec;
  dy_last_annualized: Dec;
  yoc_12m: Dec;
  regularity_12m: string;
}

export interface Position {
  ticker: string;
  name: string | null;
  segment: string | null;
  asset_class: string | null;
  quantity: string;
  avg_price: Dec;
  total_cost: Dec;
  cost_known: boolean;
  pending_quantity: string;
  pending_cost: string;
  planned_quantity: string;
  planned_cost: string;
  price: Dec;
  price_date: string | null;
  price_source: string | null;
  market_value: Dec;
  unrealized_pnl: Dec;
  unrealized_pct: Dec;
  realized_pnl: string;
  dividends_received: string;
  amortizations_received: string;
  weight: Dec;
  weight_basis: string;
  dividend_stats: DividendStatsLite;
}

export interface Summary {
  reference_date: string;
  invested_confirmed: string;
  invested_pending: string;
  invested_total: string;
  market_value: string;
  market_value_complete: boolean;
  unrealized_pnl: string;
  unrealized_pct: Dec;
  realized_pnl: string;
  dividends_month: string;
  dividends_last_month: string;
  dividends_ytd: string;
  dividends_12m: string;
  dividends_total: string;
  dy_last_month: Dec;
  dy_last_month_annualized: Dec;
  dy_12m: Dec;
  yoc_12m: Dec;
  total_return: { absolute: string; pct: Dec; method: string } | null;
  positions_count: number;
  cost_unknown: string[];
  missing_prices: string[];
  price_sources: string[];
  benchmarks: { available: boolean; reason: string };
  notes: string[];
  definitions: Record<string, string>;
}

export interface Allocation {
  basis: string;
  by_asset: { label: string; weight: string }[];
  by_segment: { label: string; weight: string }[];
  by_class: { label: string; weight: string }[];
  notes: string[];
}

export interface ConfirmedIncome {
  category: "CONFIRMADA";
  status: "A_RECEBER" | "PAGO_NAO_REGISTRADO";
  kind: string;
  ticker: string;
  com_date: string;
  ex_date: string | null;
  payment_date: string;
  amount_per_share: string;
  eligible_quantity: string;
  amount: string;
  provisional_quantity: boolean;
  announcement_id: number;
  source: string;
}

export interface Alert {
  severity: string;
  type: string;
  title: string;
  message: string;
}

export interface LedgerRow {
  month: string;
  is_current: boolean;
  is_future: boolean;
  planned_contribution: string;
  planned_extra: string;
  own_contribution: string;
  own_source: string;
  dividends_received: string;
  dividends_reinvested: string;
  amortizations: string;
  sales: string;
  carry_in: string;
  available: string;
  spent: string;
  planned_orders: string;
  carry_out: string;
}

export interface Transaction {
  id: number;
  type: string;
  status: string;
  trade_date: string;
  settlement_date: string | null;
  ticker: string | null;
  quantity: Dec;
  unit_price: Dec;
  fees: string;
  total_amount: Dec;
  ratio_from: Dec;
  ratio_to: Dec;
  announcement_id: number | null;
  source: string;
  notes: string | null;
}
