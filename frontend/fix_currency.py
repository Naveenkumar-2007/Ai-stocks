import re
import os

filepath = r"c:\Users\navee\Downloads\Ai-insights-main\Ai-stocks-main\frontend\src\pages\Prediction.jsx"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add getCurrencySymbol
curr_func = """
const getCurrencySymbol = (currency) => {
  if (!currency) return '$';
  const map = {
    'USD': '$',
    'INR': '₹',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CAD': 'CA$',
    'AUD': 'A$',
    'HKD': 'HK$',
    'CNY': '¥',
    'SGD': 'S$'
  };
  return map[currency.toUpperCase()] || currency + ' ';
};
"""
content = content.replace("const formatCurrencyCompact", curr_func + "\nconst formatCurrencyCompact")

# Update formatCurrencyCompact
content = re.sub(r'const formatCurrencyCompact = \(value\) => \{', r'const formatCurrencyCompact = (value, curr = "$") => {', content)
content = content.replace('return `$${(numeric / 1e12).toFixed(2)}T`;', 'return `${curr}${(numeric / 1e12).toFixed(2)}T`;')
content = content.replace('return `$${(numeric / 1e9).toFixed(2)}B`;', 'return `${curr}${(numeric / 1e9).toFixed(2)}B`;')
content = content.replace('return `$${(numeric / 1e6).toFixed(2)}M`;', 'return `${curr}${(numeric / 1e6).toFixed(2)}M`;')
content = content.replace('return `$${(numeric / 1e3).toFixed(2)}K`;', 'return `${curr}${(numeric / 1e3).toFixed(2)}K`;')
content = content.replace('return `$${numeric.toFixed(2)}`;', 'return `${curr}${numeric.toFixed(2)}`;')

# Update TradingViewStyleChart to accept currencySymbol
content = content.replace('const TradingViewStyleChart = ({ candles = [], sma20 = [], predictions = [], isProfit }) => {', 
                          'const TradingViewStyleChart = ({ candles = [], sma20 = [], predictions = [], isProfit, currencySymbol = "$" }) => {')
content = content.replace('const fmt = (v) => `$${Number(v).toFixed(2)}`;', 'const fmt = (v) => `${currencySymbol}${Number(v).toFixed(2)}`;')

# Define currencySymbol in Prediction component
prediction_start = """const Prediction = () => {
  const { user } = useAuth();
  const [symbol, setSymbol] = useState('');
  const [stockData, setStockData] = useState(null);"""

prediction_start_new = """const Prediction = () => {
  const { user } = useAuth();
  const [symbol, setSymbol] = useState('');
  const [stockData, setStockData] = useState(null);
  const currencySymbol = getCurrencySymbol(stockData?.currency);"""

content = content.replace(prediction_start, prediction_start_new)

# Update formatCurrencyCompact calls in Prediction
content = re.sub(r'formatCurrencyCompact\((stockData\.market_cap.*?)\)', r'formatCurrencyCompact(\1, currencySymbol)', content)

# Replace $ in JSX and string interpolations inside Prediction body
# Look for places where $ is used directly in text or in template strings

# Replace in formatters
content = content.replace('tickFormatter={(val) => `$${val.toFixed(0)}`}', 'tickFormatter={(val) => `${currencySymbol}${val.toFixed(0)}`}')
content = content.replace('tickFormatter={(val) => `$${val.toFixed(2)}`}', 'tickFormatter={(val) => `${currencySymbol}${val.toFixed(2)}`}')

# Replace text nodes
content = content.replace('>${row.price}</td>', '>{currencySymbol}{row.price}</td>')
content = content.replace('>${row.priceLow} – ${row.priceHigh}', '>{currencySymbol}{row.priceLow} – {currencySymbol}{row.priceHigh}')
content = content.replace('>${row.change}', '>{currencySymbol}{row.change}')
content = content.replace('>\n                                  Change: <span className="font-bold">{change >= 0 ? \'+\' : \'\'}${change.toFixed(2)}', '>\n                                  Change: <span className="font-bold">{change >= 0 ? \'+\' : \'\'}{currencySymbol}{change.toFixed(2)}')
content = content.replace('Price: <span className="font-bold">${data.price.toFixed(2)}</span>', 'Price: <span className="font-bold">{currencySymbol}{data.price.toFixed(2)}</span>')

content = content.replace('>${stockData.indicators.ema}</span>', '>{currencySymbol}{stockData.indicators.ema}</span>')
content = content.replace('>${stockData.day_high}</span>', '>{currencySymbol}{stockData.day_high}</span>')
content = content.replace('>${stockData.day_low}</span>', '>{currencySymbol}{stockData.day_low}</span>')

content = content.replace('title={`Current: $${current.toFixed(2)}`}', 'title={`Current: ${currencySymbol}${current.toFixed(2)}`}')
content = content.replace('title={`Median: $${median.toFixed(2)}`}', 'title={`Median: ${currencySymbol}${median.toFixed(2)}`}')
content = content.replace('>${low.toFixed(2)}</span>', '>{currencySymbol}{low.toFixed(2)}</span>')
content = content.replace('>${median.toFixed(2)}</span>', '>{currencySymbol}{median.toFixed(2)}</span>')
content = content.replace('>${high.toFixed(2)}</span>', '>{currencySymbol}{high.toFixed(2)}</span>')
content = content.replace('>${low.toFixed(2)}</p>', '>{currencySymbol}{low.toFixed(2)}</p>')
content = content.replace('>${q25.toFixed(2)}</p>', '>{currencySymbol}{q25.toFixed(2)}</p>')
content = content.replace('>${q75.toFixed(2)}</p>', '>{currencySymbol}{q75.toFixed(2)}</p>')
content = content.replace('>${high.toFixed(2)}</p>', '>{currencySymbol}{high.toFixed(2)}</p>')

# Update Tooltips (with currency passed via props if they're used outside)
# Wait, CustomTooltip and CandlestickTooltip are not currently being rendered in this file with a curr prop? Let's check how they're called.
# Actually, the user doesn't use CustomTooltip or CandlestickTooltip actively right now. The charts have inline tooltips.
# Let's fix the inline tooltips first:
content = content.replace('<span className="text-xs font-bold text-blue-400">${open.toFixed(2)}</span>', '<span className="text-xs font-bold text-blue-400">{currencySymbol}{open.toFixed(2)}</span>')
content = content.replace('<span className="text-xs font-bold text-green-400">${high.toFixed(2)}</span>', '<span className="text-xs font-bold text-green-400">{currencySymbol}{high.toFixed(2)}</span>')
content = content.replace('<span className="text-xs font-bold text-red-400">${low.toFixed(2)}</span>', '<span className="text-xs font-bold text-red-400">{currencySymbol}{low.toFixed(2)}</span>')
content = content.replace('>${close.toFixed(2)}\n', '>{currencySymbol}{close.toFixed(2)}\n')

# Pass currency to TradingViewStyleChart
content = content.replace('<TradingViewStyleChart\n                  candles={stockData.technical_chart?.candles}', '<TradingViewStyleChart\n                  currencySymbol={currencySymbol}\n                  candles={stockData.technical_chart?.candles}')

with open(filepath + '.temp.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done generating temporary file")
