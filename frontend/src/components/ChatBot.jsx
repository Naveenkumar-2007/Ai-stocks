import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X, Send, TrendingUp, Activity, Mic, MicOff, Image, Smile,
    Maximize2, Minimize2, Plus, Trash2, MessageSquare, ChevronLeft,
    Menu, ThumbsUp, ThumbsDown, Sparkles, BarChart3
} from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useTheme } from '../contexts/ThemeContext';

const BACKEND_API = process.env.REACT_APP_API_URL || window.location.origin;
const API = `${BACKEND_API}/chatbot`;

const EMOJIS = {
    '📈 Trading': ['📈', '📉', '📊', '💰', '💵', '🏦', '💎', '🚀', '🔥', '⚡', '🎯', '✅', '❌'],
    '😊 Faces': ['😊', '😄', '🤔', '😎', '🤩', '😅', '🙏', '💪', '👍', '👎', '👋', '💡', '🧠'],
};

const WELCOME = {
    role: 'assistant',
    content: "Hey! 👋 I'm your **AI Trading Intelligence** assistant.\n\n📊 Stock analysis — \"Analyze Tesla\"\n📈 Market trends & strategies\n🤖 AI-powered LSTM predictions\n💰 Real-time price data\n\nWhat would you like to know?"
};

const QUICK = [
    { label: '📈 Tesla', q: 'What is the outlook for Tesla stock?' },
    { label: '💰 Nvidia', q: 'Analyze Nvidia stock' },
    { label: '🍎 Apple', q: 'How is Apple doing today?' },
    { label: '📊 Market', q: 'Give me a general market overview' },
    { label: '🤖 Models', q: 'What stocks have trained AI models?' },
];

/* ── Icon button helper (outside component to avoid re-creation) ── */
const iconBtnStyle = { width: 34, height: 34, borderRadius: 8, border: 'none', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' };

const DATAVISION_WELCOME = {
    role: 'assistant',
    content: "Hey! I'm your **Datavision Agentic RAG Assistant**.\n\n- Say **Agent, open AAPL prediction** and I will operate the Prediction page.\n- Compare watchlist stocks with Opportunity Radar.\n- Upload a chart image and include the ticker for grounded model context.\n- Understand trust score, risk, market regime, and forecast range.\n\nWhat would you like to analyze?"
};

const DATAVISION_QUICK = [
    { label: 'Agent Predict AAPL', q: 'Agent, open AAPL prediction and explain it' },
    { label: 'Compare 3 stocks', q: 'Analyze Apple, Google, and NVDA with 7 days forecast' },
    { label: 'Trust score', q: 'Explain trust score in this website' },
    { label: 'Rank watchlist', q: 'Rank watchlist AAPL MSFT NVDA TSLA GOOGL' },
    { label: 'ML models', q: 'What stocks have trained AI models?' },
];

const ChatBot = () => {
    const { isDark } = useTheme();
    const [isOpen, setIsOpen] = useState(false);
    const [isFullPage, setIsFullPage] = useState(false);
    const [showSidebar, setShowSidebar] = useState(false);
    const [chatId, setChatId] = useState(null);
    const [messages, setMessages] = useState([DATAVISION_WELCOME]);
    const [savedChats, setSavedChats] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showEmoji, setShowEmoji] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [imagePreview, setImagePreview] = useState(null);
    const [feedbackGiven, setFeedbackGiven] = useState({});
    const [pendingPredictionRequest, setPendingPredictionRequest] = useState(false);
    const endRef = useRef(null);
    const inputRef = useRef(null);
    const fileRef = useRef(null);
    const recogRef = useRef(null);

    /* ── Theme tokens ── */
    const dark = isDark;
    const pageBg = dark ? '#0b0f1a' : '#f8fafc';
    const sidebarBg = dark ? '#111827' : '#ffffff';
    const sidebarBorder = dark ? '#1e293b' : '#e2e8f0';
    const chatBg = dark ? '#0f172a' : '#ffffff';
    const text = dark ? '#e2e8f0' : '#1e293b';
    const textSoft = dark ? '#64748b' : '#64748b';
    const textMuted = dark ? '#475569' : '#94a3b8';
    const inputBg = dark ? '#1e293b' : '#f1f5f9';
    const inputBorder = dark ? '1px solid #334155' : '1px solid #e2e8f0';
    const chipBg = dark ? 'rgba(51,65,85,0.5)' : 'rgba(241,245,249,0.8)';
    const chipBorder = dark ? '1px solid #334155' : '1px solid #e2e8f0';
    const chipText = dark ? '#94a3b8' : '#475569';
    const icon = dark ? '#64748b' : '#94a3b8';
    const shadow = dark ? '0 25px 60px rgba(0,0,0,0.4)' : '0 25px 60px rgba(0,0,0,0.12)';
    const headerGrad = 'linear-gradient(135deg,#1e40af 0%,#4f46e5 50%,#7c3aed 100%)';
    const sidebarItemActive = dark ? 'rgba(59,130,246,0.12)' : 'rgba(59,130,246,0.08)';
    const sidebarItemHover = dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)';
    const userRowBg = dark ? 'rgba(30,41,59,0.5)' : 'rgba(241,245,249,0.7)';
    const userAvatarBg = dark ? 'linear-gradient(135deg,#334155,#475569)' : 'linear-gradient(135deg,#e2e8f0,#cbd5e1)';
    const userAvatarText = dark ? '#e2e8f0' : '#334155';

    const scroll = () => endRef.current?.scrollIntoView({ behavior: 'smooth' });
    useEffect(scroll, [messages]);
    useEffect(() => { if (isOpen) setTimeout(() => inputRef.current?.focus(), 100); }, [isOpen, isFullPage]);

    const loadChats = useCallback(async () => {
        try { const r = await axios.get(`${API}/chats`); setSavedChats(r.data.chats || []); }
        catch { setSavedChats([]); }
    }, []);
    useEffect(() => { if (isOpen) loadChats(); }, [isOpen, loadChats]);

    /* ── Voice ── */
    useEffect(() => {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SR) {
            const r = new SR(); r.continuous = false; r.interimResults = true; r.lang = 'en-US';
            r.onresult = (e) => setInput(Array.from(e.results).map(x => x[0].transcript).join(''));
            r.onend = () => setIsRecording(false); r.onerror = () => setIsRecording(false);
            recogRef.current = r;
        }
    }, []);
    const toggleVoice = () => {
        if (!recogRef.current) { alert('Use Chrome/Edge for voice.'); return; }
        if (isRecording) recogRef.current.stop(); else { recogRef.current.start(); setIsRecording(true); }
    };

    /* ── Image ── */
    const pickImage = (e) => {
        const f = e.target.files[0]; if (!f) return;
        if (!f.type.startsWith('image/')) return alert('Select an image');
        if (f.size > 5e6) return alert('Max 5MB');
        const r = new FileReader(); r.onload = (ev) => setImagePreview(ev.target.result); r.readAsDataURL(f);
    };
    const rmImage = () => { setImagePreview(null); if (fileRef.current) fileRef.current.value = ''; };
    const addEmoji = (em) => { setInput(p => p + em); inputRef.current?.focus(); };
    const newChat = () => { setChatId(null); setMessages([DATAVISION_WELCOME]); setShowSidebar(false); setFeedbackGiven({}); };
    const loadChat = async (id) => {
        try { const r = await axios.get(`${API}/chats/${id}`); setChatId(id); const m = r.data.messages || []; setMessages(m.length ? m : [DATAVISION_WELCOME]); setShowSidebar(false); setFeedbackGiven({}); }
        catch { /* ignore */ }
    };
    const delChat = async (id, e) => {
        e.stopPropagation();
        try { await axios.delete(`${API}/chats/${id}`); setSavedChats(p => p.filter(x => x.chat_id !== id)); if (chatId === id) newChat(); }
        catch { /* ignore */ }
    };

    const COMPANY_SYMBOLS = {
        NVIDIA: 'NVDA',
        NVDA: 'NVDA',
        APPLE: 'AAPL',
        AAPL: 'AAPL',
        MICROSOFT: 'MSFT',
        MSFT: 'MSFT',
        GOOGLE: 'GOOGL',
        GOOGL: 'GOOGL',
        ALPHABET: 'GOOGL',
        AMAZON: 'AMZN',
        AMZN: 'AMZN',
        TESLA: 'TSLA',
        TSLA: 'TSLA',
        META: 'META',
        RELIANCE: 'RELIANCE.NS',
        TCS: 'TCS.NS',
        INFOSYS: 'INFY.NS',
        INFY: 'INFY.NS',
        HDFCBANK: 'HDFCBANK.NS',
        ICICIBANK: 'ICICIBANK.NS',
    };
    const SYMBOL_STOP_WORDS = new Set([
        'HEY', 'HI', 'HELLO', 'PLEASE', 'CAN', 'YOU', 'ME', 'MY', 'THE', 'A', 'AN',
        'PREDICT', 'PREDICTION', 'FORECAST', 'TARGET', 'OUTLOOK', 'ANALYZE', 'OPEN',
        'AGENT', 'CLICK', 'RUN', 'BROWSER', 'BROWSE', 'GO', 'TO', 'DO', 'FOR', 'STOCK',
        'STOCKS', 'SHARE', 'SHARES', 'OK', 'OKAY', 'HOW', 'GIVE', 'YOUR', 'YOURSELF',
        'THIS', 'THAT', 'WHAT', 'WHEN', 'WHERE', 'WHY', 'WHICH', 'WANT', 'NEED', 'HELP',
        'USE', 'USING', 'EACH', 'DAYS', 'DAY', 'AND', 'ALSO', 'WITH', 'COMPARE', 'COMPANY',
        'COMPANYS', 'COMPANIES', 'THREE', 'TWO', 'ONE'
    ]);

    const extractPredictionSymbols = (textValue) => {
        const text = String(textValue || '').trim();
        const upper = text.toUpperCase();
        const found = [];
        const push = (symbol) => {
            const clean = String(symbol || '').trim().toUpperCase();
            if (clean && !found.includes(clean)) found.push(clean);
        };

        for (const [word, symbol] of Object.entries(COMPANY_SYMBOLS)) {
            if (new RegExp(`\\b${word}\\b`, 'i').test(text)) push(symbol);
        }

        const suffixes = upper.match(/\b[A-Z0-9]{1,15}\.[A-Z]{1,3}\b/g) || [];
        suffixes.filter((item) => !SYMBOL_STOP_WORDS.has(item)).forEach(push);

        const dollars = upper.match(/\$([A-Z]{1,8})\b/g) || [];
        dollars.map((item) => item.replace('$', '')).filter((item) => !SYMBOL_STOP_WORDS.has(item)).forEach(push);

        const tokens = upper.match(/\b[A-Z0-9]{2,10}(?:-[A-Z]{2,5})?\b/g) || [];
        tokens.filter((item) => !SYMBOL_STOP_WORDS.has(item)).forEach((item) => push(COMPANY_SYMBOLS[item] || item));

        return found.slice(0, 8);
    };

    const extractPredictionSymbol = (textValue) => {
        const symbols = extractPredictionSymbols(textValue);
        return symbols.length ? symbols[symbols.length - 1] : null;
    };

    const extractForecastDays = (textValue) => {
        const text = String(textValue || '');
        const match = text.match(/\b(1|7|14|30)\s*(day|days|d)\b/i);
        if (!match) return 7;
        const numeric = Number(match[1]);
        if (numeric <= 1) return 1;
        if (numeric <= 7) return 7;
        return 14;
    };

    const wantsPrediction = (textValue) => /\b(predict|prediction|forecast|target|outlook|analyze|analysis)\b/i.test(String(textValue || ''));
    const wantsAgentAction = (textValue) => /\b(agent|open|click|go to|run|browser|browse|predict|forecast|analyze)\b/i.test(String(textValue || ''));
    const wantsWatchlist = (textValue) => /\b(watchlist|opportunity radar|rank stocks|rank watchlist|compare stocks|compare|these stocks|3 stocks|three stocks|multiple stocks)\b/i.test(String(textValue || ''));
    const wantsPredictionHelp = (textValue) => {
        const text = String(textValue || '').toLowerCase();
        return /\bhow\s+(to|do|can)\b/.test(text) && /\b(predict|prediction|forecast|use|give my stock)\b/.test(text);
    };

    const predictionHelpReply = [
        'Yes. You can give me any listed stock symbol and I can help run the prediction.',
        '',
        'Use it like this:',
        '- **predict NVDA**',
        '- **Agent, open AAPL prediction**',
        '- **forecast RELIANCE.NS for 7 days**',
        '- **rank watchlist AAPL MSFT NVDA TSLA GOOGL**',
        '',
        'When you explicitly ask me to open/run/click/predict a ticker, I will operate the Prediction page. If you only ask how it works, I will explain instead of running a random stock.'
    ].join('\n');

    const runPredictionDesk = (symbol, nextDays = 7) => {
        const cleanSymbol = String(symbol || '').trim().toUpperCase();
        if (!cleanSymbol) return false;
        const command = { type: 'predict', symbol: cleanSymbol, days: nextDays, source: 'chatbot_agent', ts: Date.now() };
        const isPredictionPage = window.location.pathname.replace(/\/$/, '') === '/prediction';

        if (isPredictionPage) {
            window.dispatchEvent(new CustomEvent('datavision:agent-predict', { detail: command }));
            return false;
        }

        sessionStorage.setItem('prediction_agent_command', JSON.stringify(command));
        window.location.href = `/prediction?agentStock=${encodeURIComponent(cleanSymbol)}`;
        return true;
    };

    const runWatchlistDesk = (symbols = [], nextDays = 7) => {
        const cleaned = [...new Set((symbols || []).map((item) => String(item).trim().toUpperCase()).filter(Boolean))].slice(0, 8);
        const command = { type: 'watchlist', symbols: cleaned, days: nextDays, source: 'chatbot_agent', ts: Date.now() };
        const isPredictionPage = window.location.pathname.replace(/\/$/, '') === '/prediction';
        if (isPredictionPage) {
            window.dispatchEvent(new CustomEvent('datavision:agent-predict', { detail: command }));
            return false;
        }
        sessionStorage.setItem('prediction_agent_command', JSON.stringify(command));
        const query = cleaned.length
            ? `?agentAction=watchlist&symbols=${encodeURIComponent(cleaned.join(','))}&days=${nextDays}`
            : `?agentAction=watchlist&days=${nextDays}`;
        window.location.href = `/prediction${query}`;
        return true;
    };

    const formatPredictionReply = (data) => {
        const ticker = data?.ticker || data?.requested_ticker || 'the stock';
        const currency = data?.currency || 'USD';
        const symbol = currency === 'INR' ? 'Rs ' : currency === 'GBP' ? 'GBP ' : currency === 'JPY' ? 'JPY ' : currency === 'EUR' ? 'EUR ' : '$';
        const current = Number(data?.current_price);
        const predicted = Number(data?.predicted_price);
        const ready = Boolean(data?.prediction_ready);
        const status = data?.model_status || {};
        const signal = data?.recommendation?.signal || data?.ai_signal || (ready ? 'HOLD' : 'LIVE ANALYSIS');
        const trust = data?.model_trust?.score;
        const rangeLow = Array.isArray(data?.price_range_low) ? Number(data.price_range_low[data.price_range_low.length - 1]) : null;
        const rangeHigh = Array.isArray(data?.price_range_high) ? Number(data.price_range_high[data.price_range_high.length - 1]) : null;
        const lines = [
            `### ${ticker} prediction desk`,
            ready
                ? `Model status: **Ready**.`
                : `Model status: **Training / live analysis**. ETA: **${status.estimated_completion_label || 'a few minutes'}**.`,
            `Agent action: **Prediction page opened for ${ticker}**.`,
            Number.isFinite(current) ? `Live price: **${symbol}${current.toFixed(2)}**` : null,
            Number.isFinite(predicted) && ready ? `Model target: **${symbol}${predicted.toFixed(2)}**` : null,
            Number.isFinite(rangeLow) && Number.isFinite(rangeHigh) && ready ? `Forecast range: **${symbol}${Math.min(rangeLow, rangeHigh).toFixed(2)} - ${symbol}${Math.max(rangeLow, rangeHigh).toFixed(2)}**` : null,
            `Signal mode: **${String(signal).replace('TRAINING', 'LIVE ANALYSIS')}**`,
            trust != null ? `Trust score: **${trust}/100**` : null,
            !ready ? 'I am showing conservative live analysis now. The trained forecast will appear automatically when validation and calibration finish.' : null,
        ].filter(Boolean);
        return lines.join('\n');
    };

    /* ── Feedback ── */
    const sendFeedback = async (idx, rating) => {
        const userMsg = messages.slice(0, idx).reverse().find(m => m.role === 'user');
        if (!userMsg) return;
        try {
            await axios.post(`${API}/feedback`, { question: userMsg.content, answer: messages[idx].content, rating, chat_id: chatId });
            setFeedbackGiven(p => ({ ...p, [idx]: rating }));
        } catch { /* silent */ }
    };

    /* ── Send ── */
    const send = async (override) => {
        const txt = override || input;
        if (!txt.trim() && !imagePreview) return;
        const attachedImage = imagePreview;
        let display = txt;
        if (attachedImage) display = txt ? `Chart image attached\n${txt}` : 'Chart image attached';
        const userMsg = { role: 'user', content: display, image: attachedImage || null };
        const updated = [...messages, userMsg];
        setMessages(updated); setInput(''); rmImage(); setShowEmoji(false); setIsLoading(true);
        try {
            if (wantsPredictionHelp(txt)) {
                setMessages(p => [...p, { role: 'assistant', content: predictionHelpReply }]);
                return;
            }

            const requestedSymbols = extractPredictionSymbols(txt);
            const forecastDays = extractForecastDays(txt);
            const shouldCompare = (wantsWatchlist(txt) || requestedSymbols.length > 1) && (wantsPrediction(txt) || wantsAgentAction(txt) || wantsWatchlist(txt));
            const predictionSymbol = pendingPredictionRequest || wantsPrediction(txt) || wantsAgentAction(txt) ? extractPredictionSymbol(txt) : null;

            if (shouldCompare) {
                const symbols = requestedSymbols.length > 1 ? requestedSymbols : [];
                const navigated = runWatchlistDesk(symbols, forecastDays);
                const symbolText = symbols.length ? symbols.join(', ') : 'your watchlist';
                setMessages(p => [...p, {
                    role: 'assistant',
                    content: navigated
                        ? `Opening **AI Watchlist Ranking - Opportunity Radar** for **${symbolText}** now. I will run the ${forecastDays}-day comparison and show the ranking charts.`
                        : `Opening **AI Watchlist Ranking - Opportunity Radar** for **${symbolText}** and running the ${forecastDays}-day comparison now.`
                }]);
                return;
            }
            if (wantsPrediction(txt) && !predictionSymbol) {
                setPendingPredictionRequest(true);
                setMessages(p => [...p, {
                    role: 'assistant',
                    content: 'Which stock should I predict? Send a ticker like **AAPL**, **RELIANCE.NS**, **7203.T**, or **BTC-USD**.'
                }]);
                return;
            }
            if (pendingPredictionRequest && predictionSymbol) {
                setPendingPredictionRequest(false);
            }
            if ((wantsPrediction(txt) || wantsAgentAction(txt) || pendingPredictionRequest) && predictionSymbol && !attachedImage) {
                const navigated = runPredictionDesk(predictionSymbol, forecastDays);
                if (navigated) {
                    setMessages(p => [...p, {
                        role: 'assistant',
                        content: `Opening the Prediction desk for **${predictionSymbol}** now. I will run the ${forecastDays}-day forecast there and show the live chart, forecast range, trust score, risk, sentiment, and news.`
                    }]);
                    return;
                }
                const r = await axios.get(`${BACKEND_API}/api/stock/${predictionSymbol}?days=${forecastDays}`, { timeout: 120000 });
                setMessages(p => [...p, { role: 'assistant', content: formatPredictionReply(r.data) }]);
                return;
            }

            const r = await axios.post(`${API}/chat`, {
                message: attachedImage ? `${txt} [User attached a chart/image for analysis]` : txt,
                history: updated.filter(m => m.role).slice(-10).map(m => ({ role: m.role, content: m.content })),
                chat_id: chatId || undefined,
                image_data: attachedImage || undefined,
                image_mime: attachedImage ? String(attachedImage).slice(5, String(attachedImage).indexOf(';')) : undefined,
            }, { timeout: 30000 });
            if (r.data.chat_id && !chatId) setChatId(r.data.chat_id);
            setMessages(p => [...p, { role: 'assistant', content: r.data.reply || "Couldn't process that." }]);
            if (r.data?.data?.agent_action === 'open_prediction' && r.data?.data?.symbol) {
                runPredictionDesk(r.data.data.symbol, 7);
            }
            loadChats();
        } catch (err) {
            let errorMsg = "Couldn't reach the analysis engine.";
            if (err.code === 'ECONNABORTED') errorMsg = "Analysis is taking longer than expected. Please try again in 30 seconds.";
            else if (String(err.response?.data?.detail || '').includes('8001')) {
                errorMsg = "The full chat brain is still starting, but I can still help with this ML stock platform. Try asking: predict AAPL, explain trust score, or how does Opportunity Radar work?";
            }
            else if (err.response?.data?.detail) errorMsg = err.response.data.detail;
            else if (err.response?.data?.error) errorMsg = err.response.data.error;
            else if (!err.response) errorMsg = 'The analysis backend is offline. Please start the backend service and try again.';

            setMessages(p => [...p, { role: 'assistant', content: errorMsg }]);
        } finally { setIsLoading(false); }
    };

    /* ── Render message content ── */
    const renderMsg = (txt, role) => {
        if (!txt) return null;
        if (role === 'assistant') {
            return (
                <div className="chat-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{txt}</ReactMarkdown>
                </div>
            );
        }
        return <span>{txt}</span>;
    };

    /* ═══════════════════════════════════════════
       INLINE RENDER HELPERS (not components!)
       ═══════════════════════════════════════════ */

    /* ── Render sidebar content ── */
    const renderSidebar = (w) => (
        <div style={{ width: w, background: sidebarBg, borderRight: `1px solid ${sidebarBorder}`, display: 'flex', flexDirection: 'column', flexShrink: 0, overflow: 'hidden' }}>
            <div style={{ padding: 16, borderBottom: `1px solid ${sidebarBorder}` }}>
                <button onClick={newChat} style={{ width: '100%', padding: '10px 14px', borderRadius: 10, border: 'none', background: headerGrad, color: '#fff', fontSize: 13, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
                    <Plus style={{ width: 16, height: 16 }} /> New Chat
                </button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
                {savedChats.length === 0 && <p style={{ color: textMuted, fontSize: 12, textAlign: 'center', padding: 24 }}>No conversations yet</p>}
                {savedChats.map(ch => (
                    <div key={ch.chat_id} onClick={() => loadChat(ch.chat_id)}
                        style={{ padding: '10px 12px', borderRadius: 10, marginBottom: 4, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10, background: chatId === ch.chat_id ? sidebarItemActive : 'transparent', transition: 'background 0.15s' }}
                        onMouseEnter={e => { if (chatId !== ch.chat_id) e.currentTarget.style.background = sidebarItemHover; }}
                        onMouseLeave={e => { if (chatId !== ch.chat_id) e.currentTarget.style.background = 'transparent'; }}
                    >
                        <MessageSquare style={{ width: 14, height: 14, color: textSoft, flexShrink: 0 }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <p style={{ color: text, fontSize: 12, fontWeight: 500, margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ch.title}</p>
                            <p style={{ color: textMuted, fontSize: 10, margin: 0 }}>{ch.message_count} messages</p>
                        </div>
                        <button onClick={e => delChat(ch.chat_id, e)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4, borderRadius: 6, opacity: 0.4, display: 'flex' }}
                            onMouseEnter={e => e.currentTarget.style.opacity = 1} onMouseLeave={e => e.currentTarget.style.opacity = 0.4}>
                            <Trash2 style={{ width: 13, height: 13, color: '#ef4444' }} />
                        </button>
                    </div>
                ))}
            </div>
            <div style={{ padding: '12px 16px', borderTop: `1px solid ${sidebarBorder}`, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Sparkles style={{ width: 14, height: 14, color: '#a78bfa' }} />
                <span style={{ color: textMuted, fontSize: 10 }}>AI learns from your feedback</span>
            </div>
        </div>
    );

    /* ── Render messages ── */
    const renderMessages = (maxW) => (
        <div style={{ flex: 1, overflowY: 'auto', padding: isFullPage ? '24px 32px' : '12px 14px', scrollbarWidth: 'thin' }}>
            <div style={{ maxWidth: maxW, margin: '0 auto' }}>
                {messages.length <= 1 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: isFullPage ? 40 : 12, marginBottom: 20 }}>
                        {DATAVISION_QUICK.map((a, i) => (
                            <button key={i} onClick={() => send(a.q)} disabled={isLoading}
                                style={{ padding: '10px 16px', borderRadius: 12, fontSize: 13, border: chipBorder, background: chipBg, color: chipText, cursor: 'pointer', transition: 'all 0.15s', fontWeight: 500 }}>
                                {a.label}
                            </button>
                        ))}
                    </div>
                )}
                {messages.map((m, i) => (
                    <div key={i} style={{
                        padding: isFullPage ? '18px 24px' : '12px 16px',
                        marginBottom: 2,
                        borderRadius: 12,
                        background: m.role === 'user' ? userRowBg : 'transparent',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                            {m.role === 'assistant' ? (
                                <img src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'} alt="DV" style={{ width: 28, height: 28, borderRadius: 8, objectFit: 'cover', flexShrink: 0 }} />
                            ) : (
                                <div style={{ width: 28, height: 28, borderRadius: 8, background: userAvatarBg, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                                    <span style={{ fontSize: 12, color: userAvatarText, fontWeight: 700 }}>Y</span>
                                </div>
                            )}
                            <span style={{ fontSize: 13, fontWeight: 600, color: text }}>{m.role === 'assistant' ? 'Datavision ML Assistant' : 'You'}</span>
                        </div>
                        <div style={{ paddingLeft: 38, fontSize: isFullPage ? 14.5 : 13, lineHeight: 1.75, color: text, whiteSpace: m.role === 'user' ? 'pre-line' : 'normal' }}>
                            {m.image && <img src={m.image} alt="" style={{ maxWidth: 300, maxHeight: 200, borderRadius: 10, marginBottom: 8, display: 'block' }} />}
                            {renderMsg(m.content, m.role)}
                        </div>
                        {m.role === 'assistant' && i > 0 && (
                            <div style={{ paddingLeft: 38, marginTop: 8, display: 'flex', gap: 4 }}>
                                <button onClick={() => sendFeedback(i, 'thumbs_up')} title="Good answer"
                                    style={{ ...iconBtnStyle, width: 28, height: 28, borderRadius: 6, background: feedbackGiven[i] === 'thumbs_up' ? 'rgba(34,197,94,0.15)' : 'transparent' }}>
                                    <ThumbsUp style={{ width: 13, height: 13, color: feedbackGiven[i] === 'thumbs_up' ? '#22c55e' : textMuted }} />
                                </button>
                                <button onClick={() => sendFeedback(i, 'thumbs_down')} title="Bad answer"
                                    style={{ ...iconBtnStyle, width: 28, height: 28, borderRadius: 6, background: feedbackGiven[i] === 'thumbs_down' ? 'rgba(239,68,68,0.15)' : 'transparent' }}>
                                    <ThumbsDown style={{ width: 13, height: 13, color: feedbackGiven[i] === 'thumbs_down' ? '#ef4444' : textMuted }} />
                                </button>
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && (
                    <div style={{ padding: isFullPage ? '18px 24px' : '12px 16px', borderRadius: 12 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                            <img src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'} alt="DV" style={{ width: 28, height: 28, borderRadius: 8, objectFit: 'cover' }} />
                            <span style={{ fontSize: 13, fontWeight: 600, color: text }}>Datavision ML Assistant</span>
                        </div>
                        <div style={{ paddingLeft: 38, display: 'flex', alignItems: 'center', gap: 6 }}>
                            {[0, 150, 300].map((d, j) => <div key={j} style={{ width: 8, height: 8, borderRadius: '50%', background: ['#60a5fa', '#818cf8', '#a78bfa'][j], animation: `bounce 1.4s ease-in-out ${d}ms infinite` }} />)}
                            <span style={{ color: textMuted, fontSize: 12, marginLeft: 6 }}>Understanding intent and preparing the right tool...</span>
                        </div>
                    </div>
                )}
                <div ref={endRef} />
            </div>
        </div>
    );

    /* ── Render input area ── */
    const renderInput = (maxW) => (
        <div style={{ padding: '16px 20px 20px', background: chatBg, flexShrink: 0 }}>
            <div style={{ maxWidth: maxW, margin: '0 auto' }}>
                {imagePreview && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, padding: '8px 12px', borderRadius: 10, background: inputBg }}>
                        <img src={imagePreview} alt="" style={{ width: 48, height: 48, borderRadius: 8, objectFit: 'cover' }} />
                        <span style={{ color: textSoft, fontSize: 12, flex: 1 }}>Chart image attached</span>
                        <button onClick={rmImage} style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: '4px 10px', cursor: 'pointer', color: '#f87171', fontSize: 11 }}>Remove</button>
                    </div>
                )}
                <AnimatePresence>
                    {showEmoji && (
                        <motion.div initial={{ height: 0 }} animate={{ height: 100 }} exit={{ height: 0 }} style={{ overflow: 'hidden', marginBottom: 8, borderRadius: 10, background: inputBg, border: inputBorder }}>
                            <div style={{ padding: 8, overflowY: 'auto', height: '100%', display: 'flex', gap: 16 }}>
                                {Object.entries(EMOJIS).map(([cat, ems]) => (
                                    <div key={cat}>
                                        <div style={{ fontSize: 10, color: textMuted, marginBottom: 4, fontWeight: 600 }}>{cat}</div>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                                            {ems.map((em, j) => <button key={j} onClick={() => addEmoji(em)} style={{ width: 32, height: 32, border: 'none', borderRadius: 6, background: 'transparent', cursor: 'pointer', fontSize: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{em}</button>)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, borderRadius: 18, padding: '8px 10px', background: inputBg, border: inputBorder, boxShadow: dark ? 'inset 0 1px 0 rgba(255,255,255,0.04)' : '0 8px 24px rgba(15,23,42,0.06)' }}>
                    <input type="file" ref={fileRef} accept="image/*" onChange={pickImage} style={{ display: 'none' }} />
                    <button onClick={() => fileRef.current?.click()} title="Attach chart" style={iconBtnStyle}>
                        <Image style={{ width: 16, height: 16, color: icon }} />
                    </button>
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                        placeholder="Ask, upload chart, or say: Agent open AAPL..."
                        disabled={isLoading}
                        style={{ flex: 1, background: 'none', border: 'none', outline: 'none', fontSize: 15, color: text, padding: '15px 6px', opacity: isLoading ? 0.5 : 1 }}
                    />
                    <button onClick={() => setShowEmoji(!showEmoji)} style={{ ...iconBtnStyle, background: showEmoji ? 'rgba(59,130,246,0.2)' : 'transparent' }}>
                        <Smile style={{ width: 16, height: 16, color: showEmoji ? '#60a5fa' : icon }} />
                    </button>
                    <button onClick={toggleVoice} style={{ ...iconBtnStyle, animation: isRecording ? 'pulse 1.5s infinite' : 'none' }}>
                        {isRecording ? <MicOff style={{ width: 16, height: 16, color: '#f87171' }} /> : <Mic style={{ width: 16, height: 16, color: icon }} />}
                    </button>
                    <button onClick={() => send()} disabled={isLoading || (!input.trim() && !imagePreview)} style={{
                        width: 38, height: 38, borderRadius: 10, border: 'none',
                        background: (input.trim() || imagePreview) ? headerGrad : 'transparent',
                        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        opacity: (isLoading || (!input.trim() && !imagePreview)) ? 0.3 : 1, transition: 'all 0.15s',
                    }}>
                        <Send style={{ width: 16, height: 16, color: '#fff' }} />
                    </button>
                </div>
                <p style={{ textAlign: 'center', fontSize: 10, color: textMuted, margin: '8px 0 0' }}>Powered by <strong>Datavision</strong> • ML Stock Assistant • Not financial advice</p>
            </div>
        </div>
    );

    const keyframes = <style>{`@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-8px)}} @keyframes pulse{0%{opacity:1}50%{opacity:.5}100%{opacity:1}}
.chat-markdown h1,.chat-markdown h2,.chat-markdown h3{margin:12px 0 6px;font-weight:700;line-height:1.3}
.chat-markdown h1{font-size:1.3em}.chat-markdown h2{font-size:1.15em}.chat-markdown h3{font-size:1.05em}
.chat-markdown p{margin:4px 0}
.chat-markdown ul,.chat-markdown ol{margin:4px 0 4px 18px;padding:0}
.chat-markdown li{margin:2px 0}
.chat-markdown code{background:rgba(127,127,127,0.15);padding:1px 5px;border-radius:4px;font-size:0.9em;font-family:'Fira Code',monospace}
.chat-markdown pre{background:rgba(0,0,0,0.25);padding:10px 14px;border-radius:8px;overflow-x:auto;margin:8px 0}
.chat-markdown pre code{background:none;padding:0}
.chat-markdown blockquote{border-left:3px solid rgba(99,102,241,0.5);padding-left:12px;margin:6px 0;opacity:0.85}
.chat-markdown table{border-collapse:collapse;margin:8px 0;width:100%}
.chat-markdown th,.chat-markdown td{border:1px solid rgba(127,127,127,0.25);padding:6px 10px;text-align:left;font-size:0.92em}
.chat-markdown th{font-weight:600;background:rgba(127,127,127,0.1)}
.chat-markdown a{color:#60a5fa;text-decoration:none}
.chat-markdown a:hover{text-decoration:underline}
.chat-markdown hr{border:none;border-top:1px solid rgba(127,127,127,0.2);margin:10px 0}
.chat-markdown strong{font-weight:600}
`}</style>;

    /* ═══════════════════════════════════════════
       FULL-PAGE MODE — ChatGPT / Gemini layout
       ═══════════════════════════════════════════ */
    if (isFullPage) {
        return (
            <div style={{ position: 'fixed', inset: 0, zIndex: 100000, display: 'flex', background: pageBg }}>
                {renderSidebar(260)}

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                    {/* Top bar */}
                    <div style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: `1px solid ${sidebarBorder}`, background: chatBg, flexShrink: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <img src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'} alt="Datavision" style={{ width: 32, height: 32, borderRadius: 10, objectFit: 'cover' }} />
                            <div>
                                <h3 style={{ color: text, fontWeight: 700, fontSize: 15, margin: 0 }}>Datavision ML Assistant</h3>
                                <span style={{ color: textMuted, fontSize: 11 }}>Forecasts • Trust scores • Live market data</span>
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 6 }}>
                            <button onClick={newChat} title="New chat" style={iconBtnStyle}><Plus style={{ width: 16, height: 16, color: textSoft }} /></button>
                            <button onClick={() => setIsFullPage(false)} title="Minimize" style={iconBtnStyle}><Minimize2 style={{ width: 16, height: 16, color: textSoft }} /></button>
                            <button onClick={() => { setIsOpen(false); setIsFullPage(false); }} title="Close" style={iconBtnStyle}><X style={{ width: 16, height: 16, color: textSoft }} /></button>
                        </div>
                    </div>

                    {renderMessages(768)}
                    {renderInput(768)}
                </div>
                {keyframes}
            </div>
        );
    }

    /* ═══════════════════════════════════════════
       WIDGET MODE — Floating bubble + chat panel
       ═══════════════════════════════════════════ */
    return (
        <div style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 99999 }}>
            <AnimatePresence mode="wait">
                {!isOpen && (
                    <motion.button key="fab" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}
                        whileHover={{ scale: 1.1, boxShadow: '0 0 35px rgba(99,102,241,0.4)' }} whileTap={{ scale: 0.92 }}
                        onClick={() => setIsOpen(true)}
                        style={{ width: 60, height: 60, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'transparent', border: 'none', cursor: 'pointer', boxShadow: '0 8px 32px rgba(34,197,94,0.35)', position: 'relative', outline: 'none', padding: 0, overflow: 'hidden' }}>
                        <img src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'} alt="Datavision" style={{ width: 60, height: 60, borderRadius: '50%', objectFit: 'cover' }} />
                        <span style={{ position: 'absolute', top: -3, right: -3, background: 'linear-gradient(135deg,#22c55e,#10b981)', color: '#fff', fontSize: 9, fontWeight: 700, padding: '2px 5px', borderRadius: 9999 }}>AI</span>
                    </motion.button>
                )}

                {isOpen && (
                    <motion.div key="panel" initial={{ opacity: 0, y: 20, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        style={{ width: 'min(400px, calc(100vw - 24px))', height: 'min(600px, calc(100vh - 100px))', borderRadius: 20, display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: shadow, background: chatBg }}>
                        {/* Header */}
                        <div style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: headerGrad, flexShrink: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <button onClick={() => setShowSidebar(!showSidebar)} style={{ ...iconBtnStyle, background: showSidebar ? 'rgba(255,255,255,0.2)' : 'transparent' }}>
                                    {showSidebar ? <ChevronLeft style={{ width: 16, height: 16, color: '#fff' }} /> : <Menu style={{ width: 16, height: 16, color: '#fff' }} />}
                                </button>
                                <img src='/assets/logo-dark.jpg' alt="DV" style={{ width: 28, height: 28, borderRadius: 8, objectFit: 'cover' }} />
                                <h3 style={{ color: '#fff', fontWeight: 700, fontSize: 13, margin: 0 }}>Datavision Assistant</h3>
                            </div>
                            <div style={{ display: 'flex', gap: 2 }}>
                                <button onClick={newChat} title="New chat" style={iconBtnStyle}><Plus style={{ width: 14, height: 14, color: 'rgba(255,255,255,0.8)' }} /></button>
                                <button onClick={() => setIsFullPage(true)} title="Full page" style={iconBtnStyle}><Maximize2 style={{ width: 14, height: 14, color: 'rgba(255,255,255,0.8)' }} /></button>
                                <button onClick={() => setIsOpen(false)} title="Close" style={iconBtnStyle}><X style={{ width: 14, height: 14, color: 'rgba(255,255,255,0.8)' }} /></button>
                            </div>
                        </div>

                        {/* Body */}
                        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                            <AnimatePresence>
                                {showSidebar && (
                                    <motion.div initial={{ width: 0 }} animate={{ width: 180 }} exit={{ width: 0 }} style={{ overflow: 'hidden', flexShrink: 0 }}>
                                        {renderSidebar(180)}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                                {renderMessages('100%')}
                                {renderInput('100%')}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
            {keyframes}
        </div>
    );
};

export default ChatBot;
