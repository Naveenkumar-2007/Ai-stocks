import React, { useEffect, useRef, useMemo, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import '../styles/landing.css';

/* ──────────────────────────────────────────────────────────── 
   DATA
   ──────────────────────────────────────────────────────────── */

const STOCKS = [
  {
    sym: 'META', full: 'Meta Platforms, Inc.', ic: 'ic-meta',
    price: 642.73, chg: +3.28, pct: +0.51, dir: 'up',
    pred: 'Bullish', conf: 94.2,
    pts: [60, 62, 58, 64, 63, 67, 65, 70, 68, 72, 71, 75, 73, 78, 76, 80, 79, 82, 81, 85],
  },
  {
    sym: 'NVDA', full: 'NVIDIA Corporation', ic: 'ic-nvda',
    price: 138.85, chg: +5.47, pct: +4.10, dir: 'up',
    pred: 'Strong Buy', conf: 97.1,
    pts: [40, 42, 45, 43, 48, 50, 47, 52, 55, 53, 58, 60, 57, 62, 65, 63, 68, 70, 72, 75],
  },
  {
    sym: 'AMD', full: 'Advanced Micro Devices', ic: 'ic-amd',
    price: 118.32, chg: -1.54, pct: -1.28, dir: 'dn',
    pred: 'Hold', conf: 61.8,
    pts: [70, 68, 72, 69, 65, 67, 63, 66, 62, 64, 60, 63, 58, 61, 57, 60, 55, 58, 54, 56],
  },
  {
    sym: 'TSLA', full: 'Tesla, Inc.', ic: 'ic-tsla',
    price: 352.64, chg: +12.31, pct: +3.62, dir: 'up',
    pred: 'Bullish', conf: 88.5,
    pts: [50, 48, 52, 55, 53, 58, 56, 60, 63, 61, 65, 68, 66, 70, 72, 69, 74, 76, 78, 80],
  },
];

const FEATURES = [
  { icon: '⚡', title: 'Real-Time Predictions', desc: 'Live market data feeds with sub-second AI inference. Get actionable buy, hold, and sell signals as the market moves.', c: 'fc-blue' },
  { icon: '🧠', title: 'LSTM  Model', desc: 'Production LSTM neural networks  trained on years of historical price, volume, and indicator data.', c: 'fc-green' },
  { icon: '📊', title: 'Portfolio Optimization', desc: 'Automated asset allocation utilizing Modern Portfolio Theory with dynamic risk-return optimization across your holdings.', c: 'fc-purple' },
  { icon: '🛡️', title: 'Risk Analysis', desc: 'Value at Risk calculations, drawdown analysis, and volatility scoring help you understand your exposure before you enter a trade.', c: 'fc-blue' },
  { icon: '🏛️', title: 'Institutional Analytics', desc: 'Access the same caliber of technical and fundamental analysis tools used by professional trading desks and hedge funds.', c: 'fc-green' },
  { icon: '💬', title: 'AI Market Chat', desc: 'Ask our AI chatbot about any stock — get instant analysis, predictions, technicals, and news summaries in natural language.', c: 'fc-purple' },
];





/* Cinematic background is now a separate component: CinematicBackground.jsx */

/* ──────────────────────────────────────────────────────────── 
   SVG SPARKLINE
   ──────────────────────────────────────────────────────────── */

function Spark({ data, color, w = 200, h = 48 }) {
  const mn = Math.min(...data);
  const mx = Math.max(...data);
  const rng = mx - mn || 1;
  const sx = w / (data.length - 1);

  const pts = data.map((v, i) => `${i * sx},${h - ((v - mn) / rng) * h * 0.85 - h * 0.05}`).join(' ');
  const area = `0,${h} ${pts} ${w},${h}`;
  const gid = `sg${Math.random().toString(36).slice(2, 7)}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${gid})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* ──────────────────────────────────────────────────────────── 
   CANDLESTICK CHART (Hero)
   ──────────────────────────────────────────────────────────── */

/* ──────────────────────────────────────────────────────────── 
   TRADING TERMINAL VISUAL (INSTITUTIONAL GRADE)
   ──────────────────────────────────────────────────────────── */

function TradingTerminalVisual() {
  const [tape, setTape] = React.useState([
    { pr: '5342.18', sz: '1.2k', side: 'buy', t: '12:04:21' },
    { pr: '5341.05', sz: '0.8k', side: 'sell', t: '12:04:19' },
    { pr: '5342.30', sz: '2.5k', side: 'buy', t: '12:04:15' },
    { pr: '5342.25', sz: '0.1k', side: 'buy', t: '12:04:12' },
    { pr: '5340.90', sz: '4.2k', side: 'sell', t: '12:04:08' },
  ]);

  const bids = [
    { p: '5341.25', s: '12', v: 85 },
    { p: '5341.00', s: '45', v: 70 },
    { p: '5340.75', s: '120', v: 95 },
    { p: '5340.50', s: '32', v: 40 },
  ];

  const asks = [
    { p: '5342.50', s: '28', v: 60 },
    { p: '5342.75', s: '150', v: 100 },
    { p: '5343.00', s: '42', v: 50 },
    { p: '5343.25', s: '10', v: 20 },
  ];

  return (
    <div className="lp-terminal-frame">
      {/* Chrome */}
      <div className="lp-terminal-top">
        <div className="lp-terminal-tabs">
          <div className="lp-terminal-tab active">MAIN_CHART</div>
          <div className="lp-terminal-tab">ORDER_FLOW</div>
          <div className="lp-terminal-tab">SENTIMENT</div>
        </div>
        <div className="lp-term-status">
          <span className="lp-status-dot" /> LIVE_FEED: OK
        </div>
      </div>

      <div className="lp-terminal-body">
        {/* Main Chart Area */}
        <div className="lp-terminal-main">
          <div className="lp-terminal-header">
            <div className="lp-term-sym">
              <span className="sym-n">NVDA</span>
              <span className="sym-f">NVIDIA Corporation</span>
            </div>
            <div className="lp-term-price">
              <span className="p-curr">852.14</span>
              <span className="p-chg up">+4.12%</span>
            </div>
          </div>

          <div className="lp-term-canvas">
            {/* SVG Candlestick Mockup */}
            <svg viewBox="0 0 600 240" preserveAspectRatio="none" className="lp-term-svg">
              {/* Grid */}
              <path d="M0 60 H600 M0 120 H600 M0 180 H600 M100 0 V240 M200 0 V240 M300 0 V240 M400 0 V240 M500 0 V240" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />

              {/* Bollinger Bands */}
              <path d="M0 100 Q150 60 300 110 T600 90" fill="none" stroke="rgba(99,102,241,0.2)" strokeWidth="1" />
              <path d="M0 160 Q150 120 300 170 T600 150" fill="none" stroke="rgba(99,102,241,0.2)" strokeWidth="1" />

              {/* Main Moving Average */}
              <motion.path
                d="M0 130 C100 110 200 160 300 140 S500 100 600 120"
                fill="none" stroke="var(--lp-accent)" strokeWidth="2"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.5 }}
              />

              {/* Candlesticks */}
              {[...Array(18)].map((_, i) => {
                const x = 20 + i * 32;
                const h = 40 + Math.random() * 60;
                const up = Math.random() > 0.4;
                return (
                  <motion.g key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    <line x1={x} y1={120 - h / 2 - 10} x2={x} y2={120 + h / 2 + 10} stroke={up ? 'var(--lp-green)' : 'var(--lp-red)'} opacity="0.4" />
                    <rect x={x - 6} y={120 - h / 2} width="12" height={h} fill={up ? 'var(--lp-green)' : 'var(--lp-red)'} />
                  </motion.g>
                );
              })}
            </svg>

            {/* AI Highlight overlay */}
            <motion.div className="lp-terminal-ai-highlight"
              animate={{ opacity: [0.1, 0.2, 0.1] }}
              transition={{ duration: 3, repeat: Infinity }}
            />
          </div>

          <div className="lp-terminal-sub-charts">
            <div className="lp-sub-chart">
              <span className="sub-lbl">MACD (12, 26, 9)</span>
              <div className="sub-viz">
                {[...Array(24)].map((_, i) => (
                  <div key={i} className="sub-bar" style={{ height: (10 + Math.random() * 80) + '%', background: Math.random() > 0.5 ? 'var(--lp-green)' : 'var(--lp-red)' }} />
                ))}
              </div>
            </div>
            <div className="lp-sub-chart">
              <span className="sub-lbl">VOL (20)</span>
              <div className="sub-viz">
                {[...Array(24)].map((_, i) => (
                  <div key={i} className="sub-bar" style={{ height: (20 + Math.random() * 60) + '%', background: '#475569' }} />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar panels */}
        <div className="lp-terminal-side">
          {/* Order Book */}
          <div className="lp-panel lp-orderbook">
            <div className="p-head">ORDER_BOOK</div>
            <div className="p-cols"><span>PRICE</span><span>SIZE</span></div>
            <div className="p-asks">
              {[...asks].reverse().map((a, i) => (
                <div key={i} className="p-row dn">
                  <div className="row-fill" style={{ width: a.v + '%' }} />
                  <span>{a.p}</span><span>{a.s}</span>
                </div>
              ))}
            </div>
            <div className="p-spread">0.36 SPREAD</div>
            <div className="p-bids">
              {bids.map((b, i) => (
                <div key={i} className="p-row up">
                  <div className="row-fill" style={{ width: b.v + '%' }} />
                  <span>{b.p}</span><span>{b.s}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Trade Tape */}
          <div className="lp-panel lp-tape">
            <div className="p-head">TRADE_TAPE</div>
            <div className="p-scroll">
              {tape.map((t, i) => (
                <div key={i} className={`pt-row ${t.side}`}>
                  <span className="pt-p">{t.pr}</span>
                  <span className="pt-s">{t.sz}</span>
                  <span className="pt-t">{t.t}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="lp-terminal-footer">
        <div className="lp-footer-met">CPU: 12% | RAM: 4.2GB | API_LATENCY: 12ms</div>
        <div className="lp-footer-sys">SYSTEM READY // DATAVISION V2.4</div>
      </div>
    </div>
  );
}

function PremiumHeroVisual() {
  const { isDark } = useTheme();
  const heroImg = isDark ? '/assets/hero-dark.jpg' : '/assets/hero-light.png';

  return (
    <div className="lp-hero-image-wrap">
      <motion.div
        className="lp-hero-image-container"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="lp-hero-image-frame">
          <img
            src={heroImg}
            alt="Datavision Trading Interface"
            className="lp-hero-img"
          />
          <div className="lp-hero-image-overlay" />
        </div>
      </motion.div>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────── 
   BACKGROUND LINES
   ──────────────────────────────────────────────────────────── */

/* Background lines are now part of CinematicBackground canvas */

/* ──────────────────────────────────────────────────────────── 
   STOCK CARD
   ──────────────────────────────────────────────────────────── */

function StockCard({ s, i }) {
  const { isDark } = useTheme();
  const col = s.dir === 'up'
    ? (isDark ? '#00e87b' : '#10b981')
    : (isDark ? '#ff4466' : '#ef4444');

  return (
    <motion.div className="lp-stock"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ delay: i * 0.08, duration: 0.45 }}
    >
      <div className="lp-stock-top">
        <div className={`lp-stock-icon ${s.ic}`}>{s.sym.slice(0, 4)}</div>
        <div className="lp-stock-badge">AI Model</div>
      </div>
      <div className="lp-stock-name">{s.sym}</div>
      <div className="lp-stock-full">{s.full}</div>
      <div className="lp-stock-pr-row">
        <span className="lp-stock-pr">${s.price.toFixed(2)}</span>
        <span className={`lp-stock-chg ${s.dir}`}>
          {s.chg > 0 ? '+' : ''}{s.pct.toFixed(2)}%
        </span>
      </div>
      <div className="lp-spark"><Spark data={s.pts} color={col} /></div>
      <div className="lp-stock-ai">
        <div className={`lp-stock-ai-arr ${s.dir}`}>{s.dir === 'up' ? '↑' : '↓'}</div>
        <span className="lp-stock-ai-lbl">AI: {s.pred}</span>
        <span className={`lp-stock-ai-pct ${s.dir}`}>{s.conf}%</span>
      </div>
    </motion.div>
  );
}


export default function Home() {
  const nav = useNavigate();
  const { isDark } = useTheme();
  const { scrollYProgress } = useScroll();
  const heroO = useTransform(scrollYProgress, [0, 0.12], [1, 0]);
  const heroS = useTransform(scrollYProgress, [0, 0.12], [1, 0.97]);

  const fade = {
    hidden: { opacity: 0, y: 24 },
    visible: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.08, duration: 0.5 } }),
  };

  return (
    <div className="lp">

      {/* ═══ HERO ═══ */}
      <motion.section className="lp-hero" style={{ opacity: heroO }}>
        <div className="lp-hero-bg" />
        <div className="lp-grid" />
        <div className="lp-c">
          <motion.div className="lp-hero-inner" style={{ scale: heroS }}>
            {/* Left */}
            <div>
              <motion.div className="lp-hero-tag" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
                <span className="lp-hero-tag-dot" />
                Live AI Models Active
              </motion.div>

              <motion.h1 className="lp-hero-h1" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45, duration: 0.6 }}>
                <em>AI-Powered</em> Stock{'\n'}Predictions for{' '}
                Smart Investors
              </motion.h1>

              <motion.p className="lp-hero-sub" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
                Real-time market intelligence powered by LSTM deep learning and technical analysis. 176+ stocks analyzed daily across US and Indian markets.
              </motion.p>

              <motion.div className="lp-btns" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.75 }}>
                <button className="lp-btn lp-btn-fill" onClick={() => nav('/prediction')}>
                  Start Predicting →
                </button>
                <button className="lp-btn lp-btn-ghost" onClick={() => nav('/prediction')}>
                  View Live Market
                </button>
              </motion.div>

              <motion.div className="lp-hero-metrics" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}>
                <div>
                  <div className="lp-metric-val green">LSTM</div>
                  <div className="lp-metric-lbl">Deep Learning</div>
                </div>
                <div>
                  <div className="lp-metric-val">176+</div>
                  <div className="lp-metric-lbl">Stocks Tracked</div>
                </div>
                <div>
                  <div className="lp-metric-val green">&lt;2s</div>
                  <div className="lp-metric-lbl">Analysis Time</div>
                </div>
              </motion.div>
            </div>

            {/* Right */}
            <motion.div className="lp-hero-right"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5, duration: 0.7 }}
            >
              <PremiumHeroVisual />
            </motion.div>
          </motion.div>
        </div>
      </motion.section>

      {/* ═══ LIVE STOCK DASHBOARD ═══ */}
      <section className="lp-section lp-section-alt">
        <div className="lp-c">
          <motion.div className="lp-sh" initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-60px' }}>
            <motion.div className="lp-sh-tag" variants={fade}><span className="lp-sh-dot" /> Live Dashboard</motion.div>
            <motion.h2 variants={fade} custom={1}>Real-Time AI Stock Intelligence</motion.h2>
            <motion.p variants={fade} custom={2}>Our models analyze price action, volume, indicators, and sentiment to score stocks in real time.</motion.p>
          </motion.div>
          <div className="lp-stocks">
            {STOCKS.map((stock, i) => <StockCard key={stock.sym} s={stock} i={i} />)}
          </div>
        </div>
      </section>

      {/* ═══ FEATURES ═══ */}
      <section className="lp-section">
        <div className="lp-c">
          <motion.div className="lp-sh" initial="hidden" whileInView="visible" viewport={{ once: true, margin: '-60px' }}>
            <motion.div className="lp-sh-tag" variants={fade}><span className="lp-sh-dot" /> Platform Features</motion.div>
            <motion.h2 variants={fade} custom={1}>Everything You Need to Trade Smarter</motion.h2>
            <motion.p variants={fade} custom={2}>Professional-grade tools powered by machine learning — from predictions to risk management.</motion.p>
          </motion.div>
          <div className="lp-feats">
            {FEATURES.map((f, i) => (
              <motion.div key={i} className="lp-feat"
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ delay: i * 0.07, duration: 0.4 }}
              >
                <div className={`lp-feat-ic ${f.c}`}><span>{f.icon}</span></div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>


      {/* ═══ CTA ═══ */}
      <section className="lp-section">
        <div className="lp-c">
          <motion.div className="lp-cta"
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.5 }}
          >
            <motion.h2 initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.15 }}>
              Start Making Smarter Trades Today
            </motion.h2>
            <motion.p initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.3 }}>
              Create your free account and access AI-powered stock predictions, real-time analysis, and institutional-grade trading tools.
            </motion.p>
            <motion.div className="lp-cta-btns" initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.4 }}>
              <button className="lp-btn lp-btn-fill" onClick={() => nav('/register')}>Get Started Free →</button>
              <button className="lp-btn lp-btn-ghost" onClick={() => nav('/about')}>Learn More</button>
            </motion.div>
          </motion.div>
        </div>
      </section>

    </div>
  );
}
