import React, { useEffect, useRef } from 'react';
import { useTheme } from '../contexts/ThemeContext';

/* ══════════════════════════════════════════════════════════════
   CINEMATIC TRADING BACKGROUND
   
   Realistic stock market visuals:
   - Moving candlestick charts scrolling horizontally
   - Technical indicators (moving average curves)
   - Price level lines with labels
   - Volume bars at bottom
   - Chart grid
   ══════════════════════════════════════════════════════════════ */

export default function CinematicBackground() {
    const ref = useRef(null);
    const anim = useRef(null);
    const { isDark } = useTheme();
    const themeRef = useRef(isDark);
    useEffect(() => { themeRef.current = isDark; }, [isDark]);

    useEffect(() => {
        const cv = ref.current;
        if (!cv) return;
        const ctx = cv.getContext('2d');
        let W, H;

        const resize = () => {
            W = window.innerWidth;
            H = window.innerHeight;
            cv.width = W;
            cv.height = H;
        };
        resize();
        window.addEventListener('resize', resize);

        /* ── Generate realistic OHLCV data ── */
        function makeChart(length, basePrice, volatility, trend) {
            const candles = [];
            let price = basePrice;
            for (let i = 0; i < length; i++) {
                const dir = Math.random() > (0.5 - trend * 0.1) ? 1 : -1;
                const move = Math.random() * volatility;
                const open = price;
                const close = price + dir * move;
                const high = Math.max(open, close) + Math.random() * volatility * 0.5;
                const low = Math.min(open, close) - Math.random() * volatility * 0.5;
                const vol = 0.3 + Math.random() * 0.7;
                candles.push({ o: open, c: close, h: high, l: low, vol, green: close >= open });
                price = close + (Math.random() - 0.5) * volatility * 0.3;
            }
            return candles;
        }

        /* ── SMA from candles ── */
        function sma(candles, period) {
            const result = [];
            for (let i = 0; i < candles.length; i++) {
                if (i < period - 1) { result.push(null); continue; }
                let sum = 0;
                for (let j = i - period + 1; j <= i; j++) sum += candles[j].c;
                result.push(sum / period);
            }
            return result;
        }

        const candleW = 10;
        const gap = 6;
        const step = candleW + gap;

        /* ── Chart strips: 3 horizontal bands of scrolling charts ── */
        const initialCount = Math.ceil(W / step) + 10;
        const strips = [
            {
                candles: makeChart(initialCount, 180, 4, 0.15),
                y: 0, h: 0.35, speed: 0.3, offset: 0,
                label: 'NVDA', labelPrice: '$138.85',
            },
            {
                candles: makeChart(initialCount, 350, 6, 0.1),
                y: 0.33, h: 0.35, speed: 0.22, offset: 0,
                label: 'TSLA', labelPrice: '$352.64',
            },
            {
                candles: makeChart(initialCount, 640, 8, 0.05),
                y: 0.65, h: 0.35, speed: 0.18, offset: 0,
                label: 'META', labelPrice: '$642.73',
            },
        ];

        let frame = 0;

        /* ═══════════════ DRAW LOOP ═══════════════ */
        const draw = () => {
            const dark = themeRef.current;
            frame++;
            ctx.clearRect(0, 0, W, H);

            // ── OPACITY MULTIPLIER: light mode is visible, dark mode is cinematic ──
            const opMul = dark ? 1.0 : 1.0;

            // Colors
            const gridCol = dark ? 'rgba(0,180,255,0.035)' : 'rgba(0,60,140,0.045)';
            const greenC = dark ? '#00d4a0' : '#10b981';
            const redC = dark ? '#ff4d6a' : '#ef4444';
            const maCol1 = dark ? 'rgba(0,180,255,0.18)' : 'rgba(0,90,255,0.15)';
            const maCol2 = dark ? 'rgba(255,170,0,0.14)' : 'rgba(200,120,0,0.12)';
            const textCol = dark ? 'rgba(136,146,168,0.2)' : 'rgba(60,80,120,0.15)';
            const candleAlpha = dark ? 0.14 : 0.12;
            const volAlpha = dark ? 0.06 : 0.055;
            const labelAlpha = dark ? 0.12 : 0.10;

            /* ── Background grid ── */
            ctx.strokeStyle = gridCol;
            ctx.lineWidth = 0.5;
            for (let x = 0; x < W; x += 60) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
            }
            for (let y = 0; y < H; y += 60) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
            }

            /* ── Draw each chart strip ── */
            for (const strip of strips) {
                strip.offset += strip.speed;

                // Adjust candles count dynamically for viewport width
                const currentNeeded = Math.ceil(W / step) + 5;
                while (strip.candles.length < currentNeeded) {
                    const last = strip.candles[strip.candles.length - 1];
                    const dir = Math.random() > 0.45 ? 1 : -1;
                    const move = Math.random() * 4;
                    const o = last.c + (Math.random() - 0.5) * 2;
                    const c = o + dir * move;
                    const h = Math.max(o, c) + Math.random() * 2;
                    const l = Math.min(o, c) - Math.random() * 2;
                    strip.candles.push({ o, c, h, l, vol: 0.3 + Math.random() * 0.7, green: c >= o });
                }

                if (strip.offset > step) {
                    strip.offset -= step;
                    strip.candles.shift();
                    const last = strip.candles[strip.candles.length - 1];
                    const dir = Math.random() > 0.45 ? 1 : -1;
                    const move = Math.random() * 4;
                    const o = last.c + (Math.random() - 0.5) * 2;
                    const c = o + dir * move;
                    const h = Math.max(o, c) + Math.random() * 2;
                    const l = Math.min(o, c) - Math.random() * 2;
                    strip.candles.push({ o, c, h, l, vol: 0.3 + Math.random() * 0.7, green: c >= o });
                }

                const data = strip.candles;
                const topY = strip.y * H;
                const stripH = strip.h * H;
                const botY = topY + stripH;

                // Price range
                let minP = Infinity, maxP = -Infinity;
                for (const cd of data) {
                    if (cd.l < minP) minP = cd.l;
                    if (cd.h > maxP) maxP = cd.h;
                }
                const pRange = maxP - minP || 1;
                const chartTop = topY + stripH * 0.08;
                const chartBot = botY - stripH * 0.18;
                const chartH = chartBot - chartTop;

                const py = (p) => chartBot - ((p - minP) / pRange) * chartH;

                // ── Price level lines ──
                ctx.setLineDash([4, 6]);
                ctx.lineWidth = 0.5;
                const levels = 4;
                for (let i = 0; i <= levels; i++) {
                    const p = minP + (pRange / levels) * i;
                    const y = py(p);
                    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.035)';
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
                    // Price label
                    ctx.font = '500 9px JetBrains Mono, monospace';
                    ctx.fillStyle = textCol;
                    ctx.textAlign = 'right';
                    ctx.fillText(p.toFixed(1), W - 8, y - 3);
                }
                ctx.setLineDash([]);

                // ── Volume bars ──
                const volTop = chartBot + 4;
                const volH = stripH * 0.1;
                for (let i = 0; i < data.length; i++) {
                    const x = i * step - strip.offset;
                    if (x < -step || x > W + step) continue;
                    const cd = data[i];
                    const bh = cd.vol * volH;
                    const rgb = cd.green ? greenC : redC;
                    ctx.fillStyle = rgb;
                    ctx.globalAlpha = volAlpha;
                    ctx.fillRect(x - candleW / 2, volTop + volH - bh, candleW, bh);
                }
                ctx.globalAlpha = 1;

                // ── Candlesticks ──
                for (let i = 0; i < data.length; i++) {
                    const x = i * step - strip.offset;
                    if (x < -step || x > W + step) continue;
                    const cd = data[i];
                    const col = cd.green ? greenC : redC;

                    ctx.globalAlpha = candleAlpha;

                    // Wick
                    ctx.strokeStyle = col;
                    ctx.lineWidth = 1.2;
                    ctx.beginPath();
                    ctx.moveTo(x, py(cd.h));
                    ctx.lineTo(x, py(cd.l));
                    ctx.stroke();

                    // Body
                    const bodyTop = py(Math.max(cd.o, cd.c));
                    const bodyBot = py(Math.min(cd.o, cd.c));
                    const bodyH = Math.max(bodyBot - bodyTop, 1.5);
                    ctx.fillStyle = col;
                    ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH);

                    if (!cd.green) {
                        // Hollow red candles
                        ctx.globalAlpha = candleAlpha * 0.5;
                        const inner = dark ? '#060b18' : '#ffffff';
                        ctx.fillStyle = inner;
                        ctx.fillRect(x - candleW / 2 + 1.5, bodyTop + 1, candleW - 3, Math.max(bodyH - 2, 0));
                    }
                }
                ctx.globalAlpha = 1;

                // ── Moving averages (SMA 10, SMA 20) ──
                const sma10 = sma(data, 10);
                const sma20 = sma(data, 20);

                // SMA 10
                ctx.beginPath();
                ctx.strokeStyle = maCol1;
                ctx.lineWidth = 1.8;
                ctx.lineJoin = 'round';
                let started = false;
                for (let i = 0; i < data.length; i++) {
                    if (sma10[i] === null) continue;
                    const x = i * step - strip.offset;
                    const y = py(sma10[i]);
                    if (!started) { ctx.moveTo(x, y); started = true; }
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();

                // SMA 20
                ctx.beginPath();
                ctx.strokeStyle = maCol2;
                ctx.lineWidth = 1.8;
                started = false;
                for (let i = 0; i < data.length; i++) {
                    if (sma20[i] === null) continue;
                    const x = i * step - strip.offset;
                    const y = py(sma20[i]);
                    if (!started) { ctx.moveTo(x, y); started = true; }
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();

                // ── Chart label (ticker name) ──
                ctx.globalAlpha = labelAlpha;
                ctx.font = '800 14px Inter, sans-serif';
                ctx.fillStyle = dark ? '#00b8ff' : '#0066ff';
                ctx.textAlign = 'left';
                ctx.fillText(strip.label, 16, topY + 22);
                ctx.font = '600 11px JetBrains Mono, monospace';
                ctx.fillStyle = dark ? '#00d4a0' : '#10b981';
                ctx.fillText(strip.labelPrice, 16, topY + 38);
                ctx.globalAlpha = 1;
            }

            anim.current = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            window.removeEventListener('resize', resize);
            if (anim.current) cancelAnimationFrame(anim.current);
        };
    }, []);

    return (
        <canvas
            ref={ref}
            style={{
                position: 'fixed',
                inset: 0,
                zIndex: 0,
                pointerEvents: 'none',
                opacity: 1
            }}
        />
    );
}
