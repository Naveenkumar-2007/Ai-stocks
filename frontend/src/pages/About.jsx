import React from 'react';
import { Brain, TrendingUp, Shield, Zap, Target, Award, Users, BarChart3, Sparkles, LineChart, Lock, Clock } from 'lucide-react';

const About = () => {
  const features = [
    {
      icon: Brain,
      title: 'LSTM Deep Learning',
      description: 'LSTM neural networks trained on historical price, volume, and indicator data generate multi-day price forecasts for every tracked stock.',
      gradient: 'from-cyan-500 to-blue-500'
    },
    {
      icon: TrendingUp,
      title: 'Real-Time Market Intelligence',
      description: 'Live price feeds, technical indicators, and candlestick charts update continuously to keep you ahead of market movements.',
      gradient: 'from-blue-500 to-purple-500'
    },
    {
      icon: Shield,
      title: 'Enterprise-Grade Security',
      description: 'Bank-level encryption and authentication protect your data while ensuring reliable, accurate analysis you can trust.',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: Zap,
      title: 'Instant Analysis',
      description: 'High-performance computing infrastructure delivers comprehensive stock analysis and forecasts in under 2 seconds.',
      gradient: 'from-pink-500 to-rose-500'
    },
    {
      icon: Target,
      title: 'Actionable Recommendations',
      description: 'Clear BUY, SELL, or HOLD signals with confidence scores help you execute trades at the optimal time.',
      gradient: 'from-rose-500 to-orange-500'
    },
    {
      icon: BarChart3,
      title: 'Professional Visualizations',
      description: 'Interactive charts with technical overlays, volume analysis, and performance metrics make complex data intuitive.',
      gradient: 'from-orange-500 to-amber-500'
    }
  ];

  const stats = [
    { label: 'AI Model', value: 'LSTM' },
    { label: 'Stocks Tracked', value: '176+' },
    { label: 'Forecast Window', value: '5-Day' },
    { label: 'Analysis Time', value: '<2s' }
  ];

  const benefits = [
    {
      icon: Sparkles,
      text: 'Execute trades with confidence using AI-powered forecasts backed by historical accuracy metrics'
    },
    {
      icon: Clock,
      text: 'Save hours of research time with automated technical analysis and multi-day price predictions'
    },
    {
      icon: Shield,
      text: 'Minimize investment risk by understanding market sentiment, trend strength, and volatility patterns'
    },
    {
      icon: LineChart,
      text: 'Monitor unlimited stocks with customizable watchlists and detailed performance tracking'
    },
    {
      icon: TrendingUp,
      text: 'Stay informed with real-time financial news and social sentiment analysis from multiple sources'
    },
    {
      icon: Target,
      text: 'Plan strategic entries and exits using 5-day price forecasts with confidence intervals'
    }
  ];

  return (
    <div className="min-h-screen bg-transparent relative z-10">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-transparent border-b border-gray-200 dark:border-gray-800">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-blue-500/5 to-purple-500/5 dark:from-cyan-500/10 dark:via-blue-500/10 dark:to-purple-500/10" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
          <div className="text-center max-w-3xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/20 rounded-full mb-6">
              <Brain className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
              <span className="text-sm font-medium text-cyan-700 dark:text-cyan-400">AI-Powered Stock Analysis</span>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 dark:text-white mb-6">
              About <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">Datavision</span>
            </h1>
            <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 leading-relaxed">
              Empowering retail investors with institutional-grade artificial intelligence to level
              the playing field. Our platform delivers professional stock analysis, real-time market
              intelligence, and AI-powered predictions previously available only to hedge funds and
              investment banks.
            </p>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent mb-2">
                  {stat.value}
                </div>
                <div className="text-sm sm:text-base text-gray-600 dark:text-gray-400 font-medium">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        {/* Mission Statement */}
        <div className="mb-16">
          <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-3xl p-8 sm:p-12 text-white shadow-2xl">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="text-3xl sm:text-4xl font-bold mb-6">Our Mission</h2>
              <p className="text-lg sm:text-xl leading-relaxed opacity-95">
                To democratize professional investment analysis by making institutional-grade AI technology
                accessible to every investor. We believe sophisticated market intelligence shouldn't be
                exclusive to Wall Street—everyone deserves the tools to build wealth through informed,
                data-driven decisions. Our advanced algorithms process millions of data points daily to
                deliver the insights you need to succeed.
              </p>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Powerful Features
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Everything you need to make informed investment decisions, powered by advanced AI technology
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl p-6 hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
                >
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Benefits Section */}
        <div className="mb-16">
          <div className="bg-white dark:bg-gray-800 rounded-3xl border border-gray-200 dark:border-gray-700 p-8 sm:p-12">
            <div className="text-center mb-10">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-4">
                Why Choose Datavision?
              </h2>
              <p className="text-lg text-gray-600 dark:text-gray-400">
                Institutional-grade AI tools built for every investor
              </p>
            </div>

            <div className="grid sm:grid-cols-2 gap-6 max-w-4xl mx-auto">
              {benefits.map((benefit, index) => {
                const Icon = benefit.icon;
                return (
                  <div key={index} className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-10 h-10 bg-cyan-100 dark:bg-cyan-500/10 rounded-lg flex items-center justify-center">
                      <Icon className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                    </div>
                    <p className="text-gray-700 dark:text-gray-300 leading-relaxed pt-1.5">
                      {benefit.text}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="bg-gradient-to-br from-cyan-600 to-blue-600 dark:from-gray-900 dark:to-gray-800 rounded-3xl p-8 sm:p-12 text-center shadow-xl">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Ready to Start?
          </h2>
          <p className="text-lg text-white/90 dark:text-gray-300 mb-8 max-w-2xl mx-auto">
            Join thousands of investors who are already making smarter decisions with Datavision
          </p>
          <a
            href="/prediction"
            className="inline-flex items-center gap-2 px-8 py-4 bg-white dark:bg-gradient-to-r dark:from-cyan-600 dark:to-blue-600 text-cyan-600 dark:text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl hover:-translate-y-0.5"
          >
            <Brain className="w-5 h-5" />
            <span>Get Started Now</span>
          </a>
        </div>
      </div>
    </div>
  );
};

export default About;
