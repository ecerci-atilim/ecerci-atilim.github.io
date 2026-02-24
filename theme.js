/**
 * Theme System
 * Themes: academic (Classic), glass (Glassmorphism), dark (Minimal Dark)
 *
 * Each theme has:
 *  - Its own Google Font stack
 *  - Completely different border-radius, shadows, spacing
 *  - Unique decorative pseudo-elements
 *  - Distinct hover behaviours
 *  - Different font weights, sizes, letter-spacing
 */
(function () {
    'use strict';

    var STORAGE_KEY = 'site-theme';
    var DEFAULT_THEME = 'academic';
    var currentTheme = localStorage.getItem(STORAGE_KEY) || DEFAULT_THEME;

    document.documentElement.setAttribute('data-theme', currentTheme);

    /* ----------------------------------------------------------------
       1. Dynamic Google Fonts per theme
       ---------------------------------------------------------------- */
    var fontLinks = {
        academic: 'https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap',
        dark: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
        // glass uses Outfit + Plus Jakarta Sans already loaded in each page
    };

    function loadFont(theme) {
        var id = 'theme-font-link';
        var existing = document.getElementById(id);
        if (existing) existing.remove();
        if (fontLinks[theme]) {
            var link = document.createElement('link');
            link.id = id;
            link.rel = 'stylesheet';
            link.href = fontLinks[theme];
            document.head.appendChild(link);
        }
    }
    loadFont(currentTheme);

    /* ----------------------------------------------------------------
       2. Full theme CSS
       ---------------------------------------------------------------- */
    var css = '\
\n\
/* ================================================================ */\n\
/*  CLASSIC THEME                                                   */\n\
/*  Serif headings · System-like body · No effects · Institutional  */\n\
/* ================================================================ */\n\
\n\
html[data-theme="academic"] {\n\
  --bg: #f4f3ef;\n\
  --card-bg: #ffffff;\n\
  --card-border: #d8d5cd;\n\
  --card-hover: #f0efeb;\n\
  --text-primary: #1b1b30;\n\
  --text-secondary: #52526e;\n\
  --accent: #1e3a5f;\n\
  --accent-light: #2b5a8a;\n\
  --accent-bg: rgba(30, 58, 95, 0.06);\n\
  --accent-border: rgba(30, 58, 95, 0.15);\n\
  --radius: 4px;\n\
  --shadow: 0 1px 2px rgba(0,0,0,0.04);\n\
  --font-heading: "Lora", Georgia, "Times New Roman", serif;\n\
  --font-body: "Source Sans 3", "Plus Jakarta Sans", -apple-system, sans-serif;\n\
  --transition: all 0.15s ease;\n\
}\n\
\n\
/* --- Body --- */\n\
[data-theme="academic"] body {\n\
  background: var(--bg) !important;\n\
  font-family: var(--font-body) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
/* --- Kill glass effects & animated bg --- */\n\
[data-theme="academic"] .floating-shapes,\n\
[data-theme="academic"] .ambient-light { display: none !important; }\n\
\n\
[data-theme="academic"] .glass-card,\n\
[data-theme="academic"] .title-card,\n\
[data-theme="academic"] .table-container,\n\
[data-theme="academic"] .status-card,\n\
[data-theme="academic"] .back-btn,\n\
[data-theme="academic"] .gate-card {\n\
  backdrop-filter: none !important;\n\
  -webkit-backdrop-filter: none !important;\n\
}\n\
\n\
/* --- Kill ALL animations --- */\n\
[data-theme="academic"] *,\n\
[data-theme="academic"] *::before,\n\
[data-theme="academic"] *::after {\n\
  animation: none !important;\n\
  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;\n\
}\n\
[data-theme="academic"] .tool-card,\n\
[data-theme="academic"] .activity-cell {\n\
  opacity: 1 !important;\n\
}\n\
\n\
/* --- Cards --- */\n\
[data-theme="academic"] .glass-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: var(--shadow) !important;\n\
  padding: 28px !important;\n\
  position: relative;\n\
}\n\
\n\
/* Accent top-border on cards */\n\
[data-theme="academic"] .glass-card::before {\n\
  content: "" !important;\n\
  display: block !important;\n\
  position: absolute !important;\n\
  top: 0; left: 0; right: 0 !important;\n\
  height: 3px !important;\n\
  background: var(--accent) !important;\n\
  border-radius: var(--radius) var(--radius) 0 0 !important;\n\
}\n\
\n\
/* --- Headings --- */\n\
[data-theme="academic"] .header-title {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 700 !important;\n\
  font-size: clamp(1.8rem, 3.5vw, 2.8rem) !important;\n\
  letter-spacing: 0 !important;\n\
  background: none !important;\n\
  -webkit-background-clip: unset !important;\n\
  -webkit-text-fill-color: var(--text-primary) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="academic"] .profile-name {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 700 !important;\n\
  background: none !important;\n\
  -webkit-background-clip: unset !important;\n\
  -webkit-text-fill-color: var(--text-primary) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="academic"] .page-title {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 700 !important;\n\
  font-size: 1.8rem !important;\n\
  background: none !important;\n\
  -webkit-background-clip: unset !important;\n\
  -webkit-text-fill-color: var(--text-primary) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="academic"] .section-title,\n\
[data-theme="academic"] .card-title {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 600 !important;\n\
  font-size: 1.3rem !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="academic"] .header-subtitle,\n\
[data-theme="academic"] .profile-title {\n\
  font-family: var(--font-body) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-weight: 400 !important;\n\
  font-size: 1.05rem !important;\n\
}\n\
\n\
/* --- Hide emoji spans --- */\n\
[data-theme="academic"] .section-header > span:not(.section-title) { display: none !important; }\n\
[data-theme="academic"] .card-title > span { display: none !important; }\n\
\n\
/* --- Section borders --- */\n\
[data-theme="academic"] .section-header {\n\
  border-bottom: 2px solid var(--accent) !important;\n\
  padding-bottom: 12px !important;\n\
  margin-bottom: 20px !important;\n\
}\n\
[data-theme="academic"] .card-title {\n\
  border-bottom: 2px solid var(--accent) !important;\n\
  padding-bottom: 12px !important;\n\
}\n\
\n\
/* --- Institution badge --- */\n\
[data-theme="academic"] .header-institution {\n\
  background: var(--accent-bg) !important;\n\
  border: 1px solid var(--accent-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--accent) !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 500 !important;\n\
}\n\
\n\
/* --- Tool cards --- */\n\
[data-theme="academic"] .tool-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-primary) !important;\n\
  padding: 14px 16px !important;\n\
}\n\
[data-theme="academic"] .tool-card:hover {\n\
  background: var(--card-hover) !important;\n\
  border-color: var(--accent-border) !important;\n\
  transform: none !important;\n\
  box-shadow: var(--shadow) !important;\n\
}\n\
[data-theme="academic"] .tool-icon { color: var(--accent) !important; }\n\
[data-theme="academic"] .tool-card:hover .tool-icon { transform: none !important; color: var(--accent-light) !important; }\n\
[data-theme="academic"] .tool-name {\n\
  color: var(--text-primary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
\n\
/* --- Tool badges --- */\n\
[data-theme="academic"] .tool-badge {\n\
  background: var(--accent-bg) !important;\n\
  color: var(--accent) !important;\n\
  border-radius: 2px !important;\n\
  font-size: 0.6rem !important;\n\
  text-transform: uppercase !important;\n\
  letter-spacing: 0.8px !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 600 !important;\n\
}\n\
\n\
/* --- Link buttons --- */\n\
[data-theme="academic"] .link-btn {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .link-btn:hover {\n\
  background: var(--accent) !important;\n\
  color: #fff !important;\n\
  border-color: var(--accent) !important;\n\
  transform: none !important;\n\
  box-shadow: none !important;\n\
}\n\
\n\
/* --- Contact buttons --- */\n\
[data-theme="academic"] .contact-btn {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .contact-btn:hover {\n\
  background: var(--accent) !important;\n\
  color: #fff !important;\n\
  border-color: var(--accent) !important;\n\
  transform: none !important;\n\
  box-shadow: none !important;\n\
}\n\
\n\
/* --- Back button --- */\n\
[data-theme="academic"] .back-btn {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .back-btn:hover {\n\
  background: var(--card-hover) !important;\n\
  color: var(--text-primary) !important;\n\
  transform: none !important;\n\
}\n\
\n\
/* --- Avatar --- */\n\
[data-theme="academic"] .avatar-container {\n\
  background: var(--accent) !important;\n\
  box-shadow: none !important;\n\
  border: 3px solid var(--card-border) !important;\n\
  font-family: var(--font-heading) !important;\n\
}\n\
\n\
/* --- Tags --- */\n\
[data-theme="academic"] .tag {\n\
  background: var(--accent-bg) !important;\n\
  border: 1px solid var(--accent-border) !important;\n\
  color: var(--text-primary) !important;\n\
  border-radius: 2px !important;\n\
  font-family: var(--font-body) !important;\n\
  font-size: 0.85rem !important;\n\
}\n\
[data-theme="academic"] .tag:hover {\n\
  background: rgba(30, 58, 95, 0.12) !important;\n\
  transform: none !important;\n\
}\n\
\n\
[data-theme="academic"] .skill-label {\n\
  color: var(--accent) !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 600 !important;\n\
}\n\
\n\
[data-theme="academic"] .intro-text {\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
  line-height: 1.8 !important;\n\
}\n\
\n\
/* --- Footer --- */\n\
[data-theme="academic"] .footer {\n\
  color: var(--text-secondary) !important;\n\
  border-top: 1px solid var(--card-border) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
\n\
/* --- Spacing --- */\n\
[data-theme="academic"] .container {\n\
  padding-top: 40px !important;\n\
  padding-bottom: 40px !important;\n\
}\n\
[data-theme="academic"] .header-section {\n\
  margin-bottom: 36px !important;\n\
}\n\
\n\
/* --- SCHEDULE: Classic --- */\n\
[data-theme="academic"] .title-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
}\n\
[data-theme="academic"] .status-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: var(--shadow) !important;\n\
}\n\
[data-theme="academic"] .clock-display { color: var(--text-primary) !important; font-family: var(--font-body) !important; font-weight: 600 !important; }\n\
[data-theme="academic"] .date-display { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }\n\
[data-theme="academic"] .table-container {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: var(--shadow) !important;\n\
}\n\
[data-theme="academic"] .schedule-table th {\n\
  color: var(--text-secondary) !important;\n\
  border-bottom-color: var(--card-border) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .time-col {\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .activity-cell {\n\
  background: #fafaf7 !important;\n\
  border: 1px solid #eeece6 !important;\n\
  border-radius: 3px !important;\n\
}\n\
[data-theme="academic"] .activity-cell:hover { background: var(--card-hover) !important; }\n\
[data-theme="academic"] .status-active  { background: rgba(30,95,60,0.08) !important; color: #1e5f3c !important; border: 1px solid rgba(30,95,60,0.18) !important; }\n\
[data-theme="academic"] .status-busy    { background: rgba(30,58,95,0.08) !important; color: var(--accent) !important; border: 1px solid var(--accent-border) !important; }\n\
[data-theme="academic"] .status-off     { background: rgba(82,82,110,0.06) !important; color: var(--text-secondary) !important; border: 1px solid rgba(82,82,110,0.12) !important; }\n\
[data-theme="academic"] .status-leave   { background: rgba(140,70,20,0.08) !important; color: #8c4614 !important; border: 1px solid rgba(140,70,20,0.18) !important; }\n\
[data-theme="academic"] .status-badge   { border-radius: var(--radius) !important; box-shadow: none !important; font-family: var(--font-body) !important; }\n\
[data-theme="academic"] .current-slot   { border: 2px solid var(--accent) !important; box-shadow: none !important; }\n\
[data-theme="academic"] #error-banner   { background: rgba(180,40,40,0.06) !important; color: #b42828 !important; border-color: rgba(180,40,40,0.18) !important; border-radius: var(--radius) !important; }\n\
\n\
/* --- PDF GATE: Classic --- */\n\
[data-theme="academic"] .gate-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: var(--shadow) !important;\n\
}\n\
[data-theme="academic"] .gate-title { font-family: var(--font-heading) !important; color: var(--text-primary) !important; }\n\
[data-theme="academic"] .gate-desc  { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }\n\
[data-theme="academic"] .gate-icon  { color: var(--accent) !important; }\n\
[data-theme="academic"] .pin-input  {\n\
  background: #f0efeb !important; color: var(--text-primary) !important;\n\
  border: 1px solid var(--card-border) !important; border-radius: var(--radius) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .pin-input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(30,58,95,0.1) !important; }\n\
[data-theme="academic"] .submit-btn {\n\
  background: var(--accent) !important;\n\
  border-radius: var(--radius) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="academic"] .submit-btn:hover { background: var(--accent-light) !important; transform: none !important; }\n\
[data-theme="academic"] .error-msg  { background: rgba(180,40,40,0.06) !important; border-color: rgba(180,40,40,0.18) !important; color: #b42828 !important; border-radius: var(--radius) !important; }\n\
[data-theme="academic"] .back-link  { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }\n\
[data-theme="academic"] .back-link:hover { color: var(--text-primary) !important; }\n\
\n\
\n\
/* ================================================================ */\n\
/*  MINIMAL DARK THEME                                              */\n\
/*  Geometric sans · Ultra flat · Monochrome · Clean lines          */\n\
/* ================================================================ */\n\
\n\
html[data-theme="dark"] {\n\
  --bg: #0c0c0c;\n\
  --card-bg: #161616;\n\
  --card-border: #232323;\n\
  --card-hover: #1c1c1c;\n\
  --text-primary: #c8c8c8;\n\
  --text-secondary: #666666;\n\
  --accent: #7c8cf0;\n\
  --accent-light: #a5b0ff;\n\
  --accent-bg: rgba(124, 140, 240, 0.06);\n\
  --accent-border: rgba(124, 140, 240, 0.15);\n\
  --radius: 8px;\n\
  --shadow: none;\n\
  --font-heading: "Inter", -apple-system, "Segoe UI", sans-serif;\n\
  --font-body: "Inter", -apple-system, "Segoe UI", sans-serif;\n\
  --transition: all 0.12s ease;\n\
}\n\
\n\
/* --- Body --- */\n\
[data-theme="dark"] body {\n\
  background: var(--bg) !important;\n\
  font-family: var(--font-body) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
/* --- Kill glass effects & animated bg --- */\n\
[data-theme="dark"] .floating-shapes,\n\
[data-theme="dark"] .ambient-light { display: none !important; }\n\
\n\
[data-theme="dark"] .glass-card,\n\
[data-theme="dark"] .title-card,\n\
[data-theme="dark"] .table-container,\n\
[data-theme="dark"] .status-card,\n\
[data-theme="dark"] .back-btn,\n\
[data-theme="dark"] .gate-card {\n\
  backdrop-filter: none !important;\n\
  -webkit-backdrop-filter: none !important;\n\
}\n\
\n\
/* --- Disable entrance animations, keep subtle hover transitions --- */\n\
[data-theme="dark"] .header-section,\n\
[data-theme="dark"] .dashboard-header,\n\
[data-theme="dark"] .profile-header,\n\
[data-theme="dark"] .glass-card,\n\
[data-theme="dark"] .table-container {\n\
  animation: none !important;\n\
}\n\
[data-theme="dark"] .tool-card,\n\
[data-theme="dark"] .activity-cell {\n\
  opacity: 1 !important;\n\
  animation: none !important;\n\
}\n\
\n\
/* --- Cards: flat dark panels with bottom accent --- */\n\
[data-theme="dark"] .glass-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: none !important;\n\
  padding: 28px !important;\n\
  position: relative;\n\
}\n\
\n\
/* Subtle bottom accent line on cards */\n\
[data-theme="dark"] .glass-card::after {\n\
  content: "" !important;\n\
  display: block !important;\n\
  position: absolute !important;\n\
  bottom: 0; left: 20px; right: 20px !important;\n\
  height: 1px !important;\n\
  background: linear-gradient(90deg, transparent, var(--accent), transparent) !important;\n\
  opacity: 0.25 !important;\n\
  border-radius: 0 !important;\n\
}\n\
\n\
/* --- Headings --- */\n\
[data-theme="dark"] .header-title {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 300 !important;\n\
  font-size: clamp(2rem, 4vw, 3.2rem) !important;\n\
  letter-spacing: -1px !important;\n\
  background: none !important;\n\
  -webkit-background-clip: unset !important;\n\
  -webkit-text-fill-color: var(--text-primary) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="dark"] .profile-name {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 300 !important;\n\
  letter-spacing: -0.5px !important;\n\
  background: none !important;\n\
  -webkit-background-clip: unset !important;\n\
  -webkit-text-fill-color: var(--text-primary) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="dark"] .page-title {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 300 !important;\n\
  letter-spacing: -0.5px !important;\n\
  font-size: 2rem !important;\n\
  background: none !important;\n\
  -webkit-background-clip: unset !important;\n\
  -webkit-text-fill-color: var(--text-primary) !important;\n\
  color: var(--text-primary) !important;\n\
}\n\
\n\
[data-theme="dark"] .section-title,\n\
[data-theme="dark"] .card-title {\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 500 !important;\n\
  font-size: 1.2rem !important;\n\
  letter-spacing: -0.3px !important;\n\
  color: var(--text-primary) !important;\n\
  text-transform: none !important;\n\
}\n\
\n\
[data-theme="dark"] .header-subtitle,\n\
[data-theme="dark"] .profile-title {\n\
  font-family: var(--font-body) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-weight: 300 !important;\n\
  letter-spacing: 1px !important;\n\
  text-transform: uppercase !important;\n\
  font-size: 0.85rem !important;\n\
}\n\
\n\
/* --- Section borders --- */\n\
[data-theme="dark"] .section-header {\n\
  border-bottom: 1px solid var(--card-border) !important;\n\
}\n\
[data-theme="dark"] .card-title {\n\
  border-bottom: 1px solid var(--card-border) !important;\n\
}\n\
\n\
/* --- Institution badge --- */\n\
[data-theme="dark"] .header-institution {\n\
  background: var(--accent-bg) !important;\n\
  border: 1px solid var(--accent-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--accent) !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 400 !important;\n\
  font-size: 0.85rem !important;\n\
  letter-spacing: 0.5px !important;\n\
}\n\
\n\
/* --- Tool cards --- */\n\
[data-theme="dark"] .tool-card {\n\
  background: #111111 !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-primary) !important;\n\
  padding: 16px !important;\n\
}\n\
[data-theme="dark"] .tool-card:hover {\n\
  background: var(--card-hover) !important;\n\
  border-color: #333 !important;\n\
  transform: none !important;\n\
  box-shadow: none !important;\n\
}\n\
[data-theme="dark"] .tool-icon { color: var(--accent) !important; }\n\
[data-theme="dark"] .tool-card:hover .tool-icon { transform: none !important; color: var(--accent-light) !important; }\n\
[data-theme="dark"] .tool-name { color: var(--text-primary) !important; font-family: var(--font-body) !important; }\n\
\n\
/* --- Tool badges --- */\n\
[data-theme="dark"] .tool-badge {\n\
  background: var(--accent-bg) !important;\n\
  color: var(--accent) !important;\n\
  border-radius: 4px !important;\n\
  font-family: var(--font-body) !important;\n\
  font-size: 0.6rem !important;\n\
  letter-spacing: 0.8px !important;\n\
  text-transform: uppercase !important;\n\
}\n\
\n\
/* --- Link buttons --- */\n\
[data-theme="dark"] .link-btn {\n\
  background: #111111 !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="dark"] .link-btn:hover {\n\
  background: var(--accent) !important;\n\
  color: #fff !important;\n\
  border-color: var(--accent) !important;\n\
  transform: none !important;\n\
  box-shadow: none !important;\n\
}\n\
\n\
/* --- Contact buttons --- */\n\
[data-theme="dark"] .contact-btn {\n\
  background: #111111 !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="dark"] .contact-btn:hover {\n\
  background: var(--accent) !important;\n\
  color: #fff !important;\n\
  border-color: var(--accent) !important;\n\
  transform: none !important;\n\
  box-shadow: none !important;\n\
}\n\
\n\
/* --- Back button --- */\n\
[data-theme="dark"] .back-btn {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
[data-theme="dark"] .back-btn:hover {\n\
  background: var(--card-hover) !important;\n\
  color: var(--text-primary) !important;\n\
  transform: none !important;\n\
}\n\
\n\
/* --- Avatar --- */\n\
[data-theme="dark"] .avatar-container {\n\
  background: var(--card-bg) !important;\n\
  box-shadow: none !important;\n\
  border: 2px solid var(--accent) !important;\n\
  font-family: var(--font-heading) !important;\n\
  font-weight: 300 !important;\n\
  color: var(--accent) !important;\n\
}\n\
\n\
/* --- Tags --- */\n\
[data-theme="dark"] .tag {\n\
  background: var(--accent-bg) !important;\n\
  border: 1px solid var(--accent-border) !important;\n\
  color: var(--text-primary) !important;\n\
  border-radius: 4px !important;\n\
  font-family: var(--font-body) !important;\n\
  font-size: 0.85rem !important;\n\
}\n\
[data-theme="dark"] .tag:hover {\n\
  background: rgba(124,140,240,0.12) !important;\n\
  transform: none !important;\n\
}\n\
\n\
[data-theme="dark"] .skill-label {\n\
  color: var(--accent) !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 500 !important;\n\
  letter-spacing: 2px !important;\n\
  font-size: 0.75rem !important;\n\
}\n\
\n\
[data-theme="dark"] .intro-text {\n\
  color: var(--text-secondary) !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 300 !important;\n\
  line-height: 1.8 !important;\n\
}\n\
\n\
/* --- Footer --- */\n\
[data-theme="dark"] .footer {\n\
  color: var(--text-secondary) !important;\n\
  border-top-color: var(--card-border) !important;\n\
  font-family: var(--font-body) !important;\n\
}\n\
\n\
/* --- SCHEDULE: Dark --- */\n\
[data-theme="dark"] .title-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
}\n\
[data-theme="dark"] .status-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: none !important;\n\
}\n\
[data-theme="dark"] .clock-display { color: var(--text-primary) !important; font-family: "JetBrains Mono", var(--font-body), monospace !important; }\n\
[data-theme="dark"] .date-display  { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }\n\
[data-theme="dark"] .table-container {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: none !important;\n\
}\n\
[data-theme="dark"] .schedule-table th { color: var(--text-secondary) !important; border-bottom-color: var(--card-border) !important; font-family: var(--font-body) !important; }\n\
[data-theme="dark"] .time-col { color: var(--text-secondary) !important; font-family: "JetBrains Mono", var(--font-body), monospace !important; }\n\
[data-theme="dark"] .activity-cell {\n\
  background: #111111 !important;\n\
  border: 1px solid #1a1a1a !important;\n\
  border-radius: 6px !important;\n\
}\n\
[data-theme="dark"] .activity-cell:hover { background: var(--card-hover) !important; }\n\
[data-theme="dark"] .status-active  { background: rgba(74,222,128,0.08) !important; color: #4ade80 !important; border: 1px solid rgba(74,222,128,0.15) !important; }\n\
[data-theme="dark"] .status-busy    { background: rgba(124,140,240,0.08) !important; color: var(--accent) !important; border: 1px solid var(--accent-border) !important; }\n\
[data-theme="dark"] .status-off     { background: rgba(102,102,102,0.08) !important; color: var(--text-secondary) !important; border: 1px solid rgba(102,102,102,0.15) !important; }\n\
[data-theme="dark"] .status-leave   { background: rgba(251,146,60,0.08) !important; color: #fb923c !important; border: 1px solid rgba(251,146,60,0.15) !important; }\n\
[data-theme="dark"] .status-badge   { border-radius: var(--radius) !important; box-shadow: none !important; font-family: var(--font-body) !important; }\n\
[data-theme="dark"] .current-slot   { border: 1px solid var(--accent) !important; box-shadow: 0 0 8px rgba(124,140,240,0.15) !important; }\n\
[data-theme="dark"] #error-banner   { background: rgba(248,113,113,0.08) !important; color: #f87171 !important; border-color: rgba(248,113,113,0.15) !important; border-radius: var(--radius) !important; }\n\
\n\
/* --- PDF GATE: Dark --- */\n\
[data-theme="dark"] .gate-card {\n\
  background: var(--card-bg) !important;\n\
  border: 1px solid var(--card-border) !important;\n\
  border-radius: var(--radius) !important;\n\
  box-shadow: none !important;\n\
}\n\
[data-theme="dark"] .gate-title { font-family: var(--font-heading) !important; font-weight: 300 !important; color: var(--text-primary) !important; }\n\
[data-theme="dark"] .gate-desc  { color: var(--text-secondary) !important; font-family: var(--font-body) !important; font-weight: 300 !important; }\n\
[data-theme="dark"] .gate-icon  { color: var(--accent) !important; }\n\
[data-theme="dark"] .pin-input  {\n\
  background: #0c0c0c !important; color: var(--text-primary) !important;\n\
  border: 1px solid var(--card-border) !important; border-radius: var(--radius) !important;\n\
  font-family: "JetBrains Mono", monospace !important;\n\
}\n\
[data-theme="dark"] .pin-input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(124,140,240,0.1) !important; }\n\
[data-theme="dark"] .submit-btn {\n\
  background: var(--accent) !important;\n\
  border-radius: var(--radius) !important;\n\
  font-family: var(--font-body) !important;\n\
  font-weight: 500 !important;\n\
}\n\
[data-theme="dark"] .submit-btn:hover { background: var(--accent-light) !important; transform: none !important; color: #0c0c0c !important; }\n\
[data-theme="dark"] .error-msg { background: rgba(248,113,113,0.08) !important; border-color: rgba(248,113,113,0.15) !important; color: #f87171 !important; border-radius: var(--radius) !important; }\n\
[data-theme="dark"] .back-link { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }\n\
[data-theme="dark"] .back-link:hover { color: var(--text-primary) !important; }\n\
\n\
\n\
/* ================================================================ */\n\
/*  GLASSMORPHISM — uses page defaults, no overrides                */\n\
/* ================================================================ */\n\
\n\
\n\
/* ================================================================ */\n\
/*  THEME PICKER UI                                                 */\n\
/* ================================================================ */\n\
#theme-picker-btn {\n\
  position: fixed; bottom: 24px; right: 24px; z-index: 9999;\n\
  width: 42px; height: 42px; border-radius: 50%;\n\
  border: 1px solid rgba(128,128,128,0.3);\n\
  cursor: pointer;\n\
  display: flex; align-items: center; justify-content: center;\n\
  padding: 0;\n\
  transition: all 0.15s ease;\n\
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);\n\
}\n\
\n\
[data-theme="academic"] #theme-picker-btn {\n\
  background: #fff; border-color: #d8d5cd; color: #52526e;\n\
}\n\
[data-theme="academic"] #theme-picker-btn:hover { border-color: #1e3a5f; color: #1e3a5f; }\n\
\n\
[data-theme="glass"] #theme-picker-btn {\n\
  background: rgba(255,255,255,0.05); color: #a0aec0;\n\
}\n\
[data-theme="glass"] #theme-picker-btn:hover { color: #fff; border-color: #63b3ed; }\n\
\n\
[data-theme="dark"] #theme-picker-btn {\n\
  background: #161616; border-color: #232323; color: #666;\n\
}\n\
[data-theme="dark"] #theme-picker-btn:hover { border-color: #7c8cf0; color: #7c8cf0; }\n\
\n\
#theme-picker-panel {\n\
  position: fixed; bottom: 76px; right: 24px; z-index: 9999;\n\
  border-radius: 8px; padding: 6px; min-width: 175px;\n\
  display: none; flex-direction: column; gap: 2px;\n\
}\n\
#theme-picker-panel.open { display: flex; }\n\
\n\
[data-theme="academic"] #theme-picker-panel {\n\
  background: #fff; border: 1px solid #d8d5cd; box-shadow: 0 4px 16px rgba(0,0,0,0.08);\n\
}\n\
[data-theme="glass"] #theme-picker-panel {\n\
  background: rgba(30,40,55,0.95); border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 30px rgba(0,0,0,0.3);\n\
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);\n\
}\n\
[data-theme="dark"] #theme-picker-panel {\n\
  background: #161616; border: 1px solid #232323; box-shadow: 0 4px 20px rgba(0,0,0,0.5);\n\
}\n\
\n\
.theme-option {\n\
  display: flex; align-items: center; gap: 10px;\n\
  padding: 10px 14px; border-radius: 6px; cursor: pointer;\n\
  border: 1px solid transparent; background: none;\n\
  font-size: 0.85rem; font-weight: 500;\n\
  transition: all 0.12s ease; text-align: left; width: 100%;\n\
}\n\
\n\
[data-theme="academic"] .theme-option { font-family: "Source Sans 3", "Plus Jakarta Sans", sans-serif; color: #52526e; }\n\
[data-theme="academic"] .theme-option:hover { background: #f0efeb; color: #1b1b30; }\n\
[data-theme="academic"] .theme-option.active { color: #1e3a5f; border-color: #1e3a5f; }\n\
\n\
[data-theme="glass"] .theme-option { font-family: "Plus Jakarta Sans", sans-serif; color: #a0aec0; }\n\
[data-theme="glass"] .theme-option:hover { background: rgba(255,255,255,0.05); color: #fff; }\n\
[data-theme="glass"] .theme-option.active { color: #63b3ed; border-color: #63b3ed; }\n\
\n\
[data-theme="dark"] .theme-option { font-family: "Inter", sans-serif; color: #666; }\n\
[data-theme="dark"] .theme-option:hover { background: #1c1c1c; color: #c8c8c8; }\n\
[data-theme="dark"] .theme-option.active { color: #7c8cf0; border-color: #7c8cf0; }\n\
\n\
.theme-dot {\n\
  width: 14px; height: 14px; border-radius: 50%;\n\
  border: 2px solid currentColor; flex-shrink: 0;\n\
}\n\
.theme-option.active .theme-dot { background: currentColor; }\n\
';

    var styleEl = document.createElement('style');
    styleEl.id = 'theme-system-css';
    styleEl.textContent = css;
    document.head.appendChild(styleEl);

    /* ----------------------------------------------------------------
       3. Picker UI
       ---------------------------------------------------------------- */
    function setTheme(name) {
        document.documentElement.setAttribute('data-theme', name);
        localStorage.setItem(STORAGE_KEY, name);
        loadFont(name);
        var opts = document.querySelectorAll('.theme-option');
        for (var i = 0; i < opts.length; i++) {
            if (opts[i].getAttribute('data-tid') === name) {
                opts[i].classList.add('active');
            } else {
                opts[i].classList.remove('active');
            }
        }
    }

    function createPicker() {
        var btn = document.createElement('button');
        btn.id = 'theme-picker-btn';
        btn.setAttribute('aria-label', 'Change theme');
        btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22C6.49 22 2 17.51 2 12S6.49 2 12 2s10 4.04 10 9c0 3.31-2.69 6-6 6h-1.77c-.28 0-.5.22-.5.5 0 .12.05.23.13.33.41.47.64 1.06.64 1.67A2.5 2.5 0 0 1 12 22zm0-18c-4.41 0-8 3.59-8 8s3.59 8 8 8c.28 0 .5-.22.5-.5a.54.54 0 0 0-.14-.35c-.41-.46-.63-1.05-.63-1.65a2.5 2.5 0 0 1 2.5-2.5H16c2.21 0 4-1.79 4-4 0-3.86-3.59-7-8-7z"/><circle cx="6.5" cy="11.5" r="1.5"/><circle cx="9.5" cy="7.5" r="1.5"/><circle cx="14.5" cy="7.5" r="1.5"/><circle cx="17.5" cy="11.5" r="1.5"/></svg>';

        var panel = document.createElement('div');
        panel.id = 'theme-picker-panel';

        var themes = [
            { id: 'academic', label: 'Classic' },
            { id: 'glass', label: 'Glassmorphism' },
            { id: 'dark', label: 'Minimal Dark' }
        ];

        themes.forEach(function (t) {
            var opt = document.createElement('button');
            opt.className = 'theme-option' + (currentTheme === t.id ? ' active' : '');
            opt.setAttribute('data-tid', t.id);
            opt.innerHTML = '<span class="theme-dot"></span>' + t.label;
            opt.addEventListener('click', function () {
                setTheme(t.id);
                panel.classList.remove('open');
            });
            panel.appendChild(opt);
        });

        btn.addEventListener('click', function (e) {
            e.stopPropagation();
            panel.classList.toggle('open');
        });

        document.addEventListener('click', function () {
            panel.classList.remove('open');
        });
        panel.addEventListener('click', function (e) {
            e.stopPropagation();
        });

        document.body.appendChild(btn);
        document.body.appendChild(panel);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createPicker);
    } else {
        createPicker();
    }
})();
