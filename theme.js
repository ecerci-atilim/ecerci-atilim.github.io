/**
 * Theme System
 * Themes: academic (Classic - default), glass (Glassmorphism), dark (Minimal Dark)
 */
(function () {
    'use strict';

    var STORAGE_KEY = 'site-theme';
    var DEFAULT_THEME = 'academic';

    // Read saved theme immediately (before paint)
    var currentTheme = localStorage.getItem(STORAGE_KEY) || DEFAULT_THEME;
    document.documentElement.setAttribute('data-theme', currentTheme);

    // -------------------------------------------------------
    // Inject theme CSS immediately to prevent FOUC
    // -------------------------------------------------------
    var css = [
        /* ===== CLASSIC (academic) ===== */
        '[data-theme="academic"] {',
        '  --bg-gradient: #f5f5f0;',
        '  --glass-bg: #ffffff;',
        '  --glass-border: #ddd8d0;',
        '  --glass-hover: #f0efec;',
        '  --text-primary: #1a1a2e;',
        '  --text-secondary: #555566;',
        '  --accent-color: #2c5282;',
        '  --accent-gradient: linear-gradient(135deg, #2c5282 0%, #2b6cb0 100%);',
        '  --shadow-soft: 0 1px 4px rgba(0,0,0,0.06);',
        '  --transition: all 0.2s ease;',
        '  --tag-bg: rgba(44, 82, 130, 0.06);',
        '  --tag-border: rgba(44, 82, 130, 0.12);',
        '  --text-accent: #2c5282;',
        '  --success-color: #2f855a;',
        '  --warning-color: #c05621;',
        '  --danger-color: #c53030;',
        '  --card-bg: #ffffff;',
        '  --card-border: #ddd8d0;',
        '  --input-bg: #f0efec;',
        '  --warning-bg: rgba(192, 86, 33, 0.08);',
        '  --warning-text: #c05621;',
        '}',

        '[data-theme="academic"] body { background: #f5f5f0 !important; }',

        '[data-theme="academic"] .floating-shapes,',
        '[data-theme="academic"] .ambient-light { display: none !important; }',

        '[data-theme="academic"] .glass-card,',
        '[data-theme="academic"] .title-card,',
        '[data-theme="academic"] .table-container,',
        '[data-theme="academic"] .status-card,',
        '[data-theme="academic"] .back-btn {',
        '  backdrop-filter: none !important;',
        '  -webkit-backdrop-filter: none !important;',
        '}',

        '[data-theme="academic"] .header-title,',
        '[data-theme="academic"] .profile-name,',
        '[data-theme="academic"] .page-title {',
        '  font-family: Georgia, "Times New Roman", serif !important;',
        '  background: none !important;',
        '  -webkit-background-clip: unset !important;',
        '  -webkit-text-fill-color: var(--text-primary) !important;',
        '}',

        '[data-theme="academic"] .section-title,',
        '[data-theme="academic"] .card-title {',
        '  font-family: Georgia, "Times New Roman", serif !important;',
        '}',

        '[data-theme="academic"] .header-institution {',
        '  background: rgba(44, 82, 130, 0.06);',
        '  border-color: rgba(44, 82, 130, 0.15);',
        '  color: var(--accent-color);',
        '}',

        '[data-theme="academic"] .tool-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }',
        '[data-theme="academic"] .tool-card:hover .tool-icon { transform: none; color: var(--accent-color); }',

        '[data-theme="academic"] .link-btn:hover,',
        '[data-theme="academic"] .contact-btn:hover {',
        '  background: var(--accent-gradient); color: #fff;',
        '  transform: translateX(3px); box-shadow: 0 3px 10px rgba(44,82,130,0.2);',
        '}',

        '[data-theme="academic"] .avatar-container {',
        '  background: var(--accent-gradient);',
        '  box-shadow: 0 4px 15px rgba(44,82,130,0.2);',
        '}',

        '[data-theme="academic"] .back-btn { background: rgba(44,82,130,0.06); border-color: rgba(44,82,130,0.15); }',
        '[data-theme="academic"] .back-btn:hover { background: rgba(44,82,130,0.1); }',

        '[data-theme="academic"] .status-card { backdrop-filter: none !important; background: #ffffff; border-color: var(--glass-border); }',
        '[data-theme="academic"] .clock-display { color: var(--text-primary); }',

        '[data-theme="academic"] .status-active { background: rgba(47,133,90,0.1); color: #2f855a; border-color: rgba(47,133,90,0.2); }',
        '[data-theme="academic"] .status-busy   { background: rgba(44,82,130,0.1); color: #2c5282; border-color: rgba(44,82,130,0.2); }',
        '[data-theme="academic"] .status-off    { background: rgba(85,85,102,0.1); color: #555566; border-color: rgba(85,85,102,0.2); }',
        '[data-theme="academic"] .status-leave  { background: rgba(192,86,33,0.1); color: #c05621; border-color: rgba(192,86,33,0.2); }',

        '[data-theme="academic"] .tag:hover { transform: translateY(-1px); }',

        /* Hide emoji spans in academic mode */
        '[data-theme="academic"] .section-header > span { display: none; }',
        '[data-theme="academic"] .card-title > span { display: none; }',

        /* ===== MINIMAL DARK ===== */
        '[data-theme="dark"] {',
        '  --bg-gradient: #111111;',
        '  --glass-bg: #1a1a1a;',
        '  --glass-border: #2a2a2a;',
        '  --glass-hover: #222222;',
        '  --text-primary: #e0e0e0;',
        '  --text-secondary: #888888;',
        '  --accent-color: #8b9cf7;',
        '  --accent-gradient: linear-gradient(135deg, #8b9cf7 0%, #6366f1 100%);',
        '  --shadow-soft: none;',
        '  --transition: all 0.2s ease;',
        '  --tag-bg: rgba(139, 156, 247, 0.08);',
        '  --tag-border: rgba(139, 156, 247, 0.15);',
        '  --text-accent: #8b9cf7;',
        '  --success-color: #68d391;',
        '  --warning-color: #f6ad55;',
        '  --danger-color: #fc8181;',
        '  --card-bg: #1a1a1a;',
        '  --card-border: #2a2a2a;',
        '  --input-bg: #111111;',
        '  --warning-bg: rgba(246, 173, 85, 0.15);',
        '  --warning-text: #f6ad55;',
        '}',

        '[data-theme="dark"] body { background: #111111 !important; }',

        '[data-theme="dark"] .floating-shapes,',
        '[data-theme="dark"] .ambient-light { display: none !important; }',

        '[data-theme="dark"] .glass-card,',
        '[data-theme="dark"] .title-card,',
        '[data-theme="dark"] .table-container,',
        '[data-theme="dark"] .status-card,',
        '[data-theme="dark"] .back-btn {',
        '  backdrop-filter: none !important;',
        '  -webkit-backdrop-filter: none !important;',
        '}',

        '[data-theme="dark"] .header-title,',
        '[data-theme="dark"] .profile-name,',
        '[data-theme="dark"] .page-title {',
        '  background: none !important;',
        '  -webkit-background-clip: unset !important;',
        '  -webkit-text-fill-color: var(--text-primary) !important;',
        '}',

        '[data-theme="dark"] .tool-card:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }',
        '[data-theme="dark"] .tool-card:hover .tool-icon { transform: none; }',

        '[data-theme="dark"] .header-institution {',
        '  background: rgba(139,156,247,0.08);',
        '  border-color: rgba(139,156,247,0.15);',
        '  color: var(--accent-color);',
        '}',

        '[data-theme="dark"] .avatar-container {',
        '  background: var(--accent-gradient);',
        '  box-shadow: 0 4px 15px rgba(99,102,241,0.3);',
        '}',

        '[data-theme="dark"] .status-card { backdrop-filter: none !important; background: #1a1a1a; }',

        /* ===== GLASSMORPHISM (base CSS – no overrides needed) ===== */

        /* ===== Theme Picker UI ===== */
        '#theme-picker-btn {',
        '  position: fixed; bottom: 24px; right: 24px; z-index: 9999;',
        '  width: 42px; height: 42px; border-radius: 50%;',
        '  border: 1px solid var(--glass-border);',
        '  background: var(--glass-bg);',
        '  color: var(--text-secondary);',
        '  cursor: pointer;',
        '  display: flex; align-items: center; justify-content: center;',
        '  transition: all 0.2s ease;',
        '  box-shadow: 0 2px 10px rgba(0,0,0,0.12);',
        '  padding: 0;',
        '}',
        '#theme-picker-btn:hover { color: var(--text-primary); border-color: var(--accent-color); }',

        '[data-theme="academic"] #theme-picker-btn { background: #fff; border-color: #ddd8d0; }',
        '[data-theme="dark"] #theme-picker-btn { background: #1a1a1a; border-color: #2a2a2a; }',

        '#theme-picker-panel {',
        '  position: fixed; bottom: 76px; right: 24px; z-index: 9999;',
        '  background: var(--glass-bg); border: 1px solid var(--glass-border);',
        '  border-radius: 12px; padding: 8px; min-width: 170px;',
        '  display: none; flex-direction: column; gap: 4px;',
        '  box-shadow: 0 8px 30px rgba(0,0,0,0.18);',
        '}',
        '#theme-picker-panel.open { display: flex; }',

        '[data-theme="academic"] #theme-picker-panel { background: #fff; border-color: #ddd8d0; }',
        '[data-theme="dark"] #theme-picker-panel { background: #1a1a1a; border-color: #2a2a2a; }',

        '.theme-option {',
        '  display: flex; align-items: center; gap: 10px;',
        '  padding: 10px 14px; border-radius: 8px; cursor: pointer;',
        '  border: 1px solid transparent; background: none;',
        '  color: var(--text-secondary); font-family: inherit;',
        '  font-size: 0.88rem; font-weight: 500;',
        '  transition: all 0.15s ease; text-align: left; width: 100%;',
        '}',
        '.theme-option:hover { background: var(--glass-hover); color: var(--text-primary); }',
        '.theme-option.active { color: var(--accent-color); border-color: var(--accent-color); }',

        '.theme-dot {',
        '  width: 14px; height: 14px; border-radius: 50%;',
        '  border: 2px solid currentColor; flex-shrink: 0;',
        '}',
        '.theme-option.active .theme-dot { background: currentColor; }'
    ].join('\n');

    var styleEl = document.createElement('style');
    styleEl.id = 'theme-system-css';
    styleEl.textContent = css;
    document.head.appendChild(styleEl);

    // -------------------------------------------------------
    // Picker UI (created on DOMContentLoaded)
    // -------------------------------------------------------
    function setTheme(name) {
        document.documentElement.setAttribute('data-theme', name);
        localStorage.setItem(STORAGE_KEY, name);
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
        // Button
        var btn = document.createElement('button');
        btn.id = 'theme-picker-btn';
        btn.setAttribute('aria-label', 'Change theme');
        btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22C6.49 22 2 17.51 2 12S6.49 2 12 2s10 4.04 10 9c0 3.31-2.69 6-6 6h-1.77c-.28 0-.5.22-.5.5 0 .12.05.23.13.33.41.47.64 1.06.64 1.67A2.5 2.5 0 0 1 12 22zm0-18c-4.41 0-8 3.59-8 8s3.59 8 8 8c.28 0 .5-.22.5-.5a.54.54 0 0 0-.14-.35c-.41-.46-.63-1.05-.63-1.65a2.5 2.5 0 0 1 2.5-2.5H16c2.21 0 4-1.79 4-4 0-3.86-3.59-7-8-7z"/><circle cx="6.5" cy="11.5" r="1.5"/><circle cx="9.5" cy="7.5" r="1.5"/><circle cx="14.5" cy="7.5" r="1.5"/><circle cx="17.5" cy="11.5" r="1.5"/></svg>';

        // Panel
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

    // Init picker when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createPicker);
    } else {
        createPicker();
    }
})();
