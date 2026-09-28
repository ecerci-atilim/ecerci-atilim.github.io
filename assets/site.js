/* Shared behaviour: theme toggle + footer year. Loaded with `defer`. */
(function () {
  'use strict';

  var root = document.documentElement;
  var KEY = 'theme';

  function systemPrefersDark() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function current() {
    return root.getAttribute('data-theme') || (systemPrefersDark() ? 'dark' : 'light');
  }
  function apply(theme) {
    root.setAttribute('data-theme', theme);
    try { localStorage.setItem(KEY, theme); } catch (e) { /* private mode */ }
    var toggles = document.querySelectorAll('.theme-toggle');
    for (var i = 0; i < toggles.length; i++) {
      toggles[i].setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      toggles[i].setAttribute('title', theme === 'dark' ? 'Light mode' : 'Dark mode');
    }
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('.theme-toggle');
    if (!btn) return;
    apply(current() === 'dark' ? 'light' : 'dark');
  });

  var toggles = document.querySelectorAll('.theme-toggle');
  for (var i = 0; i < toggles.length; i++) {
    toggles[i].setAttribute('title', current() === 'dark' ? 'Light mode' : 'Dark mode');
  }

  var y = document.querySelector('[data-year]');
  if (y) y.textContent = String(new Date().getFullYear());
})();
