// Theme: starts from the system setting; the toggle sets an explicit light/dark theme remembered across pages.
(() => {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') root.setAttribute('data-theme', saved);
  } catch (e) { /* storage unavailable: fall back to the system setting */ }

  const darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const isDark = () => {
    const t = root.getAttribute('data-theme');
    return t ? t === 'dark' : darkQuery.matches;
  };

  function bind() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const moon = document.getElementById('icon-moon'), sun = document.getElementById('icon-sun');
    const sync = () => {
      const dark = isDark();
      moon.hidden = dark; sun.hidden = !dark;
      btn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    };
    btn.addEventListener('click', () => {
      const next = isDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* ignore */ }
      sync();
    });
    darkQuery.addEventListener('change', sync);
    sync();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
