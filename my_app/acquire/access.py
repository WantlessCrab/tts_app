# my_app/acquire/access.py
"""
Acquire Service — CAPTCHA unlock ladder, camoufox, session manager.
Phase 3 implementation.

Will contain:
  _captcha_unlock_sequence()  — stealth retry → camoufox → session → archive → Googlebot
  _browser_with_session()     — Playwright persistent context with stored domain session
  _try_unpaywall()            — Unpaywall API open-access PDF lookup
  _try_internet_archive()     — Wayback Machine snapshot lookup
  _try_mirror_bypass()        — 12ft.io URL transform
"""