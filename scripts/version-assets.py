#!/usr/bin/env python3
"""Stamp assets/*.css and assets/*.js links in the root HTML pages with a
content hash (?v=xxxxxxxx), so browsers fetch a new copy whenever a file
changes instead of reusing a stale cached one. Run after editing assets/."""
import hashlib, pathlib, re

root = pathlib.Path(__file__).resolve().parent.parent
hashes = {p.relative_to(root).as_posix(): hashlib.sha1(p.read_bytes()).hexdigest()[:8]
          for p in (root / 'assets').glob('*.*') if p.suffix in ('.css', '.js')}
pattern = re.compile(r'((?:href|src)=")(assets/[\w.-]+\.(?:css|js))(?:\?v=[0-9a-f]+)?(")')
for page in sorted(root.glob('*.html')):
    text = page.read_text(encoding='utf-8')
    new = pattern.sub(lambda m: m.group(1) + m.group(2) + ('?v=' + hashes[m.group(2)] if m.group(2) in hashes else '') + m.group(3), text)
    if new != text:
        page.write_text(new, encoding='utf-8')
        print('stamped', page.name)
