# ecerci-atilim.github.io

Personal site of Emre Cerci (Atılım University) and the tools shared with
students and colleagues. Plain static HTML — no build step. GitHub Pages
deploys `main` through `.github/workflows/static.yml`.

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | Home: short intro, list of tools, contact links |
| `about.html` | Bio, skills, contact |
| `schedule.html` | Weekly schedule (reads `schedule-config.js`) |
| `schedule-config.js` | **The only file to edit each term** — see below |
| `seating-plan.html` | Exam seating generator with printable sheets |
| `timer2.html` | Exam countdown for the projector (`timer.html` redirects here) |
| `qr-generator.html` | QR code generator |
| `mbcjr.html` | M-BCJR trellis simulator |
| `mbcjr-new.html` | Trellis path enumerator with JSON export |
| `pdf-gate.html` | PIN gate in front of `eemudek0724.pdf` |
| `ftn.html` | Standalone FTN portfolio page (own design, multilingual) |
| `assets/site.css`, `assets/site.js` | Shared design system and theme toggle |
| `legacy/` | Frozen copy of the previous design (glassmorphism + `theme.js`), untouched |
| `matlab/` | BER simulation scripts |

Every page except `ftn.html` and `legacy/*` links `assets/site.css` and
carries the same masthead. There is no templating, so a nav change means
editing each page's `<header class="masthead">` block.

## Editing the schedule

Open `schedule-config.js`:

```js
window.onLeave = false;   // true → the page shows only "Currently on leave"

window.scheduleCategories = {
    lab:    { label: "Laboratory",   color: "#a78bfa" },
    office: { label: "Office Hours", color: "#00be79" },
    break:  { label: "Break",        color: "#9ca3af" },
};

window.scheduleData = {
    monday: [
        { activity: "EE209 Laboratory", location: "B2015", start: "9:30", end: "11:30", category: "lab" },
    ],
    // tuesday … saturday
};

window.leaveDays = ["2026-11-10"];   // single days off, YYYY-MM-DD
```

Notes:

- Times are `"H:MM"` or `"HH:MM"`, 24-hour. Events snap to 30-minute rows;
  the visible range grows automatically if something starts before 08:30
  or ends after 17:30.
- Saturday only appears when it has an event or a leave day.
- On screens narrower than 760px the table becomes a per-day list.
- Commit and push; the Pages workflow deploys `main`. If the site does not
  update within a few minutes, run the workflow manually from the Actions tab.

## Design

`assets/site.css` holds the design tokens (colours, type, spacing) and the
shared components (masthead, table-of-contents list, buttons, forms, tables,
modal). Light is the default; dark follows the system preference and can be
forced with the toggle in the masthead (stored in `localStorage` as `theme`).
Fonts are IBM Plex Sans / Serif / Mono from Google Fonts.

The `legacy/` folder is self-contained: `legacy/index.html` is the old home
and links only to the old pages. It reads the live `../schedule-config.js`.
