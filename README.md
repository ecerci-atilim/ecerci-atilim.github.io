# ecerci-atilim.github.io

Personal site of Emre Cerci (Atılım University) and the tools shared with
students and colleagues. Plain static HTML — no build step. GitHub Pages
deploys `main` through `.github/workflows/static.yml`.

## Layout

| Path | What it is |
| --- | --- |
| `index.html` | Home: this week's timetable, then the list of tools |
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
| `assets/site.css`, `assets/site.js` | Shared stylesheet and theme toggle |
| `assets/schedule.js` | Reads `schedule-config.js`: timetable, playhead, rail status |
| `legacy/` | Frozen copy of the previous design (glassmorphism + `theme.js`), untouched |
| `matlab/` | BER simulation scripts |

Every page except `ftn.html` and `legacy/*` links `assets/site.css` and
carries the same left rail (`<aside class="rail">`): name, navigation, live
status, contact links, theme toggle. There is no templating, so a change to
the rail means editing that block on each page.

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

The organising idea is the time axis. A left rail is the spine of every
page; the home page opens with the week itself; the schedule is a 30-minute
ruler. One accent colour, "now" red, is reserved for the present moment (the
playhead line, today's date, the status dot) and for errors. Categories are
monochrome — solid, outline, hatched — whatever colours the config lists.
No icons, gradients, shadows or chips.

`assets/site.css` holds the tokens and all components (rail, timeline,
buttons, forms, panels, modal). Light is the default; dark follows the
system and can be forced with the toggle in the rail (stored in
`localStorage` as `theme`). Fonts are self-hosted in `assets/fonts/`
(SIL OFL 1.1): Newsreader for headings, IBM Plex Sans and Mono for the rest.
The site makes no third-party requests except the libraries the QR and
M-BCJR tools load from jsDelivr / Plotly.

`legacy/` holds the previous glassmorphism design, self-contained and
untouched; `legacy/index.html` is its home page.
