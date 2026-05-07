// schedule-config.js
// ─────────────────────────────────────────────────────────
// Simple schedule configuration with start/end times.
//
// USAGE:
//   Each day has an array of events. Each event needs:
//     - activity : Display name
//     - location : Room / place
//     - start    : Start time  "HH:MM" (24h)
//     - end      : End time    "HH:MM" (24h)
//     - category : One of the keys defined in scheduleCategories
//
//   To add a new event, just add one line — no need to fill
//   every 30-minute slot manually.
//
// CATEGORIES:
//   Define your own categories below. Each category has a
//   label (shown in the legend) and a color.
// ─────────────────────────────────────────────────────────

window.onLeave = false;

window.scheduleCategories = {
    lab:     { label: "Laboratory",   color: "#a78bfa" },  // Purple
    office:  { label: "Office Hours", color: "#00be79" },  // Green
    break:   { label: "Break",        color: "#9ca3af" },  // Gray
};

window.scheduleData = {
    monday: [
        { activity: "Lunch Break",          location: "Out of office", start: "11:30", end: "12:30", category: "break" },
    ],
    tuesday: [
        { activity: "Lunch Break",          location: "Out of office", start: "11:30", end: "12:30", category: "break" },
        { activity: "EE352 Laboratory (S1)", location: "B4015",        start: "12:30", end: "14:20", category: "lab" },
    ],
    wednesday: [
        { activity: "Lunch Break",          location: "Out of office", start: "11:30", end: "12:30", category: "break" },
        { activity: "EE214 Laboratory (S1)", location: "B2015",        start: "12:30", end: "14:20", category: "lab" },
        { activity: "EE214 Laboratory (S2)", location: "B2015",        start: "14:30", end: "16:20", category: "lab" },
    ],
    thursday: [
        { activity: "EE316 Laboratory (S1)", location: "B2013",        start: "09:30", end: "11:20", category: "lab" },
        { activity: "Lunch Break",          location: "Out of office", start: "11:30", end: "12:30", category: "break" },
        { activity: "Office Hours",         location: "2042",          start: "13:00", end: "15:00", category: "office"}
    ],
    friday: [
        { activity: "Lunch Break",          location: "Out of office", start: "11:30", end: "12:30", category: "break" },
        { activity: "EE352 Laboratory (S2)", location: "B4015",        start: "12:30", end: "14:20", category: "lab" },
        { activity: "EE316 Laboratory (S2)", location: "B2013",        start: "14:30", end: "16:20", category: "lab" },
    ],
    saturday: []
};

// Dates when you are on leave (YYYY-MM-DD format)
window.leaveDays = [
    "2026-03-13",
    "2026-04-16"
];
