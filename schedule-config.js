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
// Example:
// { activity:  "Lunch Break",
//   location:  "Out of office",
//   start:     "11:30",
//   end:       "12:30",
//   category:  "break" },

window.onLeave = false;

window.scheduleCategories = {
    lab:     { label: "Laboratory",   color: "#a78bfa" },  // Purple
    office:  { label: "Office Hours", color: "#00be79" },  // Green
    break:   { label: "Break",        color: "#9ca3af" },  // Gray
};

window.scheduleData = {
    monday: [
    ],
    tuesday: [],
    wednesday: [],
    thursday: [],
    friday: [],
    saturday: []
};

// Dates when you are on leave (YYYY-MM-DD format)
window.leaveDays = [];
