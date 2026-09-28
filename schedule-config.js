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
    monday:     [
                {
                    activity:   "Lunch Break",
                    location:   "Out of office",
                    start:      "11:00",
                    end:        "12:00",
                    category:   "break",
                }
    ],
    tuesday:    [
                {   
                    activity:   "EE209 Laboratory",
                    location:   "B2015",
                    start:      "9:30",
                    end:        "11:30",
                    category:   "lab",
                },
                {
                    activity:   "EE103 Office Hours",
                    location:   "2042",
                    start:      "12:30",
                    end:        "14:30",
                    category:   "office",
                },
                {
                    activity:   "Lunch Break",
                    location:   "Out of office",
                    start:      "11:30",
                    end:        "12:30",
                    category:   "break",
                }
    ],
    wednesday:  [
                {   
                    activity:   "EE209 Laboratory",
                    location:   "B2015",
                    start:      "9:30",
                    end:        "11:30",
                    category:   "lab",
                },
                {
                    activity:   "EE209 Office Hours",
                    location:   "2042",
                    start:      "12:30",
                    end:        "14:30",
                    category:   "office",
                },
                {
                    activity:   "Lunch Break",
                    location:   "Out of office",
                    start:      "11:30",
                    end:        "12:30",
                    category:   "break",
                }
            ],
    thursday:   [
                {   
                    activity:   "EE103 Laboratory",
                    location:   "B2015",
                    start:      "12:30",
                    end:        "14:30",
                    category:   "lab",
                },
                {
                    activity:   "Lunch Break",
                    location:   "Out of office",
                    start:      "11:30",
                    end:        "12:30",
                    category:   "break",
                }
    ],
    friday:     [ 
                {   
                    activity:   "EE103 Laboratory",
                    location:   "B2015",
                    start:      "11:30",
                    end:        "13:30",
                    category:   "lab",
                },
                {
                    activity:   "Lunch Break",
                    location:   "Out of office",
                    start:      "13:30",
                    end:        "14:30",
                    category:   "break",
                }
                ],
    saturday:   []
};

// Dates when you are on leave (YYYY-MM-DD format)
window.leaveDays = [];
