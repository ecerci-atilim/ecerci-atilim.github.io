// ===================================================================
// HAFTALIK PROGRAM VERİLERİ
// Tüm program değişikliklerini bu dosyadan yapacaksınız.
// Aktivite birden fazla zaman dilimini kaplıyorsa, her dilim için kopyalayın.
// Zaman dilimleri "HH:MM" formatında olmalıdır.
// ===================================================================
const scheduleData = {
            monday: {
                "09:30": { activity: "EE103 Laboratory (S2)", location: "B2015", color: "#cfe2ff" },
                "10:00": { activity: "EE103 Laboratory (S2)", location: "B2015", color: "#cfe2ff" },
                "10:30": { activity: "EE103 Laboratory (S2)", location: "B2015", color: "#cfe2ff" },
                "11:00": { activity: "EE103 Laboratory (S2)", location: "B2015", color: "#cfe2ff" },
                "14:30": { activity: "EE103 Laboratory (S1)", location: "B2015", color: "#deacffff" },
                "15:00": { activity: "EE103 Laboratory (S1)", location: "B2015", color: "#deacffff" },
                "15:30": { activity: "EE103 Laboratory (S1)", location: "B2015", color: "#deacffff" },
                "16:00": { activity: "EE103 Laboratory (S1)", location: "B2015", color: "#deacffff" },
                "11:30": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "12:00": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
            },
            tuesday: {
                "11:30": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "12:00": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
            },
            wednesday: {
                "11:30": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "12:00": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "09:30": { activity: "EE103 Office Hours", location: "2042", color: "#5dbeffff"},
                "10:00": { activity: "EE103 Office Hours", location: "2042", color: "#5dbeffff"},
                "10:30": { activity: "EE103 Office Hours", location: "2042", color: "#5dbeffff"},
                "11:00": { activity: "EE103 Office Hours", location: "2042", color: "#5dbeffff"},
            },
            thursday: {
                "13:30": { activity: "EE209 Laboratory (S1)", location: "B2015", color: "#b2ffd2" },
                "14:00": { activity: "EE209 Laboratory (S1)", location: "B2015", color: "#b2ffd2" },
                "14:30": { activity: "EE209 Laboratory (S1)", location: "B2015", color: "#b2ffd2" },
                "15:00": { activity: "EE209 Laboratory (S1)", location: "B2015", color: "#b2ffd2" },
                "11:30": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "12:00": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "09:30": { activity: "EE209 Office Hours", location: "2042", color: "#5dffa6ff"},
                "10:00": { activity: "EE209 Office Hours", location: "2042", color: "#5dffa6ff"},
                "10:30": { activity: "EE209 Office Hours", location: "2042", color: "#5dffa6ff"},
                "11:00": { activity: "EE209 Office Hours", location: "2042", color: "#5dffa6ff"},
            },
            friday: {
                "09:30": { activity: "EE209 Laboratory (S2)", location: "B2015", color: "#f8ff9bff" },
                "10:00": { activity: "EE209 Laboratory (S2)", location: "B2015", color: "#f8ff9bff" },
                "10:30": { activity: "EE209 Laboratory (S2)", location: "B2015", color: "#f8ff9bff" },
                "11:00": { activity: "EE209 Laboratory (S2)", location: "B2015", color: "#f8ff9bff" },
                "11:30": { activity: "EE209 Laboratory (S3)", location: "B2015", color: "#ffb2ccff" },
                "12:00": { activity: "EE209 Laboratory (S3)", location: "B2015", color: "#ffb2ccff" },
                "12:30": { activity: "EE209 Laboratory (S3)", location: "B2015", color: "#ffb2ccff" },
                "13:00": { activity: "EE209 Laboratory (S3)", location: "B2015", color: "#ffb2ccff" },
                "13:30": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
                "14:00": { activity: "Lunch Break", location: "Out of office", color: "#d6d6d6ff"},
            },
            saturday: {
            }
        };

// ===================================================================
// İZİN GÜNLERİ LİSTESİ
// İzinli olduğun günleri 'YYYY-MM-DD' formatında buraya ekle.
// Bu tarihlerde program tamamen devre dışı kalır ve "On Leave" gösterilir.
// ===================================================================
const leaveDays = [
    "2025-11-14",
    "2024-11-15",
    "2025-10-25"
];