/* Reads schedule-config.js and renders the timeline, the agenda,
   the playhead and the one-line status. Everything is Ankara time. */
window.Schedule = (function () {
  'use strict';

  var DAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  var WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  var LABEL = { monday: 'Monday', tuesday: 'Tuesday', wednesday: 'Wednesday', thursday: 'Thursday', friday: 'Friday', saturday: 'Saturday' };
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var SLOT = 30, DEF_START = 8 * 60 + 30, DEF_END = 17 * 60 + 30;

  var fmt = null;
  try { fmt = new Intl.DateTimeFormat('en-GB', { timeZone: 'Europe/Istanbul', hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }); } catch (e) {}
  function now() {
    if (!fmt) return new Date();
    var p = {}; fmt.formatToParts(new Date()).forEach(function (x) { p[x.type] = x.value; });
    return new Date(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second);
  }
  function toMin(s) { var p = String(s).split(':'); return (+p[0]) * 60 + (+p[1] || 0); }
  function pad(n) { return String(n).padStart(2, '0'); }
  function hm(m) { return pad(Math.floor(m / 60)) + ':' + pad(m % 60); }
  function ymd(d) { return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function events(day) { return ((window.scheduleData || {})[day] || []).slice().sort(function (a, b) { return toMin(a.start) - toMin(b.start); }); }
  function isLeave(d) { return Array.isArray(window.leaveDays) && window.leaveDays.indexOf(ymd(d)) !== -1; }
  function isBreak(ev) { return ev.category === 'break' || /lunch|break/i.test(ev.activity); }

  // Monochrome category styles: breaks are hatched; the others alternate solid / outline
  // in the order they are declared in scheduleCategories.
  function styleOf(cat) {
    var cats = window.scheduleCategories || {};
    if (cat === 'break' || (cats[cat] && /break/i.test(cats[cat].label || ''))) return 'hatch';
    var i = 0;
    for (var k in cats) { if (k === 'break' || /break/i.test(cats[k].label || '')) continue; if (k === cat) return i % 2 ? 'outline' : 'solid'; i++; }
    return 'solid';
  }

  function weekDates(n) {
    var out = {}, off = n.getDay() === 0 ? -6 : 1 - n.getDay();
    WEEK.forEach(function (d, i) { var x = new Date(n); x.setDate(n.getDate() + off + i); out[d] = x; });
    return out;
  }
  function visibleDays(dates) {
    return WEEK.filter(function (d) { return d !== 'saturday' || events(d).length || isLeave(dates[d]); });
  }
  function bounds(days) {
    var s = DEF_START, e = DEF_END;
    days.forEach(function (d) { events(d).forEach(function (ev) {
      s = Math.min(s, Math.floor(toMin(ev.start) / SLOT) * SLOT); e = Math.max(e, Math.ceil(toMin(ev.end) / SLOT) * SLOT); }); });
    return { start: s, end: e, slots: (e - s) / SLOT };
  }

  /* ---- timeline ---- */
  function renderTimeline(el, opts) {
    var n = now(), dates = weekDates(n), days = visibleDays(dates), b = bounds(days), today = ymd(n);
    var compact = !!(opts && opts.compact);
    el.className = 'tl' + (compact ? ' tl--compact' : '');
    el.style.setProperty('--days', days.length);
    el.style.setProperty('--slots', b.slots);
    var h = '<div class="tl-head"><div class="axis"></div>';
    days.forEach(function (d) {
      var dt = dates[d], t = ymd(dt) === today;
      h += '<div class="day' + (t ? ' is-today' : '') + '">' + (compact ? LABEL[d].slice(0, 3) : LABEL[d]) + '<small>' + MON[dt.getMonth()] + ' ' + dt.getDate() + (t ? ' · today' : '') + '</small></div>';
    });
    h += '</div>';
    // axis labels at every hour (compact: every 3 hours)
    h += '<div class="tl-axis" id="tl-axis">';
    for (var m = Math.ceil(b.start / 60) * 60; m <= b.end; m += 60) {
      if (compact && ((m / 60) % 3)) continue;
      h += '<span style="top:' + ((m - b.start) / (b.end - b.start) * 100) + '%">' + hm(m) + '</span>';
    }
    h += '</div>';
    days.forEach(function (d) {
      var dt = dates[d], t = ymd(dt) === today;
      h += '<div class="tl-day' + (t ? ' is-today' : '') + '" data-day="' + d + '">';
      if (isLeave(dt)) {
        h += '<div class="tl-leave"><span>On leave</span></div>';
      } else {
        events(d).forEach(function (ev) {
          var s = toMin(ev.start), e = toMin(ev.end);
          var top = (s - b.start) / (b.end - b.start) * 100, hgt = (e - s) / (b.end - b.start) * 100;
          h += '<div class="tl-ev tl-ev--' + styleOf(ev.category) + '" style="top:' + top + '%;height:calc(' + hgt + '% - 2px)" title="' + esc(ev.activity) + ' ' + hm(s) + '–' + hm(e) + (ev.location ? ', ' + esc(ev.location) : '') + '">' +
               '<strong>' + esc(ev.activity) + '</strong><span class="t">' + hm(s) + '–' + hm(e) + '</span>' + (ev.location ? '<span class="l">' + esc(ev.location) + '</span>' : '') + '</div>';
        });
      }
      h += '</div>';
    });
    el.innerHTML = h;
    el._bounds = b;
    el._today = today;
    updatePlayhead(el);
  }

  function updatePlayhead(el) {
    var b = el._bounds; if (!b) return;
    var n = now(), min = n.getHours() * 60 + n.getMinutes(), today = ymd(n);
    if (today !== el._today) { renderTimeline(el, { compact: el.classList.contains('tl--compact') }); return; }
    Array.prototype.forEach.call(el.querySelectorAll('.tl-now, .tl-now-label'), function (x) { x.remove(); });
    var col = el.querySelector('.tl-day.is-today');
    if (!col || min < b.start || min > b.end) return;
    var pct = (min - b.start) / (b.end - b.start) * 100;
    var line = document.createElement('div'); line.className = 'tl-now'; line.style.top = pct + '%';
    col.appendChild(line);
    var lab = document.createElement('span'); lab.className = 'tl-now-label'; lab.style.top = pct + '%'; lab.textContent = hm(min);
    el.querySelector('.tl-axis').appendChild(lab);
  }

  /* ---- agenda (phones) ---- */
  function renderAgenda(el) {
    var n = now(), dates = weekDates(n), days = visibleDays(dates), today = ymd(n), min = n.getHours() * 60 + n.getMinutes();
    var h = '';
    days.forEach(function (d) {
      var dt = dates[d], t = ymd(dt) === today;
      h += '<section class="day' + (t ? ' is-today' : '') + '"><div class="day-head">' + LABEL[d] + '<small>' + MON[dt.getMonth()] + ' ' + dt.getDate() + (t ? ' · TODAY' : '') + '</small></div>';
      if (isLeave(dt)) { h += '<div class="empty">On leave.</div>'; }
      else {
        var list = events(d);
        if (!list.length) h += '<div class="empty">Nothing scheduled.</div>';
        else {
          h += '<ul>'; var marked = false;
          list.forEach(function (ev) {
            var s = toMin(ev.start), e = toMin(ev.end);
            if (t && !marked && min < s) { h += '<li class="now-mark">' + hm(min) + '</li>'; marked = true; }
            h += '<li class="ev tl-ev--' + styleOf(ev.category) + '"><span class="t">' + hm(s) + '–' + hm(e) + '</span><span><strong>' + esc(ev.activity) + '</strong>' + (ev.location ? '<span class="l"> · ' + esc(ev.location) + '</span>' : '') + '</span></li>';
            if (t && !marked && min >= s && min < e) { h += '<li class="now-mark">' + hm(min) + '</li>'; marked = true; }
          });
          if (t && !marked && min >= DEF_START && min < DEF_END) h += '<li class="now-mark">' + hm(min) + '</li>';
          h += '</ul>';
        }
      }
      h += '</section>';
    });
    el.innerHTML = h;
  }

  /* ---- one-line status ---- */
  function nextEvent(n) {
    for (var off = 0; off < 8; off++) {
      var d = new Date(n); d.setDate(n.getDate() + off);
      if (isLeave(d)) continue;
      var list = events(DAYS[d.getDay()]);
      for (var i = 0; i < list.length; i++) {
        if (isBreak(list[i])) continue;
        if (off === 0 && toMin(list[i].start) <= n.getHours() * 60 + n.getMinutes()) continue;
        return { ev: list[i], day: DAYS[d.getDay()], off: off };
      }
    }
    return null;
  }
  function status() {
    if (window.onLeave === true) return { cls: 'leave', html: 'Currently on leave.' };
    var n = now(), day = DAYS[n.getDay()], min = n.getHours() * 60 + n.getMinutes();
    var when = function (x) { return (x.off === 0 ? 'today' : x.off === 1 ? 'tomorrow' : LABEL[x.day]) + ' at ' + hm(toMin(x.ev.start)); };
    var desc = function (ev) { return '<strong>' + esc(ev.activity) + '</strong>' + (ev.location ? ', ' + esc(ev.location) : ''); };
    if (isLeave(n)) { var n0 = nextEvent(n); return { cls: 'leave', html: 'On leave today.' + (n0 ? ' Next: ' + desc(n0.ev) + ', ' + when(n0) + '.' : '') }; }
    var list = events(day), cur = null, up = null;
    list.forEach(function (ev) { var s = toMin(ev.start), e = toMin(ev.end); if (min >= s && min < e) cur = ev; else if (s > min && !up) up = ev; });
    var working = min >= DEF_START && min < DEF_END && day !== 'sunday' && (day !== 'saturday' || list.length);
    if (cur) return { cls: isBreak(cur) ? 'break' : 'busy', html: 'Now: ' + desc(cur) + ' until ' + hm(toMin(cur.end)) + '.' };
    if (working) return { cls: 'free', html: 'Available now' + (up ? ' until ' + hm(toMin(up.start)) + ' (' + esc(up.activity) + ')' : ' for the rest of the day') + '.' };
    var nx = nextEvent(n);
    return { cls: 'off', html: 'Off duty.' + (nx ? ' Next: ' + desc(nx.ev) + ', ' + when(nx) + '.' : '') };
  }
  function renderStatus(el) {
    var s = status();
    var cls = 'rail-now' + (s.cls === 'busy' || s.cls === 'free' || s.cls === 'break' ? ' is-now' : '');
    var txt = el.querySelector('.txt');
    if (el.className !== cls) el.className = cls;
    if (txt.innerHTML !== s.html) txt.innerHTML = s.html;
    var link = el.querySelector('a');
    if (link) link.hidden = window.onLeave === true;
  }
  function legend(el) {
    var cats = window.scheduleCategories || {}, h = '';
    for (var k in cats) h += '<span><i class="' + styleOf(k) + '"></i>' + esc(cats[k].label || k) + '</span>';
    h += '<span><i class="hatch"></i>On leave</span><span><i class="now"></i>Now</span>';
    el.innerHTML = h;
  }

  return { now: now, renderTimeline: renderTimeline, updatePlayhead: updatePlayhead, renderAgenda: renderAgenda, renderStatus: renderStatus, legend: legend, status: status };
})();

// Every page carries the rail; keep its one-line status current.
(function () {
  function run() {
    var el = document.getElementById('rail-now');
    if (!el || typeof window.scheduleData === 'undefined') return;
    Schedule.renderStatus(el);
    setInterval(function () { Schedule.renderStatus(el); }, 30000);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run); else run();
})();
