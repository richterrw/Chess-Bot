/**
 * RobbieBot results log. Paste into a Google Sheet's Extensions > Apps Script
 * and deploy as a web app (Execute as: Me, Who has access: Anyone).
 * Every finished game on the website becomes a row in the "Games" sheet.
 */
var SHEET_NAME = 'Games';
var HEADERS = ['Date', 'Player', 'Bot', 'Player color', 'Result', 'How', 'Moves',
               'Takebacks', 'Bot strength', 'Clock', 'PGN'];
var RESULTS = { win: 'Won', loss: 'Lost', draw: 'Draw' };

// Works both inside a sheet (Extensions > Apps Script) and as a standalone
// script from script.new, where it creates a "RobbieBot games" sheet in Drive.
function spreadsheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  if (ss) return ss;
  var props = PropertiesService.getScriptProperties();
  var id = props.getProperty('SHEET_ID');
  if (id) return SpreadsheetApp.openById(id);
  ss = SpreadsheetApp.create('RobbieBot games');
  props.setProperty('SHEET_ID', ss.getId());
  return ss;
}

// Run this once from the editor to create the sheet and print its link.
function setup() {
  sheet_();
  Logger.log('Your results sheet: ' + spreadsheet_().getUrl());
}

function sheet_() {
  var ss = spreadsheet_();
  var sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.insertSheet(SHEET_NAME);
    sh.appendRow(HEADERS);
    sh.setFrozenRows(1);
    sh.getRange(1, 1, 1, HEADERS.length).setFontWeight('bold');
  }
  return sh;
}

// Anyone can post here, so keep values short and stop them being read as formulas.
function clean_(v, max) {
  var s = String(v === undefined || v === null ? '' : v).slice(0, max);
  return /^[=+\-@]/.test(s) ? "'" + s : s;
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  var d;
  try { d = JSON.parse(e.postData.contents); } catch (err) { return json_({ ok: false }); }
  if (!RESULTS[d.result]) return json_({ ok: false });
  var lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    sheet_().appendRow([
      new Date(), clean_(d.player, 30), clean_(d.bot, 20), clean_(d.color, 5),
      RESULTS[d.result], clean_(d.how, 40), Number(d.moves) || 0, Number(d.takebacks) || 0,
      clean_(d.strength, 20), clean_(d.clock, 10), clean_(d.pgn, 20000)
    ]);
  } finally {
    lock.releaseLock();
  }
  return json_({ ok: true });
}

// The website reads recent results from here (names, bots and results only).
function doGet() {
  var rows = sheet_().getDataRange().getValues().slice(1);
  var back = { Won: 'win', Lost: 'loss', Draw: 'draw' };
  var totals = { win: 0, draw: 0, loss: 0 };
  rows.forEach(function (r) { var k = back[r[4]]; if (k) totals[k]++; });
  var games = rows.slice(-50).reverse().map(function (r) {
    return { date: r[0] instanceof Date ? r[0].toISOString() : String(r[0]), player: String(r[1]),
             bot: String(r[2]), result: back[r[4]] || '', how: String(r[5]) };
  });
  return json_({ total: rows.length, totals: totals, games: games });
}
