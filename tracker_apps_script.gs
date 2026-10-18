/**
 * Tonight, Sorted - live tracker (Google Apps Script)
 *
 * Paste this into the tracker Google Sheet: Extensions -> Apps Script.
 * 1. Change TOKEN below to the same value as the app's TRACKER_TOKEN secret.
 * 2. Run `setup` once (Run button) and allow access when Google asks.
 * 3. Deploy -> New deployment -> type "Web app"
 *      Execute as: Me      Who has access: Anyone
 *    Copy the Web app URL (ends in /exec) into the app's TRACKER_URL secret.
 *
 * The app sends rows as JSON; this script appends them to the matching tab,
 * creating tabs and columns as needed. Requests without the token are ignored.
 */

const TOKEN = 'CHANGE-ME';

const TABS = {
  Run_Log: ['Run_ID', 'Timestamp', 'Session_ID', 'App_Version', 'Location', 'Searched_Elsewhere', 'Prompt',
            'Searches', 'Time_Checked', 'Group_Size', 'Budget', 'Places_Found', 'Picks_Returned',
            'Pick_1', 'Pick_2', 'Pick_3', 'Weak_Match', 'Planner_Question', 'Runtime_Sec',
            'Input_Tokens', 'Output_Tokens', 'Total_Tokens', 'LLM_Cost_USD', 'Google_Places_Calls',
            'Google_Routes_Calls', 'Ticketmaster_Calls', 'Agents_Called', 'Agents_Skipped',
            'Adaptations', 'Errors', 'Finish_Reason'],
  Agent_Usage: ['Run_ID', 'Timestamp', 'Agent', 'Status', 'Seconds', 'Input_Tokens', 'Output_Tokens',
                'LLM_Cost_USD', 'Cached', 'Note'],
  Recommendations: ['Run_ID', 'Timestamp', 'Rank', 'Place', 'Type', 'Found_By', 'Final_Score', 'Request_Fit',
                    'Place_Quality', 'Travel_Score', 'Review_Evidence', 'Relevance_Check', 'Rating',
                    'Reviews', 'Drive_Min', 'Open_At_Time', 'Maps_Link'],
  User_Feedback: ['Run_ID', 'Timestamp', 'Session_ID', 'Helpful', 'Feedback_Text', 'Feedback_Summary'],
};

function setup() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  Object.keys(TABS).forEach(function (name) { ensureTab_(ss, name, TABS[name]); });
  buildDashboard_(ss);
  const blank = ss.getSheetByName('Sheet1');
  if (blank && blank.getLastRow() === 0) ss.deleteSheet(blank);
}

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return reply_({ ok: false, error: 'bad json' });
  }
  if (body.token !== TOKEN) return reply_({ ok: false, error: 'bad token' });

  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const tabs = body.tabs || {};
    let written = 0;
    Object.keys(tabs).forEach(function (name) {
      const rows = tabs[name] || [];
      if (!rows.length) return;
      const sheet = ensureTab_(ss, name, TABS[name] || Object.keys(rows[0]));
      let header = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
      // Add any new columns the app started sending.
      const missing = [];
      rows.forEach(function (r) { Object.keys(r).forEach(function (k) {
        if (header.indexOf(k) < 0 && missing.indexOf(k) < 0) missing.push(k);
      }); });
      if (missing.length) {
        sheet.getRange(1, header.length + 1, 1, missing.length).setValues([missing]).setFontWeight('bold');
        header = header.concat(missing);
      }
      const values = rows.map(function (r) {
        return header.map(function (k) {
          const v = r[k];
          if (v === null || v === undefined) return '';
          // Leading quote keeps prompts like "=..." or "+1" as plain text.
          if (typeof v === 'string' && /^[=+\-@]/.test(v)) return "'" + v;
          return v;
        });
      });
      sheet.getRange(sheet.getLastRow() + 1, 1, values.length, header.length).setValues(values);
      written += values.length;
    });
    return reply_({ ok: true, rows: written });
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return reply_({ ok: true, message: 'Tonight, Sorted tracker is running.' });
}

function ensureTab_(ss, name, header) {
  let sheet = ss.getSheetByName(name);
  if (!sheet) sheet = ss.insertSheet(name);
  if (sheet.getLastRow() === 0) {
    sheet.getRange(1, 1, 1, header.length).setValues([header]).setFontWeight('bold');
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function col_(tab, name) {
  // A whole column found by its header name, so formulas survive new columns.
  return 'INDEX(' + tab + '!A:AZ,0,MATCH("' + name + '",' + tab + '!1:1,0))';
}

function buildDashboard_(ss) {
  let d = ss.getSheetByName('Dashboard');
  if (!d) d = ss.insertSheet('Dashboard', 0);
  d.clear();
  const rows = [
    ['Tonight, Sorted - live tracker', ''],
    ['', ''],
    ['Searches run', '=COUNTA(' + col_('Run_Log', 'Run_ID') + ')-1'],
    ['Visitors (sessions)', '=IFERROR(COUNTUNIQUE(FILTER(' + col_('Run_Log', 'Session_ID') + ',ROW(' + col_('Run_Log', 'Session_ID') + ')>1)),0)'],
    ['Avg places found per search', '=IFERROR(AVERAGE(' + col_('Run_Log', 'Places_Found') + '),"")'],
    ['Avg run time (sec)', '=IFERROR(AVERAGE(' + col_('Run_Log', 'Runtime_Sec') + '),"")'],
    ['Avg tokens per search', '=IFERROR(AVERAGE(' + col_('Run_Log', 'Total_Tokens') + '),"")'],
    ['Avg LLM cost per search (USD)', '=IFERROR(AVERAGE(' + col_('Run_Log', 'LLM_Cost_USD') + '),"")'],
    ['Avg Google Places calls per search', '=IFERROR(AVERAGE(' + col_('Run_Log', 'Google_Places_Calls') + '),"")'],
    ['Searches with a weak match', '=IFERROR(COUNTIF(' + col_('Run_Log', 'Weak_Match') + ',"Yes")/(COUNTA(' + col_('Run_Log', 'Run_ID') + ')-1),"")'],
    ['Searches with an error', '=IFERROR((COUNTIF(' + col_('Run_Log', 'Errors') + ',"?*")-1)/(COUNTA(' + col_('Run_Log', 'Run_ID') + ')-1),"")'],
    ['Feedback: helpful (Yes)', '=COUNTIF(' + col_('User_Feedback', 'Helpful') + ',"Yes")'],
    ['Feedback: not helpful (No)', '=COUNTIF(' + col_('User_Feedback', 'Helpful') + ',"No")'],
    ['Helpful rate', '=IFERROR(B12/(B12+B13),"")'],
    ['Slowest agent on average', '=IFERROR(INDEX(QUERY(Agent_Usage!A:J,"select C, avg(E) where C is not null group by C order by avg(E) desc limit 1",1),2,1),"")'],
  ];
  d.getRange(1, 1, rows.length, 2).setValues(rows);
  d.getRange('A1').setFontWeight('bold').setFontSize(14);
  d.getRange('A3:A15').setFontWeight('bold');
  d.getRange('B10:B11').setNumberFormat('0.0%');
  d.getRange('B14').setNumberFormat('0.0%');
  d.getRange('B8').setNumberFormat('$0.0000');
  d.getRange('B5:B7').setNumberFormat('0.0');
  d.getRange('B9').setNumberFormat('0.0');
  d.setColumnWidth(1, 280);
  d.setColumnWidth(2, 160);
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
