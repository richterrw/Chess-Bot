# Saving friends' games to a Google Sheet

1. Open https://sheets.new and name the sheet "RobbieBot games".
2. Click **Extensions → Apps Script**. Delete what's there, paste in `Code.gs`
   from this folder, and click **Save**.
3. Click **Deploy → New deployment**. Click the gear next to "Select type" and
   choose **Web app**. Set **Execute as: Me** and **Who has access: Anyone**,
   then click **Deploy** and allow access when Google asks.
4. Copy the **Web app URL** (it ends in `/exec`) and put it in `web/config.js`.

Every finished game then appears as a row in the sheet's "Games" tab, with the
player's name, which bot, the result, how it ended and the full PGN.
