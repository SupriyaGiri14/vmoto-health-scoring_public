#!/usr/bin/env bash
#
# update_dashboard.sh
#
# Combines the steps you'd otherwise run by hand every time new ride
# files are added:
#
#   1. Re-export the small CSVs from everything currently in data/raw/
#   2. Commit the updated CSVs
#   3. Push to GitHub (Streamlit Cloud auto-redeploys after this)
#
# What this does NOT do: download new files from Firebase. You still
# need to put new .txt files into data/raw/ yourself before running
# this script -- see the note at the bottom for why that part isn't
# automated (yet).
#
# Usage:
#   ./update_dashboard.sh
#
# (On Windows Git Bash, run it the same way: ./update_dashboard.sh)

set -e  # stop immediately if any step fails, rather than continuing

echo "Step 1/3: Exporting dashboard data from data/raw/ ..."
python3 src/export_dashboard_data.py data/raw/ app/data/

echo ""
echo "Step 2/3: Running tests to make sure nothing is broken ..."
python3 -m unittest discover tests

echo ""
echo "Step 3/3: Committing and pushing ..."
git add app/data/
if git diff --cached --quiet; then
    echo "No changes to the dashboard data -- nothing new to push."
else
    git commit -m "Update dashboard data ($(date +%Y-%m-%d))"
    git push
    echo ""
    echo "Done. Streamlit Cloud will automatically redeploy in a minute or two."
fi

# --------------------------------------------------------------
# Why the Firebase download step isn't automated here:
#
# Automating that too is possible, but needs a Firebase "service
# account" key -- a credential file that lets a script log into
# Firebase Storage without a human clicking through the console.
# Setting that up requires admin-level access to the Firebase
# project (Project Settings -> Service Accounts), and the key must
# be stored securely (e.g. as an encrypted GitHub Actions secret),
# not just left in a plain file.
#
# If you have that level of access and want this fully automated
# (no manual download step at all, running on a schedule), that's a
# separate, bigger piece of work -- let me know and we can build it.
# --------------------------------------------------------------
