#!/usr/bin/env bash

# watch-and-run.sh
# Usage:
#   ./watch-and-run.sh [directory_to_watch] [command_to_run]
#
# Defaults:
#   directory_to_watch = current directory (.)
#   command_to_run     = echo "Change detected"

# Directory to watch (defaults to current directory)
WATCH_DIR="${1:-.}"

# Command to execute on change (everything after the first argument)
shift
if [ $# -gt 0 ]; then
  CMD="$*"
else
  CMD='echo "Change detected in '$WATCH_DIR'"'
fi

# Ensure inotifywait is installed
if ! command -v inotifywait &>/dev/null; then
  echo "Error: inotifywait not found. Install inotify-tools (e.g. 'sudo apt install inotify-tools')."
  exit 1
fi

echo "Watching directory: $WATCH_DIR"
echo "Will run on change: $CMD"
echo "Press [CTRL+C] to stop."

# Listen for create, modify, delete, move events on .java files only
inotifywait -m -r \
  --event modify,create,delete,move \
  --format '%T %w%f %e' --timefmt '%F %T' \
  --include '.*\.java$' \
  "$WATCH_DIR" \
| while read -r timestamp file events; do
    echo "[$timestamp] $events on $file"
    # run the user-specified command
    echo $CMD
    eval "$CMD"
  done
