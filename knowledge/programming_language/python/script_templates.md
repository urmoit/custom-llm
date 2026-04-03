# Python Script Templates

## File Organizer Script
Purpose: Sort files by extension into subfolders.

Requirements:
- Input: target directory path
- Create folders like `images`, `docs`, `archives`, `other`
- Move files safely and skip directories
- Print summary counts

## Log Analyzer Script
Purpose: Parse log files and count levels (`INFO`, `WARNING`, `ERROR`).

Requirements:
- Input: log file path
- Regex parse per line
- Output count table and top error messages
- Handle missing file and encoding issues

## CSV Report Script
Purpose: Read CSV data and compute totals/averages by group.

Requirements:
- Input: CSV path and target column
- Group by category column
- Output sorted summary table
- Optional JSON export

## REST API Client Script
Purpose: Call an HTTP API and save normalized JSON.

Requirements:
- Use `urllib.request` (or `requests` if available)
- Timeout and retry behavior
- Handle non-200 responses
- Save result to output file

## Folder Backup Script
Purpose: Incremental backup from source to destination.

Requirements:
- Compare by modified time and size
- Copy only changed files
- Preserve folder structure
- Write a backup report log

## SQLite Task Tracker Script
Purpose: Manage tasks with persistent storage.

Requirements:
- Commands: add, list, done, delete
- Use `sqlite3`
- Validate IDs and show clear messages
- Keep schema migration-safe
