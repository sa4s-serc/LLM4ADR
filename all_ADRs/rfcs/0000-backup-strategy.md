# 0000 - Backup strategy

## Status
[status]: #status

Proposed

## Summary
[summary]: #summary

The OpenFairDB database should be backed up automatically to prevent data loss
and to be able to restore a recent version of the database after spam attacks.

## Context
[context]: #context

All contents of *kartevonmorgen.org* are stored in an SQLite database, i.e. a single file
on the server. Currently this file is backed up manually, every few days or sometimes
weeks. The backup is stored both on the server and on private, external storage media.

## Decision
[decision]: #decision

- A daily backup is generated automatically, compressed and stored on the server in a dedicated folder
- The backup folder is synchronized periodically (daily/weekly?) via rsync or syncthing with external storage
- NTH: Old backups are selectively deleted, e.g. only te most recent 30 daily backups
are kept, for previous month only the last daily backup ist kept.

## Consequences
[consequences]: #consequences

Which system initiates the synchronization of the backup folder, i.e. where to store the credentials?

## References
[references]: #references
