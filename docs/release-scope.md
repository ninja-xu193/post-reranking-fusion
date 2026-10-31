# Release scope

Included: independently organized method code, synthetic fixtures, unit tests, installation metadata and method documentation.

Excluded: research scores and result tables, training outputs, model weights, saved fitted coefficients, real benchmark records, raw passages, paper drafts, response letters, machine-specific paths, credentials and the development repository's history.

The MIT license applies to this code. Check each dataset and scorer's terms independently before distributing any input or derived cache. A private repository is not permission to upload excluded material.

The release checker admits a small set of source, documentation and configuration locations. It rejects symlinks, unknown payloads, oversized files and common sensitive strings. Generated Python bytecode and installation metadata are ignored by the working-tree scan and never belong in a release. Before committing, inspect `git diff --cached --name-only` and run the checker again. Never initialize a release by pushing the original research directory.
