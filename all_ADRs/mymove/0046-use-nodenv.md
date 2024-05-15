---
title: 0046 Use nodenv to manage Node versions in development
---
# Use [nodenv](https://github.com/nodenv/nodenv) to manage Node versions in development

:::warning

**Replaced by [ADR 0081](0081-use-asdf-to-manage-node-and-golang-versions-in-development.md)**

:::

`nodenv` makes it easy for developers to upgrade Node in their development
environments.

## Considered Alternatives

* Docker-based development environment setup
* Homebrew-based solution where we'd need to pin the Homebrew version of Node.


## Decision Outcome

Use `nodenv` to manage local Node versions. It's widely used, regularly updated,
and allows folks to have multiple Node versions on their system. The
Docker-based development environment would provide more a consistent local
dependency story, but would add too much overhead.
