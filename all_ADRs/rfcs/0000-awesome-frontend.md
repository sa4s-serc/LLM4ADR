# 0000 - Awesome frontend

## Status
[status]: #status

Proposed

## Summary
[summary]: #summary

The Karte von morgen [frontend](https://github.com/kartevonmorgen/kartevonmorgen)
shall be accessible from multiple clients (browser, mobile) and the code shall be maintainable.

## Context
[context]: #context

The current frontend is in a bad state.
The following issues need to be solved to change the situation:

- **Maintainability**
  Due to the untyped programming language JavaScript most types of code refactoring are risky to realize.
  This gets emphasized by external dependencies that easily can break the web application.
  Often a simple dependency update leads to multiple hours of fighting the dependency hell.
  Unfortunately a handful of bugs don't occur during the refactoring process but during the
  runtime (because JS can't check correctness at compile time).

- **Complexity**
  The current complexity of the codebase is grown due to a lot of dirty quick fixes.
  This not only makes it hard for the community to contribute to the project
  but it also makes it costly to introduce new features.

- **Tests**
  There are few tests but most of the code is untested.
  This makes it even harder to refactor and maintain the project.

- **File size**
  The packed JavaScript bundle size was 5.3 MB some time ago.
  In the meanwhile its shrunken to 1.5 MB but that's still a bunch of bytes that
  needs to be downloaded at once.

## Decision
[decision]: #decision

We will port the whole frontend to [Rust](https://rust-lang.org). The migration
can be divided into the following stages:
- Port the existing admin views to WASM as the first use case. All new admin views
will be implemented with the new technology. This first stage serves as test balloon
to verify the feasibility of the plan.
- Create a read-only view of the regular Kvm front-end that allows to browse
the map as well as searching and selecting entries.
- Complete the front-end by addint the editing functionality, i.e. to create and
modify entries. This may include includes extension like additional actions for
logged in users depending on their role.

The result will be a [WASM](https://en.wikipedia.org/wiki/WebAssembly) module.

We made this decision based on following reasons.
We expect to achieve

- a much better **maintainability** because:
  - Rust is statically typed
  - refactoring is easy (the compiler does a very good job)
  - the package manager [Cargo](https://doc.rust-lang.org/stable/cargo/) provides
    a reliable dependency management
  - most mistakes will popup at compile-time
- a reduced **complexity** because we can get rid of most legacy code
- a more efficient **testing** because we can focus on business logic
  (instead of testing trivial things like it is required in JavaScript)
- a mobile friendly **file size** because the app is packed in a binary format
  (instead of text like it is with JavaScript)
- a better API integration because we can **share** modules with the backend (OpenFairDB)
  that is also written in Rust.

## Consequences
[consequences]: #consequences

The decision has the following consequences:

- developers need to know or learn Rust
- [Leaflet](https://leafletjs.com/) can't be integrated directly
- we can't use component libraries that are based on React
- the app won't work with old browsers
- the admin frontend of the OpenFairDB can be merged into the frontend
- the existing code base is migrated in stages and both front-ends need
to be online side-by-side during the migration. New functionality is
only added to the migrated version.

## References
[references]: #references

- http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions
