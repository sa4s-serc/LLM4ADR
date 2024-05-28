# 0000 - Make Events searchable

## Status
[status]: #status

Proposed

## Summary
[summary]: #summary

Events are not included in search and filter so far. If Events are activated, all of them are always visible, which is not usable for our clients. We need to make them searchable.

## Context
[context]: #context

Since the the beginning of the year, https://pioneersofchange.org/landkarte/ is using the event fuction at Kvm. In Minsk Hackers developed the entry form for new events. As soon as this is online and in use by many, the map will be useless if you can not filter these events.

Events do not only need to be searchable, but the results must also be filtered by bounding box and sorted by start/end dates.

## Decision
[decision]: #decision

We will improve the backend-API so that events are searchable till the end of the year.
At least for hashtags it has to work!

## Consequences
[consequences]: #consequences

Events are searchable. Is the frontend more complex? Does it need to send two search request for every search?

A new search index for events has to be created and populated upon service startup. The code for entries must
be partitioned. All common code that is needed for both entries and events must be shared to avoid code
duplication. This requires a substantial, internal refactoring.

The search API for entries and events should be harmonized.

## References
[references]: #references

- Can we avoid having an own API for events but somhow to use the same so that the complexity in the frontend does not increase? Should the frontend "feel" as if there where only one databasis? https://github.com/kartevonmorgen/kartevonmorgen/issues/575
