# 0000 - Metatags

## Status
[status]: #status

Proposed

## Summary
[summary]: #summary

The high diversity of possible hashtags make it quite difficult to find all entries for one topic.
We can introduce Meta-tags: They are a container for different tags and are as ease to use as one
tag but bring many more results.

## Context
[context]: #context

> This section describes the forces at play, including technological, political, social, and project local. These forces are probably in tension, and should be called out as such. The language in this section is value-neutral. It is simply describing facts.

## Motivation
[motivation]: #motivation

Many movements want to use the map for one topic,
but do not want to add one unig tag to all entries of that topic, to make this topic visible on one map.

- The "Social Economy" is a gerenal topic of our map. It can be #socent, #gwö, #b-corp, #sustainabel-finance...
  But there is no hashtag everyone is using and it is impossible to teach all active mappers to use one common hashtag.
- #GWÖ is a commonly used hashtag, but some users just use special world like #energiefeld for a #gwö-regionalgroup.
  If #gwö becomes a Meta Tag, that does not matter
- We can solve **spelling errors**. If #urbang-ardening contains the hashtags #urbangardening,  #urban-guardening and #stadtgarten,
  we automatically did a translation and connected synonyms and wrong written tags.
- #Futurefashion wants to create a map of all fair clothing-shops.
  Until know a user would have to add all tags like #futurefashion #fairfashion #kleidung

# Future possibilities
[future-possibilities]: #future-possibilities

- When we do legends and categories on any iframes, it can become really usefull to subgroup most of our content under 10 to 20 keywords.
  https://github.com/kartevonmorgen/kartevonmorgen/issues/372
- The Glossary of change can improve the search results

## Decision
[decision]: #decision

> This section describes our response to these forces. It is stated in full sentences, with active voice. "We will ..."

Tags will still remain plain tags without any metadata. Additionally we will introduce directed relationships from
a *source* to a *target* tag with the following roles:

- *generalization*: The source tag is a specialization of the target tag. Examples: target = `#social-economy`, sources = `#socent`/`#gwö`/`#b-corp`/`#b-corp`/`sustainabel-finance`
- *synonym*: The terms of both tags name the *same* concept. The *source* tag is the *alias* term, the *target* tag
is the preferred *canonical* term. Example: target = `#urban-gardening`, source = `#urbangardening` (alternative name), source = `#urban-guardening` (misspelling), source = `#stadtgarten` (german synonym).
- *i18n*: A special kind of synonym relationship between canonical terms of different languages? See [Consequences](consequences).

## Consequences
[consequences]: #consequences

> This section describes the resulting context, after applying the decision. All consequences should be listed here, not just the "positive" ones. A particular decision may have positive, negative, and neutral consequences, but all of them affect the team and project in the future.

Only tags that are not the *source* of a *synonym* relationship should be proposed to avoid fragmentation of the tagging space. In the long term synonymous tags could be replaced by the corresponding preferred canonical tag.

Tags could be split into (overlapping) groups according to *generalization* relationships. The target tags of generalization relationships could be used as a starting point for assigning tags or for selecting searchable tags.

### Open issues

#### How to consider tag relationships for search requests?

- Search for a generalized tag and include all results for specialized tags?
- Search for a synonym and include all results for both the canonical tag as well as all results for other synonyms?

#### Allow transitive relationships?

Relationships: Tag A -> Tag B -> Tag C

- Generalizations over multiple levels might be useful
- Allow only a single level for synonyms?

#### How to handle synonyms in different languages?

If synonyms are used for associating tags from different languages we have
to agree on a canonical language, i.e. *English*.  The actual language of
the synonyms is unknown, i.e. filtering tags by language is not possible.
If only canonical tags are proposed then users will always see the tags
of the canonical language instead of their local language.

One option would be to use the *synonym* relationship only within a single language.
The canonical terms in different languages could be linked by a separate relationship,
i.e. *i18n*. The language of a tag becomes a separate property or relation of each tag
with the cardinality 0..*, i.e. any number of languages per tag is possible.

## References
[references]: #references

- KVM Issue: https://github.com/kartevonmorgen/kartevonmorgen/issues/598
