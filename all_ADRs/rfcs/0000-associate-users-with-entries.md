# 0000 - Associate users with entries

## Status
[status]: #status

Proposed

## Summary
[summary]: #summary

Upon creation entries shall be associated with their *creator* if the user is logged in. Creators may decide if the entry should be moderated, i.e. if scouts must confirm edits before they become visible. Creators shall automatically be subscribed to any changes of created entries, unless they decide to disable this feature when creating the entry. Accepting the copyright again for every created entry is not required while
logged in.

> Nutzerrollen und Admininterface

## Context
[context]: #context

> This section describes the forces at play, including technological, political, social, and project local. These forces are probably in tension, and should be called out as such. The language in this section is value-neutral. It is simply describing facts.

### MVP
> Wenn ein Nutzer eingeloggt ist, wird er mit dem Eintrag assoziiert:
> - Er wird standardmäßig über zukünftige Änderungen anderer Nutzer informiert
Siehe Ziel 3). Kann dies aber über eine Checkbox vor dem Speichern abwählen.
> - Er muss keinen Copyright lizenz akzeptieren, weil er das einmal bei der Registrierung gemacht hat.
> - In der Fußzeile für jeden Eintrag wird der Username oder die Nutzer-ID
angezeigt, von dem eingeloggten Nutzer, der es zuletzt editiert hat.
> - ??? Notmoderationscheckbox: User kann auswählen, dass
anonyme Änderungen erst sichtbar werden, wenn er oder ein
anderer RegPi/ThemPi die änderung bestätigt (über link in
Notification mail. Bestätigungsmail muss sich von anderen
'Eintrag bearbeitet'-Mail unterscheiden zB. im Betreff:
Aktualisierung wartet auf Freischaltung: Name des Eintrags)

### NTH
> - Eingeloggte Nutzer können einen Filter setzten "Zeige nur
veränderte Einträge" und sehen dann nur Änderungen
> - Alle geänderten Einträge die auf Freischaltung warten, sind als
Pinfarbe rot markiert

## Decision
[decision]: #decision

> This section describes our response to these forces. It is stated in full sentences, with active voice. "We will ..."

## Consequences
[consequences]: #consequences

> This section describes the resulting context, after applying the decision. All consequences should be listed here, not just the "positive" ones. A particular decision may have positive, negative, and neutral consequences, but all of them affect the team and project in the future.

## References
[references]: #references

- http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions
