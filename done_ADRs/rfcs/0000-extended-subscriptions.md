# 0000 - Extended subscriptions

## Status
[status]: #status

Proposed

## Summary
[summary]: #summary

Users shall be able to subscribe for changes in selected regions and for selected tags.

> Notification für Änderungen in Regionen und Stichwörtern

> Moderator*innen werden bei Änderungen und neuen
Bewertungen in ihrer Stadt (Regionalpilot*innen) bzw. zu
ihrem Stichwort (Themenpilot*innen) benachrichtigt

## Context
[context]: #context

> This section describes the forces at play, including technological, political, social, and project local. These forces are probably in tension, and should be called out as such. The language in this section is value-neutral. It is simply describing facts.

> MVP:
> - Nicht nur bei Änderungen, sondern auch bei Bewertungen
werden Subscriber informiert
> - Am Ende jeder Notification Mail gibt es einen Link, worüber
das Abonnement beendet werden kann
> - Notification-Mails werden nur alle 30 min. verschickt, jeweils
mit der bis dahin letzten änderung (Das verhindert eine
Mailflut, wenn jemand 20 mal hintereinander den selber
Beitrag editiert)
> - Wird ein Eintrag gelöscht, wird den Piloten der Löschgrund
mitgeteilt.

> NTH:
> - Notifications von gelöschten Beiträgen erhalten nur Piloten
und leute, die den einzelnen Beitrag abonniert haben.
Normale "Flächenabonnementen" werden nicht über
löschungen informiert.

> MVP: Wird derzeit in Weißrussland entwickelt. Siehe
http://mapa.falanster.by/
> - Über Sharing-Button rechts mittig kann der Nutzer folgende
Optionen wählen
#Kartenansicht abonnieren (Pop-up zum mailadresse
eingeben.)
#URL-Teilen (Pop-up mit aktueller URL der Karte)
#Karte einbetten (Pop-up mit Iframe-Code)
#Download (Popup mit Feld für Mailadresse, an die der
Downloadlink geschickt werden soll. Eingeloggte Nutzer sehen
Download-Link direkt. Evtl. Downloadbutton nur für eingeloggte
Nutzer zeigen.
> - Beim Abonnieren wird nicht nur der Bereich berücksichtigt,
sondern auch die gewählten Hashtags. Hat der Nutzer dagegen
nur einen einzelnen Eintrag geöffnet, wird nur dieser abonniert.
(also ganz logisch wird immer das abonniert bzw. geteilt, was
gerade in der URL steht.)
> - Über einen Kurztext unter der Mail-Eingabe-Zeile wird dann

> NTH:
> - Beim Abonnieren werden nicht nur Hashtags berücksichtigt
sondern auch suchbegriffe
> - Umfassende Suchfunktion, die mehrere Suchbegriffe
gleichzeitig filtern kann als und-Suche bzw. mit Komma als
Oder-Suche

## Decision
[decision]: #decision

- The service will be implemented as an isolated component within OpenFairDB
- A modification time stamp needs to be added to entries in the search index
- The search index allows filtering by modification time stamp
- Subscriptions consist of
    - Name
    - Selector of either
        - either entry id
        - or search query
            - Bounding box (mandatory)
            - List of tags (might be empty)
            - Search text (optional)
    - Max. modification time stamp of the most recent entry that has been processed (initial: creation time stamp of the subscription)
- Subscriptions are owned by users, one user per subscription
- Users may create multiple subscriptions
- All subscriptions are processed independently
- Subscriptions will be processed periodically starting at a configurable time, i.e. daily at 7 a.m. (to avoid disctractions by nightly emails)
- For each subscription a search request will be submitted (if selector is not an entry id)
- The number of search results is limited to 500, applying the default scoring
- The search results are collected into an e-mail, one per subscription
- The email contains:
    - name of the subscription in the subject
    - for each modified entry:
        - modification time stamp
        - title of entry
        - link to entry
- The e-mail contents are submitted to external e-mail service provider that sends them.
Sending e-mails via *sendmail* locally is not recommended.

## Consequences
[consequences]: #consequences

The requirement to include tags and a search text to the subscription entails that the search
index must be used for determining the modified entries. Currently recently modified entries
are retrieved from the database without any filtering by bounding box, tags, or a search text.

The special case that handles subscribing for single entries requires a separate code path.
An alternative would be to subscribe for single entries by a small or empty bounding box.
But then the modification of moving an entry out of this bounding box would be missed.

Migrating existing subscriptions adds additional costs. We recommend to discard all existing
subscriptions and start with a fresh database.

## References
[references]: #references
