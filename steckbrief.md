---
titel: "ChewML – Lebensmittelklassifikation aus AirPods-IMU-Daten"
semester: "SoSe 2026"
team: ["Jonah Karstens"]
frage: "Lassen sich Kaubewegungen, die über die Bewegungssensoren handelsüblicher AirPods Pro erfasst werden, nutzen, um Essen von Nicht-Essen zu unterscheiden und das gekaute Lebensmittel zu klassifizieren?"
domaene: "Ernährung / Activity Recognition (Earable Computing)"
sensorik: ["AirPods Pro (IMU: Beschleunigung, Gyroskop, Euler-Winkel)", "Sensor Logger App (iOS, HTTP-Stream)"]
datenart: "Eigene Aufnahmen einer Versuchsperson bei ~50 Hz, jede Aufnahme genau eine Aktivität, Label über Dateinamen. Klassen: Apfel, Kaugummi, Skyr, Still, Essen (generisch)"
datenmenge: "103 Aufnahmen, 1522 Fenster à 10 s, eine Person"
methode: ["10-s-Fenster + 52 Merkmale (Statistik, Kauband 0,5–4 Hz, Kaudynamik)", "Movement-Segment-Exclusion mit adaptivem Schwellwert", "Zwei-Stufen-Modell: Random Forest (Still vs. Essen) → SVM (Lebensmittel)", "Gruppenbewusste Permutation Importance, je Falte neu bestimmt", "Leave-One-Session-Out", "Mehrheitsvotum pro Mahlzeit", "Vergleich mit 1D-CNN auf Rohdaten"]
ergebnis: "LOSO: Stufe 1 94,7 % Accuracy, Stufe 2 88,7 % pro Fenster und 95,0 % pro Mahlzeit (76/80); End-to-End 87,5 % pro Fenster. Skyr ist die am besten trennbare Klasse, Apfel↔Kaugummi der Hauptfehler"
limitation: "Alle Aufnahmen stammen von einer Person – Generalisierung auf andere Personen ungeprüft. Merkmalsauswahl instabil (14–48 von 52 Merkmalen je Falte). Der Vergleich mit dem 1D-CNN endet unentschieden"
tags: ["Wearable", "Earables", "IMU", "Klassifikation", "Random Forest", "SVM", "Echtzeit", "Web-App"]
repo: "https://github.com/karstensjonah-design/ML4CS-group-ChewML"
zustimmung_veroeffentlichung: "ja"
---
[Live-Demo der Webanwendung: Klassifikation beim Kaugummikauen und im Ruhezustand](https://www.youtube.com/watch?v=APkBl8B37zE)

[Vorstellungsvideo von ChewML (2 min)](https://www.youtube.com/watch?v=n9Op2ja_2zE)

Das Projekt untersucht, ob sich Lebensmittel allein aus den Bewegungssensoren
unveränderter AirPods Pro erkennen lassen, ohne Mikrofon, Kamera oder
Forschungshardware. Kaubewegungen werden über die Sensor-Logger-App aufgezeichnet,
in nicht überlappende 10-Sekunden-Fenster zerlegt und in 52 Merkmale überführt,
darunter physikalisch motivierte Kaudynamik-Merkmale wie Jerk, Kaurate und
Autokorrelation. Ein hierarchisches Modell trennt zunächst Stillsitzen von Essen
(Random Forest) und ordnet Essfenster dann einem von drei Lebensmitteln zu (SVM).

Bewertet wird sessionübergreifend per Leave-One-Session-Out, weil die klassische
Bewertung pro Fenster mit 93,8 % deutlich überschätzt. LOSO liefert 88,7 % pro
Fenster; durch Mehrheitsabstimmung über alle Fenster einer Mahlzeit steigt die
Genauigkeit auf 95,0 %, also 76 von 80 Mahlzeiten. Skyr ist die am besten trennbare Klasse, während sich Apfel
und Kaugummi gegenseitig verwechseln.

Zusätzlich entstand eine Live-Webanwendung, die den Sensorstrom in Echtzeit
entgegennimmt, alle zwei Sekunden klassifiziert und Mahlzeiten über einen
Zustandsautomaten mit Mehrheitsvotum automatisch erkennt. Wichtigster offener
Punkt ist die Datenbasis: Sämtliche Aufnahmen stammen von einer Person, sodass die
Übertragbarkeit auf andere Nutzer*innen mit anderer Anatomie und anderen
Kaugewohnheiten noch ungeprüft ist.
