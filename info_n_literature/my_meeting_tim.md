## worum geht das

OCGNN 

Hypersphere

node embedding

idee: voroptimierung in vorwissen

wurde sogar vorschlagen auf graph ebene

idee: indikatoren oder logical contrains quasi logische bedingungen
based on prior knowledge

HAUPTidee: wir erweitern die OCGNN 

in dem paper von tim wird genau eine art von ocgnns benutzt. nimm die


## aus dem meeting email:
die vorhin besprochene Kernidee ist die Erweiterung von One-Class Graph Anomaly Detection (OCC) auf Graph-Level unter Einbezug logischer Constraints. Ziel solcher Graph OCC Verfahren ist es also, aus einer Menge normaler Graphen ein Modell zu lernen, das in der Lage ist, normale von  anomalen Graphen zu unterscheiden (siehe OCGIN/OCGTL Paper für Details). In unserem Ansatz wollen wir die gelernten Repräsentationen zusätzlich durch logische Constraints leiten, die Vorwissen in Form von Logikformeln kodieren. So können wir das Modell gezielt beeinflussen, indem bestimmte Graphen – durch das (teilweise) Erfüllen dieser Formeln – als eher normal, anomal oder ähnlich/unähnlich zu einander eingestuft werden.

Technisch lässt sich so etwas zum Beispiel durch eine Anpassung der Verlustfunktion oder eine Erweiterung der Modellarchitektur erreichen, sodass die Constraints während des Trainings und/oder der Inferenz berücksichtigt werden.

Als grundlegende Methoden würde ich mit OCGIN und OCGTL (ein Ensemble aus mehreren OCGIN-Instanzen) arbeiten. Für den Einstieg schlage ich vor, dass du das öffentliche OCGTL-Repository lokal klonst, es in ein eigenes Repository überführst, die Experimente für (1) OCGTL und (2) ein einfaches OCGIN bei dir lauffähig bekommst und dich generell in die Codebasis einliest sowie mit der Methoden aus den beiden Papern weiter vertraut machst.
Paper OCGIN: https://arxiv.org/pdf/2012.12931
Paper OCGTL: https://arxiv.org/abs/2205.13845
Implementierung OCGTL: https://github.com/boschresearch/GraphLevel-AnomalyDetection

Für erste Ideen zur Integration logischer Constraints könnten wir z.B. erstmal annehmen, dass zu jedem Graphen ein zusätzlicher Constraint-Vektor mit Werten je in [0,1] vorliegt (der nicht direkt Teil des Graphen ist, welcher vom Modell verarbeitet ist). Dessen Einträge könnten dann angeben, in welchem Maß die einzelnen Constraints erfüllt sind (z.B. der Graph beinhaltet ein bestimmtes Graphlet häufig). Auf dieser Grundlage lässt sich dann überlegen, wie diese Vektoren die Repräsentationen während Training und Inferenz beeinflussen können.

Bei der Literaturrecherche würde ich bezüglich Graph Anomaly Detection erstmal bei OCGIN und OCGTL bleiben und untersuchen, wie logische Constraints integriert werden könnten.

Ein paar Keywords z.B. für Google Scholar dazu:
- (Logical) Constraints in Deep Learning
- (Logical) Constraints in Graph Neural Networks
- Neuro Symbolic AI

Eine ältere (2022), aber gute erste Übersicht: https://arxiv.org/pdf/2205.00523
Aktuelle Arbeiten von der Erstautorin könnten interessant sein, z.B hier: https://arxiv.org/pdf/2402.04823 (siehe dort auch related Work)

Ich würde also vorschlagen, zunächst OCGTL / OCGIN vollständig zu verstehen und lauffähig zu machen und parallel erste grobe Ideen zur Integration der logischen Constraints zu sammeln (also erstmal zu den Constraints High-Level Ideen sammeln).


## Buecher

da du ja meintest, dass du Graphen und GNNs generell ziemlich spannend findest, anbei ein paar Buchempfehlungen, falls du dazu was zu lesen suchst:

Ein cooles Buch zu Graphen im allgemeinen:
https://networksciencebook.com/

Ein ebenfalls sehr cooles Buch wenn man GNNs genauer verstehen will:
https://www.cs.mcgill.ca/~wlh/grl_book/

Viele Grüße
Tim

