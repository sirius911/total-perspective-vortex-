# total-perspective-vortex-
Brain computer interface with machine learning based on electoencephalographic data.

## Physionet
https://physionet.org/content/eegmmidb/1.0.0/

## Synthèse de l'experience

Cet ensemble de données se compose de plus de 1500 enregistrements EEG d'une ou deux minutes, obtenus auprès de 109 volontaires, comme décrit ci-dessous.

## Protocole expérimental
Les sujets ont effectué différentes tâches de motricité et d'imagerie pendant que l'EEG à 64 canaux était enregistré à l'aide du système BCI2000 (http://www.bci2000.org). Chaque sujet a effectué 14 parcours expérimentaux : deux parcours de référence d'une minute (un avec les yeux ouverts, un avec les yeux fermés), et trois parcours de deux minutes pour chacune des quatre tâches suivantes :

1. Une cible apparaît à gauche ou à droite de l'écran. Le sujet **ouvre** et **ferme** le poing correspondant jusqu'à ce que la cible disparaisse. Ensuite, le sujet se détend.
2. Une cible apparaît à gauche ou à droite de l'écran. Le sujet **imagine** qu'il ouvre et ferme le poing correspondant jusqu'à ce que la cible disparaisse. Ensuite, le sujet se détend.
3. Une cible apparaît en haut ou en bas de l'écran. Le sujet **ouvre** et **ferme** les deux poings (si la cible est en haut) ou les deux pieds (si la cible est en bas) jusqu'à ce que la cible disparaisse. Ensuite, le sujet se détend.
4. Une cible apparaît en haut ou en bas de l'écran. Le sujet **imagine** qu'il ouvre et ferme les deux poings (si la cible est en haut) ou les deux pieds (si la cible est en bas) jusqu'à ce que la cible disparaisse. Le sujet se détend ensuite.

En résumé, les séries expérimentales ont été les suivantes :

1. Base, yeux ouverts
2. Base, yeux fermés
3. Tâche 1 (ouvrir et fermer le poing gauche ou droit)
4. Tâche 2 (imaginer ouvrir et fermer le poing gauche ou droit)
5. Tâche 3 (ouvrir et fermer les deux poings ou les deux pieds)
6. Tâche 4 (imaginer ouvrir et fermer les deux poings ou les deux pieds)
7. Tâche 1
8. Tâche 2
9. Tâche 3
10. Tâche 4
11. Tâche 1
12. Tâche 2
13. Tâche 3
14. Tâche 4

Les données sont fournies ici au format EDF+ (contenant 64 signaux EEG, chacun échantillonné à 160 échantillons par seconde, et un canal d'annotation).

Chaque annotation comprend l'un des trois codes (T0, T1 ou T2) :
- T0 correspond au repos
- T1 correspond à l'apparition d'un mouvement (réel ou imaginaire) :
    - du poing gauche (dans les séries 3, 4, 7, 8, 11 et 12)
    - des deux poings (séries 5, 6, 9, 10, 13 et 14)
- T2 correspond à l'apparition du mouvement (réel ou imaginé) :
    - du poing droit (pour les séries 3, 4, 7, 8, 11 et 12)
    - des deux pieds (dans les séries 5, 6, 9, 10, 13 et 14)

Dans les versions de ces fichiers au format BCI2000, qui peuvent être obtenues auprès des contributeurs de cet ensemble de données, ces annotations sont codées sous la forme de valeurs 0, 1 ou 2 dans la variable d'état TargetCode.

## Montage
Les EEG ont été enregistrés à partir de 64 électrodes selon le système international 10-10 (à l'exclusion des électrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9 et P10), comme le montre cette figure PDF. 
[64_channel_sharbrough.pdf](https://github.com/sirius911/total-perspective-vortex-/files/11384192/64_channel_sharbrough.pdf)
Les numéros figurant sous le nom de chaque électrode indiquent l'ordre dans lequel elles apparaissent dans les enregistrements ; il convient de noter que les signaux dans les enregistrements sont numérotés de 0 à 63, tandis que les numéros dans la figure vont de 1 à 64.

## Remerciements
Cet ensemble de données a été créé et versé dans la PhysioBank par Gerwin Schalk (schalk at wadsworth dot org) et ses collègues du BCI R&D Program, Wadsworth Center, New York State Department of Health, Albany, NY. W.A. Sarnacki a recueilli les données. Aditya Joshi a compilé l'ensemble des données et préparé la documentation. D.J. McFarland et J.R. Wolpaw étaient respectivement responsables de la conception expérimentale et de la supervision du projet. Ce travail a été soutenu par des subventions du NIH/NIBIB ((EB006356 (GS) et EB00856 (JRW et GS)).

https://github.com/thervieu/42total-perspective-vortex

##Pour la première installation veuillez supprimer le fichier utils/path.json, pour que le main crée les dossiers dans votre systeme
