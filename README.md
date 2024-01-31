# OC_Project8 (D. Desoubzdanne-Dumont le 31/01/2024)
## Projet OpenClassroom parcours Data Scientist
### Description du déploiement cloud et du Notebook DataBricks

Le choix a été fait de déployer le modèle sur la plateforme cloud Azure (Microsoft).
Plusieurs fichiers ou informations sont à noter :
- **les fichiers images tests (inputs)** sont stockés dans un container blob et téléchargeables à l'adresse suivante (fichier zip) : https://projet8images.blob.core.windows.net/zipfile/fruits.zip
- **le notebook Databricks** permettant d'excuter le script PySpark est consultable dans la branche databricks de ce repo, ou à l'adresse suivante : https://adb-8983437898642533.13.azuredatabricks.net/browse/folders/163836705254556?o=8983437898642533
- **le fichier csv (ouputs)** des données transformées après ACP est stocké dans un container blob et téléchargeable à l'adresse suivante : https://projet8images.blob.core.windows.net/mytables/table.csv
- le pdf ici présent est la présentation de ma soutenance pour ce projet : Desoubzdanne_Denis_3_presentation_122023.pdf



A noter que l'API doit être installée sous une **Web App Azure Linux en Python 3.11**.
L'URL de l'application est la suivante : **https://basicwebappvl.azurewebsites.net**. Bien entendu, pour qu'elle fonctionne, elle nécessite d'être activée!

Concernant le script de l'API, **différentes "routes" sont mises en place** pour interagir avec l'utilisateur de l'application (interface du conseiller clientèle).
**Différentes méthodes GET** pour obtenir :
- la liste des ids des nouveaux clients pour vérifier que le client est bien enregistré
- la liste des features qui sont utilisées pour la modélisation
- envoyer le numéro du client sélectionné et recevoir ses informations (format json, données brutes)
- envoyer le numéro du client sélectionné et recevoir ses informations (format json, données transformées)
- envoyer le json d'un client (données transformées) et recevoir sa prédiction de classement
- envoyer le numéro d'un client et recevoir son explicabilité locale (Shap values)
- envoyer le nom d'une feature et recevoir les valueurs de cette features pour les individus classés 0 et ceux classés 1


### Tests et workflows
La plateforme d'hébergement choisie est **Azure Web App**.  
Afin de mettre en place un **processus d'intégration/amélioration continues**, le code est hébergé sur des **repo Git distants** et le déploiement réalisé par les **actions Github** communiquant avec l'hébergeur. De cette manière, des modifications peuvent être réalisées puis contrôlées d'abord dans un **environnement virtuel local** défini, puis éventuellement déployées dans une **nouvelle branche** avant d'être envoyées à la branche principale.  
Il a été décidé de séparer complètement le déploiement de l'API de celui de l'application et des projets Github distincts ont été créés :
- pour l'API : https://github.com/DDesou/Projet7_VL
- pour l'interface utilisateur : https://github.com/DDesou/Projet7_Streamlit

L'API est déployée (ou modifiée) après que les tests unitaires aient été validés (**tests Pytest** intégrés dans le déploiement). Les scripts des test effectués sont contenus dans le fichier test_main.py.  
L'application Streamlit est déployée (ou modifiée) après s'être assuré que toutes les modifications ont été d'abord enregistrées, afin d'éviter les bugs éventuels au moment du changement de version.
